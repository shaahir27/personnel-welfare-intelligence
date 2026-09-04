# `backend/medical/`

## What this module does

Runs the **medical booking domain**: a doctor roster, published availability,
appointments booked by personnel themselves, and one prescription note per
completed visit.

| File | Job |
| --- | --- |
| `identity.py` | Keep the medical and welfare identifier namespaces disjoint, and refuse the wrong one at the boundary. |
| `store.py` | The SQLite store — roster, slots, appointments, notes. |

Routes live in `backend/api/routes/medical.py`; the roster and diary are seeded
by `scripts/seed_medical_roster.py`.

## The gap this fills

`intervention_library.json` had no medical intervention at all. The closest was
`counseling_referral`, whose own description says *"this is a welfare visit, not
an inquiry"* — deliberately non-clinical. So a High-band case with a real
physical dimension was only ever pointed at a welfare conversation. Sustained
duty load with no recovery has effects — sleep, blood pressure, musculoskeletal
strain — that a welfare conversation is not the right instrument for.

The library now carries `medical_referral`, and this is what it refers to.

---

## Why this is a second subsystem rather than a feature

Three reasons, in increasing order of how much they matter.

**It is the first transactional thing here.** Everything else is "the batch
pipeline computes once, the API serves static JSON". A slot is either taken or
it is not, two people must not both get it, and the answer changes between one
request and the next. That needs a writable store with a uniqueness constraint
the database enforces.

**It needs two roles that do not exist on the welfare side.** `medical_officer`
and `establishment_admin` are not welfare roles and hold no welfare permission.
Crucially, `welfare_officer` and `commander` hold no permission *here* —
medical confidentiality is a stricter and separate boundary than welfare-risk
confidentiality, so the roles crossing it are separate too.

**It needs a real identity, and the welfare system deliberately has none.**
This is the one worth understanding properly.

### The identity problem, and how it is solved

The welfare system holds everything against `pseudonym_id` — `PSN` plus 16 hex
characters, an HMAC of the person's `personnel_id` keyed by a salt kept in a
vault the API server never opens. That is what makes the privacy claim true: if
the analytics store is taken, it yields pseudonymous records and nothing else.

You cannot schedule a human being for a real appointment against a pseudonym
nobody in the clinic can resolve. So the medical domain uses the service
identity, and the two domains use **disjoint identifier namespaces**:

| Domain | Shape | Example |
| --- | --- | --- |
| Welfare | `PSN` + 16 hex | `PSNa1b2c3d4e5f60718` |
| Medical | `P` + 5 digits | `P00123` |

That is not a naming convention — it is the enforcement point. Every medical
route runs the incoming token subject through
`identity.require_service_identity()`, so a pseudonym carried in by accident is
refused with its own message rather than being quietly used as a key. In the
other direction a service identity fails `rbac.require_self` on every welfare
route, because it will not equal any pseudonym.

**What this claims, precisely.** It does not make the link impossible for
someone holding the vault — the vault exists so an authorised officer *can*
find a person, through `scripts/reidentify.py`, with a stated purpose, audited.
What it makes impossible is the link happening **by accident**: a helper that
starts passing a subject through, a join written in a hurry, a route reusing the
wrong path parameter. Those are how this kind of separation actually fails, and
a namespace check catches all of them at the boundary.

---

## The three rules that make this safe

Each one maps to something the codebase already treats as load-bearing.

### 1. Booking is open to everyone, always — never gated by risk

If only High-band people could book, the booking button would itself disclose
the band to anyone watching. There is no risk check in any route here, and
there could not be: this package cannot read the processed store.

The same reasoning rules out a priority queue. Offering faster appointments to
higher-risk people would leak the score through booking order. `open_slots()`
orders by start time and nothing else, and says so in its docstring so nobody
adds an ordering later thinking it is an improvement.

### 2. The doctor never sees the risk score or the signals

A clinician treating somebody differently because "the algorithm flagged them"
is exactly the stigmatisation PS technical challenge #2 warns about. There is no
route that returns a welfare score to a `medical_officer`, and the schedule
payload says so in its own `scope_note`.

### 3. Sharing context is opt-in, per appointment, off by default

Not a profile setting, not a checkbox ticked once and forgotten: a decision made
at the moment of booking, for that visit. `share_context` must arrive as boolean
`true`; absent, null or any non-boolean is treated as "no", because consent is
something a person gives and never something a malformed field gives on their
behalf. `context_note` is discarded outright unless the flag is set, so a note
cannot be attached to an appointment the person did not consent to share.

Where a person did not share, the field is **absent** from the doctor's view
rather than empty — an empty "shared context" box on a screen reads as the
person having nothing to say, which is a different statement from their not
having been asked to.

This is where PS technical challenge #4 (privacy and consent) is actually
honoured rather than claimed.

---

## Access rules

| Role | Sees |
| --- | --- |
| `personnel` | The roster, open slots, their own appointments and their own notes. |
| `medical_officer` | Their own schedule, and one note per completed visit. Never a welfare score, band or indicator. |
| `establishment_admin` | The doctor roster and availability. Never appointment content, a reason, or any note. |
| `welfare_officer` | **Nothing here.** |
| `commander` | **Nothing here.** |

The last two are the point, and they are not configuration: no handler in
`routes/medical.py` lists those roles, so a welfare officer holding a perfectly
valid token gets a 403 from every route in the domain. `settings`
also lists `appointment_id`, `prescription_id` and `doctor_id` in
`COMMANDER_FORBIDDEN_FIELDS`, so even a payload built by mistake could not be
served to a commander.

---

## Storage

`data/medical_records.sqlite3` — its own file, for the same reason the identity
vault has one. Nothing here is ever joined against `identity_map.sqlite3` or
`data/processed/`, and no module in this package imports the models, the
behavioral engine, the processed store or the pseudonym vault.

```
doctors             (doctor_id, name, specialty, unit_id, subject, is_active)
availability_slots  (slot_id, doctor_id, starts_at, minutes, is_booked)
appointments        (appointment_id, personnel_id, doctor_id, slot_id, status,
                     reason, shared_context, context_note, created_at)
prescriptions       (prescription_id, appointment_id, note_text, issued_by, issued_at)
```

`tests/test_medical.py` asserts the import graph, so the separation stays true
rather than being a paragraph somebody once wrote.

### The concurrency detail

`book()` claims the slot with a conditional `UPDATE ... WHERE is_booked = 0` and
checks the affected row count, inside the same transaction that writes the
appointment. Two requests racing for the last slot cannot both win: the second
changes zero rows and the whole unit rolls back. Reading availability first and
then inserting would look identical and lose that race — a real double-booking
in a real clinic.

---

## Deliberately not built

Scope discipline, so this does not quietly become a hospital system.

| Not built | Why |
| --- | --- |
| Full medical history / EHR | One note per visit is the whole scope. A second note per visit is the first step to a chart, and `issue_prescription` refuses it. |
| Priority booking for high-risk personnel | Leaks the score through booking order. |
| Doctor-side risk visibility by default | Stigmatisation, PS challenge #2. |
| Chat, video consultation, billing, insurance | No PS requirement for any of it. |
| PDF prescription generation | A note is text. A PDF is a document-rendering dependency for no added clinical content. The route returns the note; a client that wants to print it can. |

---

## Running it

```bash
python scripts/seed_medical_roster.py --reset     # 3 doctors, ~2 weeks of slots
python -m backend.api.main
```

Demo accounts (published in `README.md` with the rest):

| Account | Password | Role |
| --- | --- | --- |
| `doctor` | `medical-officer-demo` | `medical_officer` (subject `MO-DEMO-01`) |
| `establishment` | `establishment-demo` | `establishment_admin` |

A person signs in to this domain with the `personnel` account and a **service
identity** as the subject (`"subject": "P00123"`), not a pseudonym — the same
account signs into the welfare app with a pseudonym, and the two sessions
cannot reach each other's data because neither namespace is accepted by the
other's routes.
