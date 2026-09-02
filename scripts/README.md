# `scripts/`

One-shot generation and setup scripts. Nothing here is imported by the served
system at runtime — these produce the corpus and populate the database, then
get out of the way.

| File | Job |
| --- | --- |
| `generate_synthetic_data.py` | Produce every raw tabular CSV. |
| `generate_voice_audio.py` | Synthesise the WAVs behind the voice-sample index. |
| `seed_db.py` | Build the SQLite databases and the pseudonymisation map from the raw CSVs, and populate demo user accounts. |
| `run_pipeline.py` | Run ingestion → features → signals → scoring end to end and write processed outputs. |

Run them in that order. Each is idempotent — re-running overwrites its own
outputs and touches nothing else.

---

## `generate_synthetic_data.py`

### What it does

Produces the entire synthetic HR corpus: units, personnel roster, leave
records, deployment history, monthly duty logs, transfers, training records,
the voluntary voice-sample index, and the synthetic ground-truth labels.

### Inputs and outputs

**Input:** nothing but `backend/config/settings.py`.

**Output:** nine CSVs in `data/raw/`.

| Table | Grain | Rows (default settings) |
| --- | --- | --- |
| `unit_capacity.csv` | one row per unit | 16 |
| `personnel.csv` | one row per person | 800 |
| `leave_records.csv` | one row per leave spell | ~13,900 |
| `deployment_history.csv` | one row per deployment spell | ~4,400 |
| `duty_logs.csv` | one row per person-month | 19,200 |
| `transfer_records.csv` | one row per transfer | ~940 |
| `training_records.csv` | one row per course attended | ~8,500 |
| `voice_samples.csv` | one row per voluntary check-in | 100 |
| `ground_truth_labels.csv` | one row per person-snapshot | 4,800 |

### How it fits into the pipeline

It is the head of everything. `backend/ingestion/hr_loader.py` reads these
CSVs; nothing upstream of this script exists.

### Design decisions and assumptions

**Correlation structure is the whole point.** The generator builds latent
drivers *first* — a unit-level `operational_tempo` and a person-level
`exposure_propensity` — and derives duty hours, leave availment, deployment
length, posting type and staffing shortfall *from* them. That is why long
deployment, low leave uptake and high duty hours co-occur in the output
instead of being independently randomised. Independently randomised columns
would produce a dataset in which nothing is learnable and a model comparison
that measures nothing.

**Every anchored figure carries its source, in `settings.py`.** The script
imports the constants rather than restating numbers, so the data and the data
dictionary cannot drift apart. The anchors actually used:

| Anchor | Source | Where it lands |
| --- | --- | --- |
| 100 days entitlement, ~75 availed, ~4.5% avail in full | MHA-reported CAPF leave figures | `build_leave_records` |
| 12–14 h/day, jawan ranks, high-operational units | Parliamentary / JPC findings on CRPF | `build_duty_logs` |
| 48 h/week (~208 h/month) | Indian labour-law standard | Reference baseline for workload deviation |
| 80%+ unable to avail weekly offs | JPC finding on CRPF | `build_duty_logs` |
| Tenure-based hard-area rotation | CAPF posting policy (policy, not a statistic) | `build_personnel` |

Transfer frequency is the deliberate exception: **no authoritative public
figure exists**, so it is generated from an assumed Poisson mean and is
flagged as an assumption both in `settings.py` and in `docs/data_dictionary.md`.
It is never presented as a sourced number.

**Why leave spells are placed by partitioning, not by a random walk.** The
first implementation walked forward from the start of the history window
drawing random gaps between spells. It systematically under-delivered against
the annual target — it ran out of calendar before it ran out of leave budget —
and produced a mean of 32 days/year against a sourced 75. The current version
partitions the window into one segment per spell and places a spell inside
each, which reproduces the sourced total by construction. Spell lengths come
from a Dirichlet split of the annual target, then get clipped to a plausible
range **and rescaled**, because clipping alone silently inflates the total
back above the anchor.

**Why the full-entitlement group is an exact count, not a coin flip.** A
per-person Bernoulli draw at p = 0.045 reproduces the figure only in
expectation; on the project's fixed seed it landed at 7.2%, which would have
put a number in the corpus that contradicts the figure the documentation
cites. The group is now selected as an exact shuffled count.

**Where operational tempo enters leave.** Tempo suppresses the *amount* of
leave (up to 15%) and separately biases *when* within each segment a spell
falls, pushing spells earlier and so lengthening the dry period before the
snapshot date. That second mechanism is what makes `days_since_last_leave`
correlate with unit conditions rather than floating free of them.

**The ground-truth label is a formula, and the code says so loudly.**
`latent_welfare_risk()` computes the synthetic target. It exists only because
this is a synthetic corpus; in a real deployment the training label would come
from validated welfare assessments conducted by qualified personnel. Every
weight in it is an assumption. It is deliberately non-linear — it contains an
overwork × recovery-deficit interaction and a saturating deployment term — for
two reasons: sustained overwork without recovery genuinely does compound, and
a purely linear target would make the model comparison in `ml/evaluation/` a
formality rather than a real test. A Gaussian noise term is included so no
model can reach R² = 1.0; a dataset a model fits perfectly proves nothing.

**Nothing in the served system imports this function.** It is generation-side
only.

### Anchor self-check

Running the script prints generated-vs-target values for every sourced anchor.
Current output:

```
mean leave days/year          73.5   target ~75
full-entitlement fraction    0.045   target ~0.045
jawan mean daily duty hrs    13.18   target (12.0, 14.0)
mean weekly-off availment    0.281   (JPC: 80%+ cannot avail)
```

If a change to the generator breaks an anchor, this banner shows it
immediately rather than letting the corpus quietly drift away from the figures
the documentation cites.

---

## `generate_voice_audio.py`

### What it does

Synthesises one 16 kHz mono WAV per row of `voice_samples.csv`, with acoustic
properties set from that row's latent strain value.

### Inputs and outputs

**Input:** `data/raw/voice_samples.csv`.
**Output:** `data/raw/voice_audio/<sample_id>.wav`, 100 files, ~15 MB total.

### How it fits into the pipeline

`backend/ingestion/voice_loader.py` reads these files;
`backend/voice_pipeline/` analyses them.

### Design decisions and assumptions

**These are not recordings of speech.** They contain no words, no language and
no content — they are glottal impulse trains passed through formant
resonators. That is appropriate, because the pipeline consuming them is
forbidden from looking at content. It measures only *how* a voice is produced.
There is no transcription or speech-to-text anywhere in this system.

**Jitter and shimmer are injected as the quantities they actually are.**
Jitter perturbs the length of each glottal *period*; shimmer perturbs each
pulse's *amplitude*. The extractor in `backend/voice_pipeline/` measures those
same quantities back, so the test in `tests/test_voice_pipeline.py` is a real
round-trip check rather than a comparison against a stored number.

**Per-person habitual pitch is drawn once and held stable** across that
person's check-ins. Without it, every sample would look like a deviation from
every other and the *personal* baseline logic — which is the entire point of
`voice_baseline.py` — would never be exercised.

**The direction of every strain → acoustics relationship is an assumption**,
chosen to be consistent with commonly described features of pressured speech
(raised and flatter pitch, faster rate, shorter pauses, greater cycle-to-cycle
perturbation). No claim is made that the coefficients are clinically
validated, and nothing in the served system treats the voice signal as a
diagnosis — it is one optional input among several, always flagged as
voluntary, and a person who never opts in is scored by exactly the same path
as one whose sample is simply missing.
