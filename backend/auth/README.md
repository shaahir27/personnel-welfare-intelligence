# `backend/auth/`

## What this module does

Establishes **who is making a request** and **what they are allowed to receive**.
Those are two different jobs and this module keeps them separate on purpose.

| File | Job |
| --- | --- |
| `credentials.py` | Check a username and password against stored PBKDF2 hashes; return the account's role. |
| `demo_accounts.json` | The three demo accounts. Hashes and salts only — no password is stored. |
| `jwt_handler.py` | Issue and verify HS256 tokens. Stdlib `hmac`+`base64`, or PyJWT when importable. |
| `rbac.py` | Read the acting principal, enforce per-role scope, and refuse any commander payload carrying individual data. |

## Inputs and outputs

**In:** request headers (`Authorization: Bearer <jwt>`), and a username/password
pair at the login route.

**Out:**

- `credentials.authenticate(username, password) -> Account`
- `jwt_handler.create_token(subject, role, expires_in) -> str`
- `jwt_handler.verify_token(token) -> JwtClaims`
- `rbac.principal_from_headers(headers) -> Principal`
- `rbac.require_role(...)`, `rbac.require_self(...)` — raise `AuthorisationError`
- `rbac.assert_commander_safe(payload)` — raise `IndividualDataLeak`

## Pipeline position

```
POST /api/auth/login ──▶ credentials.authenticate ──▶ jwt_handler.create_token ──▶ token
                                                                                    │
                              every role-scoped route ──▶ rbac.principal_from_headers
                                                            │
                                        require_role / require_self / assert_commander_safe
```

---

## Design decisions

### Why authorisation was built before authentication

An authentication bug lets the wrong person in as a commander. An authorisation
bug lets a commander see individuals. The second is the one that would break
this system's promise to the people it monitors, so it was built first, is
enforced in three redundant layers, and is the thing the test suite proves.

Both now exist. The ordering is still the right one and is worth stating,
because it is why the guarantee does not rest on the login being correct.

### Why the plain role header defaults to off

`rbac.principal_from_headers` will read a plain `X-Pwiews-Role` header when
`PWIEWS_DEBUG_AUTH=1`. That means any caller can claim any role by typing it,
which is fine at a shell and nowhere else.

It used to default to **on**, because the frontends had no way to obtain a
token — there was no login route. That made the whole JWT layer decorative: it
could verify tokens nobody was sending. Adding `POST /api/auth/login` removed
the reason, so the default is now off. A checkout does not start in the open
state.

### Why the demo passwords are published

There are three accounts, they are in `README.md`, and the corpus is 800
synthetic people. A reviewer has to be able to open both apps. Publishing the
passwords is honest about what this is; pretending a demo has a real identity
model would not be.

What is *not* pretend: the check itself. PBKDF2-HMAC-SHA256, 200,000
iterations, a distinct random salt per account, `hmac.compare_digest`
comparison, and an unknown username costs the same work as a known one with the
wrong password so response timing does not reveal which accounts exist.

### Why the personnel account can choose its subject

The demo has 800 synthetic people and no sign-up. The personal app has to be
able to look at more than one of them, so the personnel account carries
`may_choose_subject: true` and passes a pseudonym at login. The token it gets
back is scoped to that one pseudonym and the server enforces that scope on
every personal route — picking somebody is *signing in as them*, not a way to
read their record while signed in as somebody else.

That capability sits on the account record rather than in the route handler, so
it is greppable, it is false for every other account, and removing the demo
means deleting one account rather than finding a special case inside a handler.

### What a deployment replaces

`credentials.py` and `demo_accounts.json`, together. The force's own directory
service goes behind `authenticate()`, which keeps returning an `Account`.
Nothing else in the module changes, and no route changes at all.

Two other things must change with it, both flagged where they live:

- `settings.JWT_SECRET_KEY` is a constant in this build. It has to become a
  runtime-injected secret that never reaches source control.
- `PBKDF2_ITERATIONS` is 200,000 here so a login stays responsive during a
  demonstration. OWASP's current guidance for PBKDF2-HMAC-SHA256 is higher.

---

## Known limits

- **No refresh, no revocation, no session list.** A token is valid until it
  expires (`settings.JWT_EXPIRY_MINUTES`). There is nowhere to say "end this
  session now", which a real deployment needs.
- **`/api/meta` and `/api/demo/identities` are unauthenticated.** The personal
  app needs the identity list *before* it can sign in as anybody. Neither
  returns a name or a service number, and `/api/demo/identities` is a demo
  affordance that disappears with real sign-ups — but it does expose a risk
  band per pseudonym to an unauthenticated caller, which is worth knowing.
