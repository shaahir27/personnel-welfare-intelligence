# pwiews — PS Alignment Matrix

Maps every component of the SIH26186 problem statement to its implementation
in this codebase. Honest about partial coverage where it exists.

---

## A. Expected Solution Components

| PS Component | Location | Coverage |
|---|---|---|
| **Personnel Wellness Monitoring Dashboard** | `frontend/officer-dashboard/` | ✅ Full — 4 screens: queue, case detail, unit overview, near-miss panel |
| **Mobile-based Wellness and Self-Assessment App** | `frontend/personal-app/`, `backend/api/routes/personal.py` | ✅ Full — 4 screens; the loop closes: `GET /check-in` serves the tailored questions, `POST /check-in` stores the answers, `GET /check-in/history` reads them back, and each answer is compared against the indicator it was tagged to (`self_report_consistency.py`) |
| **Predictive Behavioral Analytics Engine** | `backend/behavioral_engine/behavioral_signals.py` | ✅ Full — 9 signals, 0–100 normalised, welfare framing |
| **Stress/Burnout Risk Prediction Model(s)** | `backend/models/` | ✅ Full — 8 candidates trained, Gradient Boosting selected; deployed model carries split-conformal calibrated intervals (±12.8 at 90%, 92.4% empirical coverage). **R²=0.760 and that interval are both measured against the synthetic label** — i.e. against the generator's formula, not against real welfare outcomes. See `docs/model_comparison_report.md` §5. |
| **Welfare Intervention Recommendation System** | `backend/recommendation_engine/`, `backend/db/intervention_log.py`, `backend/medical/` | ✅ Full — 10 pre-approved interventions, rule-based, deterministic, rendered on the officer case detail screen. Which one was taken is now recorded (`POST /api/officer/case/{id}/intervention`), and `medical_referral` points at a real booking subsystem where a person can book a unit doctor themselves. No effectiveness figure is computed from the outcome rows, and `docs/model_comparison_report.md` §5b explains why that would be unsupportable on a synthetic corpus. |
| **Role-based Access Control** | `backend/auth/`, `backend/medical/` | ✅ Full — five roles in two groups. `POST /api/auth/login` issues HS256 tokens against PBKDF2 hashes, verified on every role-scoped route; `POST /api/auth/logout` revokes a token so a session can be ended before it expires. `welfare_officer` / `commander` (welfare) and `medical_officer` / `establishment_admin` (medical) hold **no** permission in each other's domain — a stricter boundary, not another tier of the same one. |
| **Automated Alerts** | `backend/alerts/alert_rules.py` | ✅ Full — 4 alert rules, graduated escalation, delivered to the personal app and the officer case detail as two separate feeds |
| **Data Anonymisation and Secure Storage** | `backend/preprocessing/pseudonymize.py`, `backend/db/access_log.py` | ✅ Full — HMAC pseudonymisation, separate identity vault, audited re-id path, and a record-access log of every officer open (pseudonym only, never a name) surfaced to the individual as counts |

---

## B. Technical Challenges — Mitigations

| PS Technical Challenge | Mitigation in this codebase |
|---|---|
| **#1: Data integration from heterogeneous HR sources** | `backend/ingestion/` — schema validation, non-repairing validators reject ill-formed rows at ingestion. Pseudonymisation happens in the same pass. |
| **#2: Avoiding stigmatisation** | `settings.SIGNAL_HUMAN_LABELS` — every signal shown to a user is framed as an organisational circumstance, never a personal failing. Individual-vs-systemic classification (G) exists specifically to prevent the system turning a unit's roster problem into a list of individuals. |
| **#3: Minimising false positives / false negatives** | **A measured false-positive rate, not just a mechanism.** The corpus contains a gray-area group — 5% of personnel who look strained on every raw indicator for a documented benign reason, and whose label is dampened accordingly. Nothing the model sees identifies them (`benign_profile` is generation-only, and a test asserts it never reaches the feature matrix). **0 of 40 were classified High, against 16.2% across the rest; restricted to people the model was never fitted on, 0 of 8 against 20.4%.** Eight is a small number and §5b says so rather than overselling it. **Calibrated intervals:** every score carries a split-conformal range (±12.8 points, 90% coverage guaranteed in finite samples, verified 92.4% on unseen people, against the synthetic label); a case whose range crosses a cutoff is marked *borderline*, so a provisional High is never presented as a settled one. Separately, `barely_over_cutoff` marks a case whose point score is within 3 points of the cutoff that admitted it — "close" is a different statement from "uncertain" and they are labelled separately. Wide Moderate band (40–65) intentionally reduces false negatives. Officer visibility gated by one rule — High, or persistent *and rising* Moderate. Low-confidence alerts suppressed. See `docs/model_comparison_report.md` §5b. |
| **#4: Privacy and consent** | RBAC with three redundant layers for commander data leakage. `assert_commander_safe()` inspects every commander payload at all nesting depths. Officer reads of an individual's routes are gated by the escalation rule and written to the access log; the individual sees who (by role) opened their record and when. Voice pipeline: no transcription, acoustic only, opt-in. Check-in answers validated against the question bank and never used for scoring — so "answering is optional and does not affect your score" is structurally true, not a promise. **Per-appointment consent:** sharing welfare context with a doctor is opt-in, off by default, decided at the moment of booking rather than as a remembered setting; the note is discarded server-side unless the flag is explicitly `true`. The medical and welfare domains use disjoint identifier namespaces so neither can be joined to the other by passing an identifier along. The audited re-identification path is now actually exercised (`scripts/reidentify.py`) — it had never been called, so the audit table had zero rows and the control was a claim rather than a control. |
| **#5: Scalability and real-time processing** | Batch architecture: pipeline writes `data/processed/*.json` once; API reads at request time with O(1) lookup via `cases_by_id`. No compute at request time except the what-if simulation. |
| **#6: Explainability** | Exact Shapley by full coalition enumeration (2¹¹ = 2048 coalitions for 11 features). Local accuracy asserted on every call. Top-3 factors with human-readable labels shown to both individual and officer. Precomputed for **every** person at the latest snapshot, not a top slice — an explanation only some people can see is not an explainable system. |

---

## C. Ethical Constraints (from prompt.md)

| Constraint | Where enforced |
|---|---|
| No LLM/generative-AI anywhere in scoring, classification, or recommendation | `backend/models/` uses sklearn regressors. `backend/recommendation_engine/action_mapper.py` is a pure filter over a static JSON library. No text generation anywhere. |
| Commander routes must never return individual-identifiable data | `backend/auth/rbac.py:assert_commander_safe()` — server-side recursive scan. `tests/test_rbac_api.py` — automated proof. |
| Voice pipeline analyses HOW someone speaks, never WHAT they say | `backend/voice_pipeline/` — no transcription module exists. Extracts: F0, speaking rate, pause ratio, intensity CV, jitter, shimmer. `tests/test_voice_pipeline.py` — structural proof. |
| Sourced figures must be commented with source | `backend/config/settings.py` — every constant carries either `SOURCE:` or `ASSUMPTION:`. |

---

## D. Files Not in scope / explicitly deferred

| Item | Status | Reason |
|---|---|---|
| Voice upload API endpoint | Not built | Record button disabled in frontend; acoustic pipeline runs on batch corpus |
| SMS/email alert delivery | Not built | Out of scope per `settings.ALERT_CHANNELS = ("in_app",)` |
| Full DB layer (SQLAlchemy) | Partial | API serves precomputed JSON. SQLite holds the pseudonym vault, the record-access log, the intervention log, the medical booking store and the token denylist — five files, each at its own trust boundary. Self-assessment answers go to an append-only JSONL file |
| Intervention **outcome** tracking | Built | `POST /api/officer/case/{id}/intervention` records which action was taken, with four statuses that name what the welfare process did rather than what the person did |
| Intervention **effectiveness** analysis | Deliberately not built | On a synthetic corpus every snapshot after any intervention still comes from the same generator formula, so a before/after chart would be noise presented as evidence — or, if the generator were taught to make interventions "work", a demonstration of something scripted in. A test pins the absence so adding one is a deliberate act. `docs/model_comparison_report.md` §5b |
| Token revocation | Built | `POST /api/auth/logout` revokes the token; every later request carrying it is refused. Session *listing* and refresh are still absent |
| HR-side external validation | Harness built, data not collected | `validation/` renders case profiles for human rating and analyses what comes back (inter-rater agreement first, then model-vs-consensus Spearman). Needs calendar days of collection, not work hours |
| Daily-grain duty roster | Not built | Would make `duty_hours_change_ratio`, entropy drift and burstiness meaningful; currently duty is monthly and week-scale windows are pro-rated estimates. The largest single item still open |
| PyPI packages: FastAPI, shap, librosa, XGBoost | Using stdlib/sklearn equivalents | No package-registry access during build; equivalents documented in STATUS.md |
