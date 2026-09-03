# pwiews — PS Alignment Matrix

Maps every component of the SIH26186 problem statement to its implementation
in this codebase. Honest about partial coverage where it exists.

---

## A. Expected Solution Components

| PS Component | Location | Coverage |
|---|---|---|
| **Personnel Wellness Monitoring Dashboard** | `frontend/officer-dashboard/` | ✅ Full — 4 screens: queue, case detail, unit overview, near-miss panel |
| **Mobile-based Wellness and Self-Assessment App** | `frontend/personal-app/` | ✅ Full — 4 screens: summary, history, check-in, privacy transparency |
| **Predictive Behavioral Analytics Engine** | `backend/behavioral_engine/behavioral_signals.py` | ✅ Full — 9 signals, 0–100 normalised, welfare framing |
| **Stress/Burnout Risk Prediction Model(s)** | `backend/models/` | ✅ Full — 8 candidates trained, Gradient Boosting selected; deployed model carries split-conformal calibrated intervals (±9.9 at 90%, 91.5% empirical coverage). **R²=0.821 and that interval are both measured against the synthetic label** — i.e. against the generator's formula, not against real welfare outcomes. See `docs/model_comparison_report.md` §5. |
| **Welfare Intervention Recommendation System** | `backend/recommendation_engine/` | ✅ Full — 8 pre-approved interventions, rule-based, deterministic, rendered on the officer case detail screen |
| **Role-based Access Control** | `backend/auth/` | ✅ Full authorisation; authentication complete — `POST /api/auth/login` issues HS256 tokens against PBKDF2 hashes, verified on every role-scoped route |
| **Automated Alerts** | `backend/alerts/alert_rules.py` | ✅ Full — 4 alert rules, graduated escalation, delivered to the personal app and the officer case detail as two separate feeds |
| **Data Anonymisation and Secure Storage** | `backend/preprocessing/pseudonymize.py`, `backend/db/access_log.py` | ✅ Full — HMAC pseudonymisation, separate identity vault, audited re-id path, and a record-access log of every officer open (pseudonym only, never a name) surfaced to the individual as counts |

---

## B. Technical Challenges — Mitigations

| PS Technical Challenge | Mitigation in this codebase |
|---|---|
| **#1: Data integration from heterogeneous HR sources** | `backend/ingestion/` — schema validation, non-repairing validators reject ill-formed rows at ingestion. Pseudonymisation happens in the same pass. |
| **#2: Avoiding stigmatisation** | `settings.SIGNAL_HUMAN_LABELS` — every signal shown to a user is framed as an organisational circumstance, never a personal failing. Individual-vs-systemic classification (G) exists specifically to prevent the system turning a unit's roster problem into a list of individuals. |
| **#3: Minimising false positives / false negatives** | **Calibrated intervals:** every score carries a split-conformal range (±9.9 points, 90% coverage guaranteed in finite samples, verified 91.5% on unseen people, against the synthetic label); a case whose range crosses a cutoff is marked *borderline* on both screens and in the officer alert, so a provisional High is never presented as a settled one. Wide Moderate band (40–65) intentionally reduces false negatives. Officer visibility gated by one rule — High, or persistent *and rising* Moderate — which cut the queue from 619 to 159 of 800. Low-confidence alerts suppressed for officer/commander. Handling note appended to every case detail response. |
| **#4: Privacy and consent** | RBAC with three redundant layers for commander data leakage. `assert_commander_safe()` inspects every commander payload at all nesting depths. Officer reads of an individual's routes are gated by the escalation rule and written to the access log; the individual sees who (by role) opened their record and when. Voice pipeline: no transcription, acoustic only. Voice is opt-in. Check-in answers validated against the question bank and never used for scoring. |
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
| Full DB layer (SQLAlchemy) | Partial | API serves precomputed JSON; SQLite holds the pseudonym vault and the record-access log. Self-assessment answers go to an append-only JSONL file |
| Intervention outcome tracking | Not built | Recommendations are shown but nothing records what was done or whether it helped — so the system cannot learn which interventions work |
| Token revocation / refresh | Not built | Tokens expire and cannot be ended early |
| PyPI packages: FastAPI, shap, librosa, XGBoost | Using stdlib/sklearn equivalents | No package-registry access during build; equivalents documented in STATUS.md |
