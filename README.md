# pwiews — Personnel Welfare Intelligence and Early Warning System

> **Smart India Hackathon 2026** | Problem Statement **SIH26186** (Ministry of Home Affairs)  
> *AI-Based Predictive Personnel Stress and Welfare Monitoring System for Uniformed Forces (CRPF / CAPF).*

A proactive welfare-support tool designed to detect early indicators of operational stress, burnout, and organizational strain before crises occur. **This is not a surveillance or disciplinary tool** — a principle that strictly dictates the RBAC design, UI terminology, data collection defaults, and individual vs. systemic attribution.

---

## 🚀 Quick Start

### 1. Requirements
- Python 3.11+
- Standard ML stack: `numpy`, `pandas`, `scipy`, `scikit-learn`, `joblib`, `starlette`, `uvicorn` (or `pip install -r requirements.txt`)

### 2. End-to-End Pipeline Execution

The repository ships with the pipeline output already generated, so **step 5
alone is enough to see the system running.** The earlier steps regenerate
everything from scratch.

```bash
# 1. Generate the synthetic corpus (~15 s) and voluntary check-in audio (~10 s)
python scripts/generate_synthetic_data.py
python scripts/generate_voice_audio.py

# 2. Train & compare all 8 candidate models, register the winner (~40 s)
python scripts/train_models.py --quick        # use --cv for grouped cross-validation

# 3. Score all personnel, explain every case, generate recommendations & alerts (~6 min)
python scripts/run_pipeline.py

# 4. Run the automated test suite (211 passing unit tests)
python -m unittest discover -s tests

# 5. Serve the REST API and both frontends
python -m backend.api.main
```

> Most of step 3 is exact Shapley enumeration for all 800 people, at roughly
> 0.2 s each. It is batch work on nothing's critical path — the API only ever
> reads the result.

### 3. Accessing the Applications

Open your browser to:

| Application | URL | Description |
|---|---|---|
| **Landing Portal** | `http://127.0.0.1:8000/app/` | System overview and role launcher |
| **Personal Wellness App** | `http://127.0.0.1:8000/app/personal/` | Self-assessment, score history, notifications, transparency center |
| **Officer & Commander Dashboard** | `http://127.0.0.1:8000/app/officer/` | Prioritized welfare queue, case details, what-if simulator, unit near-misses |

Both apps sign themselves in, so there is nothing to type. The accounts they use
are these — the same ones to send to `POST /api/auth/login` when calling the API
directly:

| Account | Password | Role |
|---|---|---|
| `officer` | `welfare-officer-demo` | `welfare_officer` |
| `commander` | `commander-demo` | `commander` |
| `personnel` | `personnel-demo` | `personnel` (supply `"subject": "<pseudonym_id>"`) |

```bash
curl -X POST http://127.0.0.1:8000/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"officer","password":"welfare-officer-demo"}'
```

These are demonstration credentials over a synthetic corpus, published so the
system can be reviewed. The hashes are PBKDF2-HMAC-SHA256 in
`backend/auth/demo_accounts.json`; `backend/auth/README.md` describes what a
deployment replaces.

---

## 🏛️ Core Architecture & Highlights

1. **Zero LLM / Generative AI Path:** All scoring uses verified scikit-learn models (Gradient Boosting), explanations use exact Shapley value enumeration ($2^{11}=2048$ coalitions), and recommendations use a deterministic rule-based mapping engine. The reported $R^2 = 0.821$ on a person-disjoint held-out set measures how closely the model reproduces the **synthetic corpus's own scoring formula** — it is not evidence of predictive validity against real welfare outcomes, which would require field labels no hackathon build can have. See `docs/model_comparison_report.md` §5.
2. **Acoustic-Only Voice Pipeline:** Analyzes pitch ($F_0$), speaking rate, pause ratios, jitter, and shimmer. **Zero speech-to-text / transcription** exists by construction.
3. **Strict Privacy & Data Separation:** Direct identifiers are HMAC-SHA256 pseudonymized. Identity mapping is isolated in `data/identity_map.sqlite3` with an audited re-identification log. That file is **not committed** — it holds both the mapping and the salt that produced it, so a copy in the repository would undo the pseudonymisation for anyone who cloned it. It is created on the first pipeline run.
4. **Structural Leak Prevention:** The commander view cannot receive individual-identifiable records — enforced by `rbac.assert_commander_safe()` recursive payload scanning and proved by `tests/test_rbac_api.py`.
5. **Graduated Alerting & Recommendations:** 3-tier notification system (Personal, Officer, Commander) and 8 pre-approved operational welfare interventions, both surfaced on the screens that act on them. What a person is told about themselves and what an officer is told about that person are separate feeds, by construction.
6. **JWT Authentication:** HS256 issued by `POST /api/auth/login` against PBKDF2 credential hashes, verified on every role-scoped route. Both frontends hold tokens; the plain role header is off unless `PWIEWS_DEBUG_AUTH=1`.
7. **Calibrated Risk Intervals (split conformal prediction):** every score carries a range with a finite-sample coverage guarantee — ±9.9 points at 90% target coverage, verified at 91.5% on people the deployed model never saw. That coverage is with respect to the synthetic training label, not real welfare outcomes. When the range crosses a band cutoff the case is marked **borderline** on every screen and in the officer alert, which is the concrete answer to PS technical challenge #3 (false positives / negatives). `backend/models/conformal.py`, `docs/model_comparison_report.md` §5a.
8. **One Escalation Rule, Recorded Access:** who a welfare officer may see is defined once (`backend/post_model_analytics/escalation.py`: High, or persistent Moderate that is rising — 159 of 800 cases on this corpus, down from 619) and imported by the queue, the alert rules and the personal routes. Every officer open of a record is written to an access log (`backend/db/access_log.py`), and the individual sees the counts in their Privacy Centre.

---

## 📁 Repository Map

```
personnel-welfare-intelligence/
├── backend/
│   ├── config/settings.py       ← Single source of truth for all thresholds and cited sources
│   ├── pipeline.py              ← Master data orchestrator (ingestion → signals)
│   ├── ingestion/               ← CSV loading and schema validation
│   ├── preprocessing/           ← Cleaning, normalisation, HMAC pseudonymisation vault
│   ├── feature_engineering/     ← 14 point-in-time features, rolling windows, personal baselines
│   ├── behavioral_engine/       ← 9 normalized (0–100) behavioral stress signals
│   ├── voice_pipeline/          ← Acoustic DSP feature extraction (no speech transcription)
│   ├── models/                  ← 8 candidate models, person-disjoint training, selection, conformal calibration, exact SHAP
│   ├── post_model_analytics/    ← Risk bands + calibrated intervals, trend persistence, confidence heuristics, attribution, the escalation rule
│   ├── near_miss/               ← Unit-level organizational condition detection
│   ├── recommendation_engine/   ← 8 pre-approved interventions & rule-based action mapper
│   ├── alerts/                  ← Graduated 3-tier notification generator
│   ├── auth/                    ← RBAC, commander payload guard, JWT, credential store
│   ├── db/                      ← Record-access log (SQLite)
│   └── api/                     ← Starlette REST API, route handlers, request validation, check-in store
│
├── frontend/
│   ├── index.html               ← Landing page
│   ├── personal-app/            ← Self-assessment & welfare mobile/web app
│   ├── officer-dashboard/       ← Welfare officer queue & commander unit overview
│   └── shared/                  ← Shared API client, CSS design system, and UI utilities
│
├── data/
│   ├── raw/                     ← Raw synthetic CSVs and WAV audio files
│   ├── processed/               ← 7 precomputed JSON dashboard payloads
│   ├── schema/                  ← JSON table schemas
│   ├── responses/               ← Self-assessment answers (runtime, not committed)
│   ├── access_log.sqlite3       ← Record-access log (runtime, not committed)
│   └── identity_map.sqlite3     ← Identity vault & re-identification audit (not committed)
│
├── docs/                        ← Comprehensive documentation suite
│   ├── ps_alignment_matrix.md   ← 1-to-1 mapping of problem statement to code
│   ├── privacy_policy.md        ← Full data governance, voice protection, and rights
│   ├── model_comparison_report.md ← 8-model evaluation report and selection proof
│   └── data_dictionary.md       ← Complete data dictionary for all CSVs and JSONs
│
├── scripts/                     ← Entry points for data generation, training, and pipeline
├── tests/                       ← 211 automated unit tests verifying invariants & security
├── CodebaseGuide.md             ← Comprehensive in-depth technical walkthrough
└── STATUS.md                    ← Transparent accounting of completed vs. deferred scope
```

---

## 🧪 Testing

Run the automated test suite:

```bash
python -m unittest discover -s tests
```

Tests verify:
- Commander payload data-leak proof (`test_rbac_api.py`)
- Route-level auth and role scope, end to end over HTTP, including the officer gate on personal routes and what-if input validation (`test_api_routes.py`)
- Conformal calibration arithmetic, coverage on exchangeable data, and band certainty (`test_conformal.py`)
- The single escalation rule and its consumers (`test_escalation.py`)
- Record-access log contents, scoping and retention (`test_access_log.py`)
- Request-body validation (`test_request_parsing.py`)
- JWT creation, verification, tampering, and expiration (`test_jwt_auth.py`)
- Graduated alerting rules and confidence suppression (`test_alert_rules.py`)
- Recommendation determinism and attribution filters (`test_recommendation_engine.py`)
- Self-assessment answer validation against the question bank and per-person scoping (`test_checkin_store.py`)
- Voice pipeline DSP invariance and weight sums (`test_voice_pipeline.py`)
- Behavioral signal weights and settings contracts (`test_behavioral_engine.py`)

`test_api_routes.py` drives the real ASGI app through Starlette's `TestClient`
and needs `httpx`; it skips cleanly when that is not installed. It is the file
that would have caught the two authorisation defects fixed in this pass, both of
which were invisible to a unit test of `rbac.py` — the functions were correct
throughout, but one route did not call them and one header path bypassed them.

---

## 📖 Further Reading

- [**Codebase Guide**](CodebaseGuide.md) — Comprehensive layman and deep-dive technical explanation of every file, algorithm, and flow.
- [**Status Report**](STATUS.md) — Exact audit of completed components and environment-forced deviations.
- [**PS Alignment Matrix**](docs/ps_alignment_matrix.md) — One-to-one mapping of problem statement to code.
- [**Privacy Policy**](docs/privacy_policy.md) — Data governance, voice protection, and individual rights.
- [**Model Comparison Report**](docs/model_comparison_report.md) — Eight-model evaluation and selection proof.
- [**Data Dictionary**](docs/data_dictionary.md) — Every field in every CSV and JSON.
- [**Auth module**](backend/auth/README.md) — Login flow, the demo credential store, and what a deployment replaces.
