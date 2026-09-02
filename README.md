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

```bash
# 1. Generate the synthetic corpus (~15 s) and voluntary check-in audio (~10 s)
python scripts/generate_synthetic_data.py
python scripts/generate_voice_audio.py

# 2. Train & compare all 8 candidate models, register the winner (~40 s)
python scripts/train_models.py --quick        # use --cv for grouped cross-validation

# 3. Score all personnel, generate recommendations & alerts, write payloads (~60 s)
python scripts/run_pipeline.py

# 4. Run the automated test suite (91 passing unit tests)
python -m unittest discover -s tests

# 5. Serve the REST API and both frontends
python -m backend.api.main
```

### 3. Accessing the Applications

Open your browser to:

| Application | URL | Description |
|---|---|---|
| **Landing Portal** | `http://127.0.0.1:8000/app/` | System overview and role launcher |
| **Personal Wellness App** | `http://127.0.0.1:8000/app/personal/` | Self-assessment, score history, notifications, transparency center |
| **Officer & Commander Dashboard** | `http://127.0.0.1:8000/app/officer/` | Prioritized welfare queue, case details, what-if simulator, unit near-misses |

---

## 🏛️ Core Architecture & Highlights

1. **Zero LLM / Generative AI Path:** All scoring uses verified scikit-learn models (Gradient Boosting, $R^2 = 0.729$), explanations use exact Shapley value enumeration ($2^{10}=1024$ coalitions), and recommendations use a deterministic rule-based mapping engine.
2. **Acoustic-Only Voice Pipeline:** Analyzes pitch ($F_0$), speaking rate, pause ratios, jitter, and shimmer. **Zero speech-to-text / transcription** exists by construction.
3. **Strict Privacy & Data Separation:** Direct identifiers are HMAC-SHA256 pseudonymized. Identity mapping is isolated in `data/identity_map.sqlite3` with an audited re-identification log.
4. **Structural Leak Prevention:** The commander view cannot receive individual-identifiable records — enforced by `rbac.assert_commander_safe()` recursive payload scanning and proved by `tests/test_rbac_api.py`.
5. **Graduated Alerting & Recommendations:** 3-tier notification system (Personal, Officer, Commander) and 8 pre-approved operational welfare interventions.
6. **JWT Authentication:** HS256 token verification implemented via stdlib with PyJWT fallback.

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
│   ├── behavioral_engine/       ← 8 normalized (0–100) behavioral stress signals
│   ├── voice_pipeline/          ← Acoustic DSP feature extraction (no speech transcription)
│   ├── models/                  ← 8 candidate models, person-disjoint training, selection, exact SHAP
│   ├── post_model_analytics/    ← Risk bands, trend persistence, confidence heuristics, attribution
│   ├── near_miss/               ← Unit-level organizational condition detection
│   ├── recommendation_engine/   ← 8 pre-approved interventions & rule-based action mapper
│   ├── alerts/                  ← Graduated 3-tier notification generator
│   ├── auth/                    ← RBAC, commander payload guard, and JWT handler
│   └── api/                     ← Starlette REST API and route handlers
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
│   └── identity_map.sqlite3     ← Isolated identity database & re-identification audit log
│
├── docs/                        ← Comprehensive documentation suite
│   ├── ps_alignment_matrix.md   ← 1-to-1 mapping of problem statement to code
│   ├── privacy_policy.md        ← Full data governance, voice protection, and rights
│   ├── model_comparison_report.md ← 8-model evaluation report and selection proof
│   └── data_dictionary.md       ← Complete data dictionary for all CSVs and JSONs
│
├── scripts/                     ← Entry points for data generation, training, and pipeline
├── tests/                       ← 91 automated unit tests verifying invariants & security
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
- JWT creation, verification, tampering, and expiration (`test_jwt_auth.py`)
- Graduated alerting rules and confidence suppression (`test_alert_rules.py`)
- Recommendation determinism and attribution filters (`test_recommendation_engine.py`)
- Voice pipeline DSP invariance and weight sums (`test_voice_pipeline.py`)
- Behavioral signal weights and settings contracts (`test_behavioral_engine.py`)

---

## 📖 Further Reading

- [**Codebase Guide (`CodebaseGuide.md`)**](file:///d:/Desktop/project/personnel-welfare-intelligence/CodebaseGuide.md) — Comprehensive layman and deep-dive technical explanation of every file, algorithm, and flow.
- [**Status Report (`STATUS.md`)**](file:///d:/Desktop/project/personnel-welfare-intelligence/STATUS.md) — Exact audit of completed components and environment-forced deviations.
- [**Documentation Suite (`docs/`)**](file:///d:/Desktop/project/personnel-welfare-intelligence/docs/) — Full PS alignment, privacy policy, model comparison report, and data dictionary.
