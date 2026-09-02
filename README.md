# pwiews

**Personnel Welfare Intelligence and Early Warning System**
Smart India Hackathon — Problem Statement **SIH26186** (Ministry of Home Affairs):
*AI-Based Predictive Personnel Stress and Welfare Monitoring System for Uniformed Forces.*

A welfare-support tool. Not a disciplinary or surveillance tool — that constraint
shapes the RBAC model, the UI language, the data-collection defaults and the
individual-vs-systemic analysis, not just the marketing.

**Read `STATUS.md` first.** It is the honest account of what is built, what is
not, and how to run it.

## Quick start

```bash
python scripts/generate_synthetic_data.py   # synthetic corpus
python scripts/generate_voice_audio.py      # synthetic check-in audio
python scripts/train_models.py --quick      # train + compare 8 models, register winner
python scripts/run_pipeline.py              # score everything, write dashboard payloads
python -m backend.api.main                  # serve API + both frontends on :8000
```

Open `http://127.0.0.1:8000/app/`.

## Layout

```
data/           raw CSVs, synthetic audio, processed dashboard payloads, schemas
backend/
  config/       every threshold, each tagged SOURCE: or ASSUMPTION:
  ingestion/    schema validation and loading (validates, never repairs)
  preprocessing/cleaning, normalisation, HMAC pseudonymisation + audited vault
  feature_engineering/  point-in-time features, temporal windows, personal baselines
  behavioral_engine/    the eight behavioral signals the models actually consume
  voice_pipeline/       acoustics only — no transcription, ever
  models/       8 candidates (one module each), training, selection, registry, exact SHAP
  post_model_analytics/ risk bands, trend, confidence, individual-vs-systemic
  near_miss/    unit-level welfare near-miss detection
  api/          Starlette app, routes split by role
  auth/         role scoping + the commander no-individual-data guard
frontend/       two dependency-free ES-module apps
ml/evaluation/  metric definitions and comparison output
scripts/        generation, training and pipeline entry points
```

Every package carries a `README.md` covering what it does, its inputs and
outputs, where it sits in the pipeline, and the design decisions and assumptions
baked into it.
