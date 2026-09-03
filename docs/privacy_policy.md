# pwiews — Privacy Policy

**SIH26186 AI-Based Predictive Personnel Stress and Welfare Monitoring System**

This document describes what data the system holds about personnel, how it is
protected, who can access what, and what choices personnel have.

---

## 1. HR / Administrative Data

### What is collected
The system ingests the following HR records from existing HRMS sources:

| Source | Fields ingested | Retention |
|---|---|---|
| Duty logs | Unit ID, monthly duty hours, night shifts | 730 days |
| Leave records | Leave dates, type, duration | 730 days |
| Deployment history | Deployment location type, start/end dates | 730 days |
| Transfer records | Transfer dates | 730 days |
| Training records | Training hours per period | 730 days |
| Unit capacity | Sanctioned strength, on-strength count | 730 days |

Raw personnel identifiers (name, service number) **never enter the analytics
database**. They are replaced with an HMAC-derived pseudonym at the ingestion
boundary (`backend/preprocessing/pseudonymize.py`) using a key stored
separately from the analytics data.

### How pseudonymisation works
- Each `personnel_id` is replaced with a `pseudonym_id` derived using HMAC-SHA256
  with a project-specific secret key.
- The mapping between `personnel_id` and `pseudonym_id` is stored in a
  **separate SQLite database** (`data/identity_map.sqlite3`).
- The analytics pipeline, the scoring model, and the API can access
  `data/processed/` but **cannot open** `data/identity_map.sqlite3`.
  There is no import of the identity-map module anywhere in the analytics path.
- Re-identification (e.g., for pastoral follow-up by a welfare officer) requires
  an explicit call to the re-identification function with a logged audit record.

### Who can see what

| Role | What they see |
|---|---|
| **Personnel** (individual) | Their own risk score, trend, contributing factors, history, notifications. More than any other role — a system that tells the organisation more about you than it tells you cannot be trusted. |
| **Welfare Officer** | Pseudonymised case queue (High, or persistent Moderate that is rising — one rule, in `backend/post_model_analytics/escalation.py`). Full case detail including SHAP explanation. No name, no service number. Every open of a case is written to the record-access log. |
| **Commander** | Unit aggregates only. No individual scores, no contributing factors, no pseudonym IDs. Enforced server-side by `rbac.assert_commander_safe()`. Units below 10 personnel are suppressed entirely (small-cell suppression). |

---

## 2. Voice Data

### Voluntary participation
Voice check-in is **entirely voluntary**. Opting out:
- Has no effect on the welfare risk score.
- Is not recorded as a concern.
- Is not shown to the welfare officer or commander.

### What is recorded during a voice check-in
The system records a short speech sample for acoustic analysis **only**.

| Extracted | Not extracted |
|---|---|
| Fundamental frequency (F0) — pitch mean and variation | Any word, phoneme, or speech content |
| Speaking rate (syllables per second) | Transcription of any kind |
| Pause ratio (fraction of sample in silence) | Speaker identification beyond the session |
| Intensity coefficient of variation (scale-invariant loudness variability) | Any biometric feature not listed here |
| Cycle-level jitter and shimmer | — |

**The raw audio recording is deleted immediately after feature extraction**
(retention: 0 days per `settings.RETENTION_RAW_AUDIO_DAYS`).
The extracted acoustic features are retained for 180 days
(`settings.RETENTION_ACOUSTIC_FEATURES_DAYS`).

### How voice features are used
The 7 acoustic features above are compared to the individual's own historical
baseline (minimum 3 prior check-ins). The comparison produces a single
deviation number (0–100) called `voice_stress_signal`. This number is one of
10 inputs to the welfare risk model.

Because the comparison is to the individual's own baseline, the system is
measuring change relative to their own normal speech — not comparing them
against any population norm or clinical threshold. The voice signal is never
used as a diagnosis.

### Baseline reset
A person can request that their voice baseline be reset at any time by
contacting the welfare officer. A reset discards all stored acoustic features
and requires a new minimum of 3 check-ins before the voice signal is active.

---

## 3. Subject Rights

| Right | How to exercise |
|---|---|
| See your own data | `GET /api/personal/{pseudonym_id}/summary` and `/privacy` |
| Opt out of voice check-in | Do not record; no action required |
| Request voice baseline reset | Contact welfare officer |
| Request re-identification audit log | Contact welfare officer (logged access) |
| Understand what the score means | `/privacy` endpoint provides a full plain-language explanation |

---

## 4. What the system is NOT used for

The welfare risk score **must not** be used for:
- Disciplinary action of any kind.
- Promotion, posting, or performance decisions.
- Any sharing outside the welfare officer chain.

This constraint is stated in every case detail response via `handling_note`
and is documented in the system's PS alignment matrix.

---

## 5. Data flows (summary)

```
HRMS → ingestion → pseudonymise → feature engineering → model score
                                                            ↓
                                             Post-analytics (risk band, trend)
                                                            ↓
                                             API (read-only, precomputed)
                                                            ↓
                             Personnel app          Officer dashboard       Commander view
                             (own data only)        (pseudonymised)        (unit aggregates only)
```

Voice:
```
Audio recording → acoustic feature extraction → delete audio → baseline comparison
                                                               → voice_stress_signal (0-100)
                                                               → score model (one of 10 inputs)
```

---

## Record-access log

Every time a welfare officer opens an individual's record — case detail, the
what-if simulator, or the personal summary, history or notification feeds — the
server writes one row to `data/access_log.sqlite3`
(`backend/db/access_log.py`): timestamp, the officer's role and service
subject, the action, the pseudonym concerned, and whether the request was
granted or refused. Refusals are recorded as well as grants.

- The log carries the **pseudonym, never a name**, so it cannot become the one
  table where identity and welfare data sit together.
- It records the **fact** of access, never the payload.
- The individual sees it in the Privacy Centre as **counts and dates by role**,
  not the officer's identity: the purpose is that a person can see that their
  record was opened and when. The raw rows are for oversight.
- Retention: `settings.RETENTION_ACCESS_LOG_DAYS` (365, an assumption); the
  pipeline purges older rows on every run.
