# `backend/config/`

## What this module does

Holds every tunable number in the pwiews system in one place. Thresholds,
window sizes, cutoffs, saturation points, model-selection rules, retention
periods, RBAC allow-lists and the real-world data anchors all live in
`settings.py`. Nothing else in the codebase is permitted to hardcode a
meaningful number inline.

## Files

| File | Purpose |
| --- | --- |
| `settings.py` | The single source of truth for all configuration constants. |

## Inputs and outputs

`settings.py` takes no inputs and produces no side effects beyond computing
`Path` objects from its own location. It exposes:

- module-level constants (all upper-case, all `Final`-annotated),
- `as_dict() -> dict` — every public constant, JSON-serialisable, used by the
  API's config route and by the docs so documented thresholds cannot drift
  from the code,
- `signal_label(signal_name) -> str` — the non-judgemental display label for a
  behavioral signal.

## How it fits into the pipeline

Everything imports it; it imports nothing from the project. That direction is
deliberate — it makes circular imports structurally impossible.

## Design decisions and assumptions

**Every constant is tagged `SOURCE:` or `ASSUMPTION:` in a comment.** This is
a hard rule for this project. A figure anchored to a real published number
carries its source; a number the team chose is labelled an assumption and is
never presented as fact in any document or UI. The two are never blurred.

**Why the risk bands sit where they do.** `RISK_BAND_MODERATE_MIN = 40` and
`RISK_BAND_HIGH_MIN = 65` give a deliberately wide Moderate band. In a welfare
context a false negative (missing someone who needed support) costs more than
a false positive (offering support to someone who was coping). But a High
classification is what makes a case visible to a welfare officer, so that
threshold is set conservatively to limit unnecessary exposure of individuals —
directly addressing PS technical challenges #2 (stigmatisation) and #3 (false
positives/negatives).

**Why `MIN_UNIT_SIZE_FOR_AGGREGATE` exists.** A unit-level average over three
people is not an aggregate, it is three people. Small-cell suppression is
standard statistical disclosure control and is what stops a commander from
reverse-engineering an individual's score from unit numbers.

**Why `MODEL_SELECTION_NON_TREE_R2_MARGIN` exists.** Explainability is a
stated PS requirement, and tree models have an exact, fast SHAP path. Rather
than leaving "prefer the interpretable model" as a judgement call made in
prose after the fact, it is encoded as an inspectable rule: a non-tree model
must win by at least this margin to displace the best tree model.

**Voice settings are acoustic only.** There is no transcription setting in
this file because there is no transcription anywhere in the system. The F0
search range is the only voice constant with a real-world source; the rest are
standard DSP defaults or flagged assumptions.

**`JWT_SECRET_KEY` is a development value** and is labelled as such. A real
deployment must inject it from a secret manager.
