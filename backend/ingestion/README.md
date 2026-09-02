# `backend/ingestion/`

## What this module does

Gets raw data off disk and refuses to pass on anything that fails its schema.
Nothing here computes a feature, repairs a value, or analyses a signal — this
layer is I/O plus a validation gate, and that narrowness is deliberate.

| File | Job |
| --- | --- |
| `validators.py` | Declare what a valid raw table looks like; report violations. |
| `hr_loader.py` | Read the HR CSVs; return validated DataFrames. |
| `voice_loader.py` | Read the voice index and audio; enforce consent. |

## Inputs and outputs

### `validators.py`

**In:** a `pandas.DataFrame` and a `TableSchema`.
**Out:** a `ValidationReport` (`table`, `row_count`, `errors`, `warnings`,
`is_valid`). Also `validate_referential_integrity(tables) -> list[str]` and
`export_schemas()`, which dumps the schema definitions to
`data/schema/raw_table_schemas.json`.

Checks run per table: required columns present, values coercible to the
declared kind, nulls only where allowed, no negatives in non-negative columns,
values inside any declared closed set, primary key unique and non-null, and
`start_date <= end_date` wherever both exist.

### `hr_loader.py`

**In:** `data/raw/*.csv`.
**Out:** a `LoadResult` holding `tables` (name → DataFrame, dates parsed to
`datetime64`, numerics coerced per schema), `reports` (one per table), and
`integrity_problems`.

Public functions:

- `load_hr_tables(raw_dir, table_names, strict)` — the analytics set.
- `load_ground_truth_labels(raw_dir)` — the training target, **separately**.
- `load_uploaded_table(file_path, table_name)` — the dashboard upload path.

### `voice_loader.py`

**In:** `data/raw/voice_samples.csv` and `data/raw/voice_audio/*.wav`.
**Out:** `VoiceSample` objects (`sample_id`, `personnel_id`, `sample_date`,
`consent_version`, `sample_rate`, `waveform` as a float array in [-1, 1]).

Public functions: `load_voice_index`, `read_waveform`, `load_sample`,
`iter_samples`, `load_samples_for_person`.

## How it fits into the pipeline

```
data/raw/*.csv ──▶ hr_loader ──▶ preprocessing/clean ──▶ preprocessing/pseudonymize ──▶ feature_engineering/
data/raw/voice_audio/*.wav ──▶ voice_loader ──▶ voice_pipeline/audio_preprocess
```

`hr_loader` calls `validators`; nothing calls `hr_loader` except the
preprocessing layer, the training scripts, and the API's upload route.

## Design decisions and assumptions

**Validation reports, it never repairs.** A validator that silently fixes what
it finds cannot be trusted to tell you what was wrong. Cleaning is a separate,
later, logged step. This matters concretely for the dashboard upload flow: the
uploader is shown the validation report *before* anything is ingested, which is
only meaningful if validating has no side effects.

**Labels are loaded by a different function from everything else.**
`load_ground_truth_labels` is deliberately not part of `load_hr_tables`. The
served API has no reason to touch the training target, and in a real
deployment that table would not exist at all — the label would come from
validated welfare assessments. Keeping it on a separate function means a
served route cannot pull it in by accident.

**The voice loader enforces consent before opening the file.** A consent check
that runs after the audio has been read is not a consent check. A sample with
no recorded `consent_version` is never opened.

**`latent_strain` is dropped at the loader boundary.** The synthetic corpus
records the strain value each waveform was synthesised from. `voice_loader`
strips it on load, so no part of the served system and no model can learn from
the answer key. The column list is spelled out in `GENERATION_ONLY_COLUMNS`
rather than inferred, so adding a generation column without adding it there
shows up as an obvious omission instead of a silent leak.

**Sample-rate mismatches are refused, not resampled.** Resampling changes the
period lengths this pipeline measures, and jitter *is* a period-length
measurement. Silently resampling would corrupt the very feature the module
exists to compute, so a mismatch raises.

**Audio is normalised by the format's full scale, not the sample's peak.**
Peak-normalising every recording would make them all equally loud and destroy
the intensity feature outright.

**Over-long samples are truncated, not rejected.** A long recording is still
usable, and discarding it would penalise someone for talking.

## Environment-forced deviation

`pandera` / `great_expectations` would be the natural choice for the schema
layer. Neither is installable in the build environment — it has no
package-registry access — so `validators.py` is a small hand-rolled
equivalent covering exactly the checks this corpus needs. It is written
declaratively (`ColumnSpec` / `TableSchema` records rather than imperative
checks) specifically so that swapping in a real schema library later is a
mechanical translation rather than a rewrite.

Similarly, `soundfile` is unavailable, so WAV reading goes through
`scipy.io.wavfile`. That restricts input to uncompressed WAV, which is what
the corpus and the mobile app's upload path both produce.
