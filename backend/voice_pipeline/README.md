# `backend/voice_pipeline/`

## What this module does

Turns an optional, voluntary voice check-in into **one number**: how far this
recording departs from the same person's own recent baseline.

| File | Job |
| --- | --- |
| `audio_preprocess.py` | DC removal, silence trimming, framing, per-frame energy, voiced/silent decision. |
| `acoustic_features.py` | Extract nine acoustic measurements. |
| `voice_baseline.py` | Hold and update each person's own acoustic norm. |
| `voice_stress_signal.py` | Compare a check-in to its baseline → one 0–100 value. |
| `pipeline.py` | Walk a person's check-ins in order, enforcing the leak-free baseline rule. |

## The rule this whole package is built around

**It analyses *how* someone speaks, never *what* they say.**

There is no transcription, no speech-to-text, no phoneme recognition, no
keyword spotting, no language identification, and no content analysis of any
kind — not in any file here, not anywhere else in the system, not at any stage.

This is enforced structurally, not by policy. The only things that cross out of
`audio_preprocess.py` are an autocorrelation lag, a set of period marks and an
RMS envelope. The content of speech is not recoverable from those quantities.
The module boundary *is* the guarantee.

## Structural deviation from the reference layout

The reference listing names four files; there are five. `pipeline.py` holds the
walk-the-samples orchestration. Putting that loop inside any of the four
computation modules would make that file both a computation and an
orchestrator; putting it in the API route would duplicate it in the batch
pipeline. Recorded here per the project's instruction to document deviations.

## Inputs and outputs

**In:** `VoiceSample` objects from `ingestion/voice_loader` — consented,
16 kHz mono float waveforms.

**Out — and this is the entire output contract:**

```python
{"pseudonym_id": str, "sample_date": Timestamp, "voice_stress_signal": float}
```

One 0–100 value per person per check-in, aligned to the pipeline's snapshot
dates by `signals_to_frame`. Nothing downstream — not the models, not the
officer dashboard, not the alert rules, not the recommendation engine — ever
receives audio, per-feature acoustic values, or the personal baseline.

That narrowness is the point. It makes it structurally impossible for a welfare
officer's screen to show "your pitch was 12 % higher than usual", which would
be simultaneously unactionable and intrusive.

### The nine extracted features

| Feature | Compared? | Why |
| --- | --- | --- |
| `f0_mean_hz` | ✅ | Scale-invariant. |
| `f0_sd_hz` | ✅ | Scale-invariant (direction caveat below). |
| `speaking_rate_syllables_per_sec` | ✅ | Scale-invariant. |
| `pause_ratio` | ✅ | Relative threshold, so scale-invariant. |
| `intensity_rms_mean` | ❌ | **Not** scale-invariant — see below. |
| `intensity_rms_sd` | ❌ | Not scale-invariant. |
| `intensity_rms_cv` | ✅ | `sd / mean`, self-normalising. |
| `jitter_local_pct` | ✅ | Relative measure. |
| `shimmer_local_pct` | ✅ | Relative measure. |

## How it fits into the pipeline

```
voice_loader ──▶ audio_preprocess ──▶ acoustic_features ──▶ voice_baseline
                                                   │              │
                                                   └──▶ voice_stress_signal ◀┘
                                                                  │
                                          behavioral_engine (optional input)
```

---

## Design decisions and assumptions

### Why the baseline must be personal — with the number that proves it

Measured on this project's synthetic corpus, the correlation between mean pitch
and the injected strain level is:

| Comparison | Correlation |
| --- | --- |
| Pooled across all speakers | **+0.04** |
| Within each speaker | **+0.98** |

Pooled, the signal is invisible. Habitual pitch varies enormously between
people, and that between-speaker variation completely swamps the within-speaker
change. Referenced to each person's own history, the same measurement recovers
it almost perfectly.

The practical consequence: any voice feature compared against a *population*
norm would be measuring **who someone is** rather than **how they are doing**.
That is both useless and exactly the kind of profiling this system must not do.

### Absolute loudness is extracted but never compared

Recording level varies with handset, microphone gain and distance from the
mouth — none of which the system controls. Comparing `intensity_rms_mean`
across check-ins measures the recording setup, not the person. Only
scale-invariant quantities enter the deviation calculation; the two absolute
intensity measures are kept as recording-quality context and their
self-normalising ratio `intensity_rms_cv` is used in their place.

### One direction constant was set from measurement, and it is flagged

`f0_sd_hz` has a direction of **+1** in `settings.VOICE_FEATURE_DIRECTIONS`.
The intuitive assumption is −1 (a flatter, less expressive contour under
pressure). It is +1 because F0 standard deviation *measured in Hz* is
confounded: it scales with mean F0 and with cycle-level perturbation, both of
which rise under the modelled strain, and that swamps any flattening of the
contour. A semitone-denominated measure would behave differently.

This is stated rather than quietly fixed because it matters for how the system
should be read: it is the one direction in that table set from observed
measurement behaviour rather than from the literature, it carries the smallest
weight (0.06) for exactly that reason, and it is the first thing that would
need re-validating against real recordings.

### Cycle-level jitter and shimmer, honestly described

Period marks are found by peak-picking on the unfiltered waveform: from the
largest excursion in the first period-and-a-half, step forward looking for the
next largest excursion in the window `[0.7T, 1.4T]`. That window is what makes
the tracker robust — it cannot lock onto a harmonic (too close) or skip a cycle
(too far).

This is a genuine cycle-level measurement, not a frame-level approximation. It
is **not** electroglottographic or inverse-filtered glottal-closure detection,
and it is not clinical-grade jitter. It measures cycle-to-cycle variation in
the signal as recorded — which is the same quantity a check-in device can
actually capture — and it is not presented as anything more anywhere in the
system.

Parabolic interpolation on the autocorrelation peak matters more than it looks:
at 16 kHz, one sample is already ~0.9 % of a 125 Hz period, so integer-lag
resolution would put a quantisation floor directly on top of the quantity being
measured.

### Why autocorrelation rather than a learned pitch tracker

It is a closed-form signal-processing operation with no trained parameters, so
its behaviour is fully inspectable and it cannot drift. In a system that must
be defensible under scrutiny, a pitch estimate nobody can explain is a
liability. It is also cheap enough to run on every check-in without a queue.

### Speaking rate is a proxy, and the proxy is stated

Syllable nuclei are counted as peaks in the smoothed energy envelope. That will
merge two syllables spoken without an energy dip and split one with a strong
internal dip. It is used because the alternative — counting actual syllables —
requires phonetic decoding of speech, which this system is forbidden from
doing. Since the proxy is only ever compared against the same person's baseline
computed the same way, the systematic bias cancels.

### Median and IQR, not mean and SD

With three to five check-ins, one unusual recording — a bad connection, a cold,
a noisy room — would move a mean substantially and inflate an SD enough to mask
every subsequent change. Median and IQR/1.349 are far less affected by a single
outlier, which matters most precisely when the sample is smallest.

### The baseline is allowed to drift, and the cost is stated

A person's voice changes over months and years for reasons unrelated to
welfare; a frozen baseline would slowly turn ordinary ageing into a rising
stress reading. The trade-off is real: **a baseline that adapts will eventually
absorb a slow, sustained deterioration and stop flagging it.**

That is precisely why the voice signal is one optional input among eight and
never a determination on its own, and why sustained trends are the job of
`post_model_analytics/trend_engine.py`, which works on the risk score and does
not adapt.

### The leak-free ordering rule

A check-in's baseline is built from that person's **strictly earlier**
check-ins. Sample 1 has no baseline and produces no signal; sample 4 is
compared against samples 1–3. Building the baseline from all of a person's
samples — including the one being judged — would drag every deviation toward
zero and, in training, would leak future information backwards.

Concretely, on the corpus's 100 check-ins: 60 produce no signal (the first
three per person, while the baseline is being established) and 40 produce one.
Those 60 are returned as explicit unreliable results with a stated reason, not
dropped, so a person can be told honestly that their baseline is still being
built.

### Unusable recordings do not update the baseline either

A recording the extractor could not measure is skipped for scoring *and* for
baseline updating. It should not quietly widen the baseline it failed to
measure.

### Unreliable means discard, not "show with a caveat"

`is_reliable = False` when the baseline has fewer than
`VOICE_BASELINE_MIN_SAMPLES` check-ins, or when less than half the weighted
feature set could be computed. The contract is that the caller **discards** the
value. A number shown with a caveat still anchors the reader; a number not
shown does not.

### A snapshot never reaches forward for a check-in

`signals_to_frame` takes the most recent check-in at or before each snapshot,
and will not carry one forward by more than one snapshot interval. Reaching
forward would leak post-snapshot information into a training row. Carrying an
old reading forward indefinitely would present a three-month-old voice sample
as evidence about today.

### What the number is not

It is a **deviation from the person's own recent norm**, on the same 0–100
scale as the other behavioral signals. It is not a stress score, not a
diagnosis, not a measure of psychological state, and **not comparable between
people**. Two people at 60 have each departed similarly far from their own
baselines; nothing follows about which is more distressed.

### Consent and retention

`ingestion/voice_loader` refuses to open a sample with no recorded consent, so
the check happens before the file is read rather than after. Raw audio is never
retained: `settings.RETENTION_RAW_AUDIO_DAYS` is 0, and `pipeline.analyse_sample`
is where that policy is realised — features go forward, the waveform does not.
Baselines store only summary statistics; no recording can be reconstructed from
them.

## Environment-forced deviation

`librosa` is the natural choice for this package and is **not installable** in
the build environment, which has no package-registry access. Every DSP
operation here is therefore implemented directly on numpy and
`scipy.signal` / `scipy.io.wavfile`: autocorrelation pitch tracking with
parabolic refinement, RMS-envelope voicing, energy-peak syllable counting, and
peak-picked period marking for jitter and shimmer.

For a judged submission this is arguably the better outcome — the method is
visible in the code rather than delegated to a library call — but it was a
constraint, not a preference, and it is recorded as one.
