"""Synthesise the WAV files backing the voluntary voice check-in samples.

This is deliberately a separate script from ``generate_synthetic_data.py``:
tabular corpus generation and digital-signal synthesis are different jobs with
different failure modes, and mixing them would make both harder to reason
about.

WHY SYNTHETIC AUDIO
    Recordings of real personnel cannot be used, and no public corpus of
    uniformed-forces welfare check-ins exists. Rather than shipping a voice
    pipeline with nothing to run on, this script synthesises speech-like
    signals whose acoustic properties are set from a known latent strain
    value. That gives the pipeline in ``backend/voice_pipeline`` genuine
    audio to analyse and gives the tests a known answer to check against.

    These are NOT recordings of speech. They contain no words, no language and
    no content of any kind -- they are excitation-plus-resonator signals. That
    is fitting, because the pipeline that consumes them is forbidden from
    looking at content: it measures only how a voice is produced (pitch,
    rate, pauses, intensity, jitter, shimmer), never what is said. There is no
    transcription or speech-to-text anywhere in this system.

HOW THE SIGNAL IS BUILT
    1. A syllable/pause schedule is laid out for the requested duration.
    2. Within each voiced syllable, a glottal impulse train is built one pitch
       period at a time. Period length carries cycle-to-cycle perturbation
       (jitter); pulse amplitude carries cycle-to-cycle perturbation
       (shimmer).
    3. The impulse train is passed through a cascade of second-order formant
       resonators, giving a vowel-like timbre.
    4. A per-syllable amplitude envelope and a low noise floor are applied.

    Every acoustic property the extractor later measures is therefore a
    property that was actually put into the waveform, not metadata attached
    alongside it.

Inputs:
    ``data/raw/voice_samples.csv`` (written by generate_synthetic_data.py),
    specifically its ``sample_id``, ``personnel_id``, ``duration_sec`` and
    ``latent_strain`` columns.

Outputs:
    One 16 kHz mono 16-bit WAV per row, at ``data/raw/voice_audio/<id>.wav``.

Run:
    python scripts/generate_voice_audio.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import settings  # noqa: E402


# ASSUMPTION: a three-formant cascade with these centre frequencies and
# bandwidths gives a neutral, vowel-like timbre. The exact values do not
# matter for the pipeline (which never measures formants); they exist so the
# signal is speech-like rather than a raw buzz.
FORMANTS_HZ: Tuple[Tuple[float, float], ...] = ((700.0, 90.0), (1220.0, 110.0), (2600.0, 160.0))

# ASSUMPTION: per-person base pitch, drawn once per personnel_id so that a
# person's baseline is stable across their check-ins and the voice pipeline's
# *personal* baseline logic is genuinely exercised.
BASE_F0_RANGE_HZ: Tuple[float, float] = (95.0, 190.0)


def strain_to_acoustics(base_f0_hz: float, strain: float) -> Dict[str, float]:
    """Map a latent strain value to the acoustic parameters of one sample.

    Args:
        base_f0_hz: The speaker's own habitual pitch, so strain is expressed
            as a departure from their personal norm rather than an absolute.
        strain: Latent strain in [0, 1].

    Returns:
        Dict of synthesis parameters: ``f0_hz``, ``f0_declination``,
        ``syllable_rate``, ``pause_fraction``, ``jitter_frac``,
        ``shimmer_frac``, ``intensity_variability``.

    Assumptions:
        The direction of every relationship below is an ASSUMPTION made for
        this build, chosen to be consistent with commonly described features
        of pressured speech (raised and less variable pitch, faster rate,
        shorter pauses, greater cycle-to-cycle perturbation). No claim is made
        that these coefficients are clinically validated, and nothing in the
        served system treats the voice signal as a diagnosis -- it is one
        optional input among several, and it is flagged as voluntary
        everywhere it appears.
    """
    s = float(np.clip(strain, 0.0, 1.0))
    return {
        "f0_hz": base_f0_hz * (1.0 + 0.22 * s),
        "f0_declination": 0.10 - 0.05 * s,      # flatter contour under strain
        "syllable_rate": 3.2 + 1.8 * s,          # syllables per second
        "pause_fraction": 0.30 - 0.15 * s,       # less pausing under strain
        "jitter_frac": 0.004 + 0.016 * s,        # 0.4% -> 2.0%
        "shimmer_frac": 0.030 + 0.060 * s,       # 3% -> 9%
        "intensity_variability": 0.08 + 0.14 * s,
    }


def _formant_filter(x: np.ndarray, sample_rate: int) -> np.ndarray:
    """Pass an excitation signal through a cascade of formant resonators.

    Args:
        x: Excitation signal (the glottal impulse train).
        sample_rate: Sample rate in Hz.

    Returns:
        Filtered signal of the same length, normalised to unit peak.
    """
    y = x
    for centre, bandwidth in FORMANTS_HZ:
        r = float(np.exp(-np.pi * bandwidth / sample_rate))
        theta = 2.0 * np.pi * centre / sample_rate
        a = np.array([1.0, -2.0 * r * np.cos(theta), r * r])
        y = lfilter(np.array([1.0 - r]), a, y)
    peak = float(np.max(np.abs(y))) or 1.0
    return y / peak


def synthesise_sample(
    duration_sec: float,
    params: Dict[str, float],
    rng: np.random.Generator,
    sample_rate: int = settings.VOICE_SAMPLE_RATE_HZ,
) -> np.ndarray:
    """Synthesise one speech-like waveform with the requested acoustics.

    Args:
        duration_sec: Target duration.
        params: Output of :func:`strain_to_acoustics`.
        rng: Seeded generator.
        sample_rate: Output sample rate in Hz.

    Returns:
        Float array in [-1, 1] of length ``duration_sec * sample_rate``.

    Notes:
        Jitter is applied to the *period* of each glottal cycle and shimmer to
        its *amplitude*, which is what those terms mean, so the extractor in
        ``backend/voice_pipeline/acoustic_features.py`` measures back the same
        quantity that was injected here.
    """
    n_total = int(round(duration_sec * sample_rate))
    excitation = np.zeros(n_total, dtype=np.float64)
    envelope = np.zeros(n_total, dtype=np.float64)

    syllable_rate = params["syllable_rate"]
    pause_fraction = float(np.clip(params["pause_fraction"], 0.05, 0.55))
    voiced_seconds = duration_sec * (1.0 - pause_fraction)
    n_syllables = max(2, int(round(voiced_seconds * syllable_rate)))
    syllable_dur = voiced_seconds / n_syllables

    # Distribute the pause budget: a few longer inter-phrase pauses plus short
    # inter-syllable gaps. ASSUMPTION: one longer pause per ~4 syllables.
    pause_budget = duration_sec * pause_fraction
    n_gaps = n_syllables
    long_gap_idx = set(range(3, n_syllables, 4))
    weights = np.array([3.0 if i in long_gap_idx else 1.0 for i in range(n_gaps)])
    gap_lengths = pause_budget * weights / weights.sum()

    cursor = 0
    f0 = params["f0_hz"]
    declination = params["f0_declination"]

    for syl in range(n_syllables):
        syl_progress = syl / max(1, n_syllables - 1)
        # Pitch declines over the utterance; strain flattens the decline.
        syl_f0 = f0 * (1.0 - declination * syl_progress)
        syl_samples = int(round(syllable_dur * sample_rate))
        if cursor + syl_samples >= n_total:
            break

        # Lay down glottal pulses across this syllable.
        pos = cursor
        amp_base = 1.0
        while pos < cursor + syl_samples and pos < n_total:
            jitter = 1.0 + float(rng.normal(0.0, params["jitter_frac"]))
            period_samples = max(2, int(round(sample_rate / (syl_f0 * jitter))))
            shimmer = 1.0 + float(rng.normal(0.0, params["shimmer_frac"]))
            excitation[pos] = amp_base * max(0.05, shimmer)
            pos += period_samples

        # Per-syllable amplitude envelope (raised-cosine), with sample-level
        # intensity variability so the RMS-SD feature has something to measure.
        env = np.hanning(max(4, syl_samples))
        env = env * (1.0 + float(rng.normal(0.0, params["intensity_variability"])))
        end = min(cursor + len(env), n_total)
        envelope[cursor:end] += env[: end - cursor]

        cursor = end + int(round(gap_lengths[syl] * sample_rate))
        if cursor >= n_total:
            break

    voiced = _formant_filter(excitation, sample_rate)
    signal = voiced * np.clip(envelope, 0.0, None)

    # ASSUMPTION: a low broadband noise floor, so silence detection has to do
    # real work rather than testing for exact zeros.
    signal = signal + rng.normal(0.0, 0.002, size=n_total)
    peak = float(np.max(np.abs(signal))) or 1.0
    return np.clip(signal / peak * 0.92, -1.0, 1.0)


def generate_all(
    index_path: Path | None = None, audio_dir: Path | None = None
) -> List[Path]:
    """Synthesise a WAV for every row of the voice-sample index.

    Args:
        index_path: Path to ``voice_samples.csv``. Defaults to the raw data
            directory.
        audio_dir: Directory to write WAVs into. Defaults to
            ``data/raw/voice_audio``.

    Returns:
        List of written file paths.

    Raises:
        FileNotFoundError: If the voice-sample index has not been generated.
    """
    index_path = index_path or (settings.RAW_DATA_DIR / "voice_samples.csv")
    audio_dir = audio_dir or (settings.RAW_DATA_DIR / "voice_audio")
    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_path} not found -- run scripts/generate_synthetic_data.py first."
        )
    audio_dir.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(index_path)
    rng = np.random.default_rng(settings.RANDOM_SEED + 1)

    # One habitual pitch per person, stable across their samples.
    people = sorted(index["personnel_id"].unique())
    base_f0 = {
        pid: float(rng.uniform(*BASE_F0_RANGE_HZ)) for pid in people
    }

    written: List[Path] = []
    for _, row in index.iterrows():
        params = strain_to_acoustics(
            base_f0_hz=base_f0[row["personnel_id"]],
            strain=float(row["latent_strain"]),
        )
        wave = synthesise_sample(float(row["duration_sec"]), params, rng)
        path = audio_dir / f"{row['sample_id']}.wav"
        wavfile.write(path, settings.VOICE_SAMPLE_RATE_HZ, (wave * 32767).astype(np.int16))
        written.append(path)
    return written


if __name__ == "__main__":
    files = generate_all()
    total_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
    print(f"Synthesised {len(files)} WAV files ({total_mb:.1f} MB) -> "
          f"{settings.RAW_DATA_DIR / 'voice_audio'}")
