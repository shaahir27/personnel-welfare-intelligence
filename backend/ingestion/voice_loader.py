"""Read the voluntary voice-sample index and load audio waveforms.

One job: get consented audio off disk as a float array, and refuse everything
else. Signal processing happens in ``backend/voice_pipeline/``; this module
does not analyse anything.

Two rules are enforced here rather than downstream, because a consent check
that runs after the audio has already been read is not a consent check:

1. **No consent, no load.** A sample without a recorded consent version is
   never opened.
2. **The latent generation column never leaves this module.** The synthetic
   corpus records the strain value each waveform was synthesised from. That
   column is dropped on load, so no part of the served system -- and no model
   -- can accidentally learn from the answer key.

Pipeline position:
    ``data/raw/voice_samples.csv`` + ``data/raw/voice_audio/*.wav``
    -> **voice_loader** -> ``voice_pipeline/audio_preprocess.py``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np
import pandas as pd
from scipy.io import wavfile

from backend.config import settings
from backend.ingestion import validators

# Columns that exist only for corpus generation and must never reach the
# pipeline. Listed explicitly so adding one to the generator without adding it
# here shows up as an obvious omission rather than a silent leak.
GENERATION_ONLY_COLUMNS: Sequence[str] = ("latent_strain",)


@dataclass(frozen=True)
class VoiceSample:
    """One loaded voice check-in.

    Attributes:
        sample_id: Unique sample identifier.
        personnel_id: Owner of the sample.
        sample_date: Date the check-in was recorded.
        consent_version: Version of the consent text the person accepted.
        sample_rate: Sample rate of ``waveform`` in Hz.
        waveform: Mono float array in [-1, 1].
    """

    sample_id: str
    personnel_id: str
    sample_date: pd.Timestamp
    consent_version: str
    sample_rate: int
    waveform: np.ndarray

    @property
    def duration_sec(self) -> float:
        """Actual duration of the loaded waveform, in seconds."""
        return float(len(self.waveform)) / float(self.sample_rate)


def load_voice_index(raw_dir: Path | None = None) -> pd.DataFrame:
    """Load and validate the voice-sample index, dropping generation columns.

    Args:
        raw_dir: Directory holding the raw CSVs. Defaults to ``data/raw``.

    Returns:
        DataFrame with ``sample_id``, ``personnel_id``, ``sample_date``,
        ``consent_version``, ``duration_sec``, ``audio_path`` -- and nothing
        from :data:`GENERATION_ONLY_COLUMNS`.

    Raises:
        FileNotFoundError: If the index has not been generated.
        ValueError: If the index fails schema validation.
    """
    raw_dir = raw_dir or settings.RAW_DATA_DIR
    path = raw_dir / "voice_samples.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run scripts/generate_synthetic_data.py first."
        )
    df = pd.read_csv(path)
    df["sample_date"] = pd.to_datetime(df["sample_date"], errors="coerce", format="mixed")
    df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce")

    report = validators.validate_table(df, validators.VOICE_SAMPLES_SCHEMA)
    if not report.is_valid:
        raise ValueError(
            "voice_samples failed validation:\n  " + "\n  ".join(report.errors)
        )

    return df.drop(columns=[c for c in GENERATION_ONLY_COLUMNS if c in df.columns])


def read_waveform(
    audio_path: Path, target_sample_rate: int = settings.VOICE_SAMPLE_RATE_HZ
) -> tuple[np.ndarray, int]:
    """Read one WAV file into a mono float array.

    Args:
        audio_path: Path to the WAV file.
        target_sample_rate: Expected sample rate. A mismatch is reported
            rather than silently resampled, because resampling changes the
            very quantities the pipeline measures (period lengths, and so
            jitter) and a silent change there would corrupt the signal.

    Returns:
        Tuple of (waveform as float array in [-1, 1], sample rate in Hz).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the sample rate does not match ``target_sample_rate``.

    Assumption:
        Multi-channel input is mixed down by averaging channels. The synthetic
        corpus is mono, so this path exists only for real uploads.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    sample_rate, data = wavfile.read(audio_path)
    if sample_rate != target_sample_rate:
        raise ValueError(
            f"{audio_path.name}: sample rate {sample_rate} Hz does not match the "
            f"expected {target_sample_rate} Hz. Resampling is refused here because "
            f"it alters the cycle-level measurements this pipeline depends on."
        )

    waveform = np.asarray(data)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # Normalise integer PCM to [-1, 1] using the format's full scale, not the
    # sample's own peak -- peak normalisation would destroy the intensity
    # feature by making every recording equally loud.
    if np.issubdtype(waveform.dtype, np.integer):
        full_scale = float(np.iinfo(waveform.dtype).max)
        waveform = waveform.astype(np.float64) / full_scale
    else:
        waveform = waveform.astype(np.float64)

    return waveform, int(sample_rate)


def load_sample(row: pd.Series, raw_dir: Path | None = None) -> VoiceSample | None:
    """Load one indexed sample, honouring consent and duration limits.

    Args:
        row: One row of the voice index.
        raw_dir: Root the ``audio_path`` column is relative to.

    Returns:
        A :class:`VoiceSample`, or ``None`` when the sample is skipped
        (no consent recorded, file missing, or shorter than the configured
        minimum usable duration).

    Note:
        Samples longer than ``VOICE_MAX_DURATION_SEC`` are truncated rather
        than rejected -- a long recording is still usable, and discarding it
        would penalise the person for talking.
    """
    raw_dir = raw_dir or settings.RAW_DATA_DIR

    consent = str(row.get("consent_version", "") or "").strip()
    if not consent:
        return None

    audio_path = raw_dir / str(row["audio_path"])
    try:
        waveform, sample_rate = read_waveform(audio_path)
    except (FileNotFoundError, ValueError):
        return None

    if len(waveform) < settings.VOICE_MIN_DURATION_SEC * sample_rate:
        return None

    max_samples = int(settings.VOICE_MAX_DURATION_SEC * sample_rate)
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]

    return VoiceSample(
        sample_id=str(row["sample_id"]),
        personnel_id=str(row["personnel_id"]),
        sample_date=pd.Timestamp(row["sample_date"]),
        consent_version=consent,
        sample_rate=sample_rate,
        waveform=waveform,
    )


def iter_samples(
    index: pd.DataFrame | None = None, raw_dir: Path | None = None
) -> Iterator[VoiceSample]:
    """Iterate over every loadable sample in the index.

    Args:
        index: Voice index. Loaded from disk when omitted.
        raw_dir: Root for relative audio paths.

    Yields:
        :class:`VoiceSample` for each row that passes the consent, existence
        and duration checks. Skipped rows are simply not yielded.
    """
    raw_dir = raw_dir or settings.RAW_DATA_DIR
    index = load_voice_index(raw_dir) if index is None else index
    for _, row in index.iterrows():
        sample = load_sample(row, raw_dir)
        if sample is not None:
            yield sample


def load_samples_for_person(
    personnel_id: str, index: pd.DataFrame | None = None, raw_dir: Path | None = None
) -> List[VoiceSample]:
    """Load every consented sample belonging to one person, oldest first.

    Args:
        personnel_id: The person whose samples to load.
        index: Voice index. Loaded from disk when omitted.
        raw_dir: Root for relative audio paths.

    Returns:
        Chronologically ordered list, possibly empty. An empty list is a
        normal, expected outcome -- voice check-in is voluntary and most
        people will never opt in.
    """
    raw_dir = raw_dir or settings.RAW_DATA_DIR
    index = load_voice_index(raw_dir) if index is None else index
    mine = index[index["personnel_id"] == personnel_id].sort_values("sample_date")
    return [s for s in (load_sample(r, raw_dir) for _, r in mine.iterrows()) if s]
