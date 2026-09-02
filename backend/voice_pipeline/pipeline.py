"""Orchestrate the voice pipeline end to end.

One job: walk a person's check-ins in chronological order, extract features,
maintain the baseline, and emit one deviation value per check-in.

Why this is a fifth file (a deviation from the reference structure):
    The reference layout names four files in this package. Each of them does
    one computation. Putting the walk-the-samples loop inside any of them would
    make that file both a computation and an orchestrator; putting it in the
    API route would duplicate it in the batch pipeline. This module is the
    orchestration, and nothing else.

The leak-free ordering rule, which this module exists to enforce:
    A check-in's baseline is built from that person's **strictly earlier**
    check-ins. Sample 1 has no baseline and produces no signal. Sample 4 is
    compared against samples 1-3. Building the baseline from all of a person's
    samples -- including the one being judged -- would drag every deviation
    toward zero and, in training, would leak future information backwards.

Pipeline position:
    ``ingestion/voice_loader`` -> **pipeline** ->
    ``behavioral_engine/behavioral_signals``
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Sequence

import pandas as pd

from backend.config import settings
from backend.ingestion.voice_loader import VoiceSample, load_voice_index, load_sample
from backend.voice_pipeline import voice_baseline
from backend.voice_pipeline.acoustic_features import AcousticFeatures, extract_features
from backend.voice_pipeline.audio_preprocess import preprocess
from backend.voice_pipeline.voice_stress_signal import (
    VoiceStressResult,
    compute_voice_stress_signal,
)


def analyse_sample(sample: VoiceSample) -> AcousticFeatures:
    """Preprocess and extract features from one check-in.

    Args:
        sample: A loaded, consented voice sample.

    Returns:
        The extracted acoustic features.

    Note:
        The waveform is not retained after this call returns anywhere in the
        pipeline. ``settings.RETENTION_RAW_AUDIO_DAYS`` is 0 and this is where
        that policy is realised: features go forward, audio does not.
    """
    return extract_features(preprocess(sample.waveform, sample.sample_rate))


def process_person(
    pseudonym_id: str, samples: Sequence[VoiceSample]
) -> tuple[List[VoiceStressResult], voice_baseline.VoiceBaseline]:
    """Process one person's full check-in history in order.

    Args:
        pseudonym_id: The person's pseudonym. Raw personnel ids are mapped
            before this function is called; nothing in the voice pipeline sees
            a direct identifier.
        samples: That person's samples, oldest first.

    Returns:
        Tuple of (one result per sample, the final baseline after all samples
        have been folded in). Early samples yield unreliable results with a
        stated reason rather than being silently dropped, so a person can be
        told honestly that their baseline is still being established.

    Note:
        Unusable recordings (too little voiced material) are skipped for both
        scoring *and* baseline updating. A recording the extractor could not
        measure should not quietly widen the baseline it failed to measure.
    """
    results: List[VoiceStressResult] = []
    history: List[Dict[str, float]] = []
    baseline = voice_baseline.VoiceBaseline(pseudonym_id=pseudonym_id)

    for sample in samples:
        features = analyse_sample(sample)
        if not features.is_usable:
            results.append(
                VoiceStressResult(
                    pseudonym_id=pseudonym_id,
                    sample_date=sample.sample_date,
                    signal=float("nan"),
                    is_reliable=False,
                    reason="recording contained too little voiced material to measure",
                )
            )
            continue

        vector = features.feature_vector()

        # Score against the baseline as it stands BEFORE this sample.
        results.append(
            compute_voice_stress_signal(vector, baseline, sample.sample_date)
        )

        # Then fold this sample in. The first few samples rebuild the baseline
        # from the whole (short) history using robust statistics; once enough
        # history exists, incremental EMA updating takes over so the baseline
        # can drift slowly with the person rather than being recomputed from
        # an ever-growing store of past feature vectors.
        history.append(vector)
        if len(history) <= settings.VOICE_BASELINE_MIN_SAMPLES:
            baseline = voice_baseline.build_baseline(
                pseudonym_id,
                history,
                last_updated=str(sample.sample_date.date()),
            )
        else:
            baseline = voice_baseline.update_baseline(
                baseline, vector, sample_date=str(sample.sample_date.date())
            )

    return results, baseline


def process_all(
    id_mapper: Callable[[str], str],
    index: pd.DataFrame | None = None,
    raw_dir: Path | None = None,
) -> tuple[List[VoiceStressResult], Dict[str, voice_baseline.VoiceBaseline]]:
    """Process every consented check-in in the corpus.

    Args:
        id_mapper: Function turning a ``personnel_id`` into a ``pseudonym_id``.
            Normally ``PseudonymVault.pseudonym_for``. Passed in rather than
            imported so this module never opens the identity map itself.
        index: Voice-sample index. Loaded from disk when omitted.
        raw_dir: Root for relative audio paths.

    Returns:
        Tuple of (all results across all people, final baseline per pseudonym).
    """
    raw_dir = raw_dir or settings.RAW_DATA_DIR
    index = load_voice_index(raw_dir) if index is None else index

    all_results: List[VoiceStressResult] = []
    baselines: Dict[str, voice_baseline.VoiceBaseline] = {}

    for personnel_id, group in index.groupby("personnel_id"):
        pseudonym_id = id_mapper(str(personnel_id))
        samples = [
            s
            for s in (
                load_sample(row, raw_dir)
                for _, row in group.sort_values("sample_date").iterrows()
            )
            if s is not None
        ]
        if not samples:
            continue
        results, baseline = process_person(pseudonym_id, samples)
        all_results.extend(results)
        baselines[pseudonym_id] = baseline

    return all_results, baselines
