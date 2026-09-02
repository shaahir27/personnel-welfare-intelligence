"""Run the data stages in order, from raw CSVs to behavioral signals.

One job: sequence the stages. Every step of real work lives in its own module;
this file only calls them in the right order and carries the intermediate
results.

It exists so that the training script, the batch scoring script and the API all
build signals the *same* way. A second, slightly different assembly of these
stages somewhere else is how a model ends up scoring inputs that were not built
the way its training inputs were.

Stage order:
    ingestion -> cleaning -> pseudonymisation -> feature engineering
    -> voice pipeline (optional) -> behavioral signals
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from backend.behavioral_engine import behavioral_signals
from backend.config import settings
from backend.feature_engineering import assemble, hr_features
from backend.ingestion import hr_loader
from backend.preprocessing import clean, pseudonymize
from backend.voice_pipeline import pipeline as voice_pipeline
from backend.voice_pipeline import voice_stress_signal


@dataclass
class PipelineOutput:
    """Everything the data stages produced.

    Attributes:
        raw_tables: Validated raw tables.
        pseudonymised: Tables after cleaning and pseudonymisation.
        features: The engineered feature matrix.
        signals: The behavioral-signal matrix -- the model's input.
        vault: The pseudonym vault used, for later re-identification.
        snapshot_dates: The as-of dates everything was computed for.
        cleaning_log: What the cleaning stage changed.
        voice_results: Per-check-in voice results, including unreliable ones.
        voice_baselines: Final personal voice baseline per pseudonym.
    """

    raw_tables: Dict[str, pd.DataFrame]
    pseudonymised: Dict[str, pd.DataFrame]
    features: pd.DataFrame
    signals: pd.DataFrame
    vault: pseudonymize.PseudonymVault
    snapshot_dates: List[pd.Timestamp]
    cleaning_log: clean.CleaningLog
    voice_results: List = field(default_factory=list)
    voice_baselines: Dict = field(default_factory=dict)

    @property
    def voice_coverage(self) -> float:
        """Share of signal rows carrying a real voice reading."""
        flag = settings.VOICE_PRESENCE_FLAG_NAME
        if flag not in self.signals.columns or self.signals.empty:
            return 0.0
        return float(self.signals[flag].mean())


def run(
    raw_dir: Path | None = None,
    snapshot_dates: Sequence[pd.Timestamp] | None = None,
    include_voice: bool = True,
    vault: pseudonymize.PseudonymVault | None = None,
) -> PipelineOutput:
    """Run every data stage and return the results.

    Args:
        raw_dir: Directory holding the raw CSVs.
        snapshot_dates: As-of dates. Defaults to the project standard.
        include_voice: Whether to run the voice pipeline. When False, every
            row gets a neutral voice value and a presence flag of 0 -- the
            same path a person who never opted in takes.
        vault: Pseudonym vault to use. A default one is opened when omitted.

    Returns:
        A :class:`PipelineOutput`.

    Raises:
        ValueError: If any raw table fails validation. The pipeline runs in
            strict mode: a bad table means the run is worthless, and producing
            welfare scores from data known to be broken would be worse than
            producing none.
    """
    raw_dir = raw_dir or settings.RAW_DATA_DIR
    snapshots = list(snapshot_dates or hr_features.default_snapshot_dates())

    load_result = hr_loader.load_hr_tables(raw_dir=raw_dir, strict=True)
    cleaned, cleaning_log = clean.clean_all(load_result.tables)
    pseudonymised, vault = pseudonymize.pseudonymize_tables(cleaned, vault)

    features = assemble.build_feature_matrix(pseudonymised, snapshots)

    voice_results: List = []
    voice_baselines: Dict = {}
    voice_frame: pd.DataFrame | None = None
    if include_voice:
        voice_results, voice_baselines = voice_pipeline.process_all(
            vault.pseudonym_for, raw_dir=raw_dir
        )
        voice_frame = voice_stress_signal.signals_to_frame(voice_results, snapshots)

    signals = behavioral_signals.compute_behavioral_signals(features, voice_frame)

    return PipelineOutput(
        raw_tables=load_result.tables,
        pseudonymised=pseudonymised,
        features=features,
        signals=signals,
        vault=vault,
        snapshot_dates=snapshots,
        cleaning_log=cleaning_log,
        voice_results=voice_results,
        voice_baselines=voice_baselines,
    )


def load_labels(output: PipelineOutput, raw_dir: Path | None = None) -> pd.DataFrame:
    """Load the synthetic training labels, keyed by pseudonym.

    Args:
        output: A completed pipeline run, whose vault provides the mapping.
        raw_dir: Directory holding the raw CSVs.

    Returns:
        Frame with ``pseudonym_id``, ``snapshot_date`` and the target column.

    Note:
        Kept out of :func:`run` deliberately. Labels exist only because this
        corpus is synthetic, and only the training path has any business
        touching them. The served API calls ``run`` and never this.
    """
    labels = hr_loader.load_ground_truth_labels(raw_dir or settings.RAW_DATA_DIR)
    labels = labels.copy()
    labels["pseudonym_id"] = [
        output.vault.pseudonym_for(str(pid)) for pid in labels["personnel_id"]
    ]
    return labels[["pseudonym_id", "snapshot_date", settings.MODEL_TARGET_NAME]]
