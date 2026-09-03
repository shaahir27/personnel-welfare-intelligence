"""Metrics may not travel without a statement of what they were measured against.

Why this test exists
--------------------
Twice now the same defect has been introduced by careful people. The long-form
caveat sections in ``STATUS.md`` and ``docs/model_comparison_report.md`` were
both updated correctly when the headline number moved 0.807 -> 0.821, including
the figures derived from it. What was missed both times was the place where the
number appears with no room for a caveat beside it -- a README bullet and a
table cell in the PS alignment matrix.

That is the failure mode: the caveat survives wherever there is a paragraph to
hold it, and is lost wherever the number travels alone. Numbers travel alone
constantly -- into slides, into table cells, into a spoken answer. Prose
discipline does not scale to that; a test does.

This is a floor, not a ceiling. It asserts that a document quoting a headline
metric also carries the provenance and points at the single source of truth. It
cannot assert that the caveat sits *next to* the number, which is still a matter
of writing it well.

Companion guarantees, enforced here too:
  - ``settings.TRAINING_LABEL_NAME`` and ``settings.MODEL_TARGET_NAME`` stay
    distinct. The label is synthetic; the prediction is not claimed to be.
  - ``MODEL_TARGET_NAME`` stays in ``COMMANDER_FORBIDDEN_FIELDS``. It is the
    string ``rbac.assert_commander_safe()`` matches on, and the one every
    payload already in ``data/processed/`` uses, so renaming it would quietly
    stop the commander leak guard from catching anything.
  - ``model_registry.save()`` refuses to write a version whose metrics carry no
    provenance.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.dummy import DummyRegressor

from backend.config import settings
from backend.models import model_registry

REPO_ROOT = Path(__file__).resolve().parents[1]

# Documents a judge or reviewer reads. Each must carry the provenance and point
# at the source of truth whenever it quotes a headline metric.
JUDGE_FACING = ("README.md", "STATUS.md", "docs/ps_alignment_matrix.md")

# The one document allowed to state the caveat in full. Everything else links.
SOURCE_OF_TRUTH = "docs/model_comparison_report.md"

# A headline metric: an R-squared with a value, or a conformal coverage claim.
METRIC_PATTERNS = (
    re.compile(r"R\s*(?:²|\^2)[^0-9\n]{0,8}0\.\d+"),
    re.compile(r"\d+(?:\.\d+)?\s*%\s*(?:empirical\s+)?coverage", re.IGNORECASE),
    re.compile(r"coverage\s+guaranteed", re.IGNORECASE),
)

# Any one of these, anywhere in the file, satisfies the provenance requirement.
PROVENANCE_MARKERS = (
    "synthetic label",
    "synthetic corpus",
    "generator's formula",
    "latent_welfare_risk",
    "predictive validity",
    "formula recovery",
    settings.TRAINING_LABEL_NAME,
)


def _read(relative: str) -> str:
    """Return the text of a repository-relative file."""
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _quotes_a_metric(text: str) -> bool:
    """Whether the text states a headline metric value."""
    return any(pattern.search(text) for pattern in METRIC_PATTERNS)


def _has_marker(text: str) -> bool:
    """Whether the text carries any accepted provenance marker."""
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in PROVENANCE_MARKERS)


class TestDocumentsCarryProvenance(unittest.TestCase):

    def test_judge_facing_docs_carry_provenance_and_link_to_source(self) -> None:
        for relative in JUDGE_FACING:
            with self.subTest(document=relative):
                text = _read(relative)
                if not _quotes_a_metric(text):
                    continue
                self.assertTrue(
                    _has_marker(text),
                    f"{relative} quotes a headline metric but contains none of "
                    f"{PROVENANCE_MARKERS}. A number without its provenance is "
                    f"the defect this test exists to prevent.",
                )
                self.assertIn(
                    "model_comparison_report.md",
                    text,
                    f"{relative} quotes a headline metric but does not point at "
                    f"{SOURCE_OF_TRUTH}, where the caveat is maintained.",
                )

    def test_source_of_truth_states_the_caveat(self) -> None:
        text = _read(SOURCE_OF_TRUTH)
        self.assertTrue(_quotes_a_metric(text))
        self.assertTrue(_has_marker(text))

    def test_the_pattern_would_actually_fire(self) -> None:
        # Guards against a regex that silently matches nothing, which would make
        # every assertion above vacuously true.
        firing = (
            "R² = 0.821",
            "$R^2 = 0.821$",
            "R² 0.821",
            "verified 91.5% coverage",
            "90% coverage guaranteed",
        )
        for sample in firing:
            with self.subTest(sample=sample):
                self.assertTrue(_quotes_a_metric(sample))

        for sample in ("| Model | MAE | RMSE | R² |", "no numbers here"):
            with self.subTest(sample=sample):
                self.assertFalse(_quotes_a_metric(sample))


class TestSettingsKeepLabelAndPredictionDistinct(unittest.TestCase):

    def test_label_and_prediction_names_differ(self) -> None:
        self.assertNotEqual(settings.TRAINING_LABEL_NAME, settings.MODEL_TARGET_NAME)

    def test_training_label_names_itself_synthetic(self) -> None:
        self.assertIn("synthetic", settings.TRAINING_LABEL_NAME)

    def test_prediction_name_is_still_commander_forbidden(self) -> None:
        # rbac.assert_commander_safe() matches on this exact string, and every
        # payload in data/processed/ uses it. Renaming it without updating the
        # forbidden list would disable the leak guard silently.
        self.assertIn(settings.MODEL_TARGET_NAME, settings.COMMANDER_FORBIDDEN_FIELDS)

    def test_label_provenance_is_substantive_and_names_the_label(self) -> None:
        self.assertGreater(len(settings.LABEL_PROVENANCE), 100)
        self.assertIn(settings.TRAINING_LABEL_NAME, settings.LABEL_PROVENANCE)
        self.assertIn("model_comparison_report.md", settings.LABEL_PROVENANCE)


class TestRegistryRequiresProvenance(unittest.TestCase):

    def _save(self, registry_dir: Path, **overrides) -> Path:
        """Call ``model_registry.save`` with valid defaults plus overrides."""
        kwargs = dict(
            estimator=DummyRegressor().fit([[0.0], [1.0]], [0.0, 1.0]),
            model_name="dummy",
            display_name="Dummy",
            is_tree_based=False,
            metrics={"r2": 0.5},
            selection_reason="test",
            training_rows=2,
            training_people=2,
            feature_names=["x"],
            registry_dir=registry_dir,
            mark_current=False,
        )
        kwargs.update(overrides)
        return model_registry.save(**kwargs)

    def test_save_refuses_without_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self._save(Path(tmp))
            with self.assertRaises(ValueError):
                self._save(Path(tmp), label_provenance="   ")

    def test_save_writes_provenance_into_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            version_dir = self._save(
                Path(tmp), label_provenance=settings.LABEL_PROVENANCE
            )
            meta = json.loads(
                (version_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(meta["label_provenance"], settings.LABEL_PROVENANCE)

    def test_deployed_model_carries_provenance(self) -> None:
        current = (
            (settings.MODEL_REGISTRY_DIR / "CURRENT").read_text(encoding="utf-8").strip()
        )
        meta = json.loads(
            (settings.MODEL_REGISTRY_DIR / current / "metadata.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(meta.get("label_provenance", "").strip())


if __name__ == "__main__":
    unittest.main()
