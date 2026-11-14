"""Smoke test for scripts/fs005_phase5_full_run.py using a stub VLM.

Skipped when the SLAKE data folder is not present. Uses no real model.
"""

import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts import fs005_phase5_full_run as drv
from vlm.base import VLMClient

pytestmark = pytest.mark.skipif(
    not (drv.SLAKE_DIR / "validate.json").exists(), reason="SLAKE data not available"
)


class _Cites(VLMClient):
    """Always answers 'yes' and cites R1 (positive) plus an invalid region."""

    @property
    def model_name(self) -> str:
        return "stub"

    @property
    def device(self) -> str:
        return "cpu"

    @staticmethod
    def _resp() -> str:
        return json.dumps(
            {
                "answer": "yes",
                "findings": [
                    {"claim": "organ visible", "region_id": "R1", "status": "present",
                     "confidence": "high"},
                    {"claim": "made up", "region_id": "R99", "status": "present",
                     "confidence": "low"},
                ],
            }
        )

    def generate(self, marked_image, question, temperature=0.0, max_tokens=512):
        return self._resp()

    def sample_n(self, marked_image, question, n=3, temperature=0.7, max_tokens=512):
        return [self._resp()] * n


def _args(out: Path, split: str, track: str = "B") -> Namespace:
    return Namespace(
        model="medgemma-4b", track=track, split=split, limit=5, mock=False,
        out_dir=str(out),
    )


def test_validation_cohorts_match_committed_manifests() -> None:
    fit = drv._cohort("val_fit", "B")
    assess = drv._cohort("val_assess", "B")
    assert len(fit) == 719 and len(assess) == 172
    assert {e["img_id"] for e in fit}.isdisjoint({e["img_id"] for e in assess})


def test_test_cohort_sizes() -> None:
    assert len(drv._cohort("test", "C")) == 945
    assert len(drv._cohort("test", "B")) == 864


def test_claim_statistics_flow_through_evaluation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(drv, "_make_client", lambda args: _Cites())
    for split in ("val_fit", "val_assess", "test"):
        assert drv.cmd_generate(_args(tmp_path, split)) == 0
    # Limit the cohort for the scoring stages to the five generated questions.
    orig = drv._cohort
    monkeypatch.setattr(drv, "_cohort", lambda s, t: orig(s, t)[:5])
    assert drv.cmd_calibrate(_args(tmp_path, "val_fit")) == 0
    assert drv.cmd_evaluate(_args(tmp_path, "test")) == 0

    metrics = json.loads(
        (tmp_path / "evaluation" / "medgemma-4b" / "B" / "condition2"
         / "experiment_metrics.json").read_text("utf-8")
    )
    m = metrics["metrics"]
    # One supported (R1) and one unverifiable (R99) positive claim per sample,
    # repeated over N=5 samples -> USR_raw = 0.5, and the oracle region is cited.
    assert m["usr_raw"] == pytest.approx(0.5)
    assert m["pointing_accuracy"] == pytest.approx(1.0)
    assert m["sampling_agreement"] == pytest.approx(1.0)
