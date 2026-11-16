"""Regression test: agreement/entropy must cluster parsed answers, not raw JSON.

Earlier the generation harness clustered each sample's raw JSON response, so
five samples with the SAME answer but different findings looked like five
distinct answers (agreement 0.2, entropy 1.0). The harness must cluster the
parsed answer text only.
"""

import json

from scripts.fs005_phase5_generation import GenerationRunner
from vlm.base import VLMClient


class _SameAnswerDifferentFindings(VLMClient):
    """Returns one answer with a different finding text for every sample."""

    @property
    def model_name(self) -> str:
        return "stub"

    @property
    def device(self) -> str:
        return "cpu"

    def generate(self, marked_image, question, temperature=0.0, max_tokens=512):
        return self._one(0)

    @staticmethod
    def _one(i: int) -> str:
        return json.dumps(
            {
                "answer": "Yes",
                "findings": [
                    {
                        "claim": f"finding variant {i}",
                        "region_id": "NONE",
                        "status": "uncertain",
                        "confidence": "low",
                    }
                ],
            }
        )

    def sample_n(self, marked_image, question, n=3, temperature=0.7, max_tokens=512):
        return [self._one(i) for i in range(n)]


def test_same_answer_different_findings_is_full_agreement(tmp_path) -> None:
    runner = GenerationRunner(
        vlm_client=_SameAnswerDifferentFindings(),
        model_name="stub",
        model_revision=None,
        track="C",
        output_dir=tmp_path,
        n_samples=5,
        temperature=0.7,
        random_seed=1,
    )
    record, _ = runner.generate_or_load(
        question_id="q1", image_id="i1", question="Is it normal?", image=None
    )
    result = runner.evaluate_condition_2(record)
    assert result.agreement == 1.0
    assert result.normalized_entropy == 0.0
    assert result.majority_answer == "Yes"
    assert result.raw_answers == ["Yes"] * 5


class _NotJson(_SameAnswerDifferentFindings):
    def sample_n(self, marked_image, question, n=3, temperature=0.7, max_tokens=512):
        return ["Sure! The answer is yes."] * n


def test_unparseable_responses_are_recorded_as_failed_not_crash(tmp_path) -> None:
    """Regression: invalid JSON used to crash the run (failed record had no error)."""
    runner = GenerationRunner(
        vlm_client=_NotJson(),
        model_name="stub",
        model_revision=None,
        track="C",
        output_dir=tmp_path,
        n_samples=5,
        temperature=0.7,
        random_seed=1,
    )
    summary = runner.run_batch(
        [{"question_id": "q1", "image_id": "i1", "question": "Q?", "region_ids": []}],
        {"i1": None},
    )
    assert summary["n_failed"] == 1 and summary["n_completed"] == 0
    err = summary["progress"]["failures"][0]["error"]
    assert "samples failed" in err and "Sure! The answer is yes." in err


def test_extract_json_object_handles_fences_and_preamble() -> None:
    from scripts.fs005_phase5_generation import extract_json_object

    obj = '{"answer": "Yes", "findings": []}'
    assert extract_json_object(obj) == obj
    assert extract_json_object(f"```json\n{obj}\n```") == obj
    assert extract_json_object(f"```\n{obj}\n```") == obj
    assert extract_json_object(f"Sure, here it is: {obj} Hope that helps.") == obj
    # Non-JSON stays non-JSON: it must still fail validation, not be repaired.
    assert extract_json_object("Sure! The answer is yes.") == "Sure! The answer is yes."
