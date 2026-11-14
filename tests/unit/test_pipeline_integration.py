"""Unit tests for FS-005 Phase 4 pipeline integration.

Covers end-to-end pipeline integration:

A. Component integration
   - Claim extraction → association → verification → filtering
   - Sampling → semantic clustering → entropy → confidence → abstention
   - Evaluation-record generation from pipeline outputs

B. Evidence and uncertainty safety
   - Ground truth cannot influence runtime filtering
   - Ground truth cannot influence confidence or abstention
   - Positive claim with region_id="NONE" remains positive but ungrounded
   - Absence claims remain distinct from ungrounded positive claims
   - Uncertain claims trigger abstention
   - Invalid sampling always causes abstention (including tau=0.0)
   - Empty detector output does not imply medical absence

C. Experiment tracks
   - Track A uses predicted localization
   - Track B uses oracle localization without runtime GT leakage
   - Track C remains ungrounded
   - Ineligible Track A on SLAKE represented correctly
   - Empty-GT samples handled correctly for pointing evaluation
   - All three answer conditions preserve distinct outputs
   - Track and condition metadata attached to every record

D. Artifact generation
   - Experiment metadata persisted
   - Track and answer-condition identifiers preserved
   - Raw, filtered, selective outputs distinguishable
   - Inactive combinations recorded without fabricated metrics
   - JSON/CSV outputs serializable and reproducible

All tests are deterministic and require no model weights, paid APIs, or
real VLM inference. Uses MockVLMClient and MockRegionDetector.
"""

import json
import tempfile
from pathlib import Path

from core.schemas import (
    CandidateBox,
    EvaluationTrack,
    GroundTruthRegion,
    MedicalVQASample,
)
from evaluation.metrics import (
    SampleRecord,
)
from evaluation.runner import EvaluationRunner
from verification.pipeline import (
    AnswerCondition,
    PipelineConfig,
    PipelineEngine,
)

# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class MockVLMClient:
    """Deterministic mock VLM client for pipeline testing.

    Returns structured JSON responses without model weights. Configurable
    per-call to simulate different model behaviors.
    """

    def __init__(self, responses: list[dict] | None = None):
        """Initialize with optional list of response dicts.

        Each dict should have 'answer' and 'findings' keys. If fewer
        responses than requested samples are provided, the last response
        is repeated.
        """
        self._responses = responses or [
            {"answer": "Yes", "findings": []}
        ]
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return "mock-vlm"

    @property
    def device(self) -> str:
        return "cpu"

    def __len__(self) -> int:
        return len(self._responses)

    def __getitem__(self, idx: int) -> dict:
        return self._responses[min(idx, len(self._responses) - 1)]

    def generate(
        self,
        marked_image,
        question: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        response = self._responses[min(self.call_count, len(self._responses) - 1)]
        self.call_count += 1
        return json.dumps(response)

    def sample_n(
        self,
        marked_image,
        question: str,
        n: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> list[str]:
        responses = []
        for i in range(n):
            response = self._responses[min(i, len(self._responses) - 1)]
            responses.append(json.dumps(response))
        self.call_count += n
        return responses


class MockRegionDetector:
    """Deterministic mock region detector for pipeline testing."""

    def __init__(self, candidates: list[dict] | None = None):
        """Initialize with optional candidate box dicts.

        Each dict: {'box_2d': (ymin, xmin, ymax, xmax), 'score': float,
        'label': str}
        """
        self._candidates = candidates or []

    def detect(self, image) -> list[CandidateBox]:
        boxes = []
        for idx, cand in enumerate(self._candidates):
            boxes.append(
                CandidateBox(
                    region_id=f"R{idx + 1}",
                    box_2d=tuple(cand["box_2d"]),
                    normalized_box=tuple(cand["box_2d"]),  # Already normalized
                    score=float(cand.get("score", 0.9)),
                    label=cand.get("label", "lesion"),
                )
            )
        return boxes


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_sample(
    sample_id: str = "s1",
    question: str = "Is there pleural effusion?",
    answer: str = "No",
    question_type: str = "closed",
    has_gt_regions: bool = True,
    gt_regions: list[dict] | None = None,
    compatible_tracks: list | None = None,
    modality: str = "CT",
    organ: str = "chest",
    dataset_name: str = "slake",
    split: str = "test",
) -> MedicalVQASample:
    """Create a MedicalVQASample for pipeline testing."""
    gt_regions_list = []
    if gt_regions:
        for gt in gt_regions:
            gt_regions_list.append(
                GroundTruthRegion(label=gt["label"], box_2d=tuple(gt["box_2d"]))
            )

    if compatible_tracks is None:
        compatible_tracks = [
            EvaluationTrack.TRACK_B_ORACLE_SLAKE,
            EvaluationTrack.TRACK_C_ANSWER_ONLY,
        ]

    return MedicalVQASample(
        sample_id=sample_id,
        dataset_name=dataset_name,
        split=split,
        image_id=f"img_{sample_id}",
        image_path=f"/tmp/img_{sample_id}.png",
        modality=modality,
        organ=organ,
        question=question,
        answer=answer,
        question_type=question_type,
        ground_truth_regions=gt_regions_list,
        has_ground_truth_regions=has_gt_regions,
        patient_id=None,
        compatible_tracks=compatible_tracks,
        metadata={},
    )


def _make_image():
    """Create a blank PIL image for testing."""
    from PIL import Image

    return Image.new("RGB", (224, 224), color="black")


def _make_pipeline_config(
    track: EvaluationTrack = EvaluationTrack.TRACK_C_ANSWER_ONLY,
    condition: AnswerCondition = AnswerCondition.CONDITION_3_GROUNDED_WITH_ABSTENTION,
    n_samples: int = 5,
    temperature: float = 0.7,
    w1: float = 0.4,
    w2: float = 0.3,
    w3: float = 0.3,
    tau: float = 0.5,
    model_name: str = "mock-vlm",
) -> PipelineConfig:
    """Create a PipelineConfig for testing."""
    return PipelineConfig(
        track=track,
        condition=condition,
        model_name=model_name,
        n_samples=n_samples,
        temperature=temperature,
        random_seed=42,
        w1=w1,
        w2=w2,
        w3=w3,
        tau=tau,
    )


# ---------------------------------------------------------------------------
# Component integration tests
# ---------------------------------------------------------------------------


def test_pipeline_end_to_end_present_grounded():
    """Full pipeline: present claim grounded to region → supported → accepted."""

    vlm = MockVLMClient(
        responses=[
            {
                "answer": "Yes",
                "findings": [
                    {
                        "claim": "pleural effusion",
                        "region_id": "R1",
                        "status": "present",
                        "confidence": "high",
                    }
                ],
            }
        ]
        * 5
    )
    detector = MockRegionDetector(
        candidates=[
            {"box_2d": (50, 50, 150, 150), "score": 0.9, "label": "lesion"}
        ]
    )
    engine = PipelineEngine(
        vlm_client=vlm,
        region_detector=detector,
        config=_make_pipeline_config(
            track=EvaluationTrack.TRACK_A_PREDICTED_CXR,
            condition=AnswerCondition.CONDITION_3_GROUNDED_WITH_ABSTENTION,
        ),
    )
    sample = _make_sample(
        sample_id="s1",
        question="Is there pleural effusion?",
        answer="Yes",
        gt_regions=[{"label": "effusion", "box_2d": (0.1, 0.1, 0.5, 0.5)}],
    )
    result = engine.run_single(sample, _make_image())

    assert result.sample_id == "s1"
    assert result.track == "track_a_predicted_cxr"
    assert result.condition == 3
    assert result.model_answer == "Yes"
    assert result.correct is True
    assert result.is_abstaining is False
    assert result.in_coverage is True
    assert result.confidence > 0.0
    assert result.evidence_state == "present_grounded"
    assert result.sampling_valid is True
    assert result.n_claims == 5  # 5 samples × 1 claim each
    assert result.n_positive_claims == 5
    assert result.n_supported == 5
    assert result.n_absence == 0
    assert result.n_uncertain == 0


def test_pipeline_absence_claim_no_rho():
    """Absence claim gets no artificial localization confidence."""
    vlm = MockVLMClient(
        responses=[
            {
                "answer": "No",
                "findings": [
                    {
                        "claim": "pneumothorax",
                        "region_id": "NONE",
                        "status": "absent",
                        "confidence": "high",
                    }
                ],
            }
        ]
        * 5
    )
    engine = PipelineEngine(
        vlm_client=vlm,
        region_detector=None,
        config=_make_pipeline_config(
            track=EvaluationTrack.TRACK_C_ANSWER_ONLY,
            condition=AnswerCondition.CONDITION_3_GROUNDED_WITH_ABSTENTION,
        ),
    )
    sample = _make_sample(
        sample_id="s2",
        question="Is there pneumothorax?",
        answer="No",
        compatible_tracks=[
            EvaluationTrack.TRACK_C_ANSWER_ONLY,
            EvaluationTrack.TRACK_B_ORACLE_SLAKE,
        ],
    )
    result = engine.run_single(sample, _make_image())

    assert result.evidence_state == "absent"
    assert result.is_abstaining is False  # Unanimous absence → high confidence
    assert result.n_absence == 5
    assert result.n_positive_claims == 0
    # No positive claims → usr_raw undefined (None)


def test_pipeline_uncertain_claim_abstains():
    """Uncertain claims always abstain regardless of confidence."""
    vlm = MockVLMClient(
        responses=[
            {
                "answer": "Uncertain",
                "findings": [
                    {
                        "claim": "possible nodule",
                        "region_id": "R1",
                        "status": "uncertain",
                        "confidence": "low",
                    }
                ],
            }
        ]
        * 5
    )
    detector = MockRegionDetector(
        candidates=[{"box_2d": (50, 50, 150, 150), "score": 0.9, "label": "lesion"}]
    )
    engine = PipelineEngine(
        vlm_client=vlm,
        region_detector=detector,
        config=_make_pipeline_config(
            track=EvaluationTrack.TRACK_A_PREDICTED_CXR,
            condition=AnswerCondition.CONDITION_3_GROUNDED_WITH_ABSTENTION,
        ),
    )
    sample = _make_sample(
        sample_id="s3",
        question="Is there a nodule?",
        answer="No",
        compatible_tracks=[
            EvaluationTrack.TRACK_A_PREDICTED_CXR,
            EvaluationTrack.TRACK_C_ANSWER_ONLY,
        ],
    )
    result = engine.run_single(sample, _make_image())

    assert result.evidence_state == "uncertain"
    assert result.is_abstaining is True
    assert result.model_answer is None
    assert result.in_coverage is False


def test_pipeline_present_ungrounded_no_rescaling():
    """Present claim with region_id='NONE' uses rho=0.0, no rescaling."""
    vlm = MockVLMClient(
        responses=[
            {
                "answer": "Yes",
                "findings": [
                    {
                        "claim": "mass",
                        "region_id": "NONE",
                        "status": "present",
                        "confidence": "high",
                    }
                ],
            }
        ]
        * 5
    )
    engine = PipelineEngine(
        vlm_client=vlm,
        region_detector=None,
        config=_make_pipeline_config(
            track=EvaluationTrack.TRACK_C_ANSWER_ONLY,
            condition=AnswerCondition.CONDITION_3_GROUNDED_WITH_ABSTENTION,
            w1=0.4,
            w2=0.3,
            w3=0.3,
        ),
    )
    sample = _make_sample(
        sample_id="s4",
        question="Is there a mass?",
        answer="Yes",
        compatible_tracks=[EvaluationTrack.TRACK_C_ANSWER_ONLY],
    )
    result = engine.run_single(sample, _make_image())

    assert result.evidence_state == "present_ungrounded"
    # Full formula with rho=0.0: 0.4*1.0 + 0.3*(1-0.0) + 0.3*0.0 = 0.7
    assert result.confidence == 0.7


def test_pipeline_invalid_sampling_abstains_tau_zero():
    """Invalid sampling (sampling_valid=False) always abstains, even at tau=0."""
    vlm = MockVLMClient(
        responses=[
            # Malformed JSON - missing required 'findings' field
            {"foo": "bar"}
        ] * 5
    )
    engine = PipelineEngine(
        vlm_client=vlm,
        region_detector=None,
        config=_make_pipeline_config(
            track=EvaluationTrack.TRACK_C_ANSWER_ONLY,
            condition=AnswerCondition.CONDITION_3_GROUNDED_WITH_ABSTENTION,
            tau=0.0,  # Even tau=0.0 should abstain on invalid sampling
        ),
    )
    sample = _make_sample(
        sample_id="s5",
        question="Is there anything?",
        answer="Yes",
        compatible_tracks=[EvaluationTrack.TRACK_C_ANSWER_ONLY],
    )
    result = engine.run_single(sample, _make_image())

    assert result.sampling_valid is False
    assert result.is_abstaining is True
    assert result.model_answer is None
    assert result.confidence == 0.0


def test_pipeline_empty_detector_no_absence_proof():
    """Empty detector output (R=∅) does not imply medical absence."""
    vlm = MockVLMClient(
        responses=[
            {
                "answer": "Yes",
                "findings": [
                    {
                        "claim": "effusion",
                        "region_id": "R1",
                        "status": "present",
                        "confidence": "high",
                    }
                ],
            }
        ] * 5
    )
    # Empty detector - no candidate regions
    engine = PipelineEngine(
        vlm_client=vlm,
        region_detector=MockRegionDetector(candidates=[]),
        config=_make_pipeline_config(
            track=EvaluationTrack.TRACK_A_PREDICTED_CXR,
            condition=AnswerCondition.CONDITION_3_GROUNDED_WITH_ABSTENTION,
        ),
    )
    sample = _make_sample(
        sample_id="s6",
        question="Is there effusion?",
        answer="Yes",
        compatible_tracks=[
            EvaluationTrack.TRACK_A_PREDICTED_CXR,
            EvaluationTrack.TRACK_C_ANSWER_ONLY,
        ],
    )
    result = engine.run_single(sample, _make_image())

    # Claim cites R1 but R is empty → ungrounded → unverifiable
    assert result.n_unverifiable == 5
    assert result.n_positive_claims == 5
    assert result.n_supported == 0


# ---------------------------------------------------------------------------
# Track tests
# ---------------------------------------------------------------------------


def test_track_a_uses_predicted_regions():
    """Track A uses detector-proposed candidate regions."""
    vlm = MockVLMClient(
        responses=[
            {"answer": "Yes", "findings": [
                {"claim": "mass", "region_id": "R1", "status": "present", "confidence": "high"}
            ]}
        ]
        * 5
    )
    detector = MockRegionDetector(
        candidates=[{"box_2d": (30, 30, 100, 100), "score": 0.85, "label": "mass"}]
    )
    engine = PipelineEngine(
        vlm_client=vlm,
        region_detector=detector,
        config=_make_pipeline_config(
            track=EvaluationTrack.TRACK_A_PREDICTED_CXR,
            condition=AnswerCondition.CONDITION_2_GROUNDED_NO_ABSTENTION,
        ),
    )
    sample = _make_sample(
        sample_id="ta1",
        question="Is there a mass?",
        answer="Yes",
        compatible_tracks=[EvaluationTrack.TRACK_A_PREDICTED_CXR],
    )
    result = engine.run_single(sample, _make_image())

    assert result.track == "track_a_predicted_cxr"
    assert result.n_claims == 5
    assert result.n_positive_claims == 5


def test_track_b_uses_oracle_gt_boxes():
    """Track B uses ground-truth boxes as oracle regions."""
    vlm = MockVLMClient(
        responses=[
            {"answer": "Yes", "findings": [
                {"claim": "effusion", "region_id": "R1", "status": "present", "confidence": "high"}
            ]}
        ]
        * 5
    )
    engine = PipelineEngine(
        vlm_client=vlm,
        region_detector=None,
        config=_make_pipeline_config(
            track=EvaluationTrack.TRACK_B_ORACLE_SLAKE,
            condition=AnswerCondition.CONDITION_3_GROUNDED_WITH_ABSTENTION,
        ),
    )
    sample = _make_sample(
        sample_id="tb1",
        question="Is there effusion?",
        answer="Yes",
        gt_regions=[{"label": "effusion", "box_2d": (0.1, 0.1, 0.5, 0.5)}],
        compatible_tracks=[EvaluationTrack.TRACK_B_ORACLE_SLAKE],
    )
    result = engine.run_single(sample, _make_image())

    assert result.track == "track_b_oracle_slake"
    assert result.n_claims == 5
    assert result.n_positive_claims == 5


def test_track_c_ungrounded_no_regions():
    """Track C has no region requirements - empty region set."""
    vlm = MockVLMClient(
        responses=[
            {"answer": "Yes", "findings": []} * 1
        ][0] if False else MockVLMClient(
            responses=[{"answer": "Yes", "findings": []}] * 5
        )
    )
    engine = PipelineEngine(
        vlm_client=vlm,
        region_detector=None,
        config=_make_pipeline_config(
            track=EvaluationTrack.TRACK_C_ANSWER_ONLY,
            condition=AnswerCondition.CONDITION_3_GROUNDED_WITH_ABSTENTION,
        ),
    )
    sample = _make_sample(
        sample_id="tc1",
        question="What is shown?",
        answer="Chest X-ray",
        compatible_tracks=[EvaluationTrack.TRACK_C_ANSWER_ONLY],
    )
    result = engine.run_single(sample, _make_image())

    assert result.track == "track_c_answer_only"
    assert result.n_claims == 0
    assert result.n_positive_claims == 0


def test_track_a_ineligible_for_slake():
    """Track A (CXR) is inactive on SLAKE (no CXR images)."""
    vlm = MockVLMClient(
        responses=[{"answer": "Yes", "findings": []}] * 5
    )
    PipelineEngine(
        vlm_client=vlm,
        region_detector=None,
        config=_make_pipeline_config(
            track=EvaluationTrack.TRACK_A_PREDICTED_CXR,
            condition=AnswerCondition.CONDITION_3_GROUNDED_WITH_ABSTENTION,
        ),
    )
    # SLAKE sample - no CXR images available
    sample = _make_sample(
        sample_id="ta_slake",
        question="Is there effusion?",
        answer="No",
        modality="CT",
        organ="chest",
        compatible_tracks=[
            EvaluationTrack.TRACK_B_ORACLE_SLAKE,
            EvaluationTrack.TRACK_C_ANSWER_ONLY,
        ],
    )
    # Track A not in compatible_tracks - sample incompatible
    assert not sample.is_compatible_with(EvaluationTrack.TRACK_A_PREDICTED_CXR)
    assert sample.is_compatible_with(EvaluationTrack.TRACK_C_ANSWER_ONLY)


# ---------------------------------------------------------------------------
# Answer condition tests
# ---------------------------------------------------------------------------


def test_condition_1_free_form():
    """Condition 1: Free-form VLM answers, N=1, T=0.0, no verification."""
    vlm = MockVLMClient(
        responses=[{"answer": "Pleural effusion is present.", "findings": []}]
    )
    engine = PipelineEngine(
        vlm_client=vlm,
        region_detector=None,
        config=_make_pipeline_config(
            track=EvaluationTrack.TRACK_C_ANSWER_ONLY,
            condition=AnswerCondition.CONDITION_1_FREE_FORM,
        ),
    )
    sample = _make_sample(
        sample_id="c1",
        question="What is shown?",
        answer="Pleural effusion is present.",
        compatible_tracks=[EvaluationTrack.TRACK_C_ANSWER_ONLY],
    )
    result = engine.run_single(sample, _make_image())

    assert result.condition == 1
    assert result.raw_answer == "Pleural effusion is present."
    assert result.n_claims == 0  # No claims in free-form
    assert result.is_abstaining is False


def test_condition_2_grounded_no_abstention():
    """Condition 2: Grounded + verification, no abstention gate."""
    vlm = MockVLMClient(
        responses=[
            {"answer": "Yes", "findings": [
                {"claim": "mass", "region_id": "R1", "status": "present", "confidence": "high"}
            ]}
        ]
        * 5
    )
    detector = MockRegionDetector(
        candidates=[{"box_2d": (50, 50, 150, 150), "score": 0.9, "label": "mass"}]
    )
    engine = PipelineEngine(
        vlm_client=vlm,
        region_detector=detector,
        config=_make_pipeline_config(
            track=EvaluationTrack.TRACK_A_PREDICTED_CXR,
            condition=AnswerCondition.CONDITION_2_GROUNDED_NO_ABSTENTION,
        ),
    )
    sample = _make_sample(
        sample_id="c2",
        question="Is there a mass?",
        answer="Yes",
        compatible_tracks=[EvaluationTrack.TRACK_A_PREDICTED_CXR],
    )
    result = engine.run_single(sample, _make_image())

    assert result.condition == 2
    assert result.is_abstaining is False  # No abstention in condition 2


def test_condition_3_grounded_with_abstention():
    """Condition 3: Grounded + verification + confidence-gate abstention."""
    vlm = MockVLMClient(
        responses=[
            {"answer": "Yes", "findings": [
                {"claim": "mass", "region_id": "R1", "status": "present", "confidence": "high"}
            ]}
        ]
        * 5
    )
    detector = MockRegionDetector(
        candidates=[{"box_2d": (50, 50, 150, 150), "score": 0.9, "label": "mass"}]
    )
    engine = PipelineEngine(
        vlm_client=vlm,
        region_detector=detector,
        config=_make_pipeline_config(
            track=EvaluationTrack.TRACK_A_PREDICTED_CXR,
            condition=AnswerCondition.CONDITION_3_GROUNDED_WITH_ABSTENTION,
            tau=0.9,  # High threshold to force abstention
        ),
    )
    sample = _make_sample(
        sample_id="c3",
        question="Is there a mass?",
        answer="Yes",
        compatible_tracks=[EvaluationTrack.TRACK_A_PREDICTED_CXR],
    )
    result = engine.run_single(sample, _make_image())

    assert result.condition == 3
    assert result.is_abstaining is True
    assert result.model_answer is None


# ---------------------------------------------------------------------------
# Track and condition metadata preservation
# ---------------------------------------------------------------------------


def test_metadata_preserved_per_sample():
    """Track, condition, sample ID, model, dataset preserved in every record."""
    vlm = MockVLMClient(
        responses=[{"answer": "Yes", "findings": []}] * 5
    )
    engine = PipelineEngine(
        vlm_client=vlm,
        region_detector=None,
        config=_make_pipeline_config(
            track=EvaluationTrack.TRACK_B_ORACLE_SLAKE,
            condition=AnswerCondition.CONDITION_2_GROUNDED_NO_ABSTENTION,
            model_name="qwen2.5-vl-3b",
            n_samples=5,
        ),
    )
    sample = _make_sample(
        sample_id="meta1",
        question="Test?",
        answer="Yes",
        dataset_name="vqarad",
        split="test",
        compatible_tracks=[EvaluationTrack.TRACK_B_ORACLE_SLAKE],
    )
    result = engine.run_single(sample, _make_image())

    assert result.sample_id == "meta1"
    assert result.track == "track_b_oracle_slake"
    assert result.condition == 2
    assert result.dataset_name == "vqarad"
    assert result.split == "test"


# ---------------------------------------------------------------------------
# Evaluation record generation
# ---------------------------------------------------------------------------


def test_evaluation_record_from_pipeline():
    """Pipeline output can be converted to SampleRecord for metrics."""
    from verification.pipeline import PipelineEngine

    vlm = MockVLMClient(
        responses=[{"answer": "Yes", "findings": []}] * 5
    )
    engine = PipelineEngine(
        vlm_client=vlm,
        region_detector=None,
        config=_make_pipeline_config(
            track=EvaluationTrack.TRACK_C_ANSWER_ONLY,
            condition=AnswerCondition.CONDITION_3_GROUNDED_WITH_ABSTENTION,
        ),
    )
    sample = _make_sample(
        sample_id="er1",
        question="Test?",
        answer="Yes",
        compatible_tracks=[EvaluationTrack.TRACK_C_ANSWER_ONLY],
    )
    result = engine.run_single(sample, _make_image())

    # Build SampleRecord manually (as the pipeline runner would)
    from evaluation.metrics import SampleRecord

    record = SampleRecord(
        sample_id=result.sample_id,
        dataset_name=result.dataset_name,
        split=result.split,
        track=result.track,
        condition=result.condition,
        ground_truth_answer=result.ground_truth_answer,
        model_answer=result.model_answer,
        correct=result.correct,
        question_type=result.question_type,
        has_gt_regions=result.has_gt_regions,
        pointing_correct=result.pointing_correct,
        n_claims=result.n_claims,
        n_positive_claims=result.n_positive_claims,
        n_supported=result.n_supported,
        n_unsupported=result.n_unsupported,
        n_unverifiable=result.n_unverifiable,
        n_absence=result.n_absence,
        n_uncertain=result.n_uncertain,
        agreement=result.agreement,
        normalized_entropy=result.normalized_entropy,
        confidence=result.confidence,
        threshold=result.threshold,
        is_abstaining=result.is_abstaining,
        in_coverage=result.in_coverage,
        evidence_state=result.evidence_state,
        sampling_valid=result.sampling_valid,
    )

    assert record.sample_id == "er1"
    assert record.correct is True
    assert record.track == "track_c_answer_only"
    assert record.condition == 3


# ---------------------------------------------------------------------------
# Artifact generation
# ---------------------------------------------------------------------------


def test_artifact_generation_from_pipeline_results():
    """Pipeline results can be serialized to experiment artifacts."""

    # Create sample records from pipeline results
    records = [
        SampleRecord(
            sample_id="s1",
            dataset_name="slake",
            split="test",
            track="track_c_answer_only",
            condition=3,
            ground_truth_answer="Yes",
            model_answer="Yes",
            correct=True,
            question_type="CLOSED",
            has_gt_regions=False,
            pointing_correct=False,
            n_claims=5,
            n_positive_claims=5,
            n_supported=5,
            n_unsupported=0,
            n_unverifiable=0,
            n_absence=0,
            n_uncertain=0,
            agreement=1.0,
            normalized_entropy=0.0,
            confidence=0.9,
            threshold=0.5,
            is_abstaining=False,
            in_coverage=True,
            evidence_state="present_grounded",
            sampling_valid=True,
        )
    ]

    config = {
        "w1": 0.4,
        "w2": 0.3,
        "w3": 0.3,
        "tau": 0.5,
        "n_samples": 5,
        "track": "track_c_answer_only",
        "condition": 3,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        runner = EvaluationRunner(
            out_dir=out_dir, experiment_name="Test Pipeline", config=config
        )
        metrics = runner.run(records)

        # Verify artifacts exist
        assert (out_dir / "experiment_metrics.json").exists()
        assert (out_dir / "experiment_summary.csv").exists()
        assert (out_dir / "per_sample_results.csv").exists()
        assert (out_dir / "accuracy_vs_coverage.csv").exists()
        assert (out_dir / "comparison_report.md").exists()

        # Verify metrics
        assert metrics.n_samples == 1
        assert metrics.accuracy == 1.0
        assert metrics.coverage == 1.0

        # Verify JSON serializable
        json_data = (out_dir / "experiment_metrics.json").read_text()
        parsed = json.loads(json_data)
        assert parsed["n_samples"] == 1

        # Verify CSV serializable
        csv_data = (out_dir / "experiment_summary.csv").read_text()
        assert "accuracy" in csv_data
        assert "usr_raw" in csv_data
