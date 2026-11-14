"""Unit tests for FS-005 Phase 1 claim extraction and region verification.

Covers claim extraction, claim-to-region association, runtime region
verification (ground-truth-free), evaluation-time IoU isolation, unsupported
claim filtering, and ADR 0004 Option 3 negative/absence claim handling.

All tests are deterministic and require no model weights, paid APIs, or
real VLM inference.

Key invariants tested:
- Ground-truth annotations CANNOT influence prediction-time claim filtering.
- Evaluation-time IoU is isolated from runtime verification.
- region_id="NONE" does NOT automatically imply absence (only status="absent" does).
- Negative claims remain distinct from ungrounded positive claims.
- Structural validity is not presented as clinical evidence support.
- Empty detections do not prove absence.
"""

import pytest

from core.schemas import (
    CandidateBox,
    GroundTruthRegion,
    VLMStructuredOutput,
)
from verification.claim_associator import AssociatedClaim, associate_claims
from verification.claim_extractor import ExtractedClaim, extract_claims
from verification.claim_filter import filter_claims
from verification.evidence_verifier import (
    VerifiedClaim,
    compute_iou,
    evaluate_grounding_iou,
    evaluate_grounding_iou_with_boxes,
    verify_association,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _candidate_box(
    region_id: str,
    box_2d: tuple[int, int, int, int],
    normalized_box: tuple[float, float, float, float],
    score: float = 0.9,
    label: str = "lesion",
) -> CandidateBox:
    return CandidateBox(
        region_id=region_id,
        box_2d=box_2d,
        normalized_box=normalized_box,
        score=score,
        label=label,
    )


def _ground_truth_region(
    label: str,
    box_2d: tuple[float, float, float, float],
) -> GroundTruthRegion:
    return GroundTruthRegion(label=label, box_2d=box_2d)


def _structured_output(
    answer: str,
    findings: list[dict],
) -> VLMStructuredOutput:
    return VLMStructuredOutput.model_validate({"answer": answer, "findings": findings})


def _extracted_claim(
    claim: str,
    region_id: str,
    status: str = "present",
    confidence: str = "high",
) -> ExtractedClaim:
    return ExtractedClaim(
        claim=claim,
        region_id=region_id,
        status=status,  # type: ignore[arg-type]
        confidence=confidence,  # type: ignore[arg-type]
        raw_claim_text=claim,
    )


def _associated_claim(
    claim: str,
    region_id: str,
    status: str = "present",
    is_grounded: bool = True,
    cited_box: CandidateBox | None = None,
    rejection_reason: str | None = None,
) -> AssociatedClaim:
    return AssociatedClaim(
        claim=claim,
        region_id=region_id,
        status=status,  # type: ignore[arg-type]
        confidence="high",
        is_grounded=is_grounded,
        cited_box=cited_box,
        rejection_reason=rejection_reason,
    )


def _verified_claim(
    claim: str,
    region_id: str,
    status: str = "present",
    is_grounded: bool = True,
    localizer_score: float | None = 0.9,
    is_supported: bool = True,
    is_absence_claim: bool = False,
    is_uncertain_claim: bool = False,
    verification_path: str = "positive_grounding",
    iou_with_ground_truth: float | None = None,
) -> VerifiedClaim:
    return VerifiedClaim(
        claim=claim,
        region_id=region_id,
        status=status,  # type: ignore[arg-type]
        is_grounded=is_grounded,
        localizer_score=localizer_score,
        is_supported=is_supported,
        is_absence_claim=is_absence_claim,
        is_uncertain_claim=is_uncertain_claim,
        verification_path=verification_path,  # type: ignore[arg-type]
        iou_with_ground_truth=iou_with_ground_truth,
    )


# ---------------------------------------------------------------------------
# Claim extraction tests
# ---------------------------------------------------------------------------


def test_extract_claims_valid_multiple() -> None:
    output = _structured_output(
        "Yes, right lower zone opacity.",
        [
            {"claim": "opacity", "region_id": "R2", "status": "present", "confidence": "high"},
            {
                "claim": "pneumothorax",
                "region_id": "NONE",
                "status": "absent",
                "confidence": "high",
            },
        ],
    )
    result = extract_claims(output, raw_response="raw json")
    assert result.parse_success is True
    assert result.error is None
    assert result.answer_text == "Yes, right lower zone opacity."
    assert len(result.claims) == 2
    assert result.claims[0].claim == "opacity"
    assert result.claims[0].region_id == "R2"
    assert result.claims[0].status == "present"
    assert result.claims[1].claim == "pneumothorax"
    assert result.claims[1].region_id == "NONE"
    assert result.claims[1].status == "absent"


def test_extract_claims_positive_presence() -> None:
    output = _structured_output(
        "There is a mass in the right lung.",
        [{"claim": "mass", "region_id": "R1", "status": "present", "confidence": "medium"}],
    )
    result = extract_claims(output)
    assert len(result.claims) == 1
    assert result.claims[0].status == "present"
    assert result.claims[0].region_id == "R1"


def test_extract_claims_negative_absence() -> None:
    output = _structured_output(
        "No pleural effusion is visible.",
        [
            {
                "claim": "pleural effusion",
                "region_id": "NONE",
                "status": "absent",
                "confidence": "high",
            }
        ],
    )
    result = extract_claims(output)
    assert len(result.claims) == 1
    assert result.claims[0].status == "absent"
    assert result.claims[0].region_id == "NONE"


def test_extract_claims_empty_response() -> None:
    result = extract_claims(None, raw_response="")
    assert result.parse_success is False
    assert result.error is not None
    assert result.answer_text == ""
    assert result.claims == []


def test_extract_claims_malformed_response() -> None:
    result = extract_claims(None, raw_response="not json at all {{{")
    assert result.parse_success is False
    assert result.error == "parse_failure"


def test_extract_claims_preserves_answer() -> None:
    output = _structured_output(
        "  Yes,   right lower zone opacity.  ",
        [{"claim": "opacity", "region_id": "R2", "status": "present", "confidence": "high"}],
    )
    result = extract_claims(output)
    assert result.answer_text == "Yes, right lower zone opacity."


def test_extract_claims_empty_findings() -> None:
    output = _structured_output("Normal study.", [])
    result = extract_claims(output)
    assert result.parse_success is True
    assert result.answer_text == "Normal study."
    assert result.claims == []


# ---------------------------------------------------------------------------
# Claim-to-region association tests
# ---------------------------------------------------------------------------


def test_associate_claims_valid_region() -> None:
    boxes = [_candidate_box("R1", (10, 10, 50, 50), (0.1, 0.1, 0.5, 0.5))]
    claims = [_extracted_claim("mass", "R1")]
    associated = associate_claims(claims, boxes)
    assert len(associated) == 1
    assert associated[0].is_grounded is True
    assert associated[0].cited_box is not None
    assert associated[0].cited_box.region_id == "R1"
    assert associated[0].rejection_reason is None


def test_associate_claims_multiple_candidate_regions() -> None:
    boxes = [
        _candidate_box("R1", (10, 10, 50, 50), (0.1, 0.1, 0.5, 0.5)),
        _candidate_box("R2", (60, 60, 90, 90), (0.6, 0.6, 0.9, 0.9)),
    ]
    claims = [
        _extracted_claim("mass", "R1"),
        _extracted_claim("effusion", "R2"),
    ]
    associated = associate_claims(claims, boxes)
    assert len(associated) == 2
    assert associated[0].cited_box.region_id == "R1"
    assert associated[1].cited_box.region_id == "R2"


def test_associate_claims_unknown_region_rejected() -> None:
    boxes = [_candidate_box("R1", (10, 10, 50, 50), (0.1, 0.1, 0.5, 0.5))]
    claims = [_extracted_claim("mass", "R99")]
    associated = associate_claims(claims, boxes)
    assert len(associated) == 1
    assert associated[0].is_grounded is False
    assert associated[0].cited_box is None
    assert "R99" in associated[0].rejection_reason


def test_associate_claims_empty_region_collection() -> None:
    claims = [_extracted_claim("mass", "R1")]
    associated = associate_claims(claims, [])
    assert len(associated) == 1
    assert associated[0].is_grounded is False
    assert associated[0].cited_box is None
    assert "empty" in associated[0].rejection_reason


def test_associate_claims_none_claim_grounded_when_empty() -> None:
    # A NONE claim is structurally grounded even when R is empty.
    claims = [_extracted_claim("effusion", "NONE", status="absent")]
    associated = associate_claims(claims, [])
    assert len(associated) == 1
    assert associated[0].is_grounded is True
    assert associated[0].cited_box is None


def test_associate_claims_preserves_region_ids_and_provenance() -> None:
    box = _candidate_box("R1", (10, 10, 50, 50), (0.1, 0.1, 0.5, 0.5), score=0.87)
    claims = [_extracted_claim("mass", "R1")]
    associated = associate_claims(claims, [box])
    assert associated[0].cited_box is box  # same object provenance preserved
    assert associated[0].cited_box.score == 0.87


# ---------------------------------------------------------------------------
# Negative-claim classification tests (Issue 2)
# ---------------------------------------------------------------------------


def test_classify_positive_claim_valid_region_not_absence() -> None:
    """Positive claim with valid region is NOT an absence claim."""
    assoc = _associated_claim("mass", "R1", status="present", is_grounded=True,
                              cited_box=_candidate_box("R1", (10, 10, 50, 50), (0.1, 0.1, 0.5, 0.5)))
    verified = verify_association([assoc])
    assert len(verified) == 1
    assert verified[0].is_absence_claim is False
    assert verified[0].is_uncertain_claim is False
    assert verified[0].verification_path == "positive_grounding"
    assert verified[0].is_supported is True


def test_classify_positive_claim_with_none_is_not_absence() -> None:
    """status='present' + region_id='NONE' is positive but ungrounded, NOT absence."""
    assoc = _associated_claim("mass", "NONE", status="present", is_grounded=True,
                              cited_box=None)
    verified = verify_association([assoc])
    assert len(verified) == 1
    assert verified[0].is_absence_claim is False  # status is 'present', not 'absent'
    assert verified[0].is_uncertain_claim is False
    assert verified[0].verification_path == "positive_grounding"
    assert verified[0].is_supported is False  # positive claim, no box → not supported
    assert verified[0].is_grounded is True  # NONE is structurally valid


def test_classify_negative_claim_with_none_is_absence() -> None:
    """status='absent' + region_id='NONE' IS an absence claim."""
    assoc = _associated_claim("effusion", "NONE", status="absent", is_grounded=True,
                              cited_box=None)
    verified = verify_association([assoc])
    assert len(verified) == 1
    assert verified[0].is_absence_claim is True
    assert verified[0].verification_path == "absence_answer"
    assert verified[0].localizer_score is None


def test_classify_negative_claim_with_empty_regions_is_absence() -> None:
    """Absence claim with empty R is still an absence claim, not a verified negative."""
    assoc = _associated_claim("effusion", "NONE", status="absent", is_grounded=True,
                              cited_box=None)
    verified = verify_association([assoc])
    assert verified[0].is_absence_claim is True
    assert verified[0].localizer_score is None
    assert verified[0].iou_with_ground_truth is None  # runtime: no GT access


def test_classify_uncertain_claim_preserved() -> None:
    """Uncertain claim keeps distinct meaning, not classified as absence."""
    assoc = _associated_claim("possible mass", "R1", status="uncertain", is_grounded=True,
                              cited_box=_candidate_box("R1", (10, 10, 50, 50), (0.1, 0.1, 0.5, 0.5)))
    verified = verify_association([assoc])
    assert len(verified) == 1
    assert verified[0].is_absence_claim is False
    assert verified[0].is_uncertain_claim is True
    assert verified[0].verification_path == "uncertain"


def test_classify_mixed_claims() -> None:
    """Mixed positive, negative, uncertain claims classified distinctly."""
    box_r1 = _candidate_box("R1", (10, 10, 50, 50), (0.1, 0.1, 0.5, 0.5))
    assocs = [
        _associated_claim("mass", "R1", status="present", is_grounded=True, cited_box=box_r1),
        _associated_claim("effusion", "NONE", status="absent", is_grounded=True, cited_box=None),
        _associated_claim("possible nodule", "R1", status="uncertain", is_grounded=True, cited_box=box_r1),
    ]
    verified = verify_association(assocs)
    assert len(verified) == 3
    assert verified[0].is_absence_claim is False and verified[0].is_uncertain_claim is False
    assert verified[1].is_absence_claim is True
    assert verified[2].is_uncertain_claim is True


# ---------------------------------------------------------------------------
# Runtime verification is ground-truth-free (Issue 1)
# ---------------------------------------------------------------------------


def test_runtime_verification_does_not_access_ground_truth() -> None:
    """verify_association takes no ground-truth parameter — GT cannot influence outputs."""
    import inspect
    sig = inspect.signature(verify_association)
    params = list(sig.parameters.keys())
    assert "ground_truth_regions" not in params
    assert "gt_" not in "".join(params)


def test_runtime_verified_claim_has_no_gt_iou() -> None:
    """Runtime VerifiedClaim always has iou_with_ground_truth=None."""
    assoc = _associated_claim("mass", "R1", status="present", is_grounded=True,
                              cited_box=_candidate_box("R1", (10, 10, 50, 50), (0.1, 0.1, 0.5, 0.5)))
    verified = verify_association([assoc])
    assert verified[0].iou_with_ground_truth is None


def test_runtime_filtering_cannot_use_gt_iou() -> None:
    """filter_claims receives only VerifiedClaim (no GT), so GT IoU cannot influence filtering."""
    import inspect
    sig = inspect.signature(filter_claims)
    params = list(sig.parameters.keys())
    assert "ground_truth_regions" not in params


def test_supported_is_structural_not_clinical() -> None:
    """is_supported at runtime reflects structural validity, not clinical evidence support."""
    # A structurally valid positive claim (grounded to R1) is is_supported=True
    # at runtime, even though clinical support requires GT IoU >= 0.5.
    box = _candidate_box("R1", (10, 10, 50, 50), (0.1, 0.1, 0.5, 0.5))
    assoc = _associated_claim("mass", "R1", status="present", is_grounded=True, cited_box=box)
    verified = verify_association([assoc])
    assert verified[0].is_supported is True  # structural validity
    assert verified[0].iou_with_ground_truth is None  # no clinical evidence at runtime


# ---------------------------------------------------------------------------
# Evaluation-time IoU isolation (Issue 1)
# ---------------------------------------------------------------------------


def test_evaluate_grounding_iou_is_explicitly_evaluation_only() -> None:
    """evaluate_grounding_iou_with_boxes takes GT — explicitly evaluation-only."""
    import inspect
    sig = inspect.signature(evaluate_grounding_iou_with_boxes)
    params = list(sig.parameters.keys())
    assert "ground_truth_regions" in params


def test_evaluation_iou_supported_positive() -> None:
    """Evaluation IoU >= 0.5 → meets_evaluation_threshold=True."""
    box_r1 = _candidate_box("R1", (10, 10, 50, 50), (0.1, 0.1, 0.5, 0.5))
    assocs = [_associated_claim("mass", "R1", status="present", is_grounded=True, cited_box=box_r1)]
    gts = [_ground_truth_region("mass", (0.1, 0.1, 0.5, 0.5))]
    evals = evaluate_grounding_iou_with_boxes(assocs, gts, iou_threshold=0.5)
    assert len(evals) == 1
    assert evals[0].iou_with_ground_truth == pytest.approx(1.0)
    assert evals[0].meets_evaluation_threshold is True


def test_evaluation_iou_unsupported_low_iou() -> None:
    """Evaluation IoU < 0.5 → meets_evaluation_threshold=False."""
    box_r1 = _candidate_box("R1", (10, 10, 50, 50), (0.1, 0.1, 0.5, 0.5))
    assocs = [_associated_claim("mass", "R1", status="present", is_grounded=True, cited_box=box_r1)]
    gts = [_ground_truth_region("mass", (0.7, 0.7, 0.9, 0.9))]  # no overlap
    evals = evaluate_grounding_iou_with_boxes(assocs, gts, iou_threshold=0.5)
    assert evals[0].iou_with_ground_truth == pytest.approx(0.0)
    assert evals[0].meets_evaluation_threshold is False


def test_evaluation_iou_isolation_from_runtime() -> None:
    """Evaluation IoU results do not modify runtime VerifiedClaim.is_supported."""
    box_r1 = _candidate_box("R1", (10, 10, 50, 50), (0.1, 0.1, 0.5, 0.5))
    assocs = [_associated_claim("mass", "R1", status="present", is_grounded=True, cited_box=box_r1)]
    verified = verify_association(assocs)
    runtime_supported = verified[0].is_supported  # structural: True

    # Now run evaluation with GT that gives IoU=0 (no overlap)
    gts = [_ground_truth_region("mass", (0.7, 0.7, 0.9, 0.9))]
    evals = evaluate_grounding_iou_with_boxes(assocs, gts, iou_threshold=0.5)

    # Runtime is_supported unchanged by evaluation
    assert verified[0].is_supported == runtime_supported
    # Evaluation independently reports IoU=0, threshold not met
    assert evals[0].iou_with_ground_truth == pytest.approx(0.0)
    assert evals[0].meets_evaluation_threshold is False


def test_evaluation_absence_claim_not_evaluated_against_gt() -> None:
    """Absence claims get IoU=0.0, meets_threshold=False in evaluation."""
    assocs = [_associated_claim("effusion", "NONE", status="absent", is_grounded=True, cited_box=None)]
    gts = [_ground_truth_region("effusion", (0.1, 0.1, 0.5, 0.5))]
    evals = evaluate_grounding_iou_with_boxes(assocs, gts, iou_threshold=0.5)
    assert evals[0].iou_with_ground_truth == pytest.approx(0.0)
    assert evals[0].meets_evaluation_threshold is False


def test_evaluate_grounding_iou_no_gt_param() -> None:
    """evaluate_grounding_iou (without boxes) accepts GT but cannot compute real IoU."""
    import inspect
    sig = inspect.signature(evaluate_grounding_iou)
    params = list(sig.parameters.keys())
    assert "ground_truth_regions" in params


# ---------------------------------------------------------------------------
# Region evidence verification (compute_iou utility)
# ---------------------------------------------------------------------------


def test_compute_iou_overlapping() -> None:
    iou = compute_iou((0.1, 0.1, 0.5, 0.5), (0.3, 0.3, 0.7, 0.7))
    assert iou == pytest.approx(0.04 / 0.28)


def test_compute_iou_no_overlap() -> None:
    assert compute_iou((0.1, 0.1, 0.3, 0.3), (0.7, 0.7, 0.9, 0.9)) == 0.0


def test_compute_iou_identical() -> None:
    assert compute_iou((0.2, 0.2, 0.6, 0.6), (0.2, 0.2, 0.6, 0.6)) == 1.0


def test_compute_iou_contained() -> None:
    iou = compute_iou((0.1, 0.1, 0.5, 0.5), (0.2, 0.2, 0.4, 0.4))
    assert iou == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Unsupported claim filtering (structural, GT-free)
# ---------------------------------------------------------------------------


def test_filter_claims_supported_retained() -> None:
    claims = [_verified_claim("mass", "R1", is_supported=True)]
    result = filter_claims(claims)
    assert len(result.supported_claims) == 1
    assert len(result.unsupported_claims) == 0
    assert result.usr_raw == pytest.approx(0.0)
    assert result.usr_filtered == pytest.approx(0.0)


def test_filter_claims_unsupported_filtered() -> None:
    # Present + NONE: structurally ungrounded positive claim → unsupported
    claims = [_verified_claim("mass", "NONE", status="present", is_grounded=True,
                              localizer_score=None, is_supported=False,
                              is_absence_claim=False)]
    result = filter_claims(claims)
    assert len(result.unsupported_claims) == 1
    assert result.usr_raw == pytest.approx(1.0)


def test_filter_claims_absence_excluded_from_usr() -> None:
    """Absence claims excluded from USR denominators (ADR 0004 Option 3)."""
    claims = [
        _verified_claim("mass", "R1", is_supported=True, is_absence_claim=False),
        _verified_claim("effusion", "NONE", status="absent", is_grounded=True,
                        localizer_score=None, is_supported=True,
                        is_absence_claim=True, verification_path="absence_answer"),
    ]
    result = filter_claims(claims)
    assert len(result.absence_claims) == 1
    assert len(result.supported_claims) == 1
    assert len(result.unsupported_claims) == 0
    assert len(result.unverifiable_claims) == 0
    # n_positive = 1 (only the mass claim)
    assert result.n_positive_claims == 1
    assert result.usr_raw == pytest.approx(0.0)
    assert result.usr_filtered == pytest.approx(0.0)


def test_filter_claims_mixed() -> None:
    claims = [
        _verified_claim("mass", "R1", is_supported=True, is_absence_claim=False),
        _verified_claim("effusion", "NONE", status="present", is_grounded=True,
                        localizer_score=None, is_supported=False,
                        is_absence_claim=False),  # present+NONE → unsupported positive
        _verified_claim("nodule", "R99", is_grounded=False, localizer_score=None,
                        is_supported=False, is_absence_claim=False),  # unverifiable
    ]
    result = filter_claims(claims)
    assert len(result.supported_claims) == 1
    assert len(result.unsupported_claims) == 1  # present+NONE
    assert len(result.unverifiable_claims) == 1  # R99
    # usr_raw = (1 unsupported + 1 unverifiable) / 3 = 2/3
    assert result.usr_raw == pytest.approx(2 / 3)
    # usr_filtered = 1 unsupported / (3 - 1 unverifiable) = 1/2
    assert result.usr_filtered == pytest.approx(0.5)


def test_filter_claims_uncertain_separated() -> None:
    """Uncertain claims separated from positive/negative, not in USR denominator."""
    claims = [
        _verified_claim("mass", "R1", is_supported=True, is_absence_claim=False),
        _verified_claim("possible nodule", "R1", status="uncertain", is_grounded=True,
                        localizer_score=0.9, is_supported=True,
                        is_absence_claim=False, is_uncertain_claim=True,
                        verification_path="uncertain"),
    ]
    result = filter_claims(claims)
    assert len(result.uncertain_claims) == 1
    assert len(result.supported_claims) == 1
    # n_positive = 1 (uncertain excluded from USR denominator)
    assert result.n_positive_claims == 1


def test_filter_claims_empty() -> None:
    result = filter_claims([])
    assert result.usr_raw == 0.0
    assert result.usr_filtered == 0.0
    assert result.n_claims == 0


def test_filter_claims_original_records_preserved() -> None:
    original = _verified_claim("mass", "R1", is_supported=True, localizer_score=0.8)
    result = filter_claims([original])
    assert result.supported_claims[0] is original
    assert result.supported_claims[0].localizer_score == 0.8


# ---------------------------------------------------------------------------
# ADR 0004 Option 3 - negative/absence claims
# ---------------------------------------------------------------------------


def test_option3_negative_claim_preserved_when_regions_empty() -> None:
    output = _structured_output(
        "No pneumothorax is visible.",
        [
            {
                "claim": "pneumothorax",
                "region_id": "NONE",
                "status": "absent",
                "confidence": "high",
            }
        ],
    )
    extraction = extract_claims(output, raw_response="raw")
    assert extraction.parse_success is True
    assert len(extraction.claims) == 1
    assert extraction.claims[0].status == "absent"
    assert extraction.claims[0].region_id == "NONE"


def test_option3_negative_claim_no_localization_confidence() -> None:
    assoc = _associated_claim("pneumothorax", "NONE", status="absent", is_grounded=True,
                              cited_box=None)
    verified = verify_association([assoc])
    assert verified[0].is_absence_claim is True
    assert verified[0].localizer_score is None
    assert verified[0].iou_with_ground_truth is None
    assert verified[0].verification_path == "absence_answer"


def test_option3_negative_claim_excluded_from_grounding_metrics() -> None:
    claims = [
        _verified_claim("mass", "R1", is_supported=True, verification_path="positive_grounding",
                        is_absence_claim=False),
        _verified_claim("effusion", "NONE", status="absent", is_grounded=True,
                        localizer_score=None, is_supported=True,
                        is_absence_claim=True, verification_path="absence_answer"),
    ]
    result = filter_claims(claims)
    assert len(result.absence_claims) == 1
    assert len(result.supported_claims) == 1
    assert result.n_positive_claims == 1
    assert result.usr_raw == pytest.approx(0.0)
    assert result.usr_filtered == pytest.approx(0.0)


def test_option3_empty_detector_output_not_verified_negative() -> None:
    """Empty R with absence claim: NOT a clinically verified negative finding."""
    assoc = _associated_claim("effusion", "NONE", status="absent", is_grounded=True,
                              cited_box=None)
    verified = verify_association([assoc])
    assert verified[0].is_grounded is True  # structurally valid (NONE token)
    assert verified[0].localizer_score is None  # no artificial confidence
    assert verified[0].iou_with_ground_truth is None  # no GT at runtime


def test_option3_positive_and_negative_distinct_paths() -> None:
    box_r1 = _candidate_box("R1", (10, 10, 50, 50), (0.1, 0.1, 0.5, 0.5), score=0.9)
    positive_assoc = _associated_claim("mass", "R1", status="present", is_grounded=True,
                                       cited_box=box_r1)
    negative_assoc = _associated_claim("effusion", "NONE", status="absent", is_grounded=True,
                                       cited_box=None)
    verified = verify_association([positive_assoc, negative_assoc])
    pos = verified[0]
    neg = verified[1]
    assert pos.verification_path == "positive_grounding"
    assert pos.is_absence_claim is False
    assert pos.localizer_score == 0.9
    assert pos.is_supported is True  # structural
    assert neg.verification_path == "absence_answer"
    assert neg.is_absence_claim is True
    assert neg.localizer_score is None


def test_negative_claim_distinct_from_unsupported_positive() -> None:
    """Negative claim (absent+NONE) ≠ unsupported positive claim (present+NONE)."""
    assocs = [
        _associated_claim("effusion", "NONE", status="absent", is_grounded=True, cited_box=None),
        _associated_claim("mass", "NONE", status="present", is_grounded=True, cited_box=None),
    ]
    verified = verify_association(assocs)
    neg = verified[0]
    pos_ungrounded = verified[1]
    assert neg.is_absence_claim is True
    assert pos_ungrounded.is_absence_claim is False
    assert pos_ungrounded.is_supported is False  # positive without box
    # Different verification paths
    assert neg.verification_path == "absence_answer"
    assert pos_ungrounded.verification_path == "positive_grounding"


def test_empty_detections_do_not_prove_absence() -> None:
    """R=∅ does not make an absence claim clinically verified."""
    assoc = _associated_claim("effusion", "NONE", status="absent", is_grounded=True,
                              cited_box=None)
    verified = verify_association([assoc])
    # Structurally grounded (NONE valid) but no localization confidence
    assert verified[0].is_grounded is True
    assert verified[0].localizer_score is None
    assert verified[0].iou_with_ground_truth is None
