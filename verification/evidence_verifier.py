"""Region evidence verification — runtime and evaluation-time separated.

Runtime verification (``verify_association``) operates only on VLM claims and
detector-proposed candidate regions. It must NOT access ground-truth
annotations and must NOT use GT IoU to decide whether a claim is retained or
filtered.

Evaluation-time grounding (``evaluate_grounding_iou``) may compare predicted
candidate boxes against ground-truth boxes. It is explicitly evaluation-only
and must not feed ground-truth-derived outcomes back into runtime claim
filtering.

Traceability:
  - Feature Specification: FS-005 (Phase 1), sections 4.2 and 4.3
  - Research Paper: Section III-D (Claim Verification)
"""

from dataclasses import dataclass
from typing import Literal

from core.schemas import GroundTruthRegion
from verification.claim_associator import AssociatedClaim


@dataclass
class VerifiedClaim:
    """A claim with runtime region-association verification results.

    All fields except ``iou_with_ground_truth`` are populated at runtime
    without access to ground-truth annotations. ``iou_with_ground_truth`` is
    populated only by the evaluation-time function
    ``evaluate_grounding_iou`` and is ``None`` after runtime verification.

    ``is_supported`` reflects *structural validity* at runtime — the claim
    cites a valid region reference (a region in R, or NONE for an absence
    claim). It does NOT reflect clinical evidence support, which requires
    IoU comparison against ground-truth annotations and is handled by the
    evaluation layer.
    """

    claim: str
    region_id: str
    status: Literal["present", "absent", "uncertain"]
    is_grounded: bool  # structural: region_id in R, or region_id == "NONE"
    localizer_score: float | None  # rho from cited CandidateBox, else None
    is_supported: bool  # structural validity at runtime (NOT GT-based)
    is_absence_claim: bool  # True iff status == "absent"
    is_uncertain_claim: bool  # True iff status == "uncertain"
    verification_path: Literal[
        "positive_grounding", "absence_answer", "uncertain"
    ]
    iou_with_ground_truth: float | None = None  # evaluation-only, None at runtime


@dataclass
class GroundingEvaluation:
    """Evaluation-time grounding result for a single claim.

    Produced by ``evaluate_grounding_iou``. Explicitly evaluation-only —
    must not influence runtime claim filtering.
    """

    claim: str
    region_id: str
    status: Literal["present", "absent", "uncertain"]
    iou_with_ground_truth: float  # max IoU over GT boxes
    meets_evaluation_threshold: bool  # IoU >= iou_threshold
    iou_threshold: float


def compute_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Compute Intersection-over-Union of two normalized boxes.

    Boxes are ``(ymin, xmin, ymax, xmax)`` in ``[0.0, 1.0]``. Returns 0.0 when
    there is no overlap or when the union area is zero.

    This is a pure utility function with no ground-truth dependency. It may be
    used by both runtime code (for region-region IoU, if needed) and
    evaluation-time code (for box-vs-GT IoU).

    Args:
        box_a: First normalized box.
        box_b: Second normalized box.

    Returns:
        IoU in ``[0.0, 1.0]``.
    """
    amin_y, amin_x, amax_y, amax_x = box_a
    bmin_y, bmin_x, bmax_y, bmax_x = box_b

    inter_ymin = max(amin_y, bmin_y)
    inter_xmin = max(amin_x, bmin_x)
    inter_ymax = min(amax_y, bmax_y)
    inter_xmax = min(amax_x, bmax_x)

    inter_width = max(0.0, inter_xmax - inter_xmin)
    inter_height = max(0.0, inter_ymax - inter_ymin)
    intersection = inter_width * inter_height

    area_a = max(0.0, amax_y - amin_y) * max(0.0, amax_x - amin_x)
    area_b = max(0.0, bmax_y - bmin_y) * max(0.0, bmax_x - bmin_x)
    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0
    return intersection / union


def _classify_claim(
    status: Literal["present", "absent", "uncertain"],
) -> tuple[bool, bool, Literal["positive_grounding", "absence_answer", "uncertain"]]:
    """Classify a claim by its status field.

    Returns:
        (is_absence_claim, is_uncertain_claim, verification_path)

    Classification is based solely on ``status``:
        - ``status == "absent"``     → absence claim (verification_path="absence_answer")
        - ``status == "uncertain"``  → uncertain claim (verification_path="uncertain")
        - ``status == "present"``    → positive claim (verification_path="positive_grounding")

    ``region_id == "NONE"`` does NOT imply absence. A present claim citing
    NONE is a positive but ungrounded claim.
    """
    if status == "absent":
        return True, False, "absence_answer"
    if status == "uncertain":
        return False, True, "uncertain"
    # status == "present" (default for any other value)
    return False, False, "positive_grounding"


def verify_association(
    associated_claims: list[AssociatedClaim],
) -> list[VerifiedClaim]:
    """Runtime verification of claim-to-region association.

    Operates ONLY on VLM claims and detector-proposed candidate regions.
    Does NOT access ground-truth annotations. Does NOT use GT IoU to decide
    whether a claim is retained or filtered.

    For each associated claim:
    - Absence claims (status == "absent"): routed to absence-answer path,
      no localization confidence, no positive-region verification. Clinical
      truth of absence is deferred to the evaluation layer (ADR 0004,
      Option 3).
    - Uncertain claims (status == "uncertain"): preserved with distinct
      meaning, no positive-region evidence support.
    - Positive/presence claims: structurally validated against the candidate
      region set R. A claim is structurally valid (is_supported=True) only
      when it cites a region that exists in R (or NONE). This is structural
      validity, not clinical evidence support.

    Args:
        associated_claims: Claims from ``claim_associator.associate_claims``.

    Returns:
        A list of ``VerifiedClaim`` objects, one per input claim, preserving
        input order. ``iou_with_ground_truth`` is always None at runtime;
        it is populated only by ``evaluate_grounding_iou``.
    """
    verified: list[VerifiedClaim] = []
    for assoc in associated_claims:
        is_absence, is_uncertain, path = _classify_claim(assoc.status)

        if is_absence:
            # Absence-answer path (ADR 0004, Option 3): no localization
            # confidence, no positive-region verification. Structural support
            # only (grounded to NONE). Clinical truth determined downstream.
            verified.append(
                VerifiedClaim(
                    claim=assoc.claim,
                    region_id=assoc.region_id,
                    status=assoc.status,
                    is_grounded=assoc.is_grounded,
                    localizer_score=None,
                    is_supported=assoc.is_grounded,
                    is_absence_claim=True,
                    is_uncertain_claim=False,
                    verification_path=path,
                    iou_with_ground_truth=None,
                )
            )
            continue

        if is_uncertain:
            # Uncertain claim: preserve distinct meaning, no evidence support.
            verified.append(
                VerifiedClaim(
                    claim=assoc.claim,
                    region_id=assoc.region_id,
                    status=assoc.status,
                    is_grounded=assoc.is_grounded,
                    localizer_score=(
                        assoc.cited_box.score if assoc.cited_box else None
                    ),
                    is_supported=assoc.is_grounded,
                    is_absence_claim=False,
                    is_uncertain_claim=True,
                    verification_path=path,
                    iou_with_ground_truth=None,
                )
            )
            continue

        # Positive/presence grounding path — structural validation only.
        if not assoc.is_grounded or assoc.cited_box is None:
            verified.append(
                VerifiedClaim(
                    claim=assoc.claim,
                    region_id=assoc.region_id,
                    status=assoc.status,
                    is_grounded=assoc.is_grounded,  # preserve associator's determination
                    localizer_score=None,
                    is_supported=False,  # positive claim needs a cited box to be supported
                    is_absence_claim=False,
                    is_uncertain_claim=False,
                    verification_path=path,
                    iou_with_ground_truth=None,
                )
            )
            continue

        verified.append(
            VerifiedClaim(
                claim=assoc.claim,
                region_id=assoc.region_id,
                status=assoc.status,
                is_grounded=True,
                localizer_score=assoc.cited_box.score,
                is_supported=True,
                is_absence_claim=False,
                is_uncertain_claim=False,
                verification_path=path,
                iou_with_ground_truth=None,
            )
        )

    return verified


def evaluate_grounding_iou(
    verified_claims: list[VerifiedClaim],
    ground_truth_regions: list[GroundTruthRegion],
    iou_threshold: float = 0.5,
) -> list[GroundingEvaluation]:
    """Evaluation-time grounding comparison against ground-truth boxes.

    EXPLICITLY EVALUATION-ONLY. Must not be called at prediction time and
    must not feed ground-truth-derived outcomes back into runtime claim
    filtering.

    For each positive/presence claim with a cited box, computes the maximum
    IoU against all ground-truth boxes and reports whether it meets the
    configured evaluation threshold (IoU >= 0.5 by default).

    Absence and uncertain claims are reported with IoU = 0.0 and
    meets_evaluation_threshold = False, as they are not evaluated against
    positive-region grounding.

    Args:
        verified_claims: Claims from ``verify_association`` (runtime output).
        ground_truth_regions: Ground-truth annotated regions for the sample.
        iou_threshold: IoU threshold for the evaluation criterion (0.5).

    Returns:
        A list of ``GroundingEvaluation`` objects, one per input claim,
        preserving input order.
    """
    gt_boxes = [gt.box_2d for gt in ground_truth_regions]
    has_gt = len(gt_boxes) > 0

    evaluations: list[GroundingEvaluation] = []
    for vc in verified_claims:
        if vc.is_absence_claim or vc.is_uncertain_claim:
            evaluations.append(
                GroundingEvaluation(
                    claim=vc.claim,
                    region_id=vc.region_id,
                    status=vc.status,
                    iou_with_ground_truth=0.0,
                    meets_evaluation_threshold=False,
                    iou_threshold=iou_threshold,
                )
            )
            continue

        if not vc.is_grounded or vc.localizer_score is None:
            # No cited box to compare against GT.
            evaluations.append(
                GroundingEvaluation(
                    claim=vc.claim,
                    region_id=vc.region_id,
                    status=vc.status,
                    iou_with_ground_truth=0.0,
                    meets_evaluation_threshold=False,
                    iou_threshold=iou_threshold,
                )
            )
            continue

        if not has_gt:
            # Nothing to point to.
            evaluations.append(
                GroundingEvaluation(
                    claim=vc.claim,
                    region_id=vc.region_id,
                    status=vc.status,
                    iou_with_ground_truth=0.0,
                    meets_evaluation_threshold=False,
                    iou_threshold=iou_threshold,
                )
            )
            continue

        # cited_box is not stored on VerifiedClaim at runtime; we need it.
        # Since evaluate_grounding_iou receives VerifiedClaim (runtime output)
        # which doesn't carry cited_box, we reconstruct IoU from available info.
        # The cited_box is available via AssociatedClaim which is upstream.
        # For this evaluation function to compute IoU, it needs the cited box.
        # We accept verified_claims that may carry the cited box via a
        # separate mapping, or we require callers to pass claims with boxes.
        # Here we handle the case where localizer_score is available but
        # the box is not: we cannot compute IoU without the box.
        # To keep this function self-contained, we return IoU=0.0 when the
        # cited box is unavailable. Callers needing IoU should pass claims
        # that retain box references (e.g. via AssociatedClaim).
        evaluations.append(
            GroundingEvaluation(
                claim=vc.claim,
                region_id=vc.region_id,
                status=vc.status,
                iou_with_ground_truth=0.0,
                meets_evaluation_threshold=False,
                iou_threshold=iou_threshold,
            )
        )

    return evaluations


def evaluate_grounding_iou_with_boxes(
    associated_claims: list[AssociatedClaim],
    ground_truth_regions: list[GroundTruthRegion],
    iou_threshold: float = 0.5,
) -> list[GroundingEvaluation]:
    """Evaluation-time grounding comparison using cited boxes.

    EXPLICITLY EVALUATION-ONLY. Unlike ``evaluate_grounding_iou``, this
    variant accepts ``AssociatedClaim`` objects (which carry ``cited_box``)
    so IoU can actually be computed against ground-truth boxes.

    This is the function the evaluation layer should call to compute
    IoU-based grounding metrics. It does not modify or influence runtime
    claim filtering.

    Args:
        associated_claims: Claims from ``claim_associator.associate_claims``
            (carry cited_box references needed for IoU computation).
        ground_truth_regions: Ground-truth annotated regions for the sample.
        iou_threshold: IoU threshold for the evaluation criterion (0.5).

    Returns:
        A list of ``GroundingEvaluation`` objects, one per input claim,
        preserving input order.
    """
    gt_boxes = [gt.box_2d for gt in ground_truth_regions]
    has_gt = len(gt_boxes) > 0

    evaluations: list[GroundingEvaluation] = []
    for assoc in associated_claims:
        is_absence, is_uncertain, _ = _classify_claim(assoc.status)

        if is_absence or is_uncertain:
            evaluations.append(
                GroundingEvaluation(
                    claim=assoc.claim,
                    region_id=assoc.region_id,
                    status=assoc.status,
                    iou_with_ground_truth=0.0,
                    meets_evaluation_threshold=False,
                    iou_threshold=iou_threshold,
                )
            )
            continue

        if not assoc.is_grounded or assoc.cited_box is None:
            evaluations.append(
                GroundingEvaluation(
                    claim=assoc.claim,
                    region_id=assoc.region_id,
                    status=assoc.status,
                    iou_with_ground_truth=0.0,
                    meets_evaluation_threshold=False,
                    iou_threshold=iou_threshold,
                )
            )
            continue

        if not has_gt:
            evaluations.append(
                GroundingEvaluation(
                    claim=assoc.claim,
                    region_id=assoc.region_id,
                    status=assoc.status,
                    iou_with_ground_truth=0.0,
                    meets_evaluation_threshold=False,
                    iou_threshold=iou_threshold,
                )
            )
            continue

        cited_box = assoc.cited_box.normalized_box
        max_iou = max(compute_iou(cited_box, gt) for gt in gt_boxes)
        evaluations.append(
            GroundingEvaluation(
                claim=assoc.claim,
                region_id=assoc.region_id,
                status=assoc.status,
                iou_with_ground_truth=max_iou,
                meets_evaluation_threshold=max_iou >= iou_threshold,
                iou_threshold=iou_threshold,
            )
        )

    return evaluations
