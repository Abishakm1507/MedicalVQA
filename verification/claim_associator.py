"""Claim-to-region association.

Associates each extracted claim with a candidate region from the detector-
proposed set ``R = {R1, ..., Rk}``. Validates that a ``region_id`` names an
existing region and handles ``NONE`` (negative/absence claims). Never creates
synthetic boxes or region IDs; an empty region collection ``R = ∅`` is a
valid state, not an error.

Traceability:
  - Feature Specification: FS-005 (Phase 1), section 4.2
  - Research Paper: Section III-C (Region Labelling and Structured Answers)
"""

from dataclasses import dataclass
from typing import Literal

from core.schemas import CandidateBox
from verification.claim_extractor import ExtractedClaim


@dataclass
class AssociatedClaim:
    """An extracted claim associated (or explicitly not) with a region."""

    claim: str
    region_id: str
    status: Literal["present", "absent", "uncertain"]
    confidence: Literal["high", "medium", "low"]
    is_grounded: bool  # True iff region_id in R or region_id == "NONE"
    cited_box: CandidateBox | None  # the CandidateBox matching region_id
    rejection_reason: str | None  # why the claim is ungrounded, if any


def associate_claims(
    claims: list[ExtractedClaim],
    candidate_boxes: list[CandidateBox],
) -> list[AssociatedClaim]:
    """Associate each claim with a candidate region.

    Args:
        claims: Extracted claims from ``claim_extractor.extract_claims``.
        candidate_boxes: The active region set ``R`` proposed by the detector.

    Returns:
        A list of ``AssociatedClaim`` objects, one per input claim, preserving
        input order. Each claim is either grounded (its ``region_id`` names a
        region in ``R``, or is ``NONE``) or explicitly ungrounded with a
        ``rejection_reason``.
    """
    region_lookup = {box.region_id: box for box in candidate_boxes}
    region_ids = sorted(region_lookup.keys())
    r_is_empty = len(candidate_boxes) == 0

    associated: list[AssociatedClaim] = []
    for claim in claims:
        if claim.region_id == "NONE":
            # NONE is a valid token for absence claims; always grounded
            # structurally (no box to point to).
            associated.append(
                AssociatedClaim(
                    claim=claim.claim,
                    region_id=claim.region_id,
                    status=claim.status,
                    confidence=claim.confidence,
                    is_grounded=True,
                    cited_box=None,
                    rejection_reason=None,
                )
            )
            continue

        if r_is_empty:
            associated.append(
                AssociatedClaim(
                    claim=claim.claim,
                    region_id=claim.region_id,
                    status=claim.status,
                    confidence=claim.confidence,
                    is_grounded=False,
                    cited_box=None,
                    rejection_reason="R is empty (no candidate regions)",
                )
            )
            continue

        cited_box = region_lookup.get(claim.region_id)
        if cited_box is not None:
            associated.append(
                AssociatedClaim(
                    claim=claim.claim,
                    region_id=claim.region_id,
                    status=claim.status,
                    confidence=claim.confidence,
                    is_grounded=True,
                    cited_box=cited_box,
                    rejection_reason=None,
                )
            )
        else:
            associated.append(
                AssociatedClaim(
                    claim=claim.claim,
                    region_id=claim.region_id,
                    status=claim.status,
                    confidence=claim.confidence,
                    is_grounded=False,
                    cited_box=None,
                    rejection_reason=(
                        f"region_id {claim.region_id} not in R={region_ids}"
                    ),
                )
            )

    return associated
