"""Unsupported claim detection and filtering — runtime, ground-truth-free.

Classifies verified claims as supported, unsupported, or unverifiable based
on *structural validity* only (the claim cites a valid region reference).
Ground-truth-derived IoU is never used at runtime — it belongs exclusively
to the evaluation layer (``evaluate_grounding_iou_with_boxes``).

Separates negative/absence claims (ADR 0004, Option 3) from positive-region
grounding metrics. Produces the filtered claim set and structural USR
metrics while preserving original claims and verification records for
auditing.

Traceability:
  - Feature Specification: FS-005 (Phase 1), section 4.4
  - Research Paper: Section III-D (Claim Verification)
"""

from dataclasses import dataclass, field

from verification.evidence_verifier import VerifiedClaim


@dataclass
class FilterResult:
    """Result of runtime claim filtering.

    USR metrics here are *structural USR* — computed from runtime structural
    validity, not from ground-truth IoU. The evaluation layer will compute
    IoU-based USR separately using ``evaluate_grounding_iou_with_boxes``.
    """

    supported_claims: list[VerifiedClaim] = field(default_factory=list)
    unsupported_claims: list[VerifiedClaim] = field(default_factory=list)
    unverifiable_claims: list[VerifiedClaim] = field(default_factory=list)
    absence_claims: list[VerifiedClaim] = field(default_factory=list)
    uncertain_claims: list[VerifiedClaim] = field(default_factory=list)
    usr_raw: float = 0.0
    usr_filtered: float = 0.0
    n_claims: int = 0
    n_positive_claims: int = 0


def filter_claims(verified_claims: list[VerifiedClaim]) -> FilterResult:
    """Filter verified claims at runtime using structural validity only.

    Classification is based on ``VerifiedClaim`` fields set by
    ``verify_association`` (no ground-truth access):

    - Absence claims (status == "absent"): separated into ``absence_claims``,
      excluded from positive-region grounding metrics and denominators
      (ADR 0004, Option 3). No localization confidence assigned.
    - Uncertain claims (status == "uncertain"): separated into
      ``uncertain_claims``, preserved with distinct meaning.
    - Positive/presence claims:
        - supported: structurally valid (grounded to a region in R)
        - unsupported: NOT structurally valid for a positive claim
          (e.g. present + region_id="NONE", which is ungrounded)
        - unverifiable: ungrounded (bad region_id or R = ∅)

    Original verified claims and verification records are preserved in the
    respective lists so later evaluation can compare filtered and
    unfiltered outputs.

    Args:
        verified_claims: Claims from ``evidence_verifier.verify_association``.

    Returns:
        A ``FilterResult`` with partitioned claims and structural USR
        metrics computed over positive claims only.
    """
    supported: list[VerifiedClaim] = []
    unsupported: list[VerifiedClaim] = []
    unverifiable: list[VerifiedClaim] = []
    absence: list[VerifiedClaim] = []
    uncertain: list[VerifiedClaim] = []

    for claim in verified_claims:
        if claim.is_absence_claim:
            absence.append(claim)
            continue

        if claim.is_uncertain_claim:
            uncertain.append(claim)
            continue

        # Positive/presence claims — classified by structural validity.
        if claim.is_supported:
            supported.append(claim)
        elif claim.is_grounded:
            # Grounded but not supported: e.g. present + NONE (valid token
            # but no actual region box). Positive claim without evidence.
            unsupported.append(claim)
        else:
            unverifiable.append(claim)

    n_claims = len(verified_claims)
    # USR computed over positive claims only — absence and uncertain claims
    # are excluded from positive-region grounding denominators
    # (ADR 0004, Option 3).
    n_positive = len(supported) + len(unsupported) + len(unverifiable)

    n_unsup_raw = len(unsupported) + len(unverifiable)
    usr_raw = n_unsup_raw / n_positive if n_positive > 0 else 0.0

    n_denominator_filtered = n_positive - len(unverifiable)
    usr_filtered = (
        len(unsupported) / n_denominator_filtered
        if n_denominator_filtered > 0
        else 0.0
    )

    return FilterResult(
        supported_claims=supported,
        unsupported_claims=unsupported,
        unverifiable_claims=unverifiable,
        absence_claims=absence,
        uncertain_claims=uncertain,
        usr_raw=usr_raw,
        usr_filtered=usr_filtered,
        n_claims=n_claims,
        n_positive_claims=n_positive,
    )
