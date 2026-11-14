"""Unit tests for paired Track B/C cohort consistency validation.

Validates the audited SLAKE test cohort against the FS-005 Phase 5 protocol:
- Track B: 864 GT-annotated samples.
- Track C: 945 eligible samples.
- Paired Track B/C: exactly the same 864 question IDs (and image IDs).
- Additional Track C-only samples: 81, kept separate from paired grounding comparisons.
- Rejects mismatched paired cohorts.
"""

from scripts.fs005_phase5_validate_cohort import validate_cohort


def test_paired_cohort_is_consistent() -> None:
    result = validate_cohort()
    assert result["valid"] is True, result["errors"]
    assert result["warnings"] == [] or result["warnings"] is not None

    counts = result["counts"]
    assert counts["eligible_total"] == 945
    assert counts["track_b_gt_annotated"] == 864
    assert counts["track_c_total"] == 945
    assert counts["track_c_only_no_gt"] == 81
    assert counts["paired_track_b_c"] == 864

    expected = result["expected"]
    assert expected["eligible_total"] == 945
    assert expected["track_b_gt_annotated"] == 864
    assert expected["track_c_total"] == 945
    assert expected["track_c_only_no_gt"] == 81
    assert expected["paired_track_b_c"] == 864

    # Paired set equals Track B set and is a subset of Track C.
    assert set(result["paired_question_ids"]) == set(result["track_b_question_ids"])
    assert set(result["track_b_question_ids"]).issubset(
        set(result["track_c_only_question_ids"]) | set(result["paired_question_ids"])
    )
    # Track C-only (81) and paired (864) are disjoint.
    assert (
        set(result["track_c_only_question_ids"]) & set(result["paired_question_ids"])
        == set()
    )
    assert len(result["track_c_only_question_ids"]) + len(
        result["paired_question_ids"]
    ) == 945


def test_validation_rejects_mismatched_paired_sets(monkeypatch) -> None:
    from scripts import fs005_phase5_validate_cohort as cohort_mod

    # Force an impossible expected paired count so the mismatch check triggers.
    monkeypatch.setattr(cohort_mod, "EXPECTED_PAIRED", 999)
    result = validate_cohort()
    assert result["valid"] is False
    assert any("paired" in e.lower() for e in result["errors"])
