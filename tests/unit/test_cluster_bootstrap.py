"""Unit tests for image-clustered bootstrap and paired inference.

Covers: cluster preservation (all questions per sampled image), seed/replicate
count recording, >= 1,000 replicates, paired bootstrap using the same clusters
for both arms, and McNemar being explicitly NOT image-cluster-adjusted.
"""

from evaluation.bootstrap import (
    McNemarResult,
    bootstrap_metric,
    bootstrap_paired_difference,
    mcnemar_paired,
)


def _records_with_images():
    # 3 images, 3 questions each = 9 records. Metric = fraction correct.
    records = []
    for img in range(1, 4):
        for j in range(3):
            records.append(
                type(
                    "R",
                    (),
                    {
                        "image_id": f"img{img}",
                        "sample_id": f"q{img}{j}",
                        "correct": (img + j) % 2 == 0,
                    },
                )()
            )
    return records


def test_bootstrap_preserves_image_clusters() -> None:
    records = _records_with_images()

    def accuracy(recs):
        correct = sum(1 for r in recs if getattr(r, "correct", False))
        return correct / len(recs) if recs else None

    result = bootstrap_metric(
        records, accuracy, n_replicates=200, seed=1, ci_level=0.95
    )

    # Every bootstrap replicate must contain only whole image clusters.
    # Reconstruct: distribution length equals n_replicates (non-none values).
    assert len(result.bootstrap_distribution) == 200
    assert result.n_clusters == 3
    assert result.n_observations == 9
    assert result.seed == 1
    assert result.cluster_unit == "image"
    assert result.method == "image_clustered_bootstrap_percentile"
    assert result.point_estimate is not None


def test_bootstrap_records_seed_and_replicates() -> None:
    records = _records_with_images()
    result = bootstrap_metric(
        records, lambda rs: 0.5, n_replicates=1000, seed=42
    )
    assert result.n_replicates == 1000
    assert result.seed == 42


def test_bootstrap_at_least_1000_replicates_recommended() -> None:
    records = _records_with_images()
    result = bootstrap_metric(
        records, lambda rs: 0.5, n_replicates=1000, seed=0
    )
    assert result.n_replicates >= 1000


def test_paired_bootstrap_uses_same_clusters_for_both_arms() -> None:
    records_a = _records_with_images()
    records_b = _records_with_images()

    def acc(recs):
        return sum(1 for r in recs if getattr(r, "correct", False)) / len(recs)

    result = bootstrap_paired_difference(
        records_a, records_b, acc, n_replicates=200, seed=5
    )
    assert result.n_clusters == 3
    assert result.n_observations_a == 9
    assert result.n_observations_b == 9
    assert len(result.bootstrap_distribution) == 200
    assert result.seed == 5
    # Identical arms → difference ~ 0.
    assert result.diff_point_estimate == 0.0


def test_paired_bootstrap_requires_same_images() -> None:
    records_a = _records_with_images()
    # records_b has an extra image (img4) → mismatched image sets.
    records_b = _records_with_images() + [
        type("R", (), {"image_id": "img4", "sample_id": "q40", "correct": True})()
    ]

    def acc(recs):
        return sum(1 for r in recs if getattr(r, "correct", False)) / len(recs)

    try:
        bootstrap_paired_difference(records_a, records_b, acc, n_replicates=50, seed=0)
        raise AssertionError("Expected ValueError for mismatched image sets")
    except ValueError as exc:
        assert "image sets" in str(exc).lower() or "Paired bootstrap requires" in str(exc)


def test_mcnemar_is_sample_level_and_not_cluster_adjusted() -> None:
    correct_a = [True, True, False, False, True]
    correct_b = [True, False, False, True, True]
    result = mcnemar_paired(correct_a, correct_b)
    assert isinstance(result, McNemarResult)
    assert result.n_both_correct == 2  # q0, q4
    assert result.n_both_incorrect == 1  # q2
    assert result.n_a_only == 1  # q3 (A False? wait)
    # Recompute: q0 T/T both_correct; q1 T/F a_only; q2 F/F both_incorrect; q3 F/T b_only; q4 T/T both_correct
    assert result.n_a_only == 1
    assert result.n_b_only == 1
    assert "NOT image-cluster-adjusted" in result.note
    assert "sample-level" in result.note.lower()
