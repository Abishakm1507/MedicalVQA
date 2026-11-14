"""Image-clustered bootstrap and paired inference for FS-005 Phase 5.

Implements image-level (cluster) bootstrap confidence intervals as required by
the FS-005 statistical analysis plan:

- Resamples IMAGE IDs, not individual questions.
- Preserves ALL eligible questions belonging to each sampled image cluster.
- Preserves paired Track B/C and model observations within each sampled image
  cluster (paired bootstrap uses the same resampled clusters for both arms).
- Calculates percentile confidence intervals for predefined metrics and for
  paired differences between arms.
- Records the bootstrap seed and replicate count.
- Uses at least 1,000 replicates for the planned experiment.

Paired McNemar is provided as a secondary SAMPLE-LEVEL analysis and is
explicitly NOT image-cluster-adjusted; its results must not be interpreted as
cluster-robust.

No model inference is performed by this module. It operates on per-sample
records (e.g. ``evaluation.metrics.SampleRecord``) produced by the pipeline.

Traceability:
  - Feature Specification: FS-005 (Phase 5 statistical analysis)
  - ADR 0005: Confidence gate evaluation
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

MetricFn = Callable[[Sequence[object]], float | None]


@dataclass
class BootstrapResult:
    """Result of a single-metric image-clustered bootstrap."""

    metric_name: str
    point_estimate: float | None
    ci_lower: float | None
    ci_upper: float | None
    ci_level: float
    n_replicates: int
    seed: int
    cluster_unit: str
    n_clusters: int
    n_observations: int
    bootstrap_distribution: list[float] = field(default_factory=list)
    method: str = "image_clustered_bootstrap_percentile"


@dataclass
class PairedBootstrapResult:
    """Result of a paired image-clustered bootstrap difference between two arms."""

    metric_name: str
    diff_point_estimate: float | None
    ci_lower: float | None
    ci_upper: float | None
    ci_level: float
    n_replicates: int
    seed: int
    cluster_unit: str
    n_clusters: int
    n_observations_a: int
    n_observations_b: int
    bootstrap_distribution: list[float] = field(default_factory=list)
    method: str = "paired_image_clustered_bootstrap_percentile"


@dataclass
class McNemarResult:
    """Result of a sample-level paired McNemar test.

    NOTE: ordinary McNemar is NOT image-cluster-adjusted. It treats each
    question as an independent observation, which is invalid when multiple
    questions share the same image. Reported only as a secondary, descriptive
    analysis; image-clustered bootstrap (above) is the primary inference.
    """

    n_both_correct: int
    n_both_incorrect: int
    n_a_only: int  # arm A correct, arm B incorrect
    n_b_only: int  # arm B correct, arm A incorrect
    chi2_statistic: float | None
    p_value: float | None
    note: str = (
        "Ordinary McNemar is sample-level and NOT image-cluster-adjusted. "
        "Do not interpret as cluster-robust inference."
    )


def _chi2_sf(chi2: float, df: int = 1) -> float:
    """Survival function of the chi-square distribution (df=1 only)."""
    if chi2 < 0:
        return 1.0
    # df=1: chi2 = z^2, survival = 2 * (1 - Phi(|z|)).
    z = math.sqrt(chi2)
    # Standard normal survival via erfc.
    return math.erfc(z / math.sqrt(2.0))


def _group_by_cluster(
    records: Sequence[object], image_id_fn: Callable[[object], str | int]
) -> dict[str | int, list[object]]:
    groups: dict[str | int, list[object]] = {}
    for r in records:
        groups.setdefault(image_id_fn(r), []).append(r)
    return groups


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    n = len(s)
    if n == 1:
        return float(s[0])
    k = (n - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(s[int(k)])
    d = k - f
    return float(s[f] * (1 - d) + s[c] * d)


def _default_image_id(record: object) -> str | int:
    """Best-effort image cluster key for a record.

    Tries ``record.image_id`` first, then ``record.sample_id`` (which may not
    be image-unique; callers should supply ``image_id_fn`` whenever records
    carry a true image identifier).
    """
    if hasattr(record, "image_id"):
        return record.image_id
    if hasattr(record, "sample_id"):
        return record.sample_id
    return str(record)


def bootstrap_metric(
    records: Sequence[object],
    metric_fn: MetricFn,
    *,
    n_replicates: int = 1000,
    seed: int = 42,
    ci_level: float = 0.95,
    image_id_fn: Callable[[object], str | int] | None = None,
    metric_name: str = "metric",
) -> BootstrapResult:
    """Image-clustered bootstrap confidence interval for a metric.

    Each replicate resamples image IDs with replacement and includes every
    record whose image ID was drawn, preserving within-cluster question
    multiplicity and any paired observations stored on the records.

    Args:
        records: Per-sample records to bootstrap.
        metric_fn: Function mapping a list of records to a scalar metric.
        n_replicates: Number of bootstrap resamples (>= 1,000 recommended).
        seed: Random seed (recorded for reproducibility).
        ci_level: Confidence level in (0, 1).
        image_id_fn: Callable extracting the image cluster key from a record.
            Defaults to ``record.image_id`` or ``record.sample_id``.
        metric_name: Label for the metric.

    Returns:
        BootstrapResult with point estimate, percentile CI, seed, replicate
        count, and the full bootstrap distribution.
    """
    if n_replicates < 1:
        raise ValueError(f"n_replicates must be >= 1, got {n_replicates}")
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    if not records:
        return BootstrapResult(
            metric_name=metric_name,
            point_estimate=None,
            ci_lower=None,
            ci_upper=None,
            ci_level=ci_level,
            n_replicates=n_replicates,
            seed=seed,
            cluster_unit="image",
            n_clusters=0,
            n_observations=0,
        )

    id_fn = image_id_fn or _default_image_id
    clusters = _group_by_cluster(records, id_fn)
    cluster_keys = sorted(clusters.keys())
    n_clusters = len(cluster_keys)
    n_observations = len(records)

    rng = random.Random(seed)
    point_estimate = metric_fn(records)

    dist: list[float] = []
    tail = (1.0 - ci_level) / 2.0
    for _ in range(n_replicates):
        sampled_keys = [rng.choice(cluster_keys) for _ in range(n_clusters)]
        bootstrap_records: list[object] = []
        for key in sampled_keys:
            bootstrap_records.extend(clusters[key])
        value = metric_fn(bootstrap_records)
        if value is not None:
            dist.append(float(value))

    alpha_low = tail
    alpha_high = 1.0 - tail
    ci_lower = _percentile(dist, alpha_low) if dist else None
    ci_upper = _percentile(dist, alpha_high) if dist else None

    return BootstrapResult(
        metric_name=metric_name,
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        n_replicates=n_replicates,
        seed=seed,
        cluster_unit="image",
        n_clusters=n_clusters,
        n_observations=n_observations,
        bootstrap_distribution=dist,
    )


def bootstrap_paired_difference(
    records_a: Sequence[object],
    records_b: Sequence[object],
    metric_fn: MetricFn,
    *,
    n_replicates: int = 1000,
    seed: int = 42,
    ci_level: float = 0.95,
    image_id_fn: Callable[[object], str | int] | None = None,
    metric_name: str = "metric",
    require_same_images: bool = True,
) -> PairedBootstrapResult:
    """Paired image-clustered bootstrap difference between two arms.

    Uses the SAME resampled image clusters for both arms in each replicate so
    the difference is computed on paired cluster-level observations, preserving
    the Track B/C and model pairing within each image.

    Args:
        records_a: Per-sample records for arm A.
        records_b: Per-sample records for arm B.
        metric_fn: Function mapping a list of records to a scalar metric.
        n_replicates: Number of bootstrap resamples (>= 1,000 recommended).
        seed: Random seed (recorded for reproducibility).
        ci_level: Confidence level in (0, 1).
        image_id_fn: Callable extracting the image cluster key from a record.
        metric_name: Label for the metric.
        require_same_images: If True, raise when the two arms do not share the
            same set of image IDs (required for a valid paired comparison).

    Returns:
        PairedBootstrapResult with the point-estimate difference, percentile CI
        for the difference, seed, replicate count, and distribution.
    """
    if n_replicates < 1:
        raise ValueError(f"n_replicates must be >= 1, got {n_replicates}")
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")

    id_fn = image_id_fn or _default_image_id
    clusters_a = _group_by_cluster(records_a, id_fn)
    clusters_b = _group_by_cluster(records_b, id_fn)
    keys_a = set(clusters_a.keys())
    keys_b = set(clusters_b.keys())

    if require_same_images and keys_a != keys_b:
        only_a = sorted(keys_a - keys_b)
        only_b = sorted(keys_b - keys_a)
        raise ValueError(
            "Paired bootstrap requires identical image sets across arms. "
            f"Images only in A: {only_a}; images only in B: {only_b}"
        )

    cluster_keys = sorted(keys_a & keys_b)
    n_clusters = len(cluster_keys)
    n_obs_a = len(records_a)
    n_obs_b = len(records_b)

    rng = random.Random(seed)
    point_a = metric_fn(records_a)
    point_b = metric_fn(records_b)
    diff_estimate = (
        (point_a - point_b) if (point_a is not None and point_b is not None) else None
    )

    dist: list[float] = []
    tail = (1.0 - ci_level) / 2.0
    for _ in range(n_replicates):
        sampled_keys = [rng.choice(cluster_keys) for _ in range(n_clusters)]
        boot_a = [r for k in sampled_keys for r in clusters_a[k]]
        boot_b = [r for k in sampled_keys for r in clusters_b[k]]
        va = metric_fn(boot_a)
        vb = metric_fn(boot_b)
        if va is not None and vb is not None:
            dist.append(float(va) - float(vb))

    ci_lower = _percentile(dist, tail) if dist else None
    ci_upper = _percentile(dist, 1.0 - tail) if dist else None

    return PairedBootstrapResult(
        metric_name=metric_name,
        diff_point_estimate=diff_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        n_replicates=n_replicates,
        seed=seed,
        cluster_unit="image",
        n_clusters=n_clusters,
        n_observations_a=n_obs_a,
        n_observations_b=n_obs_b,
        bootstrap_distribution=dist,
    )


def mcnemar_paired(
    correct_a: Sequence[bool],
    correct_b: Sequence[bool],
) -> McNemarResult:
    """Sample-level paired McNemar test for two binary correctness series.

    This is ORDINARY McNemar: it treats every question as an independent
    observation and is NOT image-cluster-adjusted. It is provided only as a
    secondary descriptive analysis. The primary inference for FS-005 paired
    comparisons is the image-clustered bootstrap (see
    :func:`bootstrap_paired_difference`).

    Args:
        correct_a: Binary correctness for arm A (one value per paired sample).
        correct_b: Binary correctness for arm B (one value per paired sample).

    Returns:
        McNemarResult with the contingency counts, chi-square statistic,
        p-value, and a note that the test is not cluster-adjusted.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError(
            f"Paired series must have equal length: {len(correct_a)} vs {len(correct_b)}"
        )

    n_both_correct = 0
    n_both_incorrect = 0
    n_a_only = 0  # A correct, B incorrect
    n_b_only = 0  # B correct, A incorrect

    for ca, cb in zip(correct_a, correct_b, strict=False):
        ca = bool(ca)
        cb = bool(cb)
        if ca and cb:
            n_both_correct += 1
        elif not ca and not cb:
            n_both_incorrect += 1
        elif ca and not cb:
            n_a_only += 1
        else:
            n_b_only += 1

    discordant = n_a_only + n_b_only
    if discordant == 0:
        chi2 = None
        p_value = None
    else:
        # Continuity-corrected McNemar chi-square.
        chi2 = ((abs(n_a_only - n_b_only) - 1.0) ** 2) / discordant
        p_value = _chi2_sf(chi2, df=1)

    return McNemarResult(
        n_both_correct=n_both_correct,
        n_both_incorrect=n_both_incorrect,
        n_a_only=n_a_only,
        n_b_only=n_b_only,
        chi2_statistic=chi2,
        p_value=p_value,
    )
