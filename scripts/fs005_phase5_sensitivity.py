"""Phase 5 sensitivity analyses (CPU only, no model calls).

1. Strict-only evaluation: restrict each arm to questions whose generation
   satisfied the JSON contract WITHOUT offline salvage, and report accuracy,
   coverage and selective accuracy on that set next to the full (salvaged) set.
2. Gate-signal ablation (on Condition 2 correctness; Condition 3 labels abstained
   questions as wrong, which makes its AUROC circular): AUROC and accuracy at the gate's own coverage for
   agreement-only, entropy-only (1 - H_n) and the fused confidence, using the
   stored per-question signals. Ties are handled in expectation (a tied group at
   the cutoff is accepted proportionally).

Usage (same flags as the driver):
    python scripts/fs005_phase5_sensitivity.py --subset paper \
        --out-dir experiments/runs/fs005_phase5_full [--replicates 2000]
Writes <out-dir>/evaluation/sensitivity.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fs005_phase5_full_run as D  # noqa: E402
from evaluation import metrics as M  # noqa: E402
from evaluation.bootstrap import bootstrap_metric  # noqa: E402


def strict_qids(args, model: str, track: str, split: str) -> set[str]:
    """Question ids whose stored generation passed the contract without salvage."""
    a = argparse.Namespace(**{**vars(args), "model": model, "track": track})
    runner = D._runner(a, None, split=split)  # noqa: SLF001
    out = set()
    for e in D._cohort(split, track):  # noqa: SLF001
        gid = D.compute_generation_id(
            model_name=runner.model_name, model_revision=runner.model_revision,
            track=runner.track, question_id=e["qid"], image_id=e["img_id"],
            n_samples=runner.n_samples, temperature=runner.temperature,
            random_seed=runner.random_seed,
        )
        if runner.store.is_completed_valid(runner.model_name, runner.track, gid):
            out.add(str(e["qid"]))
    return out


def summarize(recs):
    n = len(recs)
    if n == 0:
        return {"n": 0}
    acc = sum(r.correct for r in recs) / n
    kept = [r for r in recs if r.in_coverage]
    return {
        "n": n,
        "accuracy": acc,
        "coverage": len(kept) / n,
        "selective_accuracy": (sum(r.correct for r in kept) / len(kept)) if kept else None,
    }


def acc_at_coverage(scores, correct, coverage):
    """Accuracy of the answered set when the top ``coverage`` fraction is kept (ties in expectation)."""
    n = len(scores)
    k = coverage * n
    order = sorted(range(n), key=lambda i: -scores[i])
    kept_correct, taken = 0.0, 0.0
    i = 0
    while i < n and taken < k - 1e-9:
        j = i
        while j < n and scores[order[j]] == scores[order[i]]:
            j += 1
        group = order[i:j]
        take = min(len(group), k - taken)
        frac = take / len(group)
        kept_correct += frac * sum(correct[g] for g in group)
        taken += take
        i = j
    return kept_correct / taken if taken else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="paper")
    ap.add_argument("--out-dir", default="experiments/runs/fs005_phase5_full")
    ap.add_argument("--replicates", type=int, default=2000)
    args = ap.parse_args()
    if args.subset == "paper":
        D.SUBSET_SIZES = dict(D.PAPER_SUBSET)
    img_of = {str(e["qid"]): e["img_id"] for e in D.split_mod._english_samples("test")}  # noqa: SLF001
    image_fn = lambda r: img_of[r.sample_id]  # noqa: E731
    report = {"strict": {}, "ablation": {}}

    for model in D.MODELS:
        for track in ("B", "C"):
            strict = strict_qids(args, model, track, "test")
            for cond in (2, 3):
                recs = D._load_records(args, model, track, cond)  # noqa: SLF001
                sub = [r for r in recs if r.sample_id in strict]
                report["strict"][f"{model}/track{track}/cond{cond}"] = {
                    "all_salvaged": summarize(recs), "strict_only": summarize(sub),
                }
            if track == "C":
                strict1 = strict_qids(args, model, "C", "test_c1")
                recs = D._load_records(args, model, "C", 1)  # noqa: SLF001
                sub = [r for r in recs if r.sample_id in strict1]
                report["strict"][f"{model}/trackC/cond1"] = {
                    "all_salvaged": summarize(recs), "strict_only": summarize(sub),
                }

            # Condition 2 records carry correctness without abstention. Condition 3
            # records label abstained questions as wrong, which would make AUROC
            # circular, so the ablation must use Condition 2 correctness.
            recs = D._load_records(args, model, track, 2)  # noqa: SLF001
            c3 = D._load_records(args, model, track, 3)  # noqa: SLF001
            if not recs or not c3:
                continue
            cov = sum(r.in_coverage for r in c3) / len(c3)
            variants = {
                "agreement_only": lambda r: r.agreement,
                "entropy_only": lambda r: 1.0 - r.normalized_entropy,
                "fused_frozen_gate": lambda r: r.confidence,
            }
            res = {"gate_coverage": cov, "signals": {}}
            for name, fn in variants.items():
                vr = [dataclasses.replace(r, confidence=fn(r)) for r in recs]
                b = bootstrap_metric(vr, M.compute_auroc, n_replicates=args.replicates, seed=D.SEED,
                                     image_id_fn=image_fn, metric_name="auroc")
                res["signals"][name] = {
                    "auroc": b.point_estimate, "ci95": [b.ci_lower, b.ci_upper],
                    "acc_at_gate_coverage": acc_at_coverage(
                        [fn(r) for r in recs], [r.correct for r in recs], cov),
                }
            report["ablation"][f"{model}/track{track}"] = res

    out = Path(args.out_dir) / "evaluation" / "sensitivity.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
