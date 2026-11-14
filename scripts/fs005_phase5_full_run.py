"""FS-005 Phase 5 full-cohort driver: generate -> calibrate -> evaluate.

Stages (each resumable and safe to re-run):

  generate   Run N=5 sampled VLM generations (GenerationRunner artifacts) for one
             cohort split, one model and one track. ``--split test_c1`` runs the
             single greedy (N=1, T=0) generation used for Condition 1.
  calibrate  Fit confidence weights (w1,w2,w3) and tau on the validation FIT
             split only, then assess on the image-disjoint ASSESSMENT split.
             Writes frozen.json. The test split is never read.
  evaluate   Read frozen.json, score the test split for Conditions 1-3 and write
             the metric artifacts with evaluation.runner.EvaluationRunner.
  status     Show how many artifacts exist per split.

Cohorts (all English SLAKE, pilot questions and pilot images removed):
  val_fit / val_assess   image-disjoint 80/20 split of validation (seed 42)
  test                   945 test questions (Track B: only those with GT boxes)

Research-integrity rules (AGENTS.md): no metric is estimated from placeholder
inputs; weights/tau come only from the calibrate stage; correctness is scored
against SLAKE ground truth by normalized exact match.

Known limits, also stated in the outputs:
  * Track B regions ARE the SLAKE ground-truth boxes, so pointing accuracy under
    Track B measures whether a claim cites any oracle region, not localization
    quality.
  * Condition 1 here is a greedy single sample using the same prompt contract
    as Conditions 2/3, not an unconstrained free-form prompt.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402

from core.generation_artifact import compute_generation_id  # noqa: E402
from core.schemas import CandidateBox, VLMClaim, VLMStructuredOutput  # noqa: E402
from evaluation.metrics import SampleRecord  # noqa: E402
from evaluation.runner import EvaluationRunner  # noqa: E402
from scripts import fs005_phase5_calibration_split as split_mod  # noqa: E402
from scripts.fs005_phase5_generation import GenerationRunner  # noqa: E402
from verification.claim_associator import associate_claims  # noqa: E402
from verification.claim_extractor import extract_claims  # noqa: E402
from verification.claim_filter import filter_claims  # noqa: E402
from verification.confidence_gate import ConfidenceCalibrator  # noqa: E402
from verification.evidence_verifier import verify_association  # noqa: E402

SLAKE_DIR = PROJECT_ROOT / "data" / "SLAKE"
IMG_DIR = SLAKE_DIR / "imgs"
N_SAMPLES = 5
TEMPERATURE = 0.7
SEED = 42
TRACK_LABEL = {"B": "track_b_oracle_slake", "C": "track_c_answer_only"}
MODELS = ("medgemma-4b", "qwen2.5-vl-3b")
MANIFEST_MODEL = "medgemma-4b"  # split manifests are identical across models
SPLITS = ("val_fit", "val_assess", "test", "test_c1")


# --------------------------------------------------------------------------
# Cohorts
# --------------------------------------------------------------------------


def _cohort(split: str, track: str) -> list[dict]:
    """Return raw SLAKE entries for a split/track (never overlaps pilot)."""
    if split in ("test", "test_c1"):
        entries = split_mod._excluded_by_pilot(split_mod._english_samples("test"))
    else:
        # The committed image-disjoint split manifests (validation pilot images
        # already removed) are the single source of truth for calibration data.
        name = "fit" if split == "val_fit" else "assessment"
        manifest = (
            PROJECT_ROOT / "experiments" / "runs" / "fs005_phase5" / "calibration"
            / "splits" / MANIFEST_MODEL / track / f"{name}_qids.csv"
        )
        qids = {int(x) for x in manifest.read_text("utf-8").split()[1:]}
        entries = [e for e in split_mod._english_samples("validate") if e["qid"] in qids]
    if track == "B":
        entries = [e for e in entries if split_mod._has_gt_regions(e["img_id"])]
    return sorted(entries, key=lambda e: e["qid"])


def _detections(img_id: int) -> list[dict]:
    return json.loads((IMG_DIR / f"xmlab{img_id}" / "detection.json").read_text("utf-8"))


def _gt_boxes(img_id: int) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Return [(label, (x1, y1, x2, y2))] from SLAKE XYWH detections."""
    out = []
    for det in _detections(img_id):
        label, (x, y, w, h) = next(iter(det.items()))
        out.append((label, (float(x), float(y), float(x + w), float(y + h))))
    return out


class LazyImages:
    """Mapping-like image provider: loads (and, for Track B, overlays) on demand."""

    def __init__(self, track: str, cache_size: int = 8) -> None:
        self.track = track
        self._cache: OrderedDict[str, Image.Image] = OrderedDict()
        self._cache_size = cache_size

    def get(self, key, default=None):
        key = str(key)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        image_dir = IMG_DIR / f"xmlab{key}"
        with Image.open(image_dir / "source.jpg") as src:
            image = src.convert("RGB")
        if self.track == "B":
            from scripts.fs005_phase5_ground_truth import build_ground_truth_overlay

            image, _, _ = build_ground_truth_overlay(image, image_dir / "detection.json")
        self._cache[key] = image
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return image


def _input_samples(entries: list[dict], track: str) -> list[dict]:
    samples = []
    for e in entries:
        region_ids = (
            [f"R{i + 1}" for i in range(len(_detections(e["img_id"])))] if track == "B" else []
        )
        samples.append(
            {
                "question_id": e["qid"],
                "image_id": e["img_id"],
                "question": e["question"],
                "region_ids": region_ids,
                "localizer_score": 1.0 if track == "B" else None,
            }
        )
    return samples


# --------------------------------------------------------------------------
# Runner construction / artifact access
# --------------------------------------------------------------------------


def _out(args) -> Path:
    return Path(args.out_dir)


def _runner(args, client, *, split: str, tau=0.5, w=(1 / 3, 1 / 3, 1 / 3)) -> GenerationRunner:
    greedy = split == "test_c1"
    root = _out(args) / ("condition1" if greedy else "shared")
    return GenerationRunner(
        vlm_client=client,
        model_name=args.model,
        model_revision=None,
        track="C" if greedy else args.track,
        output_dir=root,
        random_seed=SEED,
        temperature=0.0 if greedy else TEMPERATURE,
        n_samples=1 if greedy else N_SAMPLES,
        w1=w[0],
        w2=w[1],
        w3=w[2],
        tau=tau,
    )


def _load_record(runner: GenerationRunner, qid, iid):
    gid = compute_generation_id(
        model_name=runner.model_name,
        model_revision=runner.model_revision,
        track=runner.track,
        question_id=qid,
        image_id=iid,
        n_samples=runner.n_samples,
        temperature=runner.temperature,
        random_seed=runner.random_seed,
    )
    if not runner.store.is_completed_valid(runner.model_name, runner.track, gid):
        return None
    return runner.store.load(runner.model_name, runner.track, gid)


def _make_client(args):
    if args.mock:
        from vlm.mock_client import MockVLMClient

        return MockVLMClient(model_name=args.model)
    if args.model == "medgemma-4b":
        from vlm.medgemma_client import MedGemmaVLMClient

        client = MedGemmaVLMClient(
            model_id="google/medgemma-4b-it", device="cuda", quantization="4bit"
        )
    else:
        from vlm.qwen_client import QwenVLMClient

        client = QwenVLMClient(
            model_id="Qwen/Qwen2.5-VL-3B-Instruct",
            device="cuda",
            quantization="4bit",
            trust_remote_code=True,
        )
    client._load()  # noqa: SLF001
    return client


# --------------------------------------------------------------------------
# Scoring helpers
# --------------------------------------------------------------------------


def _norm(text: str | None) -> str:
    return re.sub(r"[\s\.\,;:!]+$", "", (text or "").strip().lower())


def _correct(answer: str | None, gt: str) -> bool:
    return answer is not None and _norm(answer) == _norm(gt)


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _claim_stats(record, entry: dict, track: str) -> dict[str, Any]:
    """Run the GT-free claim pipeline on stored samples; IoU pointing offline."""
    iid = entry["img_id"]
    boxes: list[CandidateBox] = []
    gt = _gt_boxes(iid) if split_mod._has_gt_regions(iid) else []
    if track == "B":
        with Image.open(IMG_DIR / f"xmlab{iid}" / "source.jpg") as im:
            w, h = im.size
        for idx, (label, (x1, y1, x2, y2)) in enumerate(gt):
            boxes.append(
                CandidateBox(
                    region_id=f"R{idx + 1}",
                    box_2d=(int(y1), int(x1), int(y2), int(x2)),
                    normalized_box=(y1 / h, x1 / w, y2 / h, x2 / w),
                    score=1.0,
                    label=label,
                )
            )
    outputs = []
    for s in record.samples:
        if s.status != "ok" or s.parsed_answer is None:
            continue
        outputs.append(
            VLMStructuredOutput(
                answer=s.parsed_answer,
                findings=[
                    VLMClaim(
                        claim=c["claim"],
                        region_id=c["region_id"],
                        status=c["status"],
                        confidence=c["confidence"],
                    )
                    for c in s.claims
                ],
            )
        )
    claims = []
    for out in outputs:
        claims.extend(extract_claims(out, raw_response=None).claims)
    associated = associate_claims(claims, boxes)
    filtered = filter_claims(verify_association(associated))

    pointing = False
    for ac in associated:
        if ac.status != "present" or ac.cited_box is None:
            continue
        cb = ac.cited_box.box_2d  # [ymin, xmin, ymax, xmax]
        cited_xyxy = (cb[1], cb[0], cb[3], cb[2])
        if any(_iou(cited_xyxy, g[1]) >= 0.5 for g in gt):
            pointing = True
            break
    return {
        "n_claims": filtered.n_claims,
        "n_positive_claims": filtered.n_positive_claims,
        "n_supported": len(filtered.supported_claims),
        "n_unsupported": len(filtered.unsupported_claims),
        "n_unverifiable": len(filtered.unverifiable_claims),
        "n_absence": len(filtered.absence_claims),
        "n_uncertain": len(filtered.uncertain_claims),
        "pointing_correct": pointing,
        "has_gt_regions": bool(gt),
    }


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def cmd_generate(args) -> int:
    entries = _cohort(args.split, args.track)
    if args.limit:
        entries = entries[: args.limit]
    track = "C" if args.split == "test_c1" else args.track
    client = _make_client(args)
    runner = _runner(args, client, split=args.split)
    samples = _input_samples(entries, track)
    print(f"[generate] {args.model} track={runner.track} split={args.split} n={len(samples)}")
    summary = runner.run_batch(samples, LazyImages(track))
    print(
        json.dumps(
            {k: summary[k] for k in summary if k not in ("condition_results", "progress")},
            indent=2,
            default=str,
        )
    )
    return 0


def _calib_dir(args) -> Path:
    return _out(args) / "calibration" / args.model / args.track


def _collect(args, split: str, w=(1 / 3, 1 / 3, 1 / 3), tau=0.5):
    runner = _runner(args, None, split=split, w=w, tau=tau)
    rows = []
    for e in _cohort(split, args.track):
        rec = _load_record(runner, e["qid"], e["img_id"])
        if rec is None:
            continue
        rho = 1.0 if args.track == "B" else None
        res = runner.evaluate_condition_2(rec, localizer_score=rho)
        rows.append((e, rec, res, rho))
    return runner, rows


def cmd_calibrate(args) -> int:
    _, fit = _collect(args, "val_fit")
    _, assess = _collect(args, "val_assess")
    if not fit:
        print("No completed val_fit artifacts; run `generate --split val_fit` first.")
        return 2

    def cols(rows):
        return (
            [r[2].agreement for r in rows],
            [r[2].normalized_entropy for r in rows],
            [r[3] for r in rows],
            [r[2].evidence_state for r in rows],
            [_correct(r[2].majority_answer, r[0]["answer"]) for r in rows],
        )

    cal = ConfidenceCalibrator().calibrate(*cols(fit))
    out = {
        "model": args.model,
        "track": args.track,
        "fitted_on": "val_fit",
        "n_fit": len(fit),
        "n_assess": len(assess),
        "w1": cal.w1,
        "w2": cal.w2,
        "w3": cal.w3,
        "tau": cal.tau,
        "fit": {
            "ece": cal.ece,
            "auroc": cal.auroc,
            "selective_risk": cal.selective_risk,
            "coverage": cal.coverage,
        },
    }
    if assess:
        from verification.confidence_gate import ConfidenceConfig, compute_confidence

        cfg = ConfidenceConfig(w1=cal.w1, w2=cal.w2, w3=cal.w3, tau=cal.tau)
        a, h, rho, es, ok = cols(assess)
        conf = [
            compute_confidence(x, y, r, cfg, s, sampling_valid=True)
            for x, y, r, s in zip(a, h, rho, es, strict=False)
        ]
        kept = [c >= cal.tau for c in conf]
        n_kept = sum(kept)
        out["assessment"] = {
            "coverage": n_kept / len(conf),
            "selective_accuracy": (
                sum(1 for k, o in zip(kept, ok, strict=False) if k and o) / n_kept
                if n_kept
                else None
            ),
            "accuracy_all": sum(ok) / len(ok),
        }
    d = _calib_dir(args)
    d.mkdir(parents=True, exist_ok=True)
    (d / "frozen.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


def cmd_evaluate(args) -> int:
    frozen_path = _calib_dir(args) / "frozen.json"
    if not frozen_path.exists():
        print(f"Missing {frozen_path}; run `calibrate` first (test split stays untouched).")
        return 2
    fz = json.loads(frozen_path.read_text("utf-8"))
    w, tau = (fz["w1"], fz["w2"], fz["w3"]), fz["tau"]

    runner, rows = _collect(args, "test", w=w, tau=tau)
    c1_runner = _runner(args, None, split="test_c1")
    records: dict[int, list[SampleRecord]] = {1: [], 2: [], 3: []}
    notes = [
        "Weights and tau frozen from validation FIT split only.",
        "Track B pointing accuracy is computed against the same boxes the model "
        "was shown (oracle regions); it is not a localization-quality measure.",
        "Condition 1 is a greedy single sample with the same prompt contract.",
    ]
    for e, rec, res2, rho in rows:
        stats = _claim_stats(rec, e, args.track)
        res3 = runner.evaluate_condition_3(rec, localizer_score=rho)
        common = dict(
            sample_id=str(e["qid"]),
            dataset_name="SLAKE",
            split="test",
            track=TRACK_LABEL[args.track],
            ground_truth_answer=e["answer"],
            question_type=e["answer_type"].upper(),
            agreement=res2.agreement,
            normalized_entropy=res2.normalized_entropy,
            confidence=res2.confidence,
            threshold=tau,
            evidence_state=res2.evidence_state,
            sampling_valid=res2.sampling_valid,
            **stats,
        )
        records[2].append(
            SampleRecord(
                condition=2,
                model_answer=res2.answer,
                correct=_correct(res2.answer, e["answer"]),
                is_abstaining=False,
                in_coverage=True,
                **common,
            )
        )
        records[3].append(
            SampleRecord(
                condition=3,
                model_answer=res3.answer,
                correct=_correct(res3.answer, e["answer"]),
                is_abstaining=res3.is_abstaining,
                in_coverage=res3.in_coverage,
                **common,
            )
        )
        if args.track == "C":
            r1 = _load_record(c1_runner, e["qid"], e["img_id"])
            if r1 is not None and r1.samples and r1.samples[0].parsed_answer is not None:
                ans = r1.samples[0].parsed_answer
                records[1].append(
                    SampleRecord(
                        sample_id=str(e["qid"]),
                        dataset_name="SLAKE",
                        split="test",
                        track=TRACK_LABEL["C"],
                        condition=1,
                        ground_truth_answer=e["answer"],
                        model_answer=ans,
                        correct=_correct(ans, e["answer"]),
                        question_type=e["answer_type"].upper(),
                        has_gt_regions=stats["has_gt_regions"],
                        pointing_correct=False,
                        n_claims=0,
                        n_positive_claims=0,
                        n_supported=0,
                        n_unsupported=0,
                        n_unverifiable=0,
                        n_absence=0,
                        n_uncertain=0,
                        agreement=0.0,
                        normalized_entropy=0.0,
                        confidence=0.0,
                        threshold=0.0,
                        is_abstaining=False,
                        in_coverage=True,
                        evidence_state="present_ungrounded",
                        sampling_valid=False,
                    )
                )
    for cond, recs in records.items():
        if not recs:
            continue
        out = _out(args) / "evaluation" / args.model / args.track / f"condition{cond}"
        cfg = {"model": args.model, "track": args.track, "condition": cond,
               "w": list(w), "tau": tau, "n_samples": N_SAMPLES,
               "temperature": TEMPERATURE, "seed": SEED}
        metrics = EvaluationRunner(out, f"FS-005 {args.model} track {args.track} cond {cond}",
                                   cfg).run(recs, exclusions=notes)
        print(f"condition {cond}: n={metrics.n_samples} acc={metrics.accuracy} "
              f"usr_raw={metrics.usr_raw} pointing={metrics.pointing_accuracy} "
              f"ece={metrics.ece} auroc={metrics.auroc} coverage={metrics.coverage}")
    return 0


def cmd_status(args) -> int:
    for model in MODELS:
        for track in ("B", "C"):
            args.model, args.track = model, track
            for split in SPLITS:
                if split == "test_c1" and track == "B":
                    continue
                runner = _runner(args, None, split=split)
                entries = _cohort(split, track)
                done = sum(
                    1 for e in entries if _load_record(runner, e["qid"], e["img_id"]) is not None
                )
                print(f"{model:15s} track {track} {split:10s} {done:5d}/{len(entries)}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stage", choices=["generate", "calibrate", "evaluate", "status"])
    p.add_argument("--model", choices=MODELS, default="medgemma-4b")
    p.add_argument("--track", choices=["B", "C"], default="C")
    p.add_argument("--split", choices=SPLITS, default="val_fit")
    p.add_argument("--limit", type=int, default=0, help="Only the first N questions (smoke tests).")
    p.add_argument("--mock", action="store_true", help="Use MockVLMClient (testing only).")
    p.add_argument("--out-dir", default="experiments/runs/fs005_phase5_full")
    args = p.parse_args()
    return {
        "generate": cmd_generate,
        "calibrate": cmd_calibrate,
        "evaluate": cmd_evaluate,
        "status": cmd_status,
    }[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
