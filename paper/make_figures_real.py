"""Regenerates the paper figures from the measured Phase 5 results (no placeholder values).

Run:  python paper/make_figures_real.py   (needs matplotlib)
Reads experiments/runs/fs005_phase5_full/evaluation/** and analysis.json.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "experiments/runs/fs005_phase5_full/evaluation"
OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"mathtext.fontset": "stix", "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8})
analysis = json.load(open(EVAL / "analysis.json"))["single"]

ARMS = [("medgemma-4b", "B", "MedGemma B"), ("medgemma-4b", "C", "MedGemma C"),
        ("qwen2.5-vl-3b", "B", "Qwen B"), ("qwen2.5-vl-3b", "C", "Qwen C")]


def box(ax, x, y, w, h, text, fc="#ebebeb"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", fc=fc, ec="black", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5)


# ---- Fig 1: pipeline schematic (see fig_pipeline.py)
from fig_pipeline import draw_pipeline

draw_pipeline(None, OUT / "fig1_architecture.png")

# ---- Fig 2: cohorts
fig, ax = plt.subplots(figsize=(3.4, 3.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 13); ax.axis("off")
ax.text(5, 12.6, "Test (evaluation)", ha="center", fontweight="bold")
box(ax, 1, 11.3, 8, 0.9, "English test questions: 1,061")
box(ax, 1, 9.9, 8, 1.0, "Pilot questions and their 9 images removed:\n945 questions, 87 images")
box(ax, 0.2, 8.4, 4.6, 1.0, "With GT regions: 864\n(Track B pool)")
box(ax, 5.2, 8.4, 4.6, 1.0, "No GT regions: 81\n(not used for B)")
box(ax, 1, 6.9, 8, 1.0, "Seeded random subset: 250 questions\n(same questions for Tracks B and C)", "#d0d0d0")
ax.text(5, 6.0, "Calibration (validation split)", ha="center", fontweight="bold")
box(ax, 1, 4.9, 8, 0.9, "English validation questions: 1,053")
box(ax, 1, 3.5, 8, 1.0, "15 pilot images removed: 891 questions\nImage-disjoint 80/20 split")
box(ax, 0.2, 2.0, 4.6, 1.0, "Fit: 719 q, 65 images\nsubset: 200 q", "#d0d0d0")
box(ax, 5.2, 2.0, 4.6, 1.0, "Assess: 172 q, 16 images\nsubset: 80 q", "#d0d0d0")
ax.text(5, 1.0, "No image in both fit and assess; test never used to fit.", ha="center", style="italic", fontsize=7)
fig.savefig(OUT / "fig2_cohorts.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---- Fig 3: accuracy by arm (Cond 1/2 with CI; Cond 3 selective accuracy)
fig, ax = plt.subplots(figsize=(3.4, 2.2))
w = 0.2
conds = [("cond1", "Cond 1 (greedy)", "#cfcfcf", "accuracy"), ("cond2", "Cond 2 (grounded)", "#8c8c8c", "accuracy"),
         ("cond3", "Cond 3 (answered only)", "#2f2f2f", "selective_accuracy")]
for j, (c, lab, col, metric) in enumerate(conds):
    xs, vs, lo, hi = [], [], [], []
    for i, (m, t, _) in enumerate(ARMS):
        d = analysis.get(f"{m}/track{t}/{c}")
        if d is None:
            continue
        p = d[metric]
        xs.append(i + (j - 1) * w); vs.append(p["point"])
        lo.append(p["point"] - p["ci95"][0]); hi.append(p["ci95"][1] - p["point"])
    ax.bar(xs, vs, w, color=col, ec="black", lw=0.5, label=lab, yerr=[lo, hi], capsize=1.5, error_kw=dict(lw=0.6))
ax.set_xticks(range(4)); ax.set_xticklabels([a[2] for a in ARMS])
ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.25); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=1)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT / "fig3_accuracy.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---- Fig 4: accuracy-coverage curves (Condition 3 sweep of tau)
fig, ax = plt.subplots(figsize=(3.4, 2.2))
styles = ["-", "--", "-.", ":"]
for (m, t, lab), ls in zip(ARMS, styles):
    d = json.load(open(EVAL / m / t / "condition3" / "experiment_metrics.json"))["accuracy_vs_coverage"]
    pts = sorted({(p["coverage"], p["selective_accuracy"]) for p in d if p["coverage"] > 0})
    ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, color="black", lw=1, label=lab)
ax.set_xlabel("Coverage"); ax.set_ylabel("Accuracy on answered"); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
ax.legend(frameon=False, fontsize=6.5, loc="lower left"); ax.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT / "fig4_risk_coverage.png", dpi=300, bbox_inches="tight"); plt.close(fig)
print("figures written to", OUT)
