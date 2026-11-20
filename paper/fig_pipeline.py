"""Draws the pipeline schematic (Fig. 1) for the paper. Pure drawing, no measured values.

Called from make_figures_real.py.  The thumbnails are schematic, not real radiographs.
"""
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

NAVY = "#14305f"
BLUE_F, BLUE_E = "#e8f0fb", "#4f7cc0"
GRN_F, GRN_E = "#e1f4ea", "#2e9468"
RED_F, RED_E = "#fbe3e3", "#c0392b"
GREY = "#555b66"
TAG = {"R1": "#2f6fd0", "R2": "#1f9d5c", "R3": "#d23030"}


def _rbox(ax, x, y, w, h, text="", fc=BLUE_F, ec=BLUE_E, fs=5.6, ls="-", lw=0.7, bold=False,
          color="black", style="normal", r=1.2, family=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
                fontweight="bold" if bold else "normal", style=style, linespacing=1.15, family=family)


def _arrow(ax, x1, y1, x2, y2, color=NAVY, lw=0.8, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=5, lw=lw,
                                 color=color, ls=ls, shrinkA=0, shrinkB=0))


def _thumb(ax, x, y, w, h, tags=False):
    ax.add_patch(Rectangle((x, y), w, h, fc="#262b33", ec="#101216", lw=0.5))
    for cx in (0.30, 0.70):
        ax.add_patch(Ellipse((x + cx * w, y + 0.50 * h), 0.30 * w, 0.64 * h, fc="#a9afb8", ec="none", alpha=0.85))
    ax.add_patch(Rectangle((x + 0.47 * w, y + 0.08 * h), 0.06 * w, 0.84 * h, fc="#d6dae0", ec="none", alpha=0.55))
    if tags:
        # (relative box x0, y0, x1, y1) of each region
        boxes = {"R1": (0.10, 0.18, 0.42, 0.88), "R2": (0.56, 0.52, 0.88, 0.90), "R3": (0.60, 0.10, 0.90, 0.50)}
        for name, (a, b, c, d) in boxes.items():
            col = TAG[name]
            ax.add_patch(Rectangle((x + a * w, y + b * h), (c - a) * w, (d - b) * h, fc="none", ec=col, lw=0.8, ls="--"))
            ax.add_patch(Rectangle((x + a * w, y + d * h - 2.6), 4.2, 2.6, fc=col, ec="none"))
            ax.text(x + a * w + 2.1, y + d * h - 1.3, name, color="white", ha="center", va="center",
                    fontsize=4.2, fontweight="bold")


def draw_pipeline(ax_factory, out_path):
    import matplotlib.pyplot as plt

    heights = [15, 22, 15, 21, 17, 30]
    gap = 3.6
    margin_top, margin_bot = 3.2, 0.8
    total = sum(heights) + gap * (len(heights) - 1) + margin_top + margin_bot
    fig, ax = plt.subplots(figsize=(3.45, 3.45 * total / 100))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, total)
    ax.axis("off")

    tops, y = [], total - margin_top
    for h in heights:
        tops.append(y)
        y -= h + gap

    titles = ["Medical image + question", "Region localizer", "Set-of-Mark overlay",
              "Vision-language inference", "Structural claim verifier", "Confidence gate and decision"]

    def panel(i):
        T, h = tops[i], heights[i]
        ax.add_patch(FancyBboxPatch((2, T - h), 96, h, boxstyle="round,pad=0,rounding_size=2",
                                    fc="white", ec=NAVY, lw=0.7))
        tw = 12 + 1.55 * len(titles[i])
        ax.add_patch(FancyBboxPatch((7, T - 2.4), tw, 4.8, boxstyle="round,pad=0,rounding_size=2.4", fc=NAVY, ec="none"))
        ax.text(7 + 5.2, T, titles[i], color="white", ha="left", va="center", fontsize=6.2, fontweight="bold")
        ax.add_patch(Circle((7, T), 3.0, fc=NAVY, ec="white", lw=1.0))
        ax.text(7, T, str(i + 1), color="white", ha="center", va="center", fontsize=6.4, fontweight="bold")
        if i > 0:
            _arrow(ax, 78, tops[i - 1] - heights[i - 1], 78, T + 0.1)
        return T

    # 1 input
    T = panel(0)
    _thumb(ax, 8, T - 13.5, 13, 9.5)
    _rbox(ax, 26, T - 12.5, 68, 7.5, "Question: Is there a pulmonary nodule\nin the left lung?", fs=5.7)

    # 2 localizer
    T = panel(1)
    _rbox(ax, 8, T - 11.0, 40, 6.0, "Faster R-CNN: candidate boxes", fc="white", ls="--", fs=5.6)
    _rbox(ax, 8, T - 18.0, 40, 6.0, "MedSAM: masks from boxes", fc="white", ls="--", fs=5.6)
    _rbox(ax, 62, T - 15.5, 32, 9.5, "Up to $k$\ncandidate regions", fc=GRN_F, ec=GRN_E, fs=5.7)
    _arrow(ax, 48, T - 8.0, 62, T - 10.0)
    _arrow(ax, 48, T - 15.0, 62, T - 12.0)
    ax.text(50, T - 20.4, "Dashed: built, not evaluated. Experiments used oracle SLAKE boxes (Track B) or none (Track C).",
            ha="center", va="center", fontsize=4.1, style="italic", color=GREY)

    # 3 set-of-mark
    T = panel(2)
    _thumb(ax, 8, T - 13.2, 21, 9.4, tags=True)
    for k, (name, x0) in enumerate([("R1", 38), ("R2", 47), ("R3", 56)]):
        col = TAG[name]
        ax.add_patch(Rectangle((x0, T - 8.2), 7, 4.4, fc=col, ec="none"))
        ax.text(x0 + 3.5, T - 6.0, name, color="white", ha="center", va="center", fontsize=5.4, fontweight="bold")
    ax.text(66, T - 6.0, "...", ha="center", va="center", fontsize=7, fontweight="bold", color=NAVY)
    _rbox(ax, 71, T - 8.2, 9, 4.4, "$R_k$", fc="white", ec=NAVY, ls="--", fs=5.6, r=0.6)
    ax.text(66, T - 11.8, "Numbered marks drawn on a copy of the image", ha="center", va="center", fontsize=5.0, color=GREY)

    # 4 VLM
    T = panel(3)
    _rbox(ax, 7, T - 10.8, 28, 6.2, "Qwen2.5-VL-3B", fs=5.7, bold=True)
    _rbox(ax, 7, T - 18.2, 28, 6.2, "MedGemma-4B", fc=GRN_F, ec=GRN_E, fs=5.7, bold=True)
    _rbox(ax, 41, T - 10.8, 20, 6.2, "$N$ = 5 sampled\nJSON replies", fc="white", fs=5.0)
    _rbox(ax, 41, T - 18.2, 20, 6.2, "$N$ = 5 sampled\nJSON replies", fc="white", fs=5.0)
    _arrow(ax, 35, T - 7.7, 41, T - 7.7)
    _arrow(ax, 35, T - 15.1, 41, T - 15.1)
    ax.text(81.5, T - 5.2, "Illustrative JSON", ha="center", va="center", fontsize=4.8, style="italic", color=GREY)
    _rbox(ax, 65, T - 18.4, 31.5, 11.6, "", fc="#f7f9fd", ec=BLUE_E)
    ax.text(66.4, T - 7.6, '{"answer": "yes",\n "findings": [\n  {"region_id": "R1",\n   "status": "present",\n   "confidence": 0.82}]}',
            ha="left", va="top", fontsize=3.7, family="DejaVu Sans Mono", linespacing=1.25)

    # 5 verifier
    T = panel(4)
    _rbox(ax, 7, T - 9.4, 25, 4.6, "Generated findings", fs=5.2)
    _rbox(ax, 7, T - 14.4, 25, 4.6, "Region ids $\\{R_1,\\ldots,R_k\\}$", fs=5.2)
    _rbox(ax, 37, T - 14.4, 38, 9.6, "Check that each cited id exists;\ndrop findings that cite none\nor an invalid id", fc=BLUE_F, fs=4.9, bold=False)
    _rbox(ax, 79, T - 13.4, 17.5, 8.0, "Findings\nwith valid\nregion ids", fc=GRN_F, ec=GRN_E, fs=4.6)
    _arrow(ax, 32, T - 7.1, 37, T - 8.3)
    _arrow(ax, 32, T - 12.1, 37, T - 11.0)
    _arrow(ax, 75, T - 9.6, 79, T - 9.4)
    ax.text(50, T - 16.0, "Structural check only: no ground truth at runtime; semantic support is not tested.",
            ha="center", va="center", fontsize=4.2, style="italic", color=GREY)

    # 6 gate + decision
    T = panel(5)
    labels = ["Answer agreement $a$", "Entropy term $1-H_n$", "Localizer score $\\rho$"]
    for k, lab in enumerate(labels):
        yy = T - 9.2 - 5.2 * k
        _rbox(ax, 8, yy, 28, 4.4, lab, fs=5.2)
        _arrow(ax, 36, yy + 2.2, 42, T - 11.5 - 1.8 * (k - 1) * 0.0 - 0.0 + (1 - k) * 1.2)
    _rbox(ax, 42, T - 19.6, 54, 14.6, "", fc=GRN_F, ec=GRN_E)
    ax.text(69, T - 8.0, "Confidence gate", ha="center", va="center", fontsize=5.8, fontweight="bold")
    ax.text(69, T - 12.4, r"$c = w_1\,a + w_2\,(1-H_n) + w_3\,\rho$", ha="center", va="center", fontsize=6.6)
    ax.text(69, T - 16.6, "weights and $\\tau$ fitted on validation, then frozen", ha="center", va="center",
            fontsize=4.6, style="italic", color=GREY)
    # branch
    ax.plot([69, 69], [T - 19.6, T - 21.3], color=NAVY, lw=0.8)
    ax.plot([30, 74], [T - 21.3, T - 21.3], color=NAVY, lw=0.8)
    _arrow(ax, 30, T - 21.3, 30, T - 22.9)
    _arrow(ax, 74, T - 21.3, 74, T - 22.9)
    ax.text(20, T - 21.3, "$c \\geq \\tau$", ha="center", va="center", fontsize=5.8, color=BLUE_E, fontweight="bold")
    ax.text(86, T - 21.3, "$c < \\tau$", ha="center", va="center", fontsize=5.8, color=RED_E, fontweight="bold")
    _rbox(ax, 8, T - 28.2, 44, 5.3, "Answer + valid region ids", fc=BLUE_F, ec=BLUE_E, fs=5.5, bold=True)
    _rbox(ax, 56, T - 28.2, 38, 5.3, "Abstain (uncertain)", fc=RED_F, ec=RED_E, fs=5.5, bold=True)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
