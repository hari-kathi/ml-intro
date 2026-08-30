# Regenerates every figure in images/ used by README.md.
# Run with:  ../../.venv/bin/python make_figures.py
import matplotlib
matplotlib.use("Agg")   # draw to files, not to a window

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Colorblind-friendly palette (same hues seaborn's "colorblind" uses),
# matching the figures in modules 01-11.
BLUE = "#0173B2"       # data / training
ORANGE = "#DE8F05"     # held-out data / thresholds
GREEN = "#029E73"      # healthy / correct
RED = "#D55E00"        # infected / errors
PURPLE = "#CC78BC"     # the CNN
GRAY = "#949494"
LIGHTBLUE = "#B7D4EA"

SAVE = dict(dpi=150, bbox_inches="tight")

rng = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Figure 1: the project pipeline, with the module where each skill was learned
# ---------------------------------------------------------------------------
def rounded_box(ax, x, y, w, h, lines, fc, ec, fontsize=10.5):
    # One pipeline stage: a rounded rectangle with 1-3 lines of centered text.
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.06",
                                facecolor=fc, edgecolor=ec, linewidth=1.8,
                                zorder=2))
    ax.text(x + w / 2, y + h / 2, lines, ha="center", va="center",
            fontsize=fontsize, zorder=3, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 arrowstyle="-|>", mutation_scale=16,
                                 linewidth=1.8, color="#333333", zorder=1))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11.5, 4.8))

    w, h = 2.5, 1.5           # box size
    y_main = 1.6              # vertical center line of the pipeline
    xs = [0.0, 3.4, 6.8, 10.2, 13.6]   # left edge of each of the 5 stages

    rounded_box(ax, xs[0], y_main, w, h,
                "27,558 cell photos\n(NIH malaria\ndataset)", LIGHTBLUE, BLUE)
    rounded_box(ax, xs[1], y_main, w, h,
                "Look, then split\ntrain / val / test\nbefore modeling", "#FFE3B3", "#9C6203")
    rounded_box(ax, xs[2], y_main, w, h,
                "Baseline:\nlogistic regression\non raw pixels", "#CDEBE2", GREEN)
    rounded_box(ax, xs[3], y_main, w, h,
                "Small CNN\non 64×64\ncolor images", "#EDD9EA", "#8E4585")
    rounded_box(ax, xs[4], y_main, w, h,
                "Evaluate like\na diagnostic:\nsensitivity first", "#F6CCB8", RED)

    # arrows between consecutive stages
    for i in range(4):
        arrow(ax, xs[i] + w + 0.07, y_main + h / 2, xs[i + 1] - 0.07, y_main + h / 2)

    # module callbacks under each stage: where you learned that skill
    modules = ["Modules 02–04", "Module 07", "Modules 06–07",
               "Modules 10–11", "Module 06"]
    for i in range(5):
        ax.text(xs[i] + w / 2, y_main - 0.55, modules[i],
                ha="center", va="center", fontsize=10, style="italic", color=GRAY)

    # the two review stages hang off the end
    ax.text(xs[4] + w / 2, y_main + h + 0.75,
            "then: review your own mistakes,\nwrite it up like a paper",
            ha="center", va="center", fontsize=10, color="#333333")
    arrow(ax, xs[4] + w / 2, y_main + h + 0.12, xs[4] + w / 2, y_main + h + 0.42)

    ax.set_xlim(-0.4, xs[4] + w + 0.4)
    ax.set_ylim(0.4, 4.6)
    ax.set_title("The capstone pipeline: every stage is a skill you already have",
                 fontsize=14, pad=10)
    ax.axis("off")
    fig.savefig("images/pipeline.png", **SAVE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: real cells from the dataset -- uninfected vs. parasitized
# ---------------------------------------------------------------------------
def fig_sample_cells():
    # Load a few real images from the (already cached) HuggingFace dataset.
    from datasets import load_dataset
    ds = load_dataset("dpdl-benchmark/malaria", split="train")

    # Hand-picked indices, verified by eye:
    # label 1 = uninfected (smooth cytoplasm), label 0 = parasitized
    # (purple-stained Plasmodium visible). See README for how we checked.
    uninfected_idx = [12434, 21637, 10200, 3474]      # all label 1
    parasitized_idx = [2467, 19194, 18002, 11916]     # all label 0

    fig, axes = plt.subplots(2, 4, figsize=(9.5, 5.4))
    for col in range(4):
        axes[0, col].imshow(ds[uninfected_idx[col]]["image"])
        axes[1, col].imshow(ds[parasitized_idx[col]]["image"])
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_ylabel("uninfected", fontsize=13, color=GREEN)
    axes[1, 0].set_ylabel("parasitized", fontsize=13, color=RED)

    # Point at the stained parasite in one cell (index 18002: dark purple
    # blob near the bottom of the cell).
    axes[1, 2].annotate("stained parasite\n(Plasmodium)",
                        xy=(72, 108), xytext=(30, 165),
                        fontsize=10.5, color=RED, ha="center",
                        arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.8),
                        annotation_clip=False)

    fig.suptitle("Eight real cells from the dataset: the parasite is visibly stained",
                 fontsize=13.5)
    fig.savefig("images/sample-cells.png", **SAVE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: the threshold trade-off -- sensitivity vs. specificity
# ---------------------------------------------------------------------------
def fig_threshold_tradeoff():
    # Simulate realistic classifier scores for the two classes: infected cells
    # mostly score high, healthy cells mostly score low, with some overlap
    # (a perfect classifier would make this figure boring -- no trade-off).
    scores_infected = rng.beta(3.2, 1.4, size=20000)   # bunched near 1
    scores_healthy = rng.beta(1.4, 3.2, size=20000)    # bunched near 0

    thresholds = np.linspace(0.001, 0.999, 400)
    sensitivity = np.zeros(len(thresholds))
    specificity = np.zeros(len(thresholds))
    for i in range(len(thresholds)):
        t = thresholds[i]
        sensitivity[i] = np.mean(scores_infected >= t)   # infected caught
        specificity[i] = np.mean(scores_healthy < t)     # healthy cleared

    fig, ax = plt.subplots(figsize=(8.6, 5))
    ax.plot(thresholds, sensitivity, color=RED, linewidth=2.5,
            label="sensitivity (infections caught)")
    ax.plot(thresholds, specificity, color=GREEN, linewidth=2.5,
            label="specificity (healthy cleared)")

    # Operating point 1: the default threshold 0.5
    s_def = np.interp(0.5, thresholds, sensitivity)
    ax.axvline(0.5, color=GRAY, linestyle=":", linewidth=1.5)
    ax.plot([0.5], [s_def], "o", color=GRAY, markersize=9)
    ax.annotate("default threshold 0.5:\ntreats both errors as equal",
                xy=(0.5, s_def), xytext=(0.58, 0.62), fontsize=10.5,
                arrowprops=dict(arrowstyle="->", color=GRAY))

    # Operating point 2: the clinic threshold -- lowest t with sens >= 0.98
    idx = np.where(sensitivity >= 0.98)[0][-1]
    t_clinic = thresholds[idx]
    ax.axvline(t_clinic, color=ORANGE, linestyle="--", linewidth=2)
    ax.plot([t_clinic], [sensitivity[idx]], "o", color=ORANGE, markersize=9)
    ax.plot([t_clinic], [specificity[idx]], "o", color=ORANGE, markersize=9)
    ax.annotate("clinic setting: flag anything\nremotely suspicious\n(98% of infections caught...)",
                xy=(t_clinic, sensitivity[idx]), xytext=(0.30, 0.88),
                fontsize=10.5, color="#9C6203",
                arrowprops=dict(arrowstyle="->", color=ORANGE))
    ax.annotate("...paid for by clearing\nfewer healthy cells\n(more false alarms)",
                xy=(t_clinic, specificity[idx]), xytext=(0.02, 0.30),
                fontsize=10.5, color="#9C6203",
                arrowprops=dict(arrowstyle="->", color=ORANGE))

    ax.set_xlabel('decision threshold ("call it infected if score ≥ t")')
    ax.set_ylabel("fraction correct within each class")
    ax.set_title("Sliding the threshold trades false alarms for missed infections",
                 fontsize=13.5)
    ax.legend(loc="center right", frameon=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.savefig("images/threshold-tradeoff.png", **SAVE)
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline()
    fig_sample_cells()
    fig_threshold_tradeoff()
    print("all figures written to images/")
