# Regenerates every figure in images/ used by README.md.
# Run with:  ../../.venv/bin/python make_figures.py
import matplotlib
matplotlib.use("Agg")   # draw to files, not to a window

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# Colorblind-friendly palette (same hues seaborn's "colorblind" uses)
BLUE = "#0173B2"      # training data
ORANGE = "#DE8F05"    # test / validation data
GREEN = "#029E73"     # "sterile" / correct
RED = "#D55E00"       # "contaminated" / wrong
GRAY = "#949494"
LIGHTBLUE = "#B7D4EA"

SAVE = dict(dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Figure 1: k-fold cross-validation as a rotation schedule
# ---------------------------------------------------------------------------
def fig_kfold():
    n_folds = 5
    fig, ax = plt.subplots(figsize=(9, 4.6))

    block_w, block_h, gap = 1.0, 0.62, 0.28
    for round_i in range(n_folds):            # one row per round
        y = (n_folds - 1 - round_i) * (block_h + gap)   # row 0 drawn at the top
        for fold_j in range(n_folds):         # one block per fold of the data
            x = fold_j * (block_w + 0.08)
            if fold_j == round_i:             # this round's held-out test fold
                color, edge = ORANGE, "#9C6203"
            else:
                color, edge = LIGHTBLUE, BLUE
            ax.add_patch(Rectangle((x, y), block_w, block_h,
                                   facecolor=color, edgecolor=edge, linewidth=1.6))
        # label each row and its resulting score
        ax.text(-0.25, y + block_h / 2, "Round " + str(round_i + 1),
                ha="right", va="center", fontsize=11)
        ax.text(n_folds * (block_w + 0.08) + 0.25, y + block_h / 2,
                "score " + str(round_i + 1),
                ha="left", va="center", fontsize=11, style="italic", color=GRAY)

    # column header: the data is chopped into 5 folds
    top = n_folds * (block_h + gap) - gap
    for fold_j in range(n_folds):
        x = fold_j * (block_w + 0.08) + block_w / 2
        ax.text(x, top + 0.30, "fold " + str(fold_j + 1),
                ha="center", va="bottom", fontsize=10, color="#333333")

    # legend blocks
    ax.add_patch(Rectangle((0.0, -1.15), 0.5, 0.4, facecolor=LIGHTBLUE,
                           edgecolor=BLUE, linewidth=1.6))
    ax.text(0.65, -0.95, "train on these", va="center", fontsize=11)
    ax.add_patch(Rectangle((3.2, -1.15), 0.5, 0.4, facecolor=ORANGE,
                           edgecolor="#9C6203", linewidth=1.6))
    ax.text(3.85, -0.95, "test on this (held out)", va="center", fontsize=11)

    ax.text(n_folds * (block_w + 0.08) / 2, -1.75,
            "5 rounds → 5 scores → report the mean (and the spread).\n"
            "Every sample gets exactly one turn in the test set — like rotating "
            "which mouse cage is the control group.",
            ha="center", va="top", fontsize=11)

    ax.set_xlim(-1.6, n_folds * (block_w + 0.08) + 1.4)
    ax.set_ylim(-2.9, top + 0.75)
    ax.set_title("5-fold cross-validation: the test set rotates", fontsize=14, pad=12)
    ax.axis("off")
    fig.savefig("images/kfold-rotation.png", **SAVE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: data leakage (scale, then split) vs. the Pipeline (split, then scale)
# ---------------------------------------------------------------------------
def box(ax, x, y, w, h, text, fc, ec, fontsize=10.5, textcolor="black"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                           linewidth=1.8, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, zorder=3, color=textcolor)


def arrow(ax, x1, y1, x2, y2, color="#333333", style="-", lw=1.8):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=16, color=color,
                                 linestyle=style, linewidth=lw, zorder=4))


def fig_leakage():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))

    # ---- left panel: the contaminated workflow -------------------------------
    ax = axes[0]
    ax.set_title("CONTAMINATED: scale first, split later", fontsize=13,
                 color=RED, pad=12)
    box(ax, 2.5, 8.6, 5, 1.1, "ALL 569 samples", LIGHTBLUE, BLUE)
    arrow(ax, 5, 8.6, 5, 7.8)
    box(ax, 1.6, 6.6, 6.8, 1.2,
        "StandardScaler learns mean & spread\nfrom ALL samples — test included",
        "#F6D5C3", RED)
    arrow(ax, 3.6, 6.6, 2.7, 5.4)
    arrow(ax, 6.4, 6.6, 7.3, 5.4)
    box(ax, 0.8, 4.2, 3.8, 1.2, "train split", LIGHTBLUE, BLUE)
    box(ax, 5.4, 4.2, 3.8, 1.2, "test split", "#FBE3C0", "#9C6203")
    # the leak: information flowing backwards from the test data into preprocessing
    ax.add_patch(FancyArrowPatch((7.3, 5.4), (6.6, 6.55),
                                 arrowstyle="-|>", mutation_scale=18,
                                 color=RED, linestyle="--", linewidth=2.4,
                                 connectionstyle="arc3,rad=0.35", zorder=5))
    ax.text(9.0, 6.1, "test-set information\nleaks into the\npreprocessing",
            fontsize=10, color=RED, ha="center", va="center")
    arrow(ax, 2.7, 4.2, 4.2, 3.1)
    box(ax, 3.1, 1.9, 3.8, 1.2, "model", LIGHTBLUE, BLUE)
    arrow(ax, 7.3, 4.2, 6.2, 3.1)
    ax.text(5, 0.9, "✗  the \"unseen\" test data already shaped the experiment",
            ha="center", fontsize=11, color=RED)
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0.3, 10.2)
    ax.axis("off")

    # ---- right panel: the sterile workflow -----------------------------------
    ax = axes[1]
    ax.set_title("STERILE: split first, scale inside a Pipeline", fontsize=13,
                 color=GREEN, pad=12)
    box(ax, 2.5, 8.6, 5, 1.1, "ALL 569 samples", LIGHTBLUE, BLUE)
    arrow(ax, 4.2, 8.6, 2.7, 7.6)
    arrow(ax, 5.8, 8.6, 7.9, 7.6)
    box(ax, 0.8, 6.4, 3.8, 1.2, "train split", LIGHTBLUE, BLUE)
    box(ax, 6.0, 6.4, 3.8, 1.2, "test split", "#FBE3C0", "#9C6203")
    # the wall keeping test data quarantined until the very end
    ax.add_patch(Rectangle((5.15, 2.7), 0.35, 4.9, facecolor=GRAY,
                           edgecolor="#555555", hatch="////", zorder=2))
    ax.text(5.32, 8.0, "wall", ha="center", fontsize=10, color="#555555")
    arrow(ax, 2.7, 6.4, 2.7, 5.6)
    box(ax, 0.6, 3.4, 4.2, 2.2,
        "Pipeline\nscaler fits on TRAIN only\n↓\nmodel fits on TRAIN only",
        "#CDEBE1", GREEN)
    arrow(ax, 2.7, 3.4, 2.7, 2.5)
    arrow(ax, 7.9, 6.4, 7.9, 2.5)
    ax.text(8.35, 4.4, "quarantined\nuntil the end", fontsize=10,
            color="#9C6203", ha="center", va="center")
    box(ax, 1.4, 1.3, 7.4, 1.2, "evaluate ONCE on the untouched test split",
        "#CDEBE1", GREEN)
    ax.text(5, 0.45, "✓  the test data never influences any fitted step",
            ha="center", fontsize=11, color=GREEN)
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0.0, 10.2)
    ax.axis("off")

    fig.suptitle("Data leakage is contamination — sterile technique for ML",
                 fontsize=15, y=1.0)
    fig.savefig("images/leakage-vs-pipeline.png", **SAVE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: an idealized validation curve with the three zones marked
# ---------------------------------------------------------------------------
def fig_validation_curve():
    fig, ax = plt.subplots(figsize=(9, 5.2))

    x = np.linspace(0, 10, 300)               # "model flexibility" axis
    # smooth made-up curves with the classic shapes:
    train = 1.0 - 0.45 * np.exp(-0.55 * x)          # keeps climbing toward 100%
    cv = 0.92 - 0.32 * np.exp(-0.9 * x) - 0.0028 * (x - 4.0) ** 2
    cv[x < 0] = np.nan

    sweet = x[np.argmax(cv)]                  # flexibility where CV score peaks

    # shade the three zones
    ax.axvspan(0, 2.1, color="#E8E8E8", alpha=0.7)
    ax.axvspan(2.1, 6.2, color="#CDEBE1", alpha=0.7)
    ax.axvspan(6.2, 10, color="#F6D5C3", alpha=0.7)
    ax.text(1.05, 0.505, "UNDERFIT\ntoo simple:\nbad on both",
            ha="center", fontsize=10.5, color="#444444")
    ax.text(4.15, 0.505, "SWEET SPOT\nbest score on\nunseen data",
            ha="center", fontsize=10.5, color=GREEN)
    ax.text(8.1, 0.505, "OVERFIT\nmemorizing: train up,\nvalidation down",
            ha="center", fontsize=10.5, color=RED)

    ax.plot(x, train, color=BLUE, linewidth=2.5, label="training score")
    ax.plot(x, cv, color=ORANGE, linewidth=2.5, label="cross-validation score")

    # mark the peak of the CV curve
    ax.axvline(sweet, color=GREEN, linestyle=":", linewidth=2)
    ax.plot([sweet], [np.nanmax(cv)], marker="*", markersize=18,
            color=GREEN, zorder=5)
    ax.annotate("pick the settings here", xy=(sweet, np.nanmax(cv)),
                xytext=(sweet + 1.1, 0.985), fontsize=11, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN))

    # show the growing train-vs-CV gap = memorization
    ax.annotate("", xy=(9.0, float(cv[np.argmin(np.abs(x - 9.0))])),
                xytext=(9.0, float(train[np.argmin(np.abs(x - 9.0))])),
                arrowprops=dict(arrowstyle="<->", color=RED, linewidth=1.8))
    ax.text(9.15, 0.845, "gap =\nmemorization", fontsize=10, color=RED, va="center")

    ax.set_xlim(0, 10)
    ax.set_ylim(0.47, 1.03)
    ax.set_xticks([])
    ax.set_xlabel("model flexibility  →   (for KNN: smaller k = more flexible)",
                  fontsize=11)
    ax.set_ylabel("accuracy", fontsize=11)
    ax.set_title("The validation curve: underfitting, the sweet spot, overfitting",
                 fontsize=14)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.95)
    fig.savefig("images/validation-curve-zones.png", **SAVE)
    plt.close(fig)


if __name__ == "__main__":
    fig_kfold()
    fig_leakage()
    fig_validation_curve()
    print("wrote images/kfold-rotation.png, images/leakage-vs-pipeline.png, "
          "images/validation-curve-zones.png")
