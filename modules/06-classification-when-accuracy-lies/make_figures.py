# Regenerates every figure in images/ used by this module's README.
# Run with the course virtualenv:
#   /Users/hari/repos/learning/ml-intro/.venv/bin/python make_figures.py

import matplotlib

matplotlib.use("Agg")  # draw to files, not to a window

import numpy as np
import matplotlib.pyplot as plt

# Seaborn's "colorblind" palette, written out as hex codes so this script
# doesn't need seaborn at all.
BLUE = "#0173b2"
ORANGE = "#de8f05"
GREEN = "#029e73"
RED = "#d55e00"
PURPLE = "#cc78bc"
GREY = "#555555"

SAVE = dict(dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Figure A: the sigmoid — how a raw score becomes a probability
# ---------------------------------------------------------------------------
def sigmoid_anatomy():
    z = np.linspace(-8, 8, 400)          # a range of raw model scores
    p = 1.0 / (1.0 + np.exp(-z))         # the sigmoid function itself

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z, p, color=BLUE, linewidth=3, zorder=3)

    # The decision threshold: a horizontal line at probability 0.5.
    ax.axhline(0.5, color=RED, linestyle="--", linewidth=1.5)
    ax.axvline(0, color=GREY, linestyle=":", linewidth=1)

    ax.annotate(
        "decision threshold (0.5)\nabove the line → predict \"disease\"\nbelow → predict \"healthy\"",
        xy=(-7.6, 0.52), fontsize=10, color=RED, va="bottom",
    )
    ax.annotate(
        "very negative score\n→ probability near 0\n(\"looks healthy\")",
        xy=(-5.5, 0.02), xytext=(-7.5, 0.20), fontsize=10, ha="left",
        arrowprops=dict(arrowstyle="->", color=GREY),
    )
    ax.annotate(
        "score = 0\n→ probability 0.5\n(a coin flip)",
        xy=(0, 0.5), xytext=(1.4, 0.32), fontsize=10,
        arrowprops=dict(arrowstyle="->", color=GREY),
    )
    ax.annotate(
        "very positive score\n→ probability near 1\n(\"looks sick\")",
        xy=(5.5, 0.98), xytext=(2.6, 0.72), fontsize=10,
        arrowprops=dict(arrowstyle="->", color=GREY),
    )

    ax.set_xlabel("model score  (a weighted sum of the patient's measurements)")
    ax.set_ylabel("predicted probability of disease")
    ax.set_title("The sigmoid: an S-curve that turns any score into a probability")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    fig.savefig("images/sigmoid-anatomy.png", **SAVE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure B: anatomy of a confusion matrix, in medical language
# ---------------------------------------------------------------------------
def confusion_matrix_anatomy():
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    # Each cell: (x, y, face color, big label, plain-language caption)
    cells = [
        # top row = patients who are ACTUALLY HEALTHY
        (1.4, 4.0, "#d3ecd9", "TRUE NEGATIVE",
         "healthy, and the test\ncorrectly cleared them\n\n(everyone goes home relieved)"),
        (5.6, 4.0, "#fbe3b6", "FALSE POSITIVE",
         "healthy, but flagged as sick\n\n(a false alarm: anxiety and\nfollow-up tests, but no one dies)"),
        # bottom row = patients who are ACTUALLY SICK
        (1.4, 0.4, "#f8c9c0", "FALSE NEGATIVE",
         "sick, but the test says healthy\n\nTHE PATIENT IS SENT HOME\nWITH HEART DISEASE"),
        (5.6, 0.4, "#d3ecd9", "TRUE POSITIVE",
         "sick, and the test caught it\n\n(treatment can start)"),
    ]
    for x, y, color, label, caption in cells:
        edge = GREY
        lw = 1.5
        if label == "FALSE NEGATIVE":
            edge = RED          # highlight the dangerous cell
            lw = 4
        rect = plt.Rectangle((x, y), 3.6, 3.0, facecolor=color,
                             edgecolor=edge, linewidth=lw)
        ax.add_patch(rect)
        ax.text(x + 1.8, y + 2.55, label, ha="center", va="center",
                fontsize=13, fontweight="bold")
        ax.text(x + 1.8, y + 1.25, caption, ha="center", va="center", fontsize=9.5)

    # Column headers (what the MODEL says)
    ax.text(5.3, 8.25, "What the model predicts", ha="center",
            fontsize=12, fontweight="bold")
    ax.text(3.2, 7.55, "\"healthy\"", ha="center", fontsize=11, style="italic")
    ax.text(7.4, 7.55, "\"disease\"", ha="center", fontsize=11, style="italic")

    # Row headers (the TRUTH)
    ax.text(0.25, 3.9, "The truth", ha="center", va="center", rotation=90,
            fontsize=12, fontweight="bold")
    ax.text(0.95, 5.5, "actually\nhealthy", ha="center", va="center",
            fontsize=10, style="italic")
    ax.text(0.95, 1.9, "actually\nsick", ha="center", va="center",
            fontsize=10, style="italic")

    ax.text(5.3, 7.9,
            "(scikit-learn draws its confusion matrix in this same layout)",
            ha="center", fontsize=9, color=GREY)

    fig.savefig("images/confusion-matrix-anatomy.png", **SAVE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure C: the threshold is a dial — two overlapping score distributions
# ---------------------------------------------------------------------------
def threshold_tradeoff():
    x = np.linspace(-5, 5, 500)

    # Two bell curves: the scores healthy patients tend to get (low) and the
    # scores sick patients tend to get (high). They overlap — that overlap is
    # why perfect tests don't exist.
    def bell(center, width):
        return np.exp(-0.5 * ((x - center) / width) ** 2)

    healthy = bell(-1.5, 1.2)
    sick = bell(1.5, 1.2)
    thr = 0.3  # where the threshold currently sits

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, healthy, color=BLUE, linewidth=2.5)
    ax.plot(x, sick, color=ORANGE, linewidth=2.5)

    # Shade the two kinds of mistake this threshold makes.
    ax.fill_between(x, sick, where=(x < thr), color=RED, alpha=0.45)
    ax.fill_between(x, healthy, where=(x >= thr), color="#e8b64c", alpha=0.55)

    # The threshold itself, drawn like a movable slider.
    ax.axvline(thr, color="black", linewidth=2.5)
    ax.annotate("", xy=(thr - 1.6, 1.06), xytext=(thr - 0.05, 1.06),
                arrowprops=dict(arrowstyle="->", color=GREEN, linewidth=2))
    ax.annotate("", xy=(thr + 1.6, 1.06), xytext=(thr + 0.05, 1.06),
                arrowprops=dict(arrowstyle="->", color=PURPLE, linewidth=2))
    ax.text(thr - 0.15, 1.12, "slide left: catch more sick patients\n(screening test — PCR-style)",
            ha="right", fontsize=9.5, color=GREEN)
    ax.text(thr + 0.15, 1.12, "slide right: fewer false alarms\n(confirmatory test)",
            ha="left", fontsize=9.5, color=PURPLE)
    ax.text(thr, -0.10, "threshold", ha="center", fontsize=11, fontweight="bold")

    # Label the curves and the shaded mistakes.
    ax.text(-3.9, 0.75, "healthy patients'\nscores", color=BLUE,
            fontsize=11, ha="center")
    ax.text(3.9, 0.75, "sick patients'\nscores", color=ORANGE,
            fontsize=11, ha="center")
    ax.annotate("false negatives:\nsick, but scored\nbelow the line",
                xy=(-0.3, 0.08), xytext=(-2.6, 0.32), fontsize=9.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.annotate("false positives:\nhealthy, but scored\nabove the line",
                xy=(1.0, 0.35), xytext=(2.6, 0.48), fontsize=9.5, color="#a06b00",
                arrowprops=dict(arrowstyle="->", color="#a06b00"))

    ax.set_xlabel("the model's score for a patient  (higher = looks more diseased)")
    ax.set_ylabel("how many patients get each score")
    ax.set_title("No threshold escapes the overlap — it only chooses WHICH mistake you make")
    ax.set_ylim(-0.18, 1.30)
    ax.set_yticks([])
    ax.grid(alpha=0.2)
    fig.savefig("images/threshold-tradeoff.png", **SAVE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure D: a tiny decision tree drawn as a plain-language flowchart
# ---------------------------------------------------------------------------
def decision_tree_flowchart():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    box_q = dict(boxstyle="round,pad=0.45", facecolor="#dbe9f6", edgecolor=BLUE, linewidth=2)
    box_ok = dict(boxstyle="round,pad=0.45", facecolor="#d3ecd9", edgecolor=GREEN, linewidth=2)
    box_bad = dict(boxstyle="round,pad=0.45", facecolor="#f8c9c0", edgecolor=RED, linewidth=2)

    def node(xy, text, style, fontsize=10.5):
        ax.text(xy[0], xy[1], text, ha="center", va="center",
                fontsize=fontsize, bbox=style)

    def arrow(parent, child, label, dx):
        ax.annotate("", xy=child, xytext=parent,
                    arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1.8))
        midx = (parent[0] + child[0]) / 2 + dx
        midy = (parent[1] + child[1]) / 2 + 0.15
        ax.text(midx, midy, label, fontsize=10, fontweight="bold", color=GREY,
                ha="center")

    # Root question (this mirrors the top of the real tree the notebook grows)
    root = (5, 7)
    n_left = (2.5, 4.4)
    n_right = (7.5, 4.4)
    leaves = [(1.1, 1.5), (3.9, 1.5), (6.1, 1.5), (8.9, 1.5)]

    node(root, "Is more than 1 major vessel\nblocked on the angiogram?", box_q, 11.5)
    node(n_left, "Did the heart pass 135 bpm\nin the exercise stress test?", box_q)
    node(n_right, "Did the exercise ECG dip\nmore than 0.9 mm (ST depression)?", box_q)

    node(leaves[0], "heart couldn't\nspeed up:\nlook closer", box_bad)
    node(leaves[1], "probably\nhealthy", box_ok)
    node(leaves[2], "borderline:\nmore tests", box_bad)
    node(leaves[3], "probably\nheart disease", box_bad)

    arrow((4.3, 6.5), (2.9, 5.0), "no", -0.5)
    arrow((5.7, 6.5), (7.1, 5.0), "yes", 0.5)
    arrow((2.0, 3.8), (1.3, 2.2), "no", -0.4)
    arrow((3.0, 3.8), (3.7, 2.2), "yes", 0.4)
    arrow((7.0, 3.8), (6.3, 2.2), "no", -0.4)
    arrow((8.0, 3.8), (8.7, 2.2), "yes", 0.4)

    ax.set_title("A decision tree is a flowchart of questions — and the model\n"
                 "writes the flowchart itself, from the data", fontsize=12)
    fig.savefig("images/decision-tree-flowchart.png", **SAVE)
    plt.close(fig)


if __name__ == "__main__":
    sigmoid_anatomy()
    confusion_matrix_anatomy()
    threshold_tradeoff()
    decision_tree_flowchart()
    print("All figures written to images/")
