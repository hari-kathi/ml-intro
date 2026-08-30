"""Regenerates every figure in docs/images/ used by the docs.

Run it from the docs/ folder with the project's Python:

    /Users/hari/repos/learning/ml-intro/.venv/bin/python make_figures.py

Every figure is drawn with matplotlib and saved as a PNG. The colors come
from the Okabe-Ito palette, which is designed to be readable for people
with color-vision deficiencies.
"""

import os

import matplotlib

# "Agg" is a matplotlib backend that draws straight to image files,
# with no window popping up -- right for a script like this.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

# Okabe-Ito colorblind-friendly palette.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
PINK = "#CC79A7"
VERMILLION = "#D55E00"
SKY = "#56B4E9"
GRAY = "#999999"

# Save images next to this script, in an images/ subfolder.
HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(HERE, "images")
os.makedirs(IMAGES, exist_ok=True)


def save(fig, name):
    """Save a figure at 150 dpi with tight margins, then close it."""
    path = os.path.join(IMAGES, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# Figure 1: nested circles -- AI contains ML contains deep learning
# ---------------------------------------------------------------------------

def fig_nested_circles():
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")   # circles stay circular, not squashed
    ax.axis("off")           # no axes -- this is a diagram, not a plot

    # Three nested circles, biggest first so the smaller ones draw on top.
    ax.add_patch(Circle((5, 4.6), 4.4, facecolor=SKY, alpha=0.25,
                        edgecolor=BLUE, linewidth=2))
    ax.add_patch(Circle((5, 3.6), 3.1, facecolor=GREEN, alpha=0.25,
                        edgecolor=GREEN, linewidth=2))
    ax.add_patch(Circle((5, 2.6), 1.8, facecolor=ORANGE, alpha=0.35,
                        edgecolor=VERMILLION, linewidth=2))

    # Ring titles.
    ax.text(5, 8.4, "Artificial intelligence", ha="center", fontsize=14,
            fontweight="bold", color=BLUE)
    ax.text(5, 6.1, "Machine learning", ha="center", fontsize=13,
            fontweight="bold", color=GREEN)
    ax.text(5, 3.55, "Deep\nlearning", ha="center", fontsize=12,
            fontweight="bold", color=VERMILLION)

    # An example inside each ring, in plain text.
    ax.text(5, 7.6,
            "any program that acts \"smart\"\ne.g. a hand-written chess bot,"
            " a rule-based spam filter",
            ha="center", fontsize=9, color="#333333")
    ax.text(5, 5.2,
            "programs that learn rules from examples\ne.g. random forest"
            " predicting heart disease  (modules 04–08)",
            ha="center", fontsize=9, color="#333333")
    ax.text(5, 2.15,
            "many-layered\nneural networks\ne.g. a CNN reading\nblood smears"
            "  (09–11)",
            ha="center", fontsize=8.5, color="#333333")

    save(fig, "ai-ml-dl-circles.png")


# ---------------------------------------------------------------------------
# Figure 2: supervised vs unsupervised, as two tiny scatter plots
# ---------------------------------------------------------------------------

def fig_supervised_vs_unsupervised():
    rng = np.random.default_rng(42)   # a seeded random generator: same picture every run

    # Make three little clouds of points, like three species of penguin.
    cloud_a = rng.normal(loc=[2.0, 2.0], scale=0.45, size=(25, 2))
    cloud_b = rng.normal(loc=[4.3, 4.2], scale=0.45, size=(25, 2))
    cloud_c = rng.normal(loc=[5.8, 1.8], scale=0.45, size=(25, 2))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))

    # Left panel: SUPERVISED. Every point has a known label (a color).
    ax = axes[0]
    ax.scatter(cloud_a[:, 0], cloud_a[:, 1], c=BLUE, s=28, label="Adelie")
    ax.scatter(cloud_b[:, 0], cloud_b[:, 1], c=ORANGE, s=28, marker="^",
               label="Gentoo")
    ax.scatter(cloud_c[:, 0], cloud_c[:, 1], c=GREEN, s=28, marker="s",
               label="Chinstrap")
    ax.legend(loc="upper left", fontsize=8, title="labels known")
    ax.set_title("Supervised: learn from labeled examples\n"
                 "(a field guide with named photos)", fontsize=10)

    # Right panel: UNSUPERVISED. Same points, but nobody told us the species.
    ax = axes[1]
    all_points = np.vstack([cloud_a, cloud_b, cloud_c])  # stack the clouds into one array
    ax.scatter(all_points[:, 0], all_points[:, 1], c=GRAY, s=28)
    # Dashed ellipses mark the groups the algorithm would discover on its own.
    for center in [(2.0, 2.0), (4.3, 4.2), (5.8, 1.8)]:
        ax.add_patch(Ellipse(center, 2.3, 2.3, facecolor="none",
                             edgecolor=PINK, linestyle="--", linewidth=1.8))
    ax.set_title("Unsupervised: find structure without labels\n"
                 "(sorting unlabeled specimens by similarity)", fontsize=10)

    # Shared cosmetic cleanup for both panels.
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("measurement 1", fontsize=9)
        ax.set_ylabel("measurement 2", fontsize=9)
        ax.set_xlim(0.5, 7.3)
        ax.set_ylim(0.3, 5.7)

    fig.suptitle("Two ways to learn from data", fontsize=13, y=1.02)
    save(fig, "supervised-vs-unsupervised.png")


# ---------------------------------------------------------------------------
# Figure 3: the course map -- modules as boxes in a flow diagram
# ---------------------------------------------------------------------------

def fig_course_map():
    fig, ax = plt.subplots(figsize=(9.5, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=9):
        """Draw one rounded module box with centered text."""
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.12",
                                    facecolor=color, alpha=0.30,
                                    edgecolor=color, linewidth=1.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color="#222222")

    def arrow(x1, y1, x2, y2):
        """Draw one flow arrow between stages."""
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=18,
                                     color="#555555", linewidth=1.6))

    # Stage labels down the left edge.
    ax.text(0.15, 10.9, "Foundations", fontsize=11, fontweight="bold",
            color=BLUE, rotation=90, va="center")
    ax.text(0.15, 7.4, "Classic ML\n(scikit-learn)", fontsize=10,
            fontweight="bold", color=GREEN, rotation=90, va="center",
            ha="center")
    ax.text(0.15, 3.9, "Neural nets\n(PyTorch)", fontsize=10,
            fontweight="bold", color=VERMILLION, rotation=90, va="center",
            ha="center")

    # Row 1: foundations.
    box(1.0, 10.3, 2.5, 1.1, "01\nNumPy\nfoundations", BLUE)
    box(4.0, 10.3, 2.5, 1.1, "02\nPandas\nessentials", BLUE)
    box(7.0, 10.3, 2.5, 1.1, "03\nData\nwrangling", BLUE)
    arrow(3.5, 10.85, 4.0, 10.85)
    arrow(6.5, 10.85, 7.0, 10.85)

    # Row 2: classic machine learning.
    box(1.0, 8.0, 2.5, 1.1, "04\nFirst ML\nmodel", GREEN)
    box(4.0, 8.0, 2.5, 1.1, "05\nRegression", GREEN)
    box(7.0, 8.0, 2.5, 1.1, "06\nClassification\nin depth", GREEN)
    box(2.5, 5.9, 2.5, 1.1, "07\nHonest-scientist\nworkflow", GREEN)
    box(5.5, 5.9, 2.5, 1.1, "08\nUnsupervised\nlearning", GREEN)
    arrow(8.25, 10.3, 2.25, 9.1)     # down from foundations
    arrow(3.5, 8.55, 4.0, 8.55)
    arrow(6.5, 8.55, 7.0, 8.55)
    arrow(8.25, 8.0, 3.75, 7.0)
    arrow(5.0, 6.45, 5.5, 6.45)

    # Row 3: neural networks.
    box(1.0, 3.6, 2.5, 1.1, "09\nNeural nets\nfrom scratch", VERMILLION)
    box(4.0, 3.6, 2.5, 1.1, "10\nPyTorch\nfundamentals", VERMILLION)
    box(7.0, 3.6, 2.5, 1.1, "11\nDeep learning\non images", VERMILLION)
    arrow(3.75, 5.9, 2.25, 4.7)      # down from classic ML
    arrow(3.5, 4.15, 4.0, 4.15)
    arrow(6.5, 4.15, 7.0, 4.15)

    # Capstone.
    box(3.0, 1.0, 4.5, 1.2, "Capstone\nmalaria detector\n(everything combined)",
        PINK, fontsize=10)
    arrow(8.25, 3.6, 5.5, 2.2)

    ax.set_title("The course map: modules 01–11 plus the capstone",
                 fontsize=13)
    save(fig, "course-map.png")


# ---------------------------------------------------------------------------
# Figure 4: the train/test split
# ---------------------------------------------------------------------------

def fig_train_test_split():
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    n = 30            # 30 little squares, each one "specimen" in the dataset
    split = 24        # first 24 go to training, last 6 to testing (80/20)

    # Top row: the full dataset before splitting, shuffled colors to hint
    # that the rows get shuffled before the split.
    for i in range(n):
        x = 0.6 + i * 0.3
        ax.add_patch(FancyBboxPatch((x, 3.2), 0.22, 0.5,
                                    boxstyle="round,pad=0.02",
                                    facecolor=GRAY, alpha=0.6,
                                    edgecolor="white"))
    ax.text(5.0, 4.1, "all 30 specimens (shuffled)", ha="center", fontsize=10)

    # Bottom row: same squares after the split -- train left, test right.
    for i in range(n):
        x = 0.6 + i * 0.3
        if i < split:
            color = BLUE      # training squares
        else:
            color = ORANGE    # test squares
        ax.add_patch(FancyBboxPatch((x, 1.2), 0.22, 0.5,
                                    boxstyle="round,pad=0.02",
                                    facecolor=color, alpha=0.75,
                                    edgecolor="white"))

    # Arrows from top row to the two groups.
    ax.add_patch(FancyArrowPatch((3.8, 3.1), (3.8, 2.0), arrowstyle="-|>",
                                 mutation_scale=16, color="#555555",
                                 linewidth=1.5))
    ax.add_patch(FancyArrowPatch((8.9, 3.1), (8.9, 2.0), arrowstyle="-|>",
                                 mutation_scale=16, color="#555555",
                                 linewidth=1.5))

    # Labels under each group.
    ax.text(4.1, 0.55, "training set (80%)\nthe model studies these",
            ha="center", fontsize=10, color=BLUE)
    ax.text(8.6, 0.55, "test set (20%)\nkept sealed until the end",
            ha="center", fontsize=10, color=VERMILLION)

    ax.set_title("train_test_split: hold some data back to grade the model"
                 " honestly", fontsize=12)
    save(fig, "train-test-split.png")


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fig_nested_circles()
    fig_supervised_vs_unsupervised()
    fig_course_map()
    fig_train_test_split()
    print("done")
