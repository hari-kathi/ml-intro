"""Regenerates every figure in images/ used by the Module 04 README.

Run with the course venv:
    /Users/hari/repos/learning/ml-intro/.venv/bin/python make_figures.py
"""

import matplotlib
matplotlib.use("Agg")   # draw to files, not to a window

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="colorblind")

# Same fixed species colors as Modules 01/02.
palette = sns.color_palette("colorblind")
SPECIES_COLORS = {"Adelie": palette[0], "Chinstrap": palette[1], "Gentoo": palette[2]}

TRAIN_COLOR = palette[0]   # blue for training rows
TEST_COLOR = palette[3]    # orange-red for test rows

SAVE = dict(dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Figure A: the supervised-learning workflow (labeled boxes and arrows)
# ---------------------------------------------------------------------------
def draw_box(ax, x, y, w, h, text, facecolor, fontsize=11, textcolor="black"):
    """Draw one rounded, labeled box centered at (x, y)."""
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                         boxstyle="round,pad=0.06",
                         facecolor=facecolor, edgecolor="0.3", linewidth=1.4)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=textcolor)


def draw_arrow(ax, x0, y0, x1, y1, color="0.3"):
    """Draw one arrow from (x0, y0) to (x1, y1)."""
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                linewidth=1.8, mutation_scale=18))


def figure_workflow():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0.9, 4.5)
    ax.axis("off")

    steps_blue = "#cfe3f5"
    model_gold = "#fde9c8"
    result_green = "#d8ecd4"

    # 1. the labeled data table
    draw_box(ax, 1.35, 2.9, 2.2, 1.5,
             "333 penguins\n\nmeasurements (X)\n+ species labels (y)", steps_blue)

    # 2. split
    draw_box(ax, 3.9, 3.7, 1.9, 0.85, "TRAINING set\n(80%)", steps_blue)
    draw_box(ax, 3.9, 1.6, 1.9, 0.85, "TEST set (20%)\nkept in a drawer", "#f5d5c8")
    draw_arrow(ax, 2.55, 3.15, 2.9, 3.6)
    draw_arrow(ax, 2.55, 2.6, 2.9, 1.85)
    ax.text(2.62, 2.68, "split", fontsize=10, style="italic", color="0.35",
            ha="center")

    # 3. train
    draw_box(ax, 6.45, 3.7, 1.75, 0.85, "TRAIN\nmodel studies\nthe examples", model_gold, fontsize=10)
    draw_arrow(ax, 4.9, 3.7, 5.5, 3.7)

    # 4. evaluate
    draw_box(ax, 6.45, 1.6, 1.75, 0.85, "EVALUATE\naccuracy on\nunseen penguins", result_green, fontsize=10)
    draw_arrow(ax, 6.45, 3.2, 6.45, 2.1)     # trained model goes down to be evaluated
    draw_arrow(ax, 4.9, 1.6, 5.5, 1.6)       # test set comes in from the drawer

    # 5. predict
    draw_box(ax, 9.35, 2.65, 2.35, 1.15,
             "PREDICT\na brand-new penguin:\nmeasurements in,\nspecies out", result_green, fontsize=10)
    draw_arrow(ax, 7.4, 3.7, 8.2, 3.1)
    ax.text(8.15, 3.75, "if accuracy\nis good", fontsize=9, style="italic",
            color="0.35", ha="center")

    ax.set_title("The supervised-learning workflow: split, train, evaluate, predict",
                 fontsize=13)
    fig.savefig("images/supervised-workflow.png", **SAVE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure B: the train/test split as a strip of penguin rows
# ---------------------------------------------------------------------------
def figure_split():
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_xlim(-0.5, 20.5)
    ax.set_ylim(-2.6, 3.4)
    ax.axis("off")

    n_rows = 20          # a mini dataset of 20 penguin rows (ours has 333)
    n_train = 16         # 80% of 20

    # Top strip: the shuffled dataset, colored by destination.
    for i in range(n_rows):
        if i < n_train:
            color = TRAIN_COLOR
        else:
            color = TEST_COLOR
        ax.add_patch(Rectangle((i, 1.6), 0.9, 0.9,
                               facecolor=color, edgecolor="white", linewidth=1.5))

    ax.text(10, 2.95, "all penguin rows, shuffled once (random_state=42 = same shuffle every run)",
            ha="center", fontsize=11, color="0.25")

    # Braces / labels under the two parts.
    ax.annotate("", xy=(0, 1.25), xytext=(n_train - 0.1, 1.25),
                arrowprops=dict(arrowstyle="-", color=TRAIN_COLOR, linewidth=3))
    ax.annotate("", xy=(n_train, 1.25), xytext=(n_rows - 0.1, 1.25),
                arrowprops=dict(arrowstyle="-", color=TEST_COLOR, linewidth=3))

    ax.text((n_train - 0.1) / 2, 0.35,
            "TRAINING set (80%)\nthe model sees these,\nanswers included",
            ha="center", va="top", fontsize=11, color=TRAIN_COLOR, weight="bold")
    ax.text(n_train + (n_rows - n_train) / 2, 0.35,
            "TEST set (20%)\nhidden until\nthe exam",
            ha="center", va="top", fontsize=11, color=TEST_COLOR, weight="bold")

    ax.text(10, -2.3,
            "stratify=y keeps the species mix the same in both pieces\n"
            "(so the exam is not accidentally all Gentoos)",
            ha="center", fontsize=10.5, style="italic", color="0.35")

    ax.set_title("train_test_split: one honest exam, set aside before studying begins",
                 fontsize=13)
    fig.savefig("images/train-test-split.png", **SAVE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figures C and D need the real data: same load + clean + split as the notebook
# ---------------------------------------------------------------------------
def load_training_data():
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split

    df = load_dataset("SIH/palmer-penguins", split="train").to_pandas().dropna()
    X = df[["bill_length_mm", "bill_depth_mm"]]
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, y_train


# ---------------------------------------------------------------------------
# Figure C: "who are my 5 nearest neighbors?" zoom with vote tally
# ---------------------------------------------------------------------------
def figure_neighbors():
    from sklearn.neighbors import NearestNeighbors

    X_train, y_train = load_training_data()

    # The mystery penguin: bill 43.0 mm long, 18.0 mm deep.
    query = pd.DataFrame({"bill_length_mm": [43.0], "bill_depth_mm": [18.0]})

    # Find the 5 training penguins closest to the query point.
    finder = NearestNeighbors(n_neighbors=5).fit(X_train)
    distances, indices = finder.kneighbors(query)
    neighbor_rows = X_train.iloc[indices[0]]
    neighbor_labels = y_train.iloc[indices[0]]

    fig, ax = plt.subplots(figsize=(10, 5))
    # Same scale on both axes, so "distance" looks like real distance and the
    # dashed neighborhood boundary is a true circle.
    ax.set_aspect("equal")

    # All training penguins, faint, colored by species.
    for species_name in ["Adelie", "Chinstrap", "Gentoo"]:
        mask = (y_train == species_name)
        ax.scatter(X_train[mask]["bill_length_mm"], X_train[mask]["bill_depth_mm"],
                   color=SPECIES_COLORS[species_name], s=35, alpha=0.35,
                   edgecolor="white", linewidth=0.5, label=species_name)

    # Lines from the query to each of its 5 neighbors, neighbors highlighted.
    for i in range(5):
        nx = neighbor_rows.iloc[i]["bill_length_mm"]
        ny = neighbor_rows.iloc[i]["bill_depth_mm"]
        ax.plot([43.0, nx], [18.0, ny], color="0.4", linewidth=1.4, zorder=2)
        ax.scatter([nx], [ny], color=SPECIES_COLORS[neighbor_labels.iloc[i]],
                   s=150, edgecolor="black", linewidth=1.5, zorder=3)

    # A dashed circle through the 5th (farthest) neighbor = the "neighborhood".
    radius = distances[0][4]
    ax.add_patch(Circle((43.0, 18.0), radius, fill=False,
                        linestyle="--", edgecolor="0.3", linewidth=1.5))

    # The mystery penguin itself: a big black star.
    ax.scatter([43.0], [18.0], marker="*", s=500, color="black", zorder=4,
               label="mystery penguin")

    # Tally the neighbors' votes with a plain loop.
    votes = {}
    for i in range(5):
        species_name = neighbor_labels.iloc[i]
        if species_name in votes:
            votes[species_name] = votes[species_name] + 1
        else:
            votes[species_name] = 1
    tally_lines = "Votes among the 5 neighbors:\n"
    for species_name in ["Adelie", "Chinstrap", "Gentoo"]:
        if species_name in votes:
            tally_lines = tally_lines + f"  {species_name}: {votes[species_name]}\n"
    winner = max(votes, key=votes.get)
    tally_lines = tally_lines + f"Verdict: {winner}"

    ax.text(0.03, 0.03, tally_lines, transform=ax.transAxes, fontsize=11,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="0.4"))

    # Zoom in around the neighborhood so the geometry is readable.
    ax.set_xlim(36, 52)
    ax.set_ylim(14.5, 21)
    ax.set_xlabel("Bill length (mm)")
    ax.set_ylabel("Bill depth (mm)")
    ax.set_title("k-NN: the 5 most similar penguins vote on the mystery penguin")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.savefig("images/five-neighbors.png", **SAVE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure D: decision boundaries for small vs large k (real fitted models)
# ---------------------------------------------------------------------------
def figure_k_small_vs_large():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.inspection import DecisionBoundaryDisplay

    X_train, y_train = load_training_data()

    # One color per class, in alphabetical class order (Adelie, Chinstrap,
    # Gentoo) -- so the shaded regions match the dots.
    region_colors = [SPECIES_COLORS["Adelie"],
                     SPECIES_COLORS["Chinstrap"],
                     SPECIES_COLORS["Gentoo"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    k_values = [1, 51]
    titles = ["k = 1: memorizes every single penguin\n(jagged, gerrymandered borders)",
              "k = 51: averages over huge crowds\n(smooth, but ignores local detail)"]

    for panel in range(2):
        ax = axes[panel]
        model = KNeighborsClassifier(n_neighbors=k_values[panel])
        model.fit(X_train, y_train)
        DecisionBoundaryDisplay.from_estimator(
            model, X_train, response_method="predict",
            multiclass_colors=region_colors, alpha=0.35, ax=ax,
            xlabel="Bill length (mm)", ylabel="Bill depth (mm)")
        for species_name in ["Adelie", "Chinstrap", "Gentoo"]:
            mask = (y_train == species_name)
            ax.scatter(X_train[mask]["bill_length_mm"],
                       X_train[mask]["bill_depth_mm"],
                       color=SPECIES_COLORS[species_name], s=30,
                       edgecolor="white", linewidth=0.5, label=species_name)
        ax.set_title(titles[panel], fontsize=11.5)

    axes[0].legend(loc="lower right", framealpha=0.9)
    fig.suptitle("The same data, two very different maps: k controls how much the model smooths",
                 fontsize=13, y=1.05)
    fig.savefig("images/k-small-vs-large.png", **SAVE)
    plt.close(fig)


if __name__ == "__main__":
    figure_workflow()
    figure_split()
    figure_neighbors()
    figure_k_small_vs_large()
    print("All 4 figures written to images/")
