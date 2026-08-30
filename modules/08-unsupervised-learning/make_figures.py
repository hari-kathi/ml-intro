"""Regenerates every figure in images/ used by the Module 08 README.

Run with the course venv:
    /Users/hari/repos/learning/ml-intro/.venv/bin/python make_figures.py
"""

import matplotlib

matplotlib.use("Agg")  # draw to files, not to a window

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import linkage, dendrogram

sns.set_theme(style="whitegrid", palette="colorblind")
palette = sns.color_palette("colorblind")

SAVE = dict(dpi=150, bbox_inches="tight")


# ----------------------------------------------------------------------------
# Figure 1: k-means, frame by frame, on synthetic blobs
# ----------------------------------------------------------------------------
def kmeans_iterations_figure():
    # Three well-separated synthetic blobs (make_blobs invents toy clustered data).
    X, _ = make_blobs(
        n_samples=180, centers=3, cluster_std=1.1, random_state=7
    )

    rng = np.random.default_rng(3)
    # Start the centers at 3 randomly chosen data points (a common initialization).
    centers = X[rng.choice(len(X), size=3, replace=False)].copy()

    def assign(points, cents):
        # For every point, find the index of the nearest center.
        labels = np.zeros(len(points), dtype=int)
        for i in range(len(points)):
            dists = np.sqrt(((cents - points[i]) ** 2).sum(axis=1))
            labels[i] = np.argmin(dists)
        return labels

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharex=True, sharey=True)

    # --- Panel 1: initial guess, nobody assigned yet ---
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], c="0.65", s=18)
    ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=260,
               c=[palette[0], palette[1], palette[2]], edgecolor="black",
               linewidth=1.5, zorder=5)
    ax.set_title("1. Drop k centers at random", fontsize=12)

    # --- Panel 2: assign each point to its nearest center ---
    labels = assign(X, centers)
    ax = axes[1]
    for k in range(3):
        pts = X[labels == k]
        ax.scatter(pts[:, 0], pts[:, 1], color=palette[k], s=18, alpha=0.8)
    ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=260,
               c=[palette[0], palette[1], palette[2]], edgecolor="black",
               linewidth=1.5, zorder=5)
    ax.set_title("2. Assign: each point joins\nits nearest center", fontsize=12)

    # --- Panel 3: move each center to the mean of its points ---
    new_centers = centers.copy()
    for k in range(3):
        new_centers[k] = X[labels == k].mean(axis=0)
    ax = axes[2]
    for k in range(3):
        pts = X[labels == k]
        ax.scatter(pts[:, 0], pts[:, 1], color=palette[k], s=18, alpha=0.8)
        # Arrow from the old center position to the new one.
        ax.annotate("", xy=new_centers[k], xytext=centers[k],
                    arrowprops=dict(arrowstyle="->", lw=2.2, color="black"))
    ax.scatter(new_centers[:, 0], new_centers[:, 1], marker="X", s=260,
               c=[palette[0], palette[1], palette[2]], edgecolor="black",
               linewidth=1.5, zorder=5)
    ax.set_title("3. Move: each center jumps to\nthe average of its members", fontsize=12)

    # --- Panel 4: repeat until the centers stop moving ---
    centers = new_centers
    for _ in range(10):  # a few more assign/move rounds until convergence
        labels = assign(X, centers)
        for k in range(3):
            centers[k] = X[labels == k].mean(axis=0)
    labels = assign(X, centers)
    ax = axes[3]
    for k in range(3):
        pts = X[labels == k]
        ax.scatter(pts[:, 0], pts[:, 1], color=palette[k], s=18, alpha=0.8)
    ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=260,
               c=[palette[0], palette[1], palette[2]], edgecolor="black",
               linewidth=1.5, zorder=5)
    ax.set_title("4. Repeat 2–3 until the centers\nstop moving: converged", fontsize=12)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("k-means is just two moves, repeated: assign, then re-average",
                 fontsize=15, y=1.04)
    fig.savefig("images/kmeans-iterations.png", **SAVE)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 2: a small dendrogram, annotated like a phylogeny
# ----------------------------------------------------------------------------
def dendrogram_figure():
    # Eight made-up "samples" with two clear families and one loner,
    # engineered so the tree has an obvious story to tell.
    rng = np.random.default_rng(11)
    family1 = rng.normal(loc=[0, 0], scale=0.35, size=(3, 2))     # tight trio
    family2 = rng.normal(loc=[4, 0.5], scale=0.5, size=(4, 2))    # looser quartet
    loner = np.array([[2.2, 4.0]])                                # far from both
    X = np.vstack([family1, family2, loner])
    names = ["A", "B", "C", "D", "E", "F", "G", "H"]

    # 'ward' linkage: at each step, fuse the two groups whose merge keeps
    # clusters tightest -- the same greedy logic as building a UPGMA tree.
    Z = linkage(X, method="ward")

    fig, ax = plt.subplots(figsize=(9, 6))
    dendrogram(Z, labels=names, ax=ax, color_threshold=3.0,
               above_threshold_color="0.4")
    ax.set_ylabel("fusion height  (how different the two sides were when they merged)")
    ax.set_title("Read a dendrogram exactly like a phylogenetic tree", fontsize=14)

    # In dendrogram coordinates the leaves sit at x = 5, 15, 25, ... left to right.
    # Annotate the first (lowest) fusion -- the most similar pair (3rd + 4th leaves).
    ax.annotate(
        "lowest fusion =\nmost similar pair\n(they merged first)",
        xy=(30, 0.12), xytext=(20, 1.9), fontsize=11, ha="left",
        arrowprops=dict(arrowstyle="->", lw=1.8, color="black"),
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6"),
    )
    # Annotate the top join -- the deepest split (the root bar).
    ax.annotate(
        "highest join = deepest split,\nlike the root of a phylogeny:\ncut just below it to get 2 groups",
        xy=(43, 7.72), xytext=(48, 6.2), fontsize=11, ha="left",
        arrowprops=dict(arrowstyle="->", lw=1.8, color="black"),
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6"),
    )
    # Annotate the loner's long branch (5th leaf, which joins very late).
    ax.annotate(
        "a long lone branch =\nan outlier, like a species\nwith no close relatives",
        xy=(45, 3.0), xytext=(3, 3.4), fontsize=11, ha="left",
        arrowprops=dict(arrowstyle="->", lw=1.8, color="black"),
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6"),
    )
    ax.set_xlabel("samples (leaves of the tree)")
    fig.savefig("images/dendrogram-phylogeny.png", **SAVE)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 3: PCA as shadow-casting
# ----------------------------------------------------------------------------
def pca_shadow_figure():
    # An elongated 3-D cloud: most of its variation lies along one tilted axis.
    rng = np.random.default_rng(5)
    t = rng.normal(size=250)              # position along the long axis
    u = rng.normal(size=250) * 0.8        # medium spread
    v = rng.normal(size=250) * 0.15       # tiny spread (the "thin" direction)
    # Build the cloud by mixing those three directions in 3-D space.
    X = np.column_stack([
        2.0 * t + 0.5 * u,
        1.2 * t - 0.8 * u,
        0.8 * t + 0.9 * u + v + 2.5,
    ])

    fig = plt.figure(figsize=(13, 5.5))

    # --- Left: the 3-D cloud and its shadow on the floor ---
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax3.scatter(X[:, 0], X[:, 1], X[:, 2], color=palette[0], s=14, alpha=0.7)
    floor = X[:, 2].min() - 3.5
    # The shadow: same x and y, but flattened onto the floor plane.
    ax3.scatter(X[:, 0], X[:, 1], np.full(len(X), floor),
                color="0.55", s=14, alpha=0.45)
    # A few dashed "light rays" from points down to their shadows.
    for i in [4, 60, 140, 220]:
        ax3.plot([X[i, 0], X[i, 0]], [X[i, 1], X[i, 1]], [X[i, 2], floor],
                 linestyle="--", color="0.4", linewidth=1.0)
    ax3.set_title("A 3-D cloud and its 2-D shadow", fontsize=13)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_zticks([])
    ax3.set_xlabel("measurement 1")
    ax3.set_ylabel("measurement 2")

    # --- Right: the shadow alone -- what PCA hands you ---
    ax2 = fig.add_subplot(1, 2, 2)
    # PCA's shadow is cast from the BEST angle: axes = directions of most spread.
    Xc = X - X.mean(axis=0)
    # Eigen-decomposition of the covariance matrix gives those directions.
    eigvals, eigvecs = np.linalg.eigh(np.cov(Xc.T))
    order = np.argsort(eigvals)[::-1]          # biggest spread first
    proj = Xc @ eigvecs[:, order[:2]]          # project onto the top-2 directions
    ax2.scatter(proj[:, 0], proj[:, 1], color=palette[0], s=16, alpha=0.7)
    ax2.set_xlabel("PC1  (direction of MOST spread)")
    ax2.set_ylabel("PC2  (most spread left over)")
    ax2.set_title("PCA picks the angle that keeps\nthe most spread in the shadow", fontsize=13)
    ax2.annotate(
        "flat cloud, almost nothing lost:\nthe thin 3rd direction was\nmostly noise anyway",
        xy=(0.03, 0.04), xycoords="axes fraction", fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6"),
    )

    fig.suptitle("PCA = shadow-casting: flatten many dimensions, lose as little as possible",
                 fontsize=15, y=1.02)
    fig.savefig("images/pca-shadow.png", **SAVE)
    plt.close(fig)


if __name__ == "__main__":
    kmeans_iterations_figure()
    dendrogram_figure()
    pca_shadow_figure()
    print("All figures written to images/")
