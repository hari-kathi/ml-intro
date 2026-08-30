"""Regenerate every figure in images/ used by the Module 11 README.

Run with the course venv:
    /Users/hari/repos/learning/ml-intro/.venv/bin/python make_figures.py
"""

import matplotlib

matplotlib.use("Agg")  # draw to files, not to a window

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from datasets import load_dataset

# Colorblind-friendly accent colors (from seaborn's "colorblind" palette)
BLUE = "#0173B2"
ORANGE = "#DE8F05"
GREEN = "#029E73"
RED = "#D55E00"

SAVE_KW = dict(dpi=150, bbox_inches="tight")


def load_one_digit():
    """Return one real MNIST digit (28x28 uint8 array) and its label."""
    ds = load_dataset("ylecun/mnist", split="train")
    example = ds[0]  # the famous first training image: a handwritten 5
    return np.array(example["image"]), example["label"]


# ---------------------------------------------------------------------------
# Figure (a): pixels as numbers — a digit with a magnified numeric patch
# ---------------------------------------------------------------------------
def fig_pixels_as_numbers(digit, label):
    fig, (ax_img, ax_zoom) = plt.subplots(1, 2, figsize=(10, 5))

    # Left panel: the digit as you'd normally see it
    ax_img.imshow(digit, cmap="gray")
    ax_img.set_title(f'What you see: a handwritten "{label}"', fontsize=12)
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    # Highlight the patch we will magnify (rows 5-12, cols 12-19: an ink edge)
    r0, c0, size = 5, 12, 8
    ax_img.add_patch(
        Rectangle((c0 - 0.5, r0 - 0.5), size, size, fill=False, edgecolor=ORANGE, lw=2.5)
    )

    # Right panel: the same patch drawn as a grid of numbers
    patch = digit[r0 : r0 + size, c0 : c0 + size]
    ax_zoom.imshow(patch, cmap="gray", vmin=0, vmax=255)
    for i in range(size):
        for j in range(size):
            value = patch[i, j]
            # white text on dark cells, black text on bright cells
            color = "white" if value < 128 else "black"
            ax_zoom.text(j, i, str(value), ha="center", va="center", fontsize=9, color=color)
    ax_zoom.set_title("What the computer sees: 0 (black) to 255 (white ink)", fontsize=12)
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])
    for spine in ax_zoom.spines.values():
        spine.set_edgecolor(ORANGE)
        spine.set_linewidth(2.5)

    fig.suptitle("An image IS numbers — a 28×28 grid of brightness values", fontsize=14, y=1.02)
    fig.savefig("images/pixels-as-numbers.png", **SAVE_KW)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure (b): a 3×3 filter sliding across a grid, three successive positions
# ---------------------------------------------------------------------------
def fig_filter_sliding():
    # A tiny 6x6 toy "image" with a bright vertical stroke in the middle
    img = np.zeros((6, 6))
    img[:, 2] = 1.0
    img[:, 3] = 1.0

    # A vertical-edge filter: bright-on-left minus bright-on-right
    filt = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

    # Full valid convolution output (4x4), computed with plain loops
    out = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            out[i, j] = np.sum(img[i : i + 3, j : j + 3] * filt)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7), gridspec_kw={"height_ratios": [1.4, 1]})

    for step in range(3):  # filter positions (0,0), (0,1), (0,2)
        ax_in = axes[0, step]
        ax_out = axes[1, step]

        # Input grid with cell values
        ax_in.imshow(img, cmap="gray", vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                color = "black" if img[i, j] > 0.5 else "0.6"
                ax_in.text(j, i, str(int(img[i, j])), ha="center", va="center",
                           fontsize=9, color=color)
        # The 3x3 window the filter currently covers
        ax_in.add_patch(Rectangle((step - 0.5, -0.5), 3, 3, fill=False,
                                  edgecolor=ORANGE, lw=3))
        ax_in.set_title(f"position {step + 1}\n(columns {step}–{step + 2})", fontsize=9)
        ax_in.set_xticks([])
        ax_in.set_yticks([])

        # Output grid: cells computed so far are filled in, the newest highlighted
        ax_out.imshow(np.zeros((4, 4)), cmap="gray", vmin=0, vmax=1)
        for j in range(step + 1):
            value = out[0, j]
            ax_out.text(j, 0, str(int(value)), ha="center", va="center",
                        fontsize=11, color=ORANGE if j == step else "white",
                        fontweight="bold")
        ax_out.add_patch(Rectangle((step - 0.5, -0.5), 1, 1, fill=False,
                                   edgecolor=ORANGE, lw=3))
        ax_out.set_title("feature map (output)", fontsize=9)
        ax_out.set_xticks([])
        ax_out.set_yticks([])

    # Show the filter itself between the rows, on the left margin
    fig.text(0.04, 0.5,
             "the 3×3 filter\n(same at every position):\n\n"
             " 1   0  -1\n 1   0  -1\n 1   0  -1",
             fontsize=11, family="monospace", ha="left", va="center",
             bbox=dict(boxstyle="round", facecolor="#FDF2E3", edgecolor=ORANGE))

    fig.suptitle("Convolution: ONE small filter slides across the whole image,\n"
                 "writing one number per position — like an enzyme scanning DNA for its motif",
                 fontsize=13)
    fig.text(0.62, 0.44, "at each position: multiply cell-by-cell, add up → ONE output number",
             ha="center", fontsize=10, style="italic")
    fig.subplots_adjust(left=0.28, top=0.82, hspace=0.45)
    fig.savefig("images/filter-sliding.png", **SAVE_KW)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure (c): CNN architecture — labeled boxes with shrinking spatial size
# ---------------------------------------------------------------------------
def fig_cnn_architecture(digit):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Each stage: (x position, box width, box height, facecolor, title, subtitle)
    stages = [
        (0.3, 1.5, 2.8, "#DDDDDD", "input image", "28×28×1"),
        (2.4, 1.5, 2.8, "#CFE3F0", "conv 3×3\n8 filters + ReLU", "28×28×8"),
        (4.5, 1.0, 1.6, "#CFE3F0", "max pool\n2×2", "14×14×8"),
        (6.1, 1.0, 1.6, "#F8E3C8", "conv 3×3\n16 filters + ReLU", "14×14×16"),
        (7.7, 0.7, 0.9, "#F8E3C8", "max pool\n2×2", "7×7×16"),
        (9.0, 0.5, 2.8, "#DDF0E5", "flatten", "784 numbers"),
        (10.3, 0.9, 2.0, "#DDF0E5", "linear\nlayer", "10 scores"),
    ]

    for x, w, h, face, title, subtitle in stages:
        y = 2.4 - h / 2  # vertically centred boxes
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor="black", lw=1.2))
        ax.text(x + w / 2, y + h + 0.35, title, ha="center", va="bottom", fontsize=9)
        ax.text(x + w / 2, y - 0.3, subtitle, ha="center", va="top", fontsize=9,
                fontweight="bold")

    # Arrows between consecutive boxes
    for k in range(len(stages) - 1):
        x_from = stages[k][0] + stages[k][1]
        x_to = stages[k + 1][0]
        ax.add_patch(FancyArrow(x_from + 0.05, 2.4, x_to - x_from - 0.25, 0,
                                width=0.02, head_width=0.15, head_length=0.12,
                                color="0.3"))

    # A real digit inside the input box
    ax_digit = ax.inset_axes([0.35 / 13, (2.4 - 1.25) / 5, 1.4 / 13, 2.5 / 5])
    ax_digit.imshow(digit, cmap="gray")
    ax_digit.set_xticks([])
    ax_digit.set_yticks([])

    # The 10 output scores, drawn as small ticks next to the last box
    for k in range(10):
        y = 1.55 + k * 0.19
        ax.add_patch(Rectangle((11.6, y), 0.35, 0.13, facecolor=GREEN if k == 5 else "#DDDDDD",
                               edgecolor="black", lw=0.6))
        ax.text(12.05, y + 0.06, str(k), fontsize=7, va="center")
    ax.text(11.8, 3.6, "one score\nper digit", ha="center", fontsize=8)

    ax.set_title("Our small CNN: the image shrinks in width and height, but grows in "
                 "channels (motif detectors)", fontsize=13)
    fig.savefig("images/cnn-architecture.png", **SAVE_KW)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure (d): MLP vs CNN — flattening destroys neighborhoods, filters keep them
# ---------------------------------------------------------------------------
def fig_mlp_vs_cnn(digit):
    fig, (ax_mlp, ax_cnn) = plt.subplots(2, 1, figsize=(10, 7))

    # --- Top panel: the MLP view ---------------------------------------
    ax_mlp.set_xlim(0, 10)
    ax_mlp.set_ylim(0, 2.4)
    ax_mlp.axis("off")
    ax_mlp.set_title("MLP: flattening puts vertical neighbors 28 columns apart —\n"
                     "the network is never told which pixels touch each other",
                     fontsize=12, fontweight="bold")

    # Small digit on the left with two vertically adjacent pixels marked
    ax_d = ax_mlp.inset_axes([0.02, 0.15, 0.18, 0.72])
    ax_d.imshow(digit, cmap="gray")
    ax_d.add_patch(Rectangle((13.5, 9.5), 1, 1, fill=False, edgecolor=ORANGE, lw=2))
    ax_d.add_patch(Rectangle((13.5, 10.5), 1, 1, fill=False, edgecolor=GREEN, lw=2))
    ax_d.set_xticks([])
    ax_d.set_yticks([])
    ax_d.set_xlabel("two neighboring pixels", fontsize=9)

    # The flattened strip: a long row of little squares
    strip_y = 1.0
    for k in range(30):
        x = 2.6 + k * 0.24
        face = "#DDDDDD"
        if k == 6:
            face = ORANGE   # pixel from row 10
        if k == 20:
            face = GREEN    # its below-neighbor lands 28 positions away
        ax_mlp.add_patch(Rectangle((x, strip_y), 0.2, 0.35, facecolor=face,
                                   edgecolor="black", lw=0.5))
    ax_mlp.text(3.6, strip_y - 0.35, "... position 294 ...", fontsize=9, color=ORANGE,
                ha="center")
    ax_mlp.text(7.0, strip_y - 0.35, "... position 322 ...", fontsize=9, color=GREEN,
                ha="center")
    ax_mlp.text(9.85, strip_y + 0.17, "→ 784\ncolumns", fontsize=9, ha="left", va="center")
    ax_mlp.annotate("flatten", xy=(2.55, strip_y + 0.6), xytext=(2.1, strip_y + 1.1),
                    fontsize=10, ha="center",
                    arrowprops=dict(arrowstyle="->", color="0.3"))

    # --- Bottom panel: the CNN view -------------------------------------
    ax_cnn.set_xlim(0, 10)
    ax_cnn.set_ylim(0, 2.4)
    ax_cnn.axis("off")
    ax_cnn.set_title("CNN: a small filter looks at a patch of touching pixels —\n"
                     "the 2-D structure of the image is built into the model",
                     fontsize=12, fontweight="bold", color=BLUE)

    ax_d2 = ax_cnn.inset_axes([0.02, 0.15, 0.18, 0.72])
    ax_d2.imshow(digit, cmap="gray")
    # A 3x3 filter window drawn on the digit, plus two ghost positions
    ax_d2.add_patch(Rectangle((11.5, 8.5), 3, 3, fill=False, edgecolor=BLUE, lw=2.5))
    ax_d2.add_patch(Rectangle((14.5, 8.5), 3, 3, fill=False, edgecolor=BLUE, lw=1, ls="--"))
    ax_d2.add_patch(Rectangle((17.5, 8.5), 3, 3, fill=False, edgecolor=BLUE, lw=1, ls=":"))
    ax_d2.set_xticks([])
    ax_d2.set_yticks([])
    ax_d2.set_xlabel("a 3×3 filter slides along", fontsize=9)

    # Zoomed 3x3 patch to show the filter sees a NEIGHBORHOOD
    patch = digit[8:11, 11:14]
    ax_p = ax_cnn.inset_axes([0.36, 0.15, 0.15, 0.55])
    ax_p.imshow(patch, cmap="gray", vmin=0, vmax=255)
    for spine in ax_p.spines.values():
        spine.set_edgecolor(BLUE)
        spine.set_linewidth(2.5)
    ax_p.set_xticks([])
    ax_p.set_yticks([])
    ax_p.set_xlabel("it sees a patch:\nneighbors stay neighbors", fontsize=9)

    ax_cnn.text(7.6, 1.0,
                "the SAME filter is reused at every position,\n"
                "so a motif is recognized anywhere it appears\n"
                "(and shifting the digit barely matters)",
                ha="center", va="center", fontsize=10)

    fig.subplots_adjust(hspace=0.5)
    fig.savefig("images/mlp-vs-cnn.png", **SAVE_KW)
    plt.close(fig)


if __name__ == "__main__":
    digit, label = load_one_digit()
    fig_pixels_as_numbers(digit, label)
    fig_filter_sliding()
    fig_cnn_architecture(digit)
    fig_mlp_vs_cnn(digit)
    print("All figures written to images/")
