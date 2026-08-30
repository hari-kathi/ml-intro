"""Regenerates every figure in images/ used by the Module 10 README.

Run from this folder with the course venv:
    ../../.venv/bin/python make_figures.py
"""

import matplotlib

matplotlib.use("Agg")  # draw to files, not to a window

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Seaborn "colorblind" palette hex codes (safe for common color-vision deficiencies).
BLUE = "#0173B2"
ORANGE = "#DE8F05"
GREEN = "#029E73"
VERMILLION = "#D55E00"
GRAY = "#949494"
INK = "#333333"

SAVE_KW = {"dpi": 150, "bbox_inches": "tight"}


def rounded_box(ax, x, y, w, h, text, facecolor, fontsize=10.5, textcolor=INK):
    """Draw a rounded rectangle centered at (x, y) with centered text."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.4,
        edgecolor=INK,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=textcolor)


# ---------------------------------------------------------------------------
# Figure 1: the training-loop cycle — five boxes in a circle
# ---------------------------------------------------------------------------
def training_loop_figure():
    fig, ax = plt.subplots(figsize=(8.6, 8.2))
    ax.set_xlim(-1.62, 1.62)
    ax.set_ylim(-1.52, 1.58)
    ax.set_aspect("equal")
    ax.axis("off")

    # The five ritual steps, in order, with the exact line of code for each.
    steps = [
        ("1. FORWARD", "predictions = model(X)", "push data through\nthe network"),
        ("2. LOSS", "loss = loss_fn(pred, y)", "one number:\nhow wrong are we?"),
        ("3. ZERO", "optimizer.zero_grad()", "erase last cycle's\ngradients"),
        ("4. BACKWARD", "loss.backward()", "autograd computes\nevery gradient"),
        ("5. STEP", "optimizer.step()", "nudge every weight\ndownhill"),
    ]
    fills = ["#D6E8F5", "#D6E8F5", "#EFEFEF", "#FAE6C8", "#FAE6C8"]

    radius = 1.06
    n = len(steps)
    angles = []
    for i in range(n):
        # start at the top (90 degrees) and go clockwise
        angles.append(np.deg2rad(90 - i * 360.0 / n))

    # boxes
    for i in range(n):
        x = radius * np.cos(angles[i])
        y = radius * np.sin(angles[i])
        title, code, gloss = steps[i]
        label = title + "\n" + code + "\n" + gloss
        rounded_box(ax, x, y, 0.94, 0.52, label, fills[i], fontsize=10)

    # arrows between consecutive boxes (and from the last back to the first)
    for i in range(n):
        j = (i + 1) % n  # wrap around: after step 5 comes step 1 again
        a0, a1 = angles[i], angles[j]
        # arrow endpoints sit just outside the boxes, along the circle
        gap = 0.44
        start = (radius + 0.26) * np.array([np.cos(a0 - gap), np.sin(a0 - gap)])
        end = (radius + 0.26) * np.array([np.cos(a1 + gap), np.sin(a1 + gap)])
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "color": INK,
                "lw": 2.2,
                "connectionstyle": "arc3,rad=0.28",
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )

    ax.text(
        0,
        0.10,
        "one epoch",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0,
        -0.12,
        "repeat until the loss\nstops falling",
        ha="center",
        va="center",
        fontsize=10.5,
        color=GRAY,
    )
    ax.set_title(
        "The training loop: five lines that never change, from here to GPT",
        fontsize=13,
        pad=14,
    )
    fig.savefig("images/training-loop.png", **SAVE_KW)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: what autograd records — computation graph for loss = (w - 3)^2
# ---------------------------------------------------------------------------
def autograd_graph_figure():
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 4.9)
    ax.axis("off")

    y_mid = 2.7
    # nodes, left to right, evenly spaced
    nodes = [
        (1.4, "w = 2.0\n(requires_grad)"),
        (4.2, "subtract\na = w − 3\na = −1.0"),
        (7.0, "square\nloss = a²"),
        (9.8, "loss = 1.0"),
    ]
    fills = ["#D6E8F5", "#EFEFEF", "#EFEFEF", "#D6E8F5"]
    for i in range(len(nodes)):
        x, label = nodes[i]
        rounded_box(ax, x, y_mid, 1.75, 1.15, label, fills[i], fontsize=10.5)

    # forward arrows (blue, on top) — values flowing left to right
    forward_pairs = [(1.4, 4.2), (4.2, 7.0), (7.0, 9.8)]
    for x0, x1 in forward_pairs:
        ax.annotate(
            "",
            xy=(x1 - 1.05, y_mid + 0.32),
            xytext=(x0 + 1.05, y_mid + 0.32),
            arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 2.4},
        )
    ax.text(5.6, y_mid + 1.05, "forward: compute values, and record each operation",
            color=BLUE, ha="center", fontsize=10.5)

    # backward arrows (orange, underneath) — gradients flowing right to left
    grads = [
        (9.8, 7.0, "dloss/dloss = 1"),
        (7.0, 4.2, "dloss/da = 2a = −2"),
        (4.2, 1.4, "dloss/dw = −2 × 1 = −2"),
    ]
    for x0, x1, label in grads:
        ax.annotate(
            "",
            xy=(x1 + 1.05, y_mid - 0.32),
            xytext=(x0 - 1.05, y_mid - 0.32),
            arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": 2.4},
        )
        ax.text((x0 + x1) / 2, y_mid - 1.12, label, color=ORANGE,
                ha="center", fontsize=10.5)

    ax.text(
        5.6,
        y_mid - 1.75,
        "backward: multiply local derivatives as you go — the chain rule, "
        "exactly what you did by hand in Module 09",
        color=ORANGE,
        ha="center",
        fontsize=10.5,
    )
    ax.text(
        1.4,
        y_mid - 2.35,
        "w.grad ends up holding −2",
        color=INK,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_title(
        "Autograd records every operation, then replays the tape backwards",
        fontsize=13,
        pad=10,
    )
    fig.savefig("images/autograd-graph.png", **SAVE_KW)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: loss-curve anatomy — healthy / too-hot / overfitting
# ---------------------------------------------------------------------------
def loss_anatomy_figure():
    rng = np.random.default_rng(42)
    epochs = np.arange(200)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharey=True)

    # (a) healthy: smooth exponential decay with a little noise
    healthy = 1.1 * np.exp(-epochs / 40) + 0.12 + rng.normal(0, 0.008, 200)
    axes[0].plot(epochs, healthy, color=BLUE, lw=2)
    axes[0].set_title("Healthy", fontsize=12)
    axes[0].annotate(
        "fast drop early...",
        xy=(22, 0.80),
        xytext=(60, 1.02),
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": INK},
    )
    axes[0].annotate(
        "...then levels off\n(converged)",
        xy=(160, 0.14),
        xytext=(95, 0.45),
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": INK},
    )

    # (b) learning rate too high: jagged, oscillating, creeping upward
    hot = 0.75 + 0.25 * np.sin(epochs / 3.1) + epochs * 0.0022
    hot = hot + rng.normal(0, 0.05, 200)
    axes[1].plot(epochs, hot, color=VERMILLION, lw=1.6)
    axes[1].set_title("Learning rate too high", fontsize=12)
    axes[1].annotate(
        "every step overshoots\nthe valley floor",
        xy=(120, 1.02),
        xytext=(28, 1.32),
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": INK},
    )

    # (c) overfitting: train keeps falling, test turns back up
    train = 1.1 * np.exp(-epochs / 30) + 0.02 + rng.normal(0, 0.006, 200)
    test = 1.1 * np.exp(-epochs / 30) + 0.30 + 0.000038 * (epochs - 45) ** 2 * (epochs > 45)
    test = test + rng.normal(0, 0.01, 200)
    axes[2].plot(epochs, train, color=BLUE, lw=2, label="training loss")
    axes[2].plot(epochs, test, color=ORANGE, lw=2, label="test loss")
    axes[2].axvline(45, color=GRAY, ls="--", lw=1.4)
    axes[2].annotate(
        "memorization starts here\n— stop training",
        xy=(45, 1.05),
        xytext=(72, 1.25),
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": INK},
    )
    axes[2].set_title("Overfitting", fontsize=12)
    axes[2].legend(loc="center right", frameon=False, fontsize=9.5)

    for ax in axes:
        ax.set_xlabel("epoch")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("loss")
    axes[0].set_ylim(0, 1.55)

    fig.suptitle("Read the loss curve like an ECG: its shape tells you what's wrong",
                 fontsize=13, y=1.04)
    fig.savefig("images/loss-curve-anatomy.png", **SAVE_KW)
    plt.close(fig)


if __name__ == "__main__":
    training_loop_figure()
    autograd_graph_figure()
    loss_anatomy_figure()
    print("Wrote images/training-loop.png, images/autograd-graph.png, "
          "images/loss-curve-anatomy.png")
