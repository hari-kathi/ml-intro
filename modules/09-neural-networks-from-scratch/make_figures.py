# Regenerates every figure in images/ used by README.md.
# Run with:  /Users/hari/repos/learning/ml-intro/.venv/bin/python make_figures.py

import matplotlib
matplotlib.use("Agg")           # draw to files, not to a window

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Ellipse

# Colorblind-friendly palette (same hues as seaborn's "colorblind")
BLUE = "#0173B2"
ORANGE = "#DE8F05"
GREEN = "#029E73"
VERMILLION = "#D55E00"
PURPLE = "#CC78BC"
GRAY = "#888888"

SAVE_KW = dict(dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Figure (a): biological neuron vs artificial neuron, side by side
# ---------------------------------------------------------------------------
def fig_neuron_bio_vs_artificial():
    fig, (ax_bio, ax_art) = plt.subplots(1, 2, figsize=(12, 5))

    # ----- left panel: a (cartoon) biological neuron -----
    ax_bio.set_xlim(0, 10)
    ax_bio.set_ylim(0, 10)
    ax_bio.axis("off")
    ax_bio.set_title("A real neuron", fontsize=13)

    # soma (cell body)
    soma = Ellipse((3.5, 5), 2.2, 2.6, facecolor="#F5D7B0",
                   edgecolor="black", linewidth=1.5, zorder=3)
    ax_bio.add_patch(soma)
    ax_bio.text(3.5, 5, "soma\n(adds up\nthe inputs)", ha="center", va="center",
                fontsize=9, zorder=4)

    # dendrites: branching lines feeding into the soma
    dendrite_tips = [(0.5, 8.5), (0.4, 6.6), (0.4, 3.4), (0.6, 1.5)]
    for tip in dendrite_tips:
        ax_bio.plot([tip[0], 2.6], [tip[1], 5 + (tip[1] - 5) * 0.35],
                    color=BLUE, linewidth=2, zorder=2)
        # a synapse knob at the tip of each dendrite
        ax_bio.add_patch(Circle(tip, 0.18, facecolor=BLUE, edgecolor="none",
                                zorder=3))
    ax_bio.text(0.4, 9.2, "dendrites: signals arrive\nthrough synapses",
                fontsize=9, color=BLUE)

    # axon: one long line out to the right
    ax_bio.plot([4.6, 8.2], [5, 5], color=GREEN, linewidth=3, zorder=2)
    ax_bio.text(6.3, 5.5, "axon", fontsize=9, color=GREEN, ha="center")
    # axon terminals branching at the end
    for dy in (-1.2, 0, 1.2):
        ax_bio.plot([8.2, 9.4], [5, 5 + dy], color=GREEN, linewidth=2, zorder=2)
        ax_bio.add_patch(Circle((9.4, 5 + dy), 0.16, facecolor=GREEN,
                                edgecolor="none", zorder=3))
    ax_bio.text(9.6, 7.0, "output:\nfires an action\npotential (or not)",
                fontsize=9, color=GREEN, ha="right")
    ax_bio.text(3.5, 2.4, "fires only if the summed\ninput passes a threshold",
                fontsize=9, ha="center", style="italic")

    # ----- right panel: the artificial neuron -----
    ax_art.set_xlim(0, 10)
    ax_art.set_ylim(0, 10)
    ax_art.axis("off")
    ax_art.set_title("The math version", fontsize=13)

    # input circles (the "dendrites")
    for y, name in [(7, "x1"), (3, "x2")]:
        ax_art.add_patch(Circle((1.2, y), 0.55, facecolor="white",
                                edgecolor=BLUE, linewidth=2, zorder=3))
        ax_art.text(1.2, y, name, ha="center", va="center", fontsize=11,
                    color=BLUE, zorder=4)
    ax_art.text(1.2, 8.6, "inputs\n(= dendrites)", ha="center", fontsize=9,
                color=BLUE)

    # weighted arrows into the sum circle (the "synapse strengths")
    ax_art.add_patch(FancyArrowPatch((1.8, 7), (4.0, 5.4), arrowstyle="-|>",
                                     mutation_scale=14, color=GRAY, linewidth=1.5))
    ax_art.add_patch(FancyArrowPatch((1.8, 3), (4.0, 4.6), arrowstyle="-|>",
                                     mutation_scale=14, color=GRAY, linewidth=1.5))
    ax_art.text(2.5, 6.1, "× w1", fontsize=10, color=VERMILLION)
    ax_art.text(2.5, 3.2, "× w2", fontsize=10, color=VERMILLION)
    ax_art.text(2.9, 1.4, "weights = synapse strengths\n(positive: excites,  negative: inhibits)",
                fontsize=9, color=VERMILLION, ha="center")

    # the sum circle (the "soma")
    ax_art.add_patch(Circle((4.7, 5), 0.8, facecolor="#F5D7B0",
                            edgecolor="black", linewidth=1.5, zorder=3))
    ax_art.text(4.7, 5, "sum\n+ b", ha="center", va="center", fontsize=10, zorder=4)
    ax_art.text(5.2, 7.6, "w1*x1 + w2*x2 + b\n(= soma adding up)", ha="center",
                fontsize=9)

    # arrow to the activation box (the "threshold")
    ax_art.add_patch(FancyArrowPatch((5.5, 5), (6.6, 5), arrowstyle="-|>",
                                     mutation_scale=14, color="black", linewidth=1.5))
    ax_art.add_patch(plt.Rectangle((6.6, 4.3), 1.9, 1.4, facecolor="white",
                                   edgecolor=GREEN, linewidth=2, zorder=3))
    ax_art.text(7.55, 5, "activation", ha="center", va="center", fontsize=9,
                color=GREEN, zorder=4)
    ax_art.text(7.55, 3.4, "(= firing threshold,\nbut smooth)", ha="center",
                fontsize=9, color=GREEN)

    # output arrow
    ax_art.add_patch(FancyArrowPatch((8.5, 5), (9.6, 5), arrowstyle="-|>",
                                     mutation_scale=14, color=GREEN, linewidth=2))
    ax_art.text(9.1, 5.6, "output", fontsize=9, color=GREEN, ha="center")

    fig.suptitle("Dendrites → inputs,  synapse strength → weight,  soma → weighted sum,  firing threshold → activation",
                 fontsize=11, y=1.02)
    fig.savefig("images/neuron-bio-vs-artificial.png", **SAVE_KW)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure (b): network anatomy — 2 inputs -> 8 hidden -> 1 output
# ---------------------------------------------------------------------------
def fig_network_anatomy():
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.set_xlim(-1.4, 9.4)
    ax.set_ylim(-2.6, 9.0)
    ax.set_aspect("equal")      # so the neuron circles render as true circles
    ax.axis("off")

    # x position of each layer, and the y positions of its neurons
    x_in, x_hid, x_out = 0.0, 4.0, 8.0
    input_ys = [2.5, 4.5]
    hidden_ys = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    out_y = 3.5

    # the one path we highlight: input 0 -> hidden neuron 5 -> output
    hi_in, hi_hid = 0, 5

    # draw all connections first (gray), highlighted path on top (orange)
    for i, yi in enumerate(input_ys):
        for j, yj in enumerate(hidden_ys):
            if i == hi_in and j == hi_hid:
                continue                        # drawn later in orange
            ax.plot([x_in, x_hid], [yi, yj], color=GRAY, linewidth=0.8,
                    alpha=0.5, zorder=1)
    for j, yj in enumerate(hidden_ys):
        if j == hi_hid:
            continue
        ax.plot([x_hid, x_out], [yj, out_y], color=GRAY, linewidth=0.8,
                alpha=0.5, zorder=1)

    # the highlighted path: one input's signal flowing through one hidden neuron
    ax.plot([x_in, x_hid], [input_ys[hi_in], hidden_ys[hi_hid]], color=ORANGE,
            linewidth=3, zorder=2)
    ax.plot([x_hid, x_out], [hidden_ys[hi_hid], out_y], color=ORANGE,
            linewidth=3, zorder=2)

    # circles for the neurons
    for yi in input_ys:
        ax.add_patch(Circle((x_in, yi), 0.38, facecolor="white", edgecolor=BLUE,
                            linewidth=2, zorder=3))
    for j, yj in enumerate(hidden_ys):
        edge = ORANGE if j == hi_hid else GREEN
        ax.add_patch(Circle((x_hid, yj), 0.30, facecolor="white", edgecolor=edge,
                            linewidth=2, zorder=3))
    ax.add_patch(Circle((x_out, out_y), 0.38, facecolor="white",
                        edgecolor=PURPLE, linewidth=2, zorder=3))

    # labels under each layer (va="top" hangs the text downward from y=-1.0)
    ax.text(x_in, -1.0, "2 inputs\n(a point's two\ncoordinates)", ha="center",
            va="top", fontsize=10, color=BLUE)
    ax.text(x_hid, -1.0, "8 hidden neurons\n(each learns its own\nstraight-line question)",
            ha="center", va="top", fontsize=10, color=GREEN)
    ax.text(x_out, -1.0, "1 output\n(probability of\n\"top moon\")", ha="center",
            va="top", fontsize=10, color=PURPLE)

    # annotate the highlighted path
    ax.annotate("follow one signal:\ninput → hidden neuron → output",
                xy=(2.0, 3.85), xytext=(-1.2, 7.6), fontsize=10, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE))

    ax.text(4.0, 8.5,
            "every line is one weight the network can tune:  2×8 + 8 + 8×1 + 1 = 33 numbers",
            ha="center", fontsize=10, style="italic")
    ax.set_title("The two-layer network we will build (about 30 lines of NumPy)",
                 fontsize=13)
    fig.savefig("images/network-anatomy.png", **SAVE_KW)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure (c): gradient descent — a ball rolling down a loss bowl
# ---------------------------------------------------------------------------
def fig_gradient_descent_bowl():
    fig, ax = plt.subplots(figsize=(8, 5.5))

    w = np.linspace(-2.5, 8.5, 300)
    loss = (w - 3) ** 2
    ax.plot(w, loss, color=BLUE, linewidth=2.5)

    # simulate a few gradient-descent steps starting at w = -1.5
    steps = [-1.5]
    for _ in range(4):
        slope = 2 * (steps[-1] - 3)             # derivative of (w-3)^2
        steps.append(steps[-1] - 0.25 * slope)  # learning rate 0.25

    for k in range(len(steps)):
        wk = steps[k]
        ax.plot(wk, (wk - 3) ** 2, "o", color=VERMILLION, markersize=11,
                zorder=3)
        if k < len(steps) - 1:
            wn = steps[k + 1]
            ax.annotate("", xy=(wn, (wn - 3) ** 2 + 0.6),
                        xytext=(wk, (wk - 3) ** 2 + 0.6),
                        arrowprops=dict(arrowstyle="-|>", color=VERMILLION,
                                        linewidth=2))

    # mark the bottom of the bowl
    ax.plot(3, 0, "*", color=GREEN, markersize=22, zorder=3)
    ax.annotate("the best weight lives here\n(slope = 0, nothing left to fix)",
                xy=(3, 0), xytext=(4.3, 6), fontsize=10, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN))

    ax.annotate("steep slope →\nbig step downhill",
                xy=(-1.1, 18.5), xytext=(-2.3, 26), fontsize=10,
                color=VERMILLION)
    ax.annotate("gentle slope →\nsmall careful step",
                xy=(2.2, 1.2), xytext=(-0.6, 8), fontsize=10, color=VERMILLION,
                arrowprops=dict(arrowstyle="->", color=VERMILLION))

    ax.text(3, 30, "each step:  new_weight = old_weight − learning_rate × slope",
            ha="center", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="#FFF3D6", edgecolor=ORANGE))

    ax.set_xlabel("weight value")
    ax.set_ylabel("loss (how wrong the model is)")
    ax.set_title("Gradient descent: feel the slope under your feet, step downhill, repeat",
                 fontsize=12)
    ax.set_ylim(-2, 34)
    fig.savefig("images/gradient-descent-bowl.png", **SAVE_KW)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure (d): activation function gallery — sigmoid, tanh, ReLU
# ---------------------------------------------------------------------------
def fig_activation_gallery():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    z = np.linspace(-6, 6, 300)

    sigmoid = 1 / (1 + np.exp(-z))
    tanh = np.tanh(z)
    relu = np.maximum(0, z)

    panels = [
        (axes[0], sigmoid, "sigmoid", BLUE,
         "squashes anything into 0…1:\na probability of firing"),
        (axes[1], tanh, "tanh", GREEN,
         "squashes into −1…+1, centered\non 0 (inhibit / excite)"),
        (axes[2], relu, "ReLU", VERMILLION,
         "silent below 0, then passes the\nsignal straight through"),
    ]
    for ax, yvals, name, color, caption in panels:
        ax.plot(z, yvals, color=color, linewidth=2.5)
        ax.axhline(0, color=GRAY, linewidth=0.8)
        ax.axvline(0, color=GRAY, linewidth=0.8)
        ax.set_title(name, fontsize=13, color=color)
        ax.set_xlabel("input (the weighted sum)")
        ax.text(0.5, -0.32, caption, transform=ax.transAxes, ha="center",
                fontsize=10)
    axes[0].set_ylabel("output")
    fig.suptitle("Three ways to bend a straight line — each is a smooth 'firing threshold'",
                 fontsize=13, y=1.04)
    fig.savefig("images/activation-gallery.png", **SAVE_KW)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure (e): the two-moons data and the failing straight-line boundary
# ---------------------------------------------------------------------------
def fig_moons_vs_line():
    from sklearn.datasets import make_moons
    from sklearn.linear_model import LogisticRegression

    X, y = make_moons(n_samples=400, noise=0.25, random_state=42)
    model = LogisticRegression()
    model.fit(X, y)
    acc = model.score(X, y)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # background: color every point of the plane by the model's prediction
    xx, yy = np.meshgrid(np.linspace(-2.0, 3.0, 300),
                         np.linspace(-1.6, 2.1, 300))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    zz = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5],
                colors=["#D6E8F5", "#FCEBD3"], alpha=0.9)
    ax.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=1.5,
               linestyles="--")

    ax.scatter(X[y == 0, 0], X[y == 0, 1], color=BLUE, s=22, label="moon 0",
               edgecolor="white", linewidth=0.4)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color=ORANGE, s=22, label="moon 1",
               edgecolor="white", linewidth=0.4)

    ax.annotate("the straight line's best effort —\nit must cut through both moons",
                xy=(1.35, 0.30), xytext=(1.35, 1.75), fontsize=10, ha="center",
                arrowprops=dict(arrowstyle="->", color="black"))

    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    acc_pct = round(acc * 100, 1)
    ax.set_title(f"Two interlocking moons: a straight line tops out at {acc_pct}% accuracy",
                 fontsize=12)
    ax.legend(loc="lower left")
    fig.savefig("images/moons-vs-line.png", **SAVE_KW)
    plt.close(fig)


if __name__ == "__main__":
    fig_neuron_bio_vs_artificial()
    fig_network_anatomy()
    fig_gradient_descent_bowl()
    fig_activation_gallery()
    fig_moons_vs_line()
    print("All figures written to images/")
