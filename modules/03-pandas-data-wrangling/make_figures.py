"""Regenerates every figure in images/ used by the Module 03 README.

Run with the course virtualenv:
    ../../.venv/bin/python make_figures.py
"""

import matplotlib

matplotlib.use("Agg")  # draw to files, not to a window

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import pandas as pd
import seaborn as sns

sns.set_theme(style="white")
palette = sns.color_palette("colorblind")

# One fixed color per country, reused across the diagrams so the eye can
# follow a value as it moves between table shapes.
INDIA = palette[1]     # orange
JAPAN = palette[2]     # green
BRAZIL = palette[4]    # purple
CHAD = palette[3]      # reddish
GRAY = (0.55, 0.55, 0.55)
HEADER = (0.90, 0.90, 0.90)


def draw_table(ax, x, y, col_widths, header, rows, cell_colors, row_h=0.9,
               title=None, fontsize=11):
    """Draw a little table with matplotlib rectangles.

    (x, y) is the top-left corner. `rows` is a list of row-value lists,
    `cell_colors` a matching list of per-cell background colors (None = white).
    Returns the total width and height drawn.
    """
    total_w = sum(col_widths)
    if title is not None:
        ax.text(x + total_w / 2, y + 0.55, title, ha="center", va="bottom",
                fontsize=fontsize + 2, fontweight="bold")
    # header row
    cx = x
    for w, label in zip(col_widths, header):
        ax.add_patch(Rectangle((cx, y - row_h), w, row_h, facecolor=HEADER,
                               edgecolor="black", linewidth=1.1))
        ax.text(cx + w / 2, y - row_h / 2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold")
        cx += w
    # data rows
    for r, (values, colors) in enumerate(zip(rows, cell_colors)):
        ry = y - (r + 2) * row_h
        cx = x
        for w, val, color in zip(col_widths, values, colors):
            face = "white" if color is None else color
            ax.add_patch(Rectangle((cx, ry), w, row_h, facecolor=face,
                                   edgecolor="black", linewidth=1.1))
            ax.text(cx + w / 2, ry + row_h / 2, val, ha="center", va="center",
                    fontsize=fontsize)
            cx += w
    return total_w, (len(rows) + 1) * row_h


def tint(color, alpha=0.35):
    """Blend a palette color toward white so text stays readable on it."""
    r, g, b = color
    return (1 - alpha + alpha * r, 1 - alpha + alpha * g, 1 - alpha + alpha * b)


# ---------------------------------------------------------------------------
# Figure 1: long vs wide (pivot / melt)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 5.6))
ax.set_xlim(0, 22)
ax.set_ylim(-8.2, 1.6)
ax.axis("off")

ind, jap = tint(INDIA), tint(JAPAN)

# LONG table: one row per (date, country) pair
long_rows = [
    ["Jan 1", "India", "100"],
    ["Jan 1", "Japan", "80"],
    ["Jan 2", "India", "120"],
    ["Jan 2", "Japan", "90"],
    ["Jan 3", "India", "150"],
    ["Jan 3", "Japan", "85"],
]
long_colors = [
    [None, ind, ind],
    [None, jap, jap],
    [None, ind, ind],
    [None, jap, jap],
    [None, ind, ind],
    [None, jap, jap],
]
draw_table(ax, 0.6, 0, [2.2, 2.4, 2.0], ["date", "country", "cases"],
           long_rows, long_colors, title="LONG  (one row per pair)")

# WIDE table: one row per date, one column per country
wide_rows = [
    ["Jan 1", "100", "80"],
    ["Jan 2", "120", "90"],
    ["Jan 3", "150", "85"],
]
wide_colors = [
    [None, ind, jap],
    [None, ind, jap],
    [None, ind, jap],
]
draw_table(ax, 14.6, -1.35, [2.2, 2.4, 2.4], ["date", "India", "Japan"],
           wide_rows, wide_colors, title="WIDE  (countries become columns)")

# arrows between the two shapes
ax.add_patch(FancyArrowPatch((8.0, -2.3), (14.0, -2.3), arrowstyle="-|>",
                             mutation_scale=26, linewidth=2.4, color="black"))
ax.text(11.0, -1.9, '.pivot(index="date",\ncolumns="country",\nvalues="cases")',
        ha="center", va="bottom", fontsize=10.5, family="monospace")
ax.add_patch(FancyArrowPatch((14.0, -5.1), (8.0, -5.1), arrowstyle="-|>",
                             mutation_scale=26, linewidth=2.4, color="black"))
ax.text(11.0, -5.5, '.melt(id_vars="date")', ha="center", va="top",
        fontsize=10.5, family="monospace")

ax.text(11.0, -7.6,
        "Same six measurements, two shapes. Colors follow each country's values:\n"
        "pivot spreads them into columns; melt stacks them back into rows.",
        ha="center", va="top", fontsize=11)

fig.savefig("images/long-vs-wide.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: how a merge matches rows on a key column
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 7))
ax.set_xlim(0, 22)
ax.set_ylim(-12.6, 1.8)
ax.axis("off")

ind, bra, cha, japc = tint(INDIA), tint(BRAZIL), tint(CHAD), tint(JAPAN)

# left table: this week's cases
draw_table(ax, 1.2, 0, [2.6, 2.0], ["country", "cases"],
           [["India", "150"], ["Brazil", "90"], ["Chad", "12"]],
           [[ind, None], [bra, None], [cha, None]],
           title="cases table")

# right table: populations (the lookup table)
draw_table(ax, 15.0, 0, [2.6, 2.8], ["country", "population"],
           [["India", "1,408 M"], ["Brazil", "214 M"], ["Japan", "125 M"]],
           [[ind, None], [bra, None], [japc, None]],
           title="population table")

# match lines between equal key values (row centers)
def row_center_y(row_index):
    return -0.9 * (row_index + 1) - 0.45

for left_row, right_row in [(0, 0), (1, 1)]:
    ax.add_patch(FancyArrowPatch((5.9, row_center_y(left_row)),
                                 (14.9, row_center_y(right_row)),
                                 arrowstyle="<|-|>", mutation_scale=16,
                                 linewidth=2.0, color="black",
                                 connectionstyle="arc3,rad=0.12"))
ax.text(10.4, -0.75, "rows match where the KEY\n(country) is equal",
        ha="center", va="bottom", fontsize=11, fontstyle="italic")

# no-match annotations
ax.annotate("no match found\nanywhere on the right",
            xy=(5.9, row_center_y(2)), xytext=(2.6, -5.9),
            fontsize=10, ha="left", va="center", color=palette[3],
            arrowprops=dict(arrowstyle="->", color=palette[3], linewidth=1.8))
ax.annotate('only in the right table:\nwith how="left" it is ignored',
            xy=(16.3, -3.75), xytext=(17.7, -5.9),
            fontsize=10, ha="center", va="center", color=GRAY,
            arrowprops=dict(arrowstyle="->", color=GRAY, linewidth=1.8))

# result table
draw_table(ax, 6.9, -7.9, [2.6, 2.0, 2.8], ["country", "cases", "population"],
           [["India", "150", "1,408 M"],
            ["Brazil", "90", "214 M"],
            ["Chad", "12", "NaN"]],
           [[ind, None, None], [bra, None, None], [cha, None, tint(palette[3], 0.5)]],
           title='pd.merge(cases, pops, on="country", how="left")')

ax.text(11.0, -12.0,
        "Every row of the LEFT table survives. Matched rows pick up the new columns;\n"
        "a row with no partner (Chad) gets NaN — missing data you must notice and handle.",
        ha="center", va="top", fontsize=11)

fig.savefig("images/merge-diagram.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: raw weekly-batch reporting vs the 7-day rolling average
# ---------------------------------------------------------------------------

# Build a synthetic epidemic with two waves (about 5 months of days).
days = np.arange(150)
true_daily = (420 * np.exp(-((days - 45) / 13.0) ** 2)      # first wave
              + 950 * np.exp(-((days - 108) / 11.0) ** 2)   # bigger second wave
              + 12)                                          # low background

# Countries in our dataset report in WEEKLY BATCHES: the whole week's total
# appears on one day, with zeros in between. Recreate that pattern here.
raw = np.zeros_like(true_daily)
for d in days:
    if d % 7 == 6 and d >= 6:                # every 7th day is "reporting day"
        raw[d] = true_daily[d - 6:d + 1].sum()  # the whole week in one number

smoothed = pd.Series(raw).rolling(7).mean()  # the dashboard fix

fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(days, raw, color=GRAY, linewidth=1.3, label="raw daily column")
ax.plot(days, smoothed, color=palette[1], linewidth=3.0,
        label="7-day rolling average")
ax.set_xlabel("Day of the outbreak")
ax.set_ylabel("New cases")
ax.set_title("The raw column is a weekly-batch comb — the 7-day average recovers the epidemic curve")
ax.legend(frameon=False, loc="upper left")

ax.annotate("a whole week's cases\ndumped on one day,\nzeros in between",
            xy=(104.5, raw[104] if raw[104] > 0 else raw[103]),
            xytext=(52, 4300), fontsize=10, ha="center",
            arrowprops=dict(arrowstyle="->", color="black", linewidth=1.4))
ax.annotate("average cases per day\nover the last 7 days —\nthe true epidemic shape",
            xy=(112, float(smoothed[112])), xytext=(138, 2800),
            fontsize=10, ha="center", color=palette[1],
            arrowprops=dict(arrowstyle="->", color=palette[1], linewidth=1.4))
sns.despine(ax=ax)

fig.savefig("images/raw-vs-smoothed.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("wrote images/long-vs-wide.png, images/merge-diagram.png, images/raw-vs-smoothed.png")
