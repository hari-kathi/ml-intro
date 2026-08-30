# Regenerates every figure in images/ used by README.md.
# Run with the course venv:  ../../.venv/bin/python make_figures.py
import matplotlib
matplotlib.use("Agg")            # draw to files, not to a window

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

sns.set_theme(style="whitegrid", palette="colorblind")
palette = sns.color_palette("colorblind")
BLUE, ORANGE, GREEN, RED = palette[0], palette[1], palette[2], palette[3]
GRAY = "#9aa0a6"

# ---------------------------------------------------------------- load penguins
# Same dataset and cleaning as Modules 02 and 04.
raw = load_dataset("SIH/palmer-penguins", split="train").to_pandas()
df = raw.dropna()

X = df[["flipper_length_mm"]]
y = df["body_mass_g"]
# Same split as the notebook, so the figure numbers match the notebook numbers.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

line_model = LinearRegression()
line_model.fit(X_train, y_train)
slope = line_model.coef_[0]           # ~50 g per mm
intercept = line_model.intercept_     # ~ -5849 g


# ============================================================================
# Figure 1: anatomy of a fitted line (slope as rise/run, intercept explained)
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 5.5))

ax.scatter(X_train["flipper_length_mm"], y_train,
           s=25, color=GRAY, alpha=0.55, edgecolor="white", linewidth=0.4,
           label="training penguins", zorder=2)

xs = np.linspace(170, 235, 100)
ax.plot(xs, slope * xs + intercept, color=ORANGE, linewidth=2.5,
        label="fitted line", zorder=3)

# --- slope triangle: run of +10 mm, rise of slope*10 grams -------------------
x0, run = 214.0, 10.0
y0 = slope * x0 + intercept
y1 = slope * (x0 + run) + intercept
ax.plot([x0, x0 + run], [y0, y0], color=BLUE, linewidth=2, zorder=4)          # run
ax.plot([x0 + run, x0 + run], [y0, y1], color=BLUE, linewidth=2, zorder=4)    # rise
ax.annotate("+10 mm of flipper", xy=(x0 + run / 2, y0 - 60),
            ha="center", va="top", fontsize=10, color=BLUE)
ax.annotate(f"+{slope * run:.0f} g of penguin\n(slope = {slope:.0f} g per mm)",
            xy=(x0 + run + 1.2, (y0 + y1) / 2),
            ha="left", va="center", fontsize=10, color=BLUE)

# --- intercept note ----------------------------------------------------------
ax.annotate(
    f"the line's equation:\nmass  =  {slope:.0f} × flipper  −  {abs(intercept):.0f}\n\n"
    f"the −{abs(intercept):.0f} is the intercept: where the line\n"
    "would cross flipper = 0 mm. No such penguin\n"
    "exists — it just anchors the line's height.",
    xy=(171, 5900), ha="left", va="top", fontsize=9.5,
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff6e6",
              edgecolor=ORANGE, linewidth=1))

ax.set_xlim(170, 235)
ax.set_xlabel("Flipper length (mm)")
ax.set_ylabel("Body mass (g)")
ax.set_title("Anatomy of a fitted line: every extra mm of flipper ≈ 50 g of penguin")
ax.legend(loc="lower right")
fig.savefig("images/fitted-line-anatomy.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ============================================================================
# Figure 2: residuals as vertical segments from each penguin to the line
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 5.5))

# A small random sample keeps the vertical segments readable.
sample = X_train.copy()
sample["body_mass_g"] = y_train
sample = sample.sample(n=30, random_state=42)

ax.plot(xs, slope * xs + intercept, color=ORANGE, linewidth=2.5,
        label="fitted line", zorder=3)

# Draw one vertical segment per penguin: from the point down/up to the line.
# Keep track of the largest POSITIVE residual so we can annotate it.
biggest_resid = None
for i in range(len(sample)):
    flip = sample["flipper_length_mm"].iloc[i]
    mass = sample["body_mass_g"].iloc[i]
    predicted = slope * flip + intercept
    ax.plot([flip, flip], [predicted, mass], color=RED, linewidth=1.4,
            alpha=0.8, zorder=2)
    if biggest_resid is None or (mass - predicted) > biggest_resid[2]:
        biggest_resid = (flip, mass, mass - predicted)

ax.scatter(sample["flipper_length_mm"], sample["body_mass_g"],
           s=45, color=BLUE, edgecolor="white", linewidth=0.6,
           label="penguins", zorder=4)

flip, mass, resid = biggest_resid
ax.annotate(
    f"residual = actual − predicted = {resid:+.0f} g\n"
    "this penguin is heavier than\nits flipper length suggests",
    xy=(flip, mass - resid / 2), xytext=(180, 5600),
    fontsize=9.5, ha="left", va="top",
    arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.2,
                    connectionstyle="arc3,rad=-0.15"),
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fdecea",
              edgecolor=RED, linewidth=1))

ax.set_xlim(178, 235)
ax.set_xlabel("Flipper length (mm)")
ax.set_ylabel("Body mass (g)")
ax.set_title("Residuals: the vertical gap between each penguin and the line")
ax.legend(loc="lower right")
fig.savefig("images/residuals-vertical-errors.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# ============================================================================
# Figure 3: underfit / just-right / overfit triptych (synthetic chick growth)
# ============================================================================
# Synthetic 'penguin chick growth' data: a smooth S-shaped growth curve plus
# measurement noise. The truth is curved, so degree 1 genuinely underfits.
rng = np.random.default_rng(42)

def growth_curve(age):
    # A logistic (S-shaped) growth curve: chicks plateau near 3200 g.
    return 200.0 + 3000.0 / (1.0 + np.exp(-(age - 26.0) / 4.0))

age_train = np.sort(rng.uniform(1, 54, 25))
mass_train = growth_curve(age_train) + rng.normal(0, 150, 25)
age_test = np.sort(rng.uniform(1, 54, 60))
mass_test = growth_curve(age_test) + rng.normal(0, 150, 60)

Xg_train = age_train.reshape(-1, 1)   # sklearn wants a 2-D column
Xg_test = age_test.reshape(-1, 1)
grid = np.linspace(1, 54, 300).reshape(-1, 1)

titles = {1: "Degree 1 — underfit\n(too rigid: misses the curve)",
          3: "Degree 3 — just right\n(follows the trend, ignores the noise)",
          12: "Degree 12 — overfit\n(chases every noisy point)"}
colors = {1: BLUE, 3: GREEN, 12: RED}

fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
degrees = [1, 3, 12]
for k in range(3):
    deg = degrees[k]
    ax = axes[k]
    model = make_pipeline(StandardScaler(), PolynomialFeatures(deg),
                          LinearRegression())
    model.fit(Xg_train, mass_train)
    train_mae = mean_absolute_error(mass_train, model.predict(Xg_train))
    test_mae = mean_absolute_error(mass_test, model.predict(Xg_test))

    ax.scatter(age_train, mass_train, s=40, color="#444444",
               edgecolor="white", linewidth=0.6, zorder=3,
               label="training chicks")
    ax.plot(grid, model.predict(grid), color=colors[deg], linewidth=2.5,
            zorder=2)
    ax.set_title(titles[deg], fontsize=11)
    ax.set_xlabel("Chick age (days)")
    ax.text(0.03, 0.97,
            f"train error: {train_mae:.0f} g\ntest error:  {test_mae:.0f} g",
            transform=ax.transAxes, ha="left", va="top", fontsize=10,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=colors[deg], linewidth=1.2))
    ax.set_ylim(-300, 3900)
axes[0].set_ylabel("Chick mass (g)")
fig.suptitle("Same 25 data points, three polynomial models — "
             "flexibility helps, then it memorizes", y=1.03, fontsize=13)
fig.savefig("images/overfit-triptych.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("Wrote 3 figures to images/")
