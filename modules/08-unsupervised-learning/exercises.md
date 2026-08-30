# Module 08 — Exercises

Work in a fresh cell at the bottom of `notebook.ipynb`, so all the variables the notebook built (`scaled`, `hidden_diagnosis`, `coords`, `clusters_all`, `sample60`, and friends) are still alive. Easiest first.

## 1. A different pair of eyes

Section 3 clustered on `radius_mean` and `texture_mean`. Do the same thing with two *different* measurements: `smoothness_mean` and `concave points_mean` (note the space in that second column name — copy it exactly).

- Make the gray "no labels" scatter first. Do you see two groups this time?
- Fit a `KMeans(n_clusters=2, n_init=10, random_state=42)` on just those two columns of `scaled` and re-draw the scatter colored by cluster, with the centers as X marks.
- Which pair of measurements separates the tumors more cleanly — this one or the notebook's? (Just judge by eye; that's allowed.)

## 2. Stretch the elbow

The notebook's elbow plot stopped at k=8.

- Re-run the inertia loop for k from 1 to 12 (still on all 30 columns of `scaled`) and plot it.
- Does any *second* elbow appear after k=2, or is it a smooth slide all the way down?
- In one sentence: why can inertia never go *up* when k increases?

## 3. A third tumor type?

Suppose a colleague insists there are really *three* kinds of tumor in this cohort.

- Fit `KMeans(n_clusters=3, n_init=10, random_state=42)` on `scaled`.
- The envelope is already open, so you're allowed to peek now: run `pd.crosstab(hidden_diagnosis, ...)` on the new labels.
- Describe what the third cluster did: did it carve out its own group of benign tumors, its own group of malignant ones, or a mixed borderline zone? Does the crosstab support or undermine your colleague's claim?

## 4. Does the tree care how you build it?

Phylogenetics has neighbor-joining, UPGMA, maximum likelihood — and they don't always agree. Hierarchical clustering has the same situation: `method="ward"` is only one **linkage rule**.

- Rebuild the dendrogram of `sample60` with `method="average"` (fuse the two groups whose *average* pairwise distance is smallest — this one actually is UPGMA).
- Plot it next to (or below) the Ward version. Does the deep two-family split survive? Do any tumors switch families or dangle as loners?
- Moral to write down in one sentence: how much should you trust a cluster that only one linkage method finds?

## 5. What is PC1 made of?

PC1 is a recipe: a weighted mix of all 30 measurements. The weights live in `pca.components_[0]` — one number per measurement, in the same order as `scaled.columns` (positive = pushes a tumor right on the map, negative = pushes it left).

- Put the weights in a Series so they're labeled: `weights = pd.Series(pca.components_[0], index=scaled.columns)`.
- Sort it with `.sort_values()` and draw a horizontal bar chart (`weights.sort_values().plot.barh(figsize=(7, 9))`).
- Which measurements dominate PC1? Does that fit the notebook's claim that PC1 is roughly "overall nucleus size and irregularity"?

## Challenge — the borderline tumors

The finale said the ~54 disagreements are the interesting part. Find them and put them on the map.

1. Build a True/False array marking the disagreeing tumors. The benign-heavy cluster is 0 and the malignant-heavy cluster is 1, so a tumor "disagrees" when it's a `"B"` in cluster 1 or an `"M"` in cluster 0. (Hint: build two masks and combine them with `|`, which means OR.)
2. Draw the PCA map (PC1 vs PC2) with all agreeing tumors in light gray, then scatter the disagreeing tumors on top — benign-called-malignant in one bold color, malignant-called-benign in another. Add a legend and a title that states the finding.
3. Look at where the disagreements live. Are they scattered randomly, or concentrated in the border zone between the two lobes? Write two or three sentences interpreting that as a biologist: what would you do next with these specific patients' slides?

A perfect answer to this challenge is a single figure a pathologist would lean in to look at.
