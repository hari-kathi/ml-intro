# Module 01 — NumPy Foundations

**Prerequisites:** basic Python (variables, lists, loops, functions).
**Dataset:** [Iris flower measurements](https://huggingface.co/datasets/scikit-learn/iris) from HuggingFace — 150 flowers from 3 species, 4 measurements each.

## Why NumPy, and why should a biologist care?

Almost every dataset you'll meet in computational biology is secretly a **matrix of numbers**:

- A gene expression experiment → rows = samples, columns = genes, values = expression levels
- A DNA sequencing run → rows = reads, columns = positions, values = quality scores
- A field study of penguins or flowers → rows = individuals, columns = measurements

Python lists are too slow and clumsy for this. **NumPy** (Numerical Python) gives you the `ndarray` — a grid of numbers that you can do math on *all at once*, without writing loops. Every serious tool you'll use later (pandas, scikit-learn, scanpy for single-cell RNA-seq, biopython) is built on top of NumPy arrays.

Our dataset is a classic from 1936: botanist Edgar Anderson measured sepals and petals of 150 iris flowers, and statistician R.A. Fisher used it to ask *"can measurements alone tell species apart?"* — which is exactly the question behind modern taxonomy, and behind machine learning classification.

## What you'll learn

1. **Arrays** — creating them, `shape`, `dtype`; a 150×4 matrix is "150 flowers × 4 phenotype measurements"
2. **Vectorization** — why `array.mean()` beats a `for` loop (we simulate 20,000 genes to see the speed difference)
3. **Indexing & slicing** — grab one flower (a row), one measurement (a column), or any block
4. **Boolean masking** — "give me only the *setosa* flowers" as a one-liner; this is the filtering idiom you'll use forever
5. **Vectorized math & broadcasting** — compute petal area for all 150 flowers in one expression; z-score standardization (the same normalization used on gene expression data)
6. **Aggregations** — `mean`, `min`, `max`, `argmax` along rows or columns
7. **Plots** — histograms and scatter plots showing that petal measurements almost perfectly separate the species

## How to work through it

Open the notebook and run each cell, in order, and *predict the output before running it*. Every non-obvious code line has a `#` comment right next to it explaining what it does — if a line looks mysterious, read its comment first.

```bash
source ../../.venv/bin/activate
jupyter lab notebook.ipynb
```

Then do [exercises.md](exercises.md) — easiest first.

## The one idea to remember

> **Don't loop over data — describe the operation on the whole array.**
> `petal_length * petal_width` computes 150 areas at once. This "think in arrays" habit is the core mental shift of scientific computing.
