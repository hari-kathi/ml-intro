# Module 02 — Pandas Essentials

**Prerequisites:** [Module 01 — NumPy Foundations](../01-numpy-foundations/README.md).
**Dataset:** [Palmer Penguins](https://huggingface.co/datasets/SIH/palmer-penguins) from HuggingFace — 344 penguins of 3 species measured by field biologists at Palmer Station, Antarctica (2007–2009).

## From arrays to DataFrames

NumPy arrays are perfect when everything is a number. But real field data is messier: this dataset mixes **numbers** (bill length, body mass), **categories** (species, island, sex), and — because it's *real* fieldwork — **missing values** (some penguins wouldn't sit still for every measurement).

**Pandas** gives you the `DataFrame`: a table where every column is labeled and can have its own type, like a lab spreadsheet that speaks Python. Underneath, each column is still a NumPy array — everything from Module 01 still applies.

The Palmer Penguins data is genuine ecology: researchers from the Palmer Station Long-Term Ecological Research program measured Adélie, Chinstrap, and Gentoo penguins across three islands. The kind of table you'd produce in any biology field course — which is exactly why it's a great practice dataset.

## What you'll learn

1. **Loading data** — from HuggingFace with `load_dataset(...)`, then `.to_pandas()`
2. **First-look ritual** — `head()`, `shape`, `info()`, `describe()`: the four commands to run on *every* new dataset before anything else
3. **Selecting** — columns by name, rows with `loc`/`iloc`
4. **Filtering** — boolean masks again, now with readable column names (`df[df["species"] == "Gentoo"]`)
5. **Missing data** — finding it (`isna`), counting it, deciding what to do about it (`dropna` vs `fillna`), and why "just delete it" is a scientific decision, not a technical one
6. **Counting categories** — `value_counts` and cross-tabulation (which species lives on which island?)
7. **New columns** — computed from old ones (bill length ÷ bill depth, a shape ratio taxonomists actually use)
8. **Group-by** — `groupby("species").mean()`: one line replaces the mask-per-species loops from Module 01. This "split → apply → combine" pattern is the single most useful thing in pandas
9. **Plots** — grouped bar charts, box plots, and a scatter plot that reveals the species clusters

## How to work through it

```bash
source ../../.venv/bin/activate
jupyter lab notebook.ipynb
```

Run each cell in order, predicting outputs first. Then do [exercises.md](exercises.md).

## The one idea to remember

> **`groupby` is "split → apply → combine".**
> Split the penguins by species, apply `mean()` to each group, combine the results into a new table. Once you see this pattern, half of data analysis becomes a one-liner.
