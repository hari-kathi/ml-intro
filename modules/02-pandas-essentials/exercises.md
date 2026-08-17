# Module 02 — Exercises

Work in a fresh cell at the bottom of `notebook.ipynb` (`df` is already loaded and cleaned). Easiest first.

## 1. First-look ritual

- How many penguins are in the cleaned dataset, and how many columns?
- What is the mean body mass in **kilograms**?
- Use `value_counts()` on the `sex` column. Is the dataset balanced?

## 2. Filtering practice

- Select all penguins on the island of Dream. How many are there?
- Select all female Chinstrap penguins with body mass above 3,500 g.
- What fraction of Gentoo penguins have a flipper length over 220 mm? (Filter, then divide counts.)

## 3. Group-by practice

- Mean flipper length per species — which species has the longest flippers?
- Mean body mass per species **and** sex (one `groupby` with two columns). Within each species, roughly how much heavier are males?
- Per island: how many penguins and what's their average bill length? (Hint: `.agg(["count", "mean"])`.)

## 4. Derived measurements

- The lesson created `bill_ratio`. Compute its mean per species. Could you identify a species from its bill ratio alone?
- Add a column `mass_kg` (body mass in kilograms) and a boolean column `is_heavy` (True if mass is above the overall median).
- Cross-tabulate `species` vs `is_heavy`. What does this tell you about Gentoos?

## 5. Missing-data reasoning

Reload the **raw** (uncleaned) dataset into a new variable.

- Which columns have missing values, and how many each?
- Drop only rows where `body_mass_g` is missing, keeping rows that merely lack `sex`. How many rows survive, compared to `dropna()` on everything?
- In one or two sentences: for a study of *mass differences between males and females*, which of those two cleaning choices is correct, and why?

## 6. Challenge: tell the story in one figure

Make a single scatter plot of bill length vs bill depth, colored by species (adapt the lesson's scatter code).

- The three species form three clouds — an "all-in-one" morphological ID chart.
- Add a title that states the finding, not the variables (e.g. "Bill shape alone nearly identifies the species").
- Bonus: mark the point for the single heaviest penguin in the dataset with a star (`ax.scatter(..., marker="*", s=300)`), and check: which cloud is it in?
