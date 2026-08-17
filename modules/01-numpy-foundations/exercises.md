# Module 01 — Exercises

Work in a fresh notebook cell at the bottom of `notebook.ipynb` (the arrays `X`, `species`, and `feature_names` are already loaded). Easiest first.

## 1. Warm-up: reading an array

- What is the sepal width (column index 1) of the 10th flower (row index 9)?
- Print the last 5 rows of `X` using slicing.
- How many flowers have a petal length greater than 5 cm? (Hint: a boolean mask, then `.sum()`.)

## 2. Species detective

- Compute the **mean of all four measurements** for *Iris-virginica* only, in one line.
- Which species has the **widest sepals** on average?
- What is the **standard deviation** of petal length within each species? Notice which species is most "uniform" — why might a botanist find that interesting?

## 3. New measurements from old

- Create `sepal_area = sepal_length * sepal_width` for all flowers.
- Create a "petal-to-sepal ratio" array: `petal_length / sepal_length`. What is its range (`min` to `max`)?
- Plot a histogram of the ratio, colored by species (copy the histogram code from the lesson and adapt it). Does this *derived* measurement separate species better or worse than raw petal length?

## 4. Find the outlier

- Use `argmin` to find the flower with the **smallest petal area**. Which species is it? Print its full row.
- Z-score the petal lengths: `(x - x.mean()) / x.std()`. How many flowers are more than 2 standard deviations from the mean? In biology, would you call these outliers or natural variation?

## 5. Challenge: a one-rule classifier (preview of machine learning)

Fisher's question: can a single measurement identify a species?

- Write a function `guess_species(petal_length)` that returns `"Iris-setosa"` if petal length < 2.5, `"Iris-versicolor"` if < 4.9, else `"Iris-virginica"`.
- Apply it to all 150 flowers (a plain `for` loop is fine here) and compute the **accuracy**: what fraction of guesses match `species`?
- You just built a decision rule by eye. In a later module, scikit-learn will learn these thresholds automatically — from the data.
