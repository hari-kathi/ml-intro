# Module 05 — Exercises

Work in a fresh cell at the bottom of `notebook.ipynb`, after running the whole notebook top to bottom — the exercises reuse the variables it built (`df`, `X_train`, `X_test`, `y_train`, `y_test`, `line_model`, and friends).

## 1. Be the model

A penguin's flipper measures **210 mm**.

- Using only the two numbers the notebook printed — the slope and the intercept — compute the model's mass prediction with plain arithmetic (`slope * 210 + intercept`).
- Now ask the model itself, the way the notebook did in section 4:

  ```python
  new_penguin = pd.DataFrame({"flipper_length_mm": [210]})
  line_model.predict(new_penguin)
  ```

  Do the two answers agree? (They must — a linear model *is* just that arithmetic.)

## 2. A worse ruler

Repeat the single-feature workflow from sections 3–6, but predict body mass from **`bill_length_mm`** instead of flipper length: split (keep `random_state=42`), fit a new `LinearRegression`, and compute the test-set MAE and R².

- How many grams off is a typical prediction now, compared with the ~290 g the flipper achieved?
- Which measurement would you tell a field team to prioritize photographing, and why?

## 3. Interrogate the residuals by species

The notebook hinted that the residual cloud hides structure. Expose it:

- Compute the training residuals again (`y_train - line_model.predict(X_train)`).
- Put them in a small table next to each penguin's species. The training penguins' species labels line up like this:

  ```python
  resid_table = df.loc[X_train.index].copy()   # the training rows, with all their columns
  resid_table["residual"] = y_train - line_model.predict(X_train)
  ```

- Use `groupby("species")["residual"].mean()` (Module 02!) to get the average residual per species.

One species is systematically over-estimated by about 190 g — the line thinks those penguins should be heavier than they are. Which species is it? As a biologist, what is the flipper-only model missing about that species' build?

## 4. Every species gets its own allometry

Fit a `LinearRegression` of mass on flipper length using **only the Gentoo penguins** (filter `df` first, like Module 02). Compare its slope to the all-species slope of ~50 g/mm.

- Is "grams per millimeter of flipper" the same exchange rate within one species as across all three?
- Why might a mixed-species slope differ from a within-species slope? (Hint: think about what *else* changes between species besides flipper length.)

## 5. Push the overfitting demo further

The notebook compared polynomial degrees 1, 3, and 12 on the 25-penguin sample. Re-run the same loop for degrees **5** and **20** as well (reuse `small_X`, `small_y`, and the pipeline recipe from section 8).

- Does training error keep falling as the degree rises?
- Does test error ever come back down?

## Challenge: the overfitting curve

Turn exercise 5 into the classic picture. For every degree from 1 to 12:

1. Fit the pipeline (`StandardScaler` → `PolynomialFeatures(degree)` → `LinearRegression`) on the 25-penguin sample.
2. Record the training MAE and the test MAE in two lists (start with two empty lists and `.append(...)` inside the loop).

Then plot both lists against degree on one figure — training error as one line, test error as another, with a legend, axis labels, and a title that states the finding (something like "Training error always flatters; test error tells the truth"). You may need `ax.set_yscale("log")` — a log y-axis — because the worst test errors are enormous.

The shape you get — training error sliding ever downward while test error swings up like a hockey stick — is the single most reproduced diagram in machine learning. You just drew it from real penguins.
