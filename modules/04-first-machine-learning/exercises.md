# Module 04 — Exercises

Work in a fresh cell at the bottom of `notebook.ipynb` — everything the notebook built (`df`, `X_train`, `knn`, `scaler_all`, `knn_all`, ...) is still alive there. Easiest first.

## 1. Reading the exam results

- The bills-only model (`knn`) scored 0.970 on the test set. Using `len(y_test)`, how many penguins is that exam, and how many did the model get wrong?
- Compute `knn.score(X_train, y_train)` minus `knn.score(X_test, y_test)`. Now fit a fresh `KNeighborsClassifier(n_neighbors=1)` on the same data and compute the same gap. Which model is the "memorizing student", and how does the gap tell you?

## 2. Invent a penguin

- Make up a penguin of your own (a `pd.DataFrame` with the same four columns as `new_penguin`) and run it through `scaler_all.transform` and `knn_all.predict`. What's the verdict?
- Use `predict_proba` to get the vote. Adjust your penguin's measurements until you find one that wins **5 votes to 0** (deep inside a species' territory — the Module 02 group means per species are a good cheat sheet).
- Now hunt for the opposite: a penguin that gets a **split vote** like 3–2 or 2–2–1. Where must such a penguin sit on the species maps?

## 3. Other feature pairs

The notebook used bill length + bill depth. Try a k=5 model on **bill length + flipper length** instead: build the new `X`, split with the same `test_size=0.2, random_state=42, stratify=y` recipe, fit, and score.

- Better or worse than the bills pair (0.970)?
- Both features are in millimeters — do you expect scaling to change the result much here? Check your prediction with `StandardScaler`.

## 4. How lucky was our split?

`random_state=42` is one particular shuffle. Fit and score the bills-only k=5 model for `random_state` values 0, 1, 2, 3, 4 (a loop that redoes split → fit → score each time, printing the test accuracy).

- How much does the "exam grade" wobble between shuffles?
- In one or two sentences: why does this wobble mean you should never brag about a single accuracy number to the third decimal place? (Module 07 fixes this properly with cross-validation.)

## 5. Break the model on purpose

Sabotage distance: make a copy of `X` (bills only) where bill depth is expressed in **centimeters** (`df["bill_depth_mm"] / 10`) while bill length stays in millimeters. Split with the usual recipe, fit k=5, and score.

- How much accuracy did the unit change destroy?
- Explain the damage in one sentence using the word "distance" — and then repair it with `StandardScaler` and confirm the repair worked.

## Challenge: the overfitting curve

The notebook looked at k = 1, 5, 51. Now sweep every odd k from 1 to 51 (odd values avoid tied votes: `range(1, 52, 2)` counts 1, 3, 5, ... — the third number is the step size). For each k, fit a bills-only model and record **both** the training accuracy and the test accuracy in two lists.

Plot both curves against k on one figure — training accuracy in one color, test accuracy in another, with a legend and labeled axes.

- Left edge of the plot: training accuracy is perfect but test accuracy isn't — memorization.
- Right edge: both curves sag together — over-averaging.
- Give the figure a title that states the finding, not the variables (something like "Perfect training scores don't survive the exam").
- Bonus: mark the k with the best *test* accuracy using `ax.axvline(..., linestyle="--")` (a vertical dashed line). Is our k=5 close to it?
