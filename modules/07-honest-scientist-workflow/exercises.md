# Module 07 — Exercises

Work in a fresh cell at the bottom of `notebook.ipynb` — the variables the lesson built (`X`, `y`, `df`, `pipe`, `cv`, `param_grid`) are all still alive there. Easiest first.

## 1. More replicates

The lesson used 5-fold cross-validation. Rerun `cross_val_score(pipe, X, y, cv=...)` with a `StratifiedKFold` using `n_splits=10` (keep `shuffle=True, random_state=42`).

- Compare the mean and spread (`.std()`) to the 5-fold result from the lesson.
- With 10 folds, what fraction of the data does each round train on — more or less than with 5 folds? Why might that nudge the scores slightly upward?

## 2. Who's shouting the loudest?

Before scaling, the distance calculation was dominated by whichever columns have the biggest numbers.

- Run `X.std().sort_values()` (sorts the columns by their spread). Which column has the largest spread? The smallest?
- Roughly how many times bigger is the largest than the smallest? (Divide them.)
- In one sentence: what does this ratio mean for an *unscaled* KNN's distance calculation?

## 3. Spot the contamination

A labmate shows you this code and proudly reports the accuracy. **Do not run it** — read it like you'd inspect someone's bench technique:

```python
scaler = StandardScaler()
X_ready = scaler.fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X_ready, y, test_size=0.2,
                                          random_state=0, stratify=y)
model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_tr, y_tr)
print("accuracy:", model.score(X_te, y_te))
```

- In one or two sentences: where exactly is the leak, and what information seeped where?
- Now write the sterile version using a `Pipeline` (three or four lines — the lesson's Section 6 is your template) and run *that*.

## 4. Screen a different model

KNN isn't the only classifier in town. Build a new pipeline with `LogisticRegression` (from `sklearn.linear_model`) as the model step — name the step `"logreg"` and create it with `LogisticRegression(max_iter=5000)` (that just gives its fitting procedure enough iterations to settle).

- Logistic regression's main hyperparameter is `C`, which controls flexibility (bigger C = more flexible). Run a `GridSearchCV` over `{"logreg__C": [0.001, 0.01, 0.1, 1, 10, 100]}` with `cv=cv`.
- Print `.best_params_` and `.best_score_`. Does the best logistic regression beat the best KNN from the lesson (~96.7%)?

## 5. The 20-peeks problem

No coding — lab-meeting reasoning. A collaborator locks away a test set (good!), tunes on the training data (good!), evaluates on the test set… gets 94%, is disappointed, tweaks the model, evaluates again, and repeats about 20 times until the test score reads 97%. They report 97%.

- Explain, in two or three sentences, why 97% is not an honest estimate — even though every individual evaluation used only "unseen" data.
- Which wet-lab sin is this most like: contaminating a culture, or running the experiment until it "works" and only publishing that run?

## 6. Challenge: the validation curve for `C`, telling one story

Make the validation-curve figure for your logistic-regression pipeline from exercise 4 (Section 8 of the notebook is your template — swap in `param_name="logreg__C"` and `param_range=[0.001, 0.01, 0.1, 1, 10, 100, 1000]`).

- Plot training and cross-validation accuracy against `C` with `ax.set_xscale("log")`.
- You should see all three zones from the README figure: both curves low on the left (underfit), a clear peak in the middle, and a widening train-vs-CV gap on the right (overfit). Mark the sweet spot with a star.
- Give the plot a title that states the finding, not the variables — then, in a final sentence, report the *honest* protocol you would now run before publishing any accuracy for this model (three steps — the lesson's Section 9).
