# scikit-learn cheat sheet

*One page for mid-exercise lookups. Taught across Modules 04–08 — see the [curriculum](../curriculum.md) for which module covers what, and the [glossary](../glossary.md) for any term.*

## The standard workflow (never changes)

```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# X = the feature table (measurements), y = the labels (answers)
X = df[["flipper_length_mm", "bill_length_mm"]]
y = df["species"]

# 1. SPLIT -- lock away a test set before any learning happens
#    random_state=42 makes the shuffle reproducible (same split every run)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=5)   # 2. choose a model
model.fit(X_train, y_train)                   # 3. FIT -- learn from training data only
predictions = model.predict(X_test)           # 4. PREDICT on unseen data
score = model.score(X_test, y_test)           # 5. SCORE (accuracy for classifiers, R^2 for regressors)
```

Every estimator in scikit-learn uses these same `.fit()` / `.predict()` / `.score()` methods — learn the workflow once, swap in any model.

## Which model for which problem?

| Problem | Model | Import | Notes |
|---|---|---|---|
| Classification ("which kind?") | k-nearest neighbors | `from sklearn.neighbors import KNeighborsClassifier` | Most intuitive; needs scaled features (Module 04) |
| Classification | Logistic regression | `from sklearn.linear_model import LogisticRegression` | Outputs probabilities; interpretable (Module 06) |
| Classification | Decision tree | `from sklearn.tree import DecisionTreeClassifier` | Readable flowchart; overfits alone (Module 06) |
| Classification | Random forest | `from sklearn.ensemble import RandomForestClassifier` | Strong default for tables; gives feature importances (Module 06) |
| Regression ("how much?") | Linear regression | `from sklearn.linear_model import LinearRegression` | Readable slope + intercept (Module 05) |
| Regression | Random forest | `from sklearn.ensemble import RandomForestRegressor` | When a straight line isn't enough |
| Clustering (no labels) | k-means | `from sklearn.cluster import KMeans` | You pick k; use the elbow plot (Module 08) |
| Dimensionality reduction | PCA | `from sklearn.decomposition import PCA` | Many features → 2-D map (Module 08) |
| Preprocessing | Standard scaler | `from sklearn.preprocessing import StandardScaler` | Mean 0, std 1 — inside a Pipeline (Module 07) |

## Metrics

```python
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay,
                             mean_absolute_error, root_mean_squared_error,
                             r2_score, RocCurveDisplay)

# --- classification (Module 06) ---
accuracy_score(y_test, predictions)          # fraction correct -- can lie on imbalanced data
confusion_matrix(y_test, predictions)        # the 2x2 truth-vs-prediction table
classification_report(y_test, predictions)   # precision + recall (=sensitivity) per class
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)   # drawn, not printed
RocCurveDisplay.from_estimator(model, X_test, y_test)          # the threshold trade-off curve
model.predict_proba(X_test)                  # probabilities, for moving the threshold yourself

# --- regression (Module 05) ---
mean_absolute_error(y_test, predictions)     # MAE: average miss, in real units (grams!)
root_mean_squared_error(y_test, predictions) # RMSE: big misses punished extra
r2_score(y_test, predictions)                # fraction of variance explained, 0..1
```

## Cross-validation (Module 07)

```python
from sklearn.model_selection import cross_val_score

# 5 rotated train/test splits -> 5 scores; report their mean and spread
scores = cross_val_score(model, X_train, y_train, cv=5)
print(scores.mean(), scores.std())
```

## Pipeline + GridSearchCV skeleton (Module 07)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# a Pipeline chains steps so the scaler is re-fitted on training folds only
# -- this is what makes data leakage impossible
pipe = Pipeline([
    ("scaler", StandardScaler()),          # step 1: put features on a common scale
    ("knn", KNeighborsClassifier()),       # step 2: the model
])

# hyperparameter grid: "stepname__parameter" (two underscores) names the knob
param_grid = {"knn__n_neighbors": [1, 3, 5, 11, 21]}

# try every value, scored by 5-fold cross-validation, keep the winner
search = GridSearchCV(pipe, param_grid, cv=5)
search.fit(X_train, y_train)

print(search.best_params_)                 # the winning settings
print(search.best_score_)                  # its cross-validated score
search.score(X_test, y_test)               # ONE final honest test -- then stop touching the test set
```

## The honest protocol, in one paragraph

Split once, at the very start. Explore, scale, tune, and cross-validate using the **training set only** (Pipelines make the scaling part automatic). Touch the test set exactly once — at the end — and report that number. If you go back and tweak after seeing the test score, the test set has silently become part of training.
