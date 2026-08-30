# Module 06 — Exercises

Work in fresh code cells at the bottom of `notebook.ipynb`, so you can reuse everything the notebook already built: the cleaned table `df`, the split (`X_train`, `X_test`, `y_train`, `y_test`), the fitted models (`logreg`, `tree`, `forest`), and the test-set probabilities (`probs` for logistic regression, `forest_probs` for the forest). Easiest first.

## 1. Read a matrix by hand (no code — paper and pencil)

A rapid antigen test is trialed on 1,000 people. The truth (from PCR) is that 80 of them are infected. The rapid test comes back positive for 56 of the infected people, and also for 19 people who are *not* infected.

1. Draw the 2×2 confusion matrix (rows = truth, columns = test result), filling in all four cells. Careful: two of the numbers you need aren't stated directly — you have to work them out.
2. Compute the sensitivity and the specificity.
3. In one sentence: is this test better suited as a screening test or as a confirmatory test, and why?

## 2. The chest-pain paradox

Using `df`, compute the fraction of patients with disease within each `chest_pain_type` (Module 02's `groupby` is perfect for this). Remember from the glossary: types 1–3 are kinds of chest pain, and type 4 means **no chest pain at all**.

Which group has by far the highest disease rate? Write one sentence of biological speculation about why the patients who *felt nothing* were the most likely to be sick. (This is a famous quirk of this dataset — think about *who gets sent for a full cardiac workup despite having no symptoms*.)

## 3. An even more aggressive screen

The notebook tried thresholds 0.3, 0.5, and 0.7 on the logistic-regression probabilities `probs`. Now try **0.2**:

1. Build the predictions with `(probs >= 0.2).astype(int)` and print the confusion matrix.
2. Compute sensitivity and specificity, and count how many sick patients are still sent home.
3. The notebook's threshold-0.3 test missed 5 sick patients. How many does 0.2 miss, and what did each newly caught patient "cost" in extra false alarms?

## 4. When can you trust a positive result?

Precision is the trustworthiness of a positive: of the patients flagged, how many were really sick? Using `precision_score`, compute the precision of the logistic model at thresholds 0.2 and 0.5.

You should see precision *drop* as the threshold falls. Explain the pattern in one sentence, using the threshold-slider figure from the README: what happens to the false-positive region as the line slides left?

## 5. Watch a tree overfit, depth by depth

The notebook compared a depth-3 tree with an unlimited one. Fill in the picture between them: with a plain `for` loop over the depths 1, 2, 3, 5, 7, and 10, fit a `DecisionTreeClassifier(max_depth=depth, random_state=42)` on the training data and print the depth, the train accuracy, and the test accuracy on one line.

At which depth does the gap between train and test accuracy start to blow up? What is the tree doing with all those extra questions, in Module 05's vocabulary?

## 6. Give the forest a fair fight

The notebook's report card shows the random forest with the best AUC but poor sensitivity — at the *default* threshold. Repeat the notebook's threshold treatment using `forest_probs` with a threshold of 0.3: confusion matrix, sensitivity, specificity.

Does the retuned forest now beat logistic-regression-at-0.3 at catching sick patients? Update your answer to "which model would you deploy?" — did it change?

## Challenge: the threshold menu, drawn for the cardiologist

Make the one figure this module has been building toward: **sensitivity and specificity as two curves, across every threshold**, so a clinician could pick a cutoff by eye.

1. Build a list of thresholds from 0.05 to 0.95 in steps of 0.05 (a plain `for` loop appending to a list is fine, or `np.arange(0.05, 1.0, 0.05)` — `arange` gives evenly spaced values, like `linspace` but you choose the step size).
2. For each threshold, compute sensitivity and specificity of the logistic model from `probs` (the notebook's threshold loop is your template) and collect them in two lists.
3. Plot both curves against the thresholds on one set of axes — sensitivity in one color, specificity in another, labeled with `ax.legend()`.
4. Add a vertical dashed line at the default 0.5, and another at the threshold *you* would choose for a first-line screening clinic.
5. Give the plot a title that states your recommendation as a finding — not "sensitivity vs threshold" but something like "Screening at 0.3 catches three-quarters of disease with few false alarms".

Sanity check: the two curves should cross somewhere — sensitivity falling as the threshold rises, specificity climbing. If yours don't, check which side of `>=` your predictions are on.
