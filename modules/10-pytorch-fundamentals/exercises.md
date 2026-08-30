# Module 10 — Exercises

Work in a fresh cell at the bottom of `notebook.ipynb`, so everything the lesson built (`X_moons_t`, `moons_model`, `heart_loader`, `train_losses`, `best_epoch`, ...) is still loaded. Easiest first.

One tip that applies to almost every exercise here: when you rerun an experiment, **rebuild the model and the optimizer from scratch** (and call `torch.manual_seed(42)` just before, for fair comparisons). A model keeps its trained weights forever — retraining an already-trained network is a different experiment than training a fresh one.

## 1. Tensor warm-up

- Make a tensor holding the expression levels of five genes: `[12.1, 0.4, 88.0, 3.5, 41.2]`. Print its `shape`, `dtype`, and `mean()`.
- Create `counts = np.array([10.0, 20.0, 30.0])` and convert it to a tensor **the wrong way** (`torch.from_numpy(counts)`). Print the dtype — why is it not `float32`?
- Now convert it the right way and confirm the dtype. (The pattern is in Section 1.)

## 2. Autograd, checked by hand

- Before running anything: for `loss = (w − 5)²` at `w = 1`, what should the gradient be? Write your answer down (chain rule: `2 × (w − 5)`).
- Now build `w = torch.tensor(1.0, requires_grad=True)`, compute the loss, call `.backward()`, and check `w.grad` against your paper answer.
- Same game with a new expression: `loss = w**2 + 3*w` at `w = 2.0`. (Derivative by hand: `2w + 3`.) Build a **fresh** `w` for this one — gradients accumulate onto old tensors, the same reason the liturgy needs `zero_grad()`.

## 3. A bigger committee

The moons network used 8 hidden neurons. Build `big_model` with **32**, using the same three-part `nn.Sequential` recipe.

- Count its parameters with the loop from Section 3. Predict the number before you run (layer 1: 32×2 + 32, layer 2: 1×32 + 1).
- Train it with the exact liturgy from Section 4 (copy the loop, swap in `big_model` and a fresh optimizer).
- Reuse the decision-boundary code from Section 4 on `big_model`. Does the bigger committee draw a smoother curve, or start tracing individual noise points?

## 4. Learning-rate clinic

Retrain the 8-neuron moons network twice from scratch (fresh model + fresh `SGD` optimizer each time, 1500 epochs each):

- once with `lr=0.001`
- once with `lr=5.0`

Record the losses of each run in its own list, plot both curves on one figure with a `label=` for each and `ax.legend()`. Diagnose each curve using the loss-curve anatomy figure in the README: which one is the crawl, which one is the fever chart? What was Module 09's name for the `lr=5.0` failure?

## 5. Batch-size experiment

Rebuild and retrain the heart model twice (fresh model, fresh Adam optimizer, 200 epochs, tracking `train_losses` like Section 6), changing only the `DataLoader`:

- `batch_size=4` (57 tiny batches per epoch)
- `batch_size=225` (the whole training set — one batch per epoch)

Plot the two training-loss curves together. Which is jumpier, and why? (Hint: a batch of 4 patients gives a noisy estimate of the true downhill direction — like judging an epidemic from 4 cases.)

## 6. Early stopping, for real

The lesson trained for 200 epochs and *watched* overfitting happen; now avoid it. Retrain a fresh heart model, but run the epoch loop only up to `best_epoch` (the low point of the test-loss curve that Section 6 found). Compute the test accuracy of this early-stopped model the same way Section 6 did. Did stopping early beat the 200-epoch model's 84% — or just tie it? (Either result is worth a sentence: what does the *loss* see that *accuracy* doesn't? Section 6's markdown has the clue — confidence.)

## Challenge: does more data actually help?

The lesson claimed "deep learning shines with lots of data." Test the claim in reverse: shrink the data and watch both models suffer — then see who suffers *more*.

For each training-set size `n` in 50, 100, 150, 225:

1. Slice the first `n` training patients: `X_train_t[:n]`, `y_train_t[:n]` for the network; `X_train[:n]`, `y_train[:n]` for the forest. (The split already shuffled the patients, so the first `n` are a random sample.)
2. Train a **fresh** heart-shaped network on that slice — use the Section 6 recipe but only up to `best_epoch` epochs (you just built early stopping; use it). A plain full-batch loop like Section 4's is fine here — no DataLoader needed.
3. Train a fresh `RandomForestClassifier(n_estimators=300, random_state=42)` on the same slice.
4. Record both models' accuracy on the **same untouched test set** (`X_test_t` / `X_test` — it never shrinks; the exam stays the exam).

Plot the two accuracy-versus-`n` lines on one figure — network in one color, forest in another, `label=` and legend, axes labeled — and give it a title that states your finding (not "accuracy vs n"!). Then answer like a biologist reading a dose-response curve: are the lines still climbing at 225, or flattening? Sketch (mentally or on the plot) where each line might sit at n = 30,000 — that extrapolation is exactly the bet Module 11 is about to place.
