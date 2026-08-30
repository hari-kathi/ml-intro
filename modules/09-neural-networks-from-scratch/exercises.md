# Module 09 — Exercises

Work in a fresh cell at the bottom of `notebook.ipynb` — everything the notebook built (`X`, `y`, `Y`, `sigmoid`, `bce_loss`, `forward`, `backward`, `update`, `train_network`, the trained `W1, b1, W2, b2`, and the plotting grid `grid_x`, `grid_y`, `grid_points`) is still in memory. Easiest first.

## 1. One neuron, by hand

Using section 2's hand-picked neuron (`w1 = 1.0`, `w2 = -1.0`, `b = 0.0`):

- Compute (with plain arithmetic, before checking in code) the weighted sum and firing probability for the input `x1 = 1.0, x2 = 1.0`. Why does this input leave the neuron exactly on the fence?
- Now set `b = 2.0` and redo input B (`x1 = 0.0, x2 = 3.0`). In neuron terms, what did raising the bias do to the firing threshold?
- Find weights `w1, w2` (keep `b = 0`) that make the neuron fire for input B but stay quiet for input A. Which synapse did you have to make excitatory, and which inhibitory?

## 2. Reading the sigmoid

- Without running code: rank these weighted sums by firing probability: `-2, 0, 0.5, 4`. Then check with `sigmoid(...)`.
- The sigmoid never outputs exactly 0 or exactly 1. Why is that a *feature* for the loss in section 6? (Hint: what is the surprise penalty for a model that is *certain* and wrong?)

## 3. Bowl experiments

Rerun the gradient-descent loop from section 7, changing one thing at a time:

- Start from `w = 8.0` instead of `-1.5`. Does the ball still find `w = 3`? Does it approach from the other side?
- Try `learning_rate = 1.0` exactly. The slope of the bowl is `2*(w - 3)`, so work out what one update does to the distance from 3 — then run it and explain the strange path you see (the ball neither settles nor escapes).
- Find, by trial, roughly the largest learning rate that still converges on this bowl.

## 4. Surprise accounting

- Use `bce_loss` to compute the loss of a "coin-flip model" that predicts 0.5 for every one of the 400 points (`np.full((400, 1), 0.5)` builds that array — `np.full` fills a given shape with one value). Compare it to the untrained network's starting loss from section 8. Was our random newborn network better than a coin flip?
- The trained network's loss was about 0.15. Using the section 6 table (penalty for p = 0.99, 0.9, ...), what rough probability is the network typically giving the true class?

## 5. Committee size (hidden layer width)

`train_network` re-creates the network from scratch, but its width is fixed by the global `n_hidden`. Make the width a proper argument: copy `train_network` into a new function `train_network_width(chosen_learning_rate, n_epochs, width)` and replace `n_hidden` with `width` inside it (also return the trained weights: `return loss_history, final_acc, tW1, tb1, tW2, tb2`).

- Train with `width = 1`, learning rate 1.0, 2000 epochs. What accuracy do you get, and why should a one-neuron committee land near the straight line's 84%?
- Train with `width = 2, 4, 16`. Print a small table of width vs accuracy. Where does adding neurons stop paying off on this dataset?

## 6. Harder and easier moons

Generate new versions of the dataset with `make_moons(n_samples=400, noise=..., random_state=42)`, keeping everything else the same:

- With `noise = 0.05` (nearly clean crescents), retrain and report accuracy. Close to 100%?
- With `noise = 0.45`, retrain. How much do the moons still matter, biologically speaking — at what noise level does the "population structure" effectively disappear into measurement error?
- Careful: `train_network` uses the *global* `X` and `Y`, so assign your new data to `X` and `Y` before calling it (and restore the originals afterward if you keep working).

## Challenge: the committee-size filmstrip

Tell the whole capacity story in one figure. Using your `train_network_width` from exercise 5, train four networks with widths **1, 2, 4, 16** (learning rate 1.0, 2000 epochs). Make a 2×2 panel figure — reuse the background-coloring recipe from section 10 with the returned weights of each network — where each panel shows that network's final decision boundary over the moons, titled with its width and accuracy.

You should see a mini evolution: width 1 draws a near-straight line, width 2 manages a single bend, width 4 hugs the moons, and width 16 looks barely better than 4 — a picture of why more capacity helps only until the model matches the shape of the data. Give the figure a `suptitle` that states that finding.
