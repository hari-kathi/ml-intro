# Module 11 — Exercises

Work in a fresh cell at the bottom of `notebook.ipynb`, after running the whole notebook once — the exercises reuse variables the notebook already built (`mnist`, `X_train`, `X_test`, `y_test`, `mlp`, `cnn`, `convolve`, `accuracy`, `make_cnn`, `train_model`, `test_predictions`, ...). Easiest first.

## 1. A field guide to one digit

Different people write the same digit very differently — like phenotypic variation within one species. Pick a digit (say 7) and display 12 different examples of it from the training set in a 3×4 grid.

Hints: loop over `range(len(y_train))`, collect the indices `i` where `y_train[i] == 7` into a list, stop once you have 12, then plot `X_train[index]` for each with `imshow`. How much variation do you see? Do any 7s have the European crossbar?

## 2. Design your own motif detector

The notebook built a vertical-edge filter and (by transposing it) a horizontal-edge one. Invent your own 3×3 filter, run `convolve(image, your_filter)` on a digit, and plot the feature map with `cmap="RdBu_r"`.

Two suggestions to try:

- a **diagonal** detector, e.g. `np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])` — which strokes of which digits light it up?
- a **blur** filter where all 9 entries are `1/9` — why does the output look like a smudged copy of the input? (Every output pixel is the *average* of its neighborhood.)

Describe in one sentence, for each filter, what motif it reports.

## 3. Shift down instead of right

The notebook shifted digits 3 pixels *right*. Repeat the experiment shifting 3 pixels *down*: `np.roll(X_test, 3, axis=1)` (axis=1 is the row axis of the `(10000, 28, 28)` array).

Compute accuracy for both `mlp` (remember to flatten and normalize, exactly like `X_test_shifted_flat` was built) and `cnn` (unsqueeze a channel dimension, like `X_test_shifted_img`). Is the down-shift more or less damaging than the right-shift? What about a 6-pixel shift — does even the CNN eventually break? Why would a *bigger* shift defeat pooling's tolerance?

## 4. Is bigger better?

Build a wider MLP with a 512-neuron hidden layer (copy the `mlp` definition, change 128 to 512), train it with `train_model(..., n_epochs=10)`, and compare its test accuracy and `count_parameters(...)` to the CNN's.

You should find it has ~40× more parameters than the CNN and still loses. Write one sentence explaining why, using the word "neighborhood" — then say which biological design principle this reminds you of (hint: a few reusable motifs beat a giant lookup table).

## 5. The most confident mistake

Find the test digit the CNN gets wrong with the *highest* confidence — its most self-assured error. Steps:

1. `with torch.no_grad():` compute `probabilities = torch.softmax(cnn(X_test_img), dim=1)` — shape `(10000, 10)`.
2. Loop over all test images; for each one the CNN got wrong (`test_predictions[i] != y_test[i]`), record the probability of its (wrong) predicted digit: `probabilities[i, test_predictions[i]].item()`.
3. Keep the index with the largest recorded probability, then `imshow` that digit with the true label, the prediction, and the confidence in the title.

Look at the image: can you, as the human pathologist double-checking the machine, see why it was fooled?

## Challenge: the learning curve — deep learning needs data

Module 10 ended with an honest lesson: on 303 heart patients, a neural network barely beat a random forest. Show the other side of that lesson: how *both* of this module's models improve as you feed them more images, and how the CNN's architectural head start persists at every dose.

1. For each training-set size `n` in `[500, 2000, 8000, 20000]`:
   - build a **fresh** MLP (copy the 784→128→10 definition; a fresh model matters — retraining an already-trained one would cheat) and a fresh CNN with `make_cnn(0.0)`,
   - train each with `train_model` on the *first n* images (`X_train_flat[:n]`, `X_train_img[:n]`, `y_train_t[:n]`) for `n_epochs=10`,
   - record both test accuracies (use `accuracy(...)` on the full test set).
2. Plot both curves on one figure: training-set size on the x-axis, test accuracy on the y-axis, one line per model, `marker="o"`, a legend, and — since the sizes span 500 to 20,000 — try `ax.set_xscale("log")` (spaces the sizes evenly by *ratio*, like a serial dilution).
3. Give the figure a title that states its story, e.g. *"More data helps both — but the CNN wins at every dose."*

Bonus question: at 500 images, roughly how many training examples per class is that? Would you trust a diagnostic trained on that few examples per condition?
