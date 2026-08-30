# Curriculum outline — from tables to deep learning

This is the roadmap for the whole course. Modules 01–02 already exist; this outline adds
modules 03–11 plus a capstone project, taking you from pandas all the way to training a
convolutional neural network that diagnoses malaria from blood-smear photos.

Every module keeps the same recipe:

- `README.md` — the lesson, written to be read *before* opening the notebook, with pictures
- `notebook.ipynb` — runnable, heavily commented code
- `exercises.md` — practice, easiest first
- `images/` — figures used by the lesson (each module has a `make_figures.py` that regenerates them)

All datasets come from HuggingFace and every dataset ID below has been verified to exist
and load. Reference material that doesn't belong to any single module lives in [`docs/`](README.md):
a big-picture map of machine learning, a glossary of jargon, and cheat sheets.

## The path at a glance

| # | Module | Tool being learned | Dataset (HuggingFace) | Biology story |
|---|--------|--------------------|------------------------|---------------|
| 01 | NumPy foundations *(done)* | numpy | `scikit-learn/iris` | Measuring 150 flowers |
| 02 | Pandas essentials *(done)* | pandas | `SIH/palmer-penguins` | Antarctic fieldwork |
| 03 | Data wrangling with pandas | pandas (advanced) | `EPI-Eval/owid-covid` | Anatomy of a pandemic |
| 04 | Your first machine-learning model | scikit-learn | `SIH/palmer-penguins` | Teaching a computer to ID penguins |
| 05 | Regression — predicting numbers | scikit-learn | `SIH/palmer-penguins` | Estimating body mass without a scale |
| 06 | Classification — when accuracy lies | scikit-learn | `buio/heart-disease` | Diagnosing heart disease |
| 07 | The honest-scientist workflow | scikit-learn | `scikit-learn/breast-cancer-wisconsin` | Tumor diagnosis, done rigorously |
| 08 | Unsupervised learning | scikit-learn | `scikit-learn/breast-cancer-wisconsin` | Discovering tumor types without labels |
| 09 | Neural networks from scratch | numpy | toy data (`make_moons`) | A brain cell made of math |
| 10 | PyTorch fundamentals | pytorch | `buio/heart-disease` | The same network, with power tools |
| 11 | Deep learning on images | pytorch | `ylecun/mnist` | How a network learns to see |
| P1 | **Capstone:** malaria detector | everything | `dpdl-benchmark/malaria` | Diagnosing malaria from blood smears |

Suggested pace: one module a week, capstone over two.

---

## Module 03 — Data wrangling with pandas

*Prereqs: 01, 02.* Real analyses spend most of their time reshaping and combining
tables, not modeling. This module uses daily COVID-19 case counts for every country
(Our World in Data, 396,995 rows) — real epidemiology at real scale.

**Notebook arc:** load the world's case data → work with **dates** (a new data type:
`datetime`) → pick countries and plot raw daily cases → smooth the noise with **rolling
averages** (7-day window, the number every dashboard showed) → reshape between **long and
wide** formats with `pivot` and `melt` (tidy data — the wide table you'd make in a
spreadsheet vs. the long table pandas loves) → **merge** in a second table of country
populations to compute cases *per million* (the only fair comparison) → `groupby` review
at scale.

**Figures:** long-vs-wide reshaping diagram; how a merge matches rows between two tables;
raw vs. 7-day-smoothed epidemic curves.

**Exercises build to:** a per-capita comparison plot of 4 countries of the learner's choice.

## Module 04 — Your first machine-learning model

*Prereqs: 02, 03.* The conceptual heart of the course: what "learning from data" means.
A field biologist can identify a penguin's species at a glance — can we teach a computer
the same skill from 344 measured penguins?

**Notebook arc:** what features and labels are (X and y) → why we must **hold out a test
set** (a student who memorizes the answer key hasn't learned biology) → `train_test_split`
→ **k-nearest-neighbors**, the most intuitive classifier there is: "find the k most
similar penguins you've seen, let them vote" → accuracy on the test set → **decision
boundary plots** (watch the map change as k grows) → why feature *scale* matters to a
distance-based model → predict a brand-new penguin.

**Figures:** the supervised-learning workflow (data → split → train → evaluate → predict);
train/test split diagram; "who are my 5 nearest neighbors?" zoom-in; decision boundaries
for small vs. large k.

## Module 05 — Regression: predicting numbers

*Prereqs: 04.* Classification answers "which kind?"; regression answers "how much?".
Weighing a penguin in the field is awkward — can we estimate body mass from a flipper
photo instead?

**Notebook arc:** scatter plot flipper length vs. body mass → fit `LinearRegression` and
*read* the line like a biologist reads allometry (slope = grams per extra millimeter) →
**residuals**: what the line gets wrong, and what patterns in the errors tell you →
measuring error in real units (**MAE, RMSE**) and as variance explained (**R²**) → add more
features (multiple regression) and interpret coefficients → **polynomial features** on a
small sample: watch a wiggly curve ace the training data and fail the test data — your
first overfit, seen with your own eyes.

**Figures:** anatomy of a fitted line (slope/intercept in penguin units); residuals as
vertical error bars; the underfit/just-right/overfit triptych.

## Module 06 — Classification in depth: when accuracy lies

*Prereqs: 04, 05.* 303 real patients from the Cleveland heart-disease study: age, blood
pressure, cholesterol, ECG results — does this patient have heart disease? In medicine a
false negative (a sick patient sent home) and a false alarm are *very* different mistakes,
and a single accuracy number hides the difference.

**Notebook arc:** logistic regression — a line that outputs *probabilities* via the
S-shaped sigmoid → the **confusion matrix**: false positives vs. false negatives →
**sensitivity and specificity** (the same words used for lab tests like PCR and antigen
tests) and their ML names, recall and precision → moving the **decision threshold**: the
screening-test trade-off, summarized by the ROC curve → **decision trees** — a flowchart of
if-questions the model designs itself, drawn in full → **random forests** — hundreds of
trees voting → which measurements mattered (feature importance).

**Figures:** sigmoid curve; annotated confusion-matrix anatomy; threshold slider
trade-off; a real fitted decision tree.

## Module 07 — The honest-scientist workflow

*Prereqs: 06.* The module about *not fooling yourself* — the ML equivalent of sterile
technique. Dataset: 569 breast-tumor samples with 30 microscope measurements each,
predicting benign vs. malignant.

**Notebook arc:** why one train/test split can mislead (shuffle luck) → **k-fold
cross-validation**: everyone gets a turn being the test set → preprocessing (scaling) and
the subtle sin of **data leakage**: scaling *before* splitting quietly copies test-set
information into training → **`Pipeline`**: bundle preprocessing + model so leakage becomes
impossible → tuning hyperparameters with **`GridSearchCV`** → **validation curves**:
under- and overfitting as a picture → final protocol: tune inside cross-validation, touch
the test set once, at the end.

**Figures:** k-fold rotation diagram; the leakage mistake vs. the pipeline fix;
validation curve with the sweet spot marked.

## Module 08 — Unsupervised learning: patterns without answers

*Prereqs: 07.* Every model so far was told the right answers. Now we hide the diagnosis
column and ask: would the data have revealed tumor groups *on its own*? This is how
biologists find cell types in single-cell RNA-seq and subtypes in tumor cohorts.

**Notebook arc:** **k-means** clustering, step by step (place centers → assign points →
move centers → repeat, animated as a figure sequence) → choosing k (elbow plot) →
**hierarchical clustering** and dendrograms — reading them exactly like phylogenetic trees
— plus the clustered **heatmap**, the iconic figure of genomics papers → **PCA**: squashing
30 measurements down to a 2-D map that keeps as much variation as possible → the reveal:
overlay the hidden diagnosis labels on our clusters — how close did unsupervised learning
get to the pathologists?

**Figures:** k-means iterations frame by frame; dendrogram annotated like a phylogeny;
PCA as shadow-casting (projection) intuition.

## Module 09 — Neural networks from scratch

*Prereqs: 05, 07 (and comfort with NumPy).* Before PyTorch automates everything, build a
neural network with bare NumPy — it's the difference between using a thermocycler and
knowing what PCR actually does.

**Notebook arc:** a real neuron vs. an artificial one (weighted inputs → threshold →
fire) → one neuron *is* logistic regression → why stacking neurons needs **activation
functions** (without them, a deep network collapses into a straight line) → **loss** as a
landscape and **gradient descent** as walking downhill in fog, visualized on a real bowl
→ build a two-layer network (~30 lines of NumPy), train it on the two-moons dataset that
no straight line can split → watch the decision boundary *curve itself around the data*
epoch by epoch → learning rate: too small crawls, too big explodes.

**Figures:** biological vs. artificial neuron side by side; network anatomy with one
path highlighted; gradient descent on the loss bowl; the boundary-morphing filmstrip.

## Module 10 — PyTorch fundamentals

*Prereqs: 09.* Everything from module 09, with power tools: **tensors** (NumPy arrays
that remember their math), **autograd** (gradients for free — the hand-derived chain rule
from last module, automated), `nn.Module`, `DataLoader`, and the optimizer.

**Notebook arc:** tensors ↔ NumPy round-trip → `requires_grad` and `.backward()` on a
tiny expression, checked against a by-hand derivative → rebuild module 09's two-moons
network in ~15 lines of PyTorch and confirm it learns the same boundary → **the training
loop liturgy** (forward → loss → `zero_grad` → `backward` → `step`), which never changes
from here to GPT → a real problem: the heart-disease patients from module 06, now with a
neural network → does it beat random forest? (Spoiler: on 303 patients, barely if at all —
deep learning needs *data*, an honest and important lesson) → running on Apple's GPU
with `.to("mps")`.

**Figures:** the training-loop cycle diagram; what autograd records (a tiny computation
graph); loss-curve anatomy (healthy vs. too-hot learning rate).

## Module 11 — Deep learning on images

*Prereqs: 10.* Images are where deep learning earned its fame. MNIST: 70,000 handwritten
digits, the fruit fly of deep learning — small enough to train in minutes on a laptop.

**Notebook arc:** an image *is* numbers (zoom into pixel values) → flatten it and train
the module-10 MLP: ~97%, but it treats pixels as unrelated columns and falls apart if a
digit shifts → **convolution**: a small filter sliding across the image detecting one
motif everywhere (like a restriction enzyme scanning a genome for its site) → **pooling**
→ build a small **CNN**, train it, beat the MLP → look inside: learned filters and feature
maps for a real digit → the confusion matrix: which digits get mistaken for which, and do
the mistakes look human? → dropout when the net starts memorizing.

**Figures:** pixels-as-numbers zoom; a filter sliding step by step; CNN architecture
(image → conv → pool → conv → pool → dense → 10 scores); feature-map gallery.

## Capstone project — the malaria detector (`projects/01-malaria-detector/`)

*Prereqs: all of the above.* 27,558 real microscope photos of single red blood cells from
Bangladeshi hospital patients, half parasitized with *Plasmodium*, half healthy
(`dpdl-benchmark/malaria`, the NIH malaria dataset). Task: build the classifier a rural
clinic could actually use, end to end, with almost no hand-holding.

**Stages:** explore and *look at* the cells (module 02–03 skills) → subsample to a
laptop-friendly training set and set up an honest train/validation/test protocol
(module 07) → baseline: logistic regression on raw pixels (module 06) → CNN
(module 11) → evaluate like a diagnostic: sensitivity, specificity, and the threshold
question "which error is worse in a clinic?" (module 06) → inspect the cells it gets
wrong and hypothesize *why, as a biologist* → write a short "methods & results" summary,
like a paper.

---

## Reference material in `docs/`

- **[The big picture](the-big-picture.md)** — what AI / ML / deep learning actually are, the
  supervised / unsupervised / regression / classification map, and which module teaches what. Read it first.
- **[Glossary](glossary.md)** — every jargon term in the course, in plain language, each with a biology analogy.
- **Cheat sheets** — one page each, for looking things up mid-exercise:
  [pandas](cheatsheets/pandas.md) · [scikit-learn](cheatsheets/sklearn.md) · [PyTorch](cheatsheets/pytorch.md)
- `images/` — figures for the docs, regenerated by `docs/make_figures.py`.
