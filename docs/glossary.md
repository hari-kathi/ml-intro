# Glossary

Every jargon term the course uses, in plain language. Each entry ends with a biology analogy in italics. Terms are alphabetical — jump by letter:

[A](#a) · [B](#b) · [C](#c) · [D](#d) · [E](#e) · [F](#f) · [G](#g) · [H](#h) · [K](#k) · [L](#l) · [M](#m) · [N](#n) · [O](#o) · [P](#p) · [R](#r) · [S](#s) · [T](#t) · [U](#u) · [V](#v) · [W](#w)

## A

**Accuracy** — The fraction of predictions a model gets right. Simple, but it can badly mislead when one outcome is much rarer than the other (Module 06).
*A test that calls every patient "healthy" is 99% accurate for a disease with 1% prevalence — and completely useless.*

**Activation function** — A small nonlinear function (like ReLU or sigmoid) applied to each neuron's output. Without it, stacking layers would collapse into one straight line, and the network couldn't learn curved patterns (Module 09).
*Like a neuron's firing threshold: the cell doesn't pass along every whisper of input, it fires only past a threshold — and that nonlinearity is what makes circuits of neurons interesting.*

**Autograd** — PyTorch's automatic bookkeeping system: it records every calculation you do and can then compute exact gradients (slopes) through all of it with one call to `.backward()` (Module 10).
*Like a lab notebook so meticulous you can trace exactly how much each early pipetting step affected the final assay result.*

## B

**Batch** — A small group of training examples (say 32) processed together in one step of training, instead of one example at a time or the whole dataset at once (Module 10).
*Like running samples through a sequencer in plates of 96 — batches are the efficient middle ground between one tube at a time and everything at once.*

**Boolean mask** — An array of True/False values used to select rows: `df[df["species"] == "Gentoo"]` keeps rows where the mask is True (Modules 01–02).
*Like a stack of specimen tags where you flag the ones that match your criterion, then pull only the flagged ones.*

## C

**Classification** — Supervised learning where the answer is a category: which species, sick or healthy, spam or not (Modules 04, 06).
*Identifying which of three penguin species a bird belongs to from its measurements.*

**Clustering** — Unsupervised learning that groups examples by similarity, without being told what the groups are (Module 08).
*Sorting a drawer of unlabeled specimens into piles that look alike — the way cell types are discovered in single-cell RNA-seq data.*

**CNN (convolutional neural network)** — A neural network built for images: it uses convolutions to detect small local patterns (edges, blobs) and combines them layer by layer into larger structures (Module 11, capstone).
*Like the visual cortex: early neurons respond to tiny edges, later ones to whole shapes like faces — or parasites in a blood cell.*

**Confusion matrix** — A 2×2 (or bigger) table of predictions vs reality, splitting errors into false positives and false negatives instead of lumping them into one accuracy number (Module 06).
*Exactly how diagnostic tests are reported: how many sick patients did we catch, how many healthy patients did we falsely alarm?*

**Convolution** — Sliding a small filter (a tiny grid of weights) across an image, checking at every position "is my pattern here?". One filter finds one motif everywhere in the image (Module 11).
*Like a restriction enzyme scanning along a genome: one recognition site, detected wherever it occurs.*

**Cross-validation** — Splitting the data into k folds and training k times, letting each fold take a turn as the test set, then averaging the scores. Far more trustworthy than a single split (Module 07).
*Like technical replicates: one measurement might be shuffle luck; five rotated measurements give you a mean and a spread.*

## D

**DataFrame** — pandas' table type: labeled columns, each with its own data type, like a lab spreadsheet that speaks Python (Module 02).
*Your field-season spreadsheet — one row per penguin, one column per measurement — as a programmable object.*

**Data leakage** — When information from the test set accidentally sneaks into training — for example, scaling the data *before* splitting it. The model then looks better than it really is (Module 07).
*Contamination: like template DNA drifting into your negative control, it invisibly inflates the result and invalidates the experiment.*

**Decision boundary** — The line (or curved surface) where a classifier switches from predicting one class to another. Plotting it shows you what the model actually believes (Modules 04, 09).
*Like the elevation contour on a mountainside above which one moss species gives way to another.*

**Decision tree** — A model that is literally a flowchart of yes/no questions about the features ("is cholesterol > 240?"), learned from data. Easy to read, prone to memorizing (Module 06).
*A dichotomous identification key — "leaves opposite? go to 3" — except the computer writes the key itself.*

**Deep learning** — Machine learning with neural networks that have many layers. Excels on raw data (images, sequences, sound) but needs lots of examples (Modules 09–11).
*The inner circle of the AI ⊃ ML ⊃ deep-learning nesting — see [the big picture](the-big-picture.md).*

**Dendrogram** — The tree diagram produced by hierarchical clustering: the height where two branches join shows how different those groups are (Module 08).
*Read it exactly like a phylogenetic tree — except the branch lengths mean "measured similarity", not evolutionary time.*

**Dropout** — During training, randomly switching off a fraction of neurons on each step, so no single neuron can be relied on too heavily. A standard defense against overfitting (Module 11).
*Like rotating which lab members run each assay so the protocol can't secretly depend on one person's wrist technique.*

## E

**Epoch** — One full pass through the entire training set. Training usually takes many epochs (Modules 09–11).
*One complete review of all your flashcards. You rarely learn the deck in a single pass.*

## F

**Feature** — One measured input variable: flipper length, cholesterol, a pixel's brightness. The columns of X. A model's inputs are its features (Module 04).
*Each measurement you take on a specimen — the columns of your field notebook.*

**Feature importance** — A score (from tree-based models like random forests) for how much each feature contributed to the predictions (Module 06).
*Like learning that bill depth does most of the work in telling Gentoos apart — the diagnostic character of the bunch.*

**Feature scaling (standardization)** — Rescaling every feature to a comparable range, typically mean 0 and standard deviation 1 (`StandardScaler`). Essential for distance-based models, where a feature measured in grams would otherwise drown one measured in millimeters (Modules 04, 07).
*Like converting all your measurements to z-scores so a 6000 g body mass can't shout over a 19 mm bill depth.*

## G

**Gradient descent** — The learning algorithm behind neural networks: compute the slope of the loss with respect to every weight, take a small downhill step, repeat until the loss stops improving (Module 09).
*Walking downhill in fog: you can't see the valley, but you can always feel which way the ground slopes under your feet.*

**GridSearchCV** — scikit-learn's tool that tries every combination of hyperparameter values you list, scoring each by cross-validation, and keeps the best (Module 07).
*A checkerboard optimization experiment: try every combination of annealing temperature and primer concentration, keep the well that worked best.*

## H

**Hierarchical clustering** — Clustering that repeatedly merges the two most similar groups, building a tree (dendrogram) from single specimens up to one big cluster (Module 08).
*The logic of building a phylogeny from trait similarity: closest pair joins first, then the next closest, upward to the root.*

**Hyperparameter** — A setting *you* choose before training (k in k-nearest neighbors, tree depth, learning rate), as opposed to the parameters the model learns from data. Tuned with tools like GridSearchCV (Module 07).
*Like PCR cycle count and annealing temperature: knobs the experimenter sets, not things the reaction figures out on its own.*

## K

**k-fold cross-validation** — See **cross-validation**; k is how many folds the data is cut into (5 and 10 are common).
*Five technical replicates instead of one.*

**k-means** — A clustering algorithm: place k cluster centers, assign each point to its nearest center, move each center to the middle of its points, repeat until nothing changes (Module 08).
*Like iteratively sorting specimens into k piles, re-choosing each pile's "type specimen", and re-sorting until the piles settle.*

**k-nearest neighbors (k-NN)** — The most intuitive classifier: to label a new example, find the k most similar training examples and let them vote (Module 04).
*Field-biologist logic: "this bird looks like the last five Chinstraps I saw, so it's probably a Chinstrap."*

## L

**Label** — The answer a supervised model learns to predict: the species, the diagnosis, the body mass. The y to the features' X (Module 04).
*The species name written on the specimen tag.*

**Layer** — A row of neurons in a neural network that all take the previous layer's outputs as their inputs. "Deep" learning = many layers (Modules 09–11).
*Like stages in a signaling cascade: each stage transforms the signal and passes it on to the next.*

**Leakage** — See **data leakage**.

**Learning rate** — How big a step gradient descent takes each update. Too small: training crawls. Too large: it overshoots the valley and the loss explodes (Modules 09–10).
*Titration step size: tiny drops take all afternoon, big pours blast straight past the endpoint.*

**Linear regression** — Fitting the best straight line through data to predict a number: body mass = intercept + slope × flipper length (Module 05).
*An allometric fit — the slope literally reads "grams of body mass per extra millimeter of flipper".*

**Logistic regression** — Despite the name, a *classifier*: it fits a line, then squashes the output through the S-shaped sigmoid so it becomes a probability between 0 and 1 (Module 06).
*Like a dose-response curve: as the dose (risk score) rises, the probability of the response climbs an S-shaped curve from 0 toward 1.*

**Loss** — A number measuring how wrong the model currently is; training means adjusting weights to push it down. Also called cost or error (Module 09).
*The "distance from healthy" your model is trying to minimize — training walks downhill on this landscape.*

## M

**MAE (mean absolute error)** — The average size of a regression model's mistakes, in the original units: "off by 310 grams on average". Easy to explain to anyone (Module 05).
*"Our field estimate of body mass is typically off by about 300 g" — a number a biologist can act on.*

**Model** — The learned rule that maps features to predictions — the output of training. Could be a line, a tree, a forest, or a neural network.
*A fitted growth curve: built from measured data points, then used to predict values you never measured.*

## N

**Neural network** — A model built from layers of simple artificial neurons. Each neuron does a weighted sum and an activation; the network's power comes from stacking many of them (Modules 09–11).
*Loosely inspired by real neural circuits: many simple units, densely connected, collectively doing something none could alone.*

**Neuron (artificial)** — The unit of a neural network: multiply each input by a weight, add them up, pass the sum through an activation function (Module 09).
*A cartoon of a real neuron: dendrites collect weighted signals, and the axon fires if the sum crosses a threshold.*

## O

**Overfitting** — When a model memorizes its training data — noise and all — instead of learning the general pattern. Symptom: great training score, poor test score (Modules 05, 07).
*The student who memorized the answer key: perfect on that exam, lost on any new question.*

## P

**PCA (principal component analysis)** — A method that squashes many measured features down to a few new axes (components) chosen to preserve as much variation as possible — often to draw a 2-D map of high-dimensional data (Module 08).
*Like the 2-D "PCA plots" in genomics papers: 20,000 genes per cell collapsed to two axes, and cell types appear as islands.*

**Pipeline** — A scikit-learn object that chains preprocessing and model into one unit, so every cross-validation fold re-fits the scaler on training data only. The structural fix that makes leakage impossible (Module 07).
*Sterile technique built into the protocol itself, so contamination can't happen even on a careless day.*

**Pooling** — In a CNN, shrinking a feature map by keeping only the strongest signal in each small window (max pooling). Makes detection care less about *exactly* where a pattern sits (Module 11).
*Like reporting "parasite present in this quadrant of the slide" rather than its exact micrometer coordinates.*

**Precision** — Of all the cases the model flagged positive, the fraction that really are positive. High precision = few false alarms (Module 06).
*Of everyone the screening test flagged, how many actually have the disease? The clinical term is positive predictive value.*

## R

**R² (R-squared)** — The fraction of the outcome's variance a regression model explains, from 0 to 1. R² = 0.76 means the model accounts for 76% of the variation in body mass (Module 05).
*The same R² you see on heritability and QTL plots: how much of the trait's variation does this predictor capture?*

**Random forest** — Hundreds of decision trees, each trained on a random slice of the data and features, voting together. One tree memorizes; a forest generalizes (Module 06).
*Like a consensus diagnosis from a hundred pathologists who each saw a different subset of the slides.*

**Recall** — Of all the truly positive cases, the fraction the model caught. High recall = few missed cases. Identical to sensitivity (Module 06).
*Of all the patients who really have the disease, how many did the test catch? Missing one (a false negative) sends a sick patient home.*

**Regression** — Supervised learning where the answer is a number: body mass, blood pressure, expression level. Contrast with classification (Module 05).
*Estimating a penguin's mass from its flipper length instead of wrestling it onto a scale.*

**Residual** — For one data point, the difference between the true value and the model's prediction — what the fitted line got wrong. Patterns in the residuals reveal what the model is missing (Module 05).
*The vertical gap between a measured data point and your fitted growth curve.*

**RMSE (root mean squared error)** — Like MAE but it squares errors before averaging, so a few large mistakes hurt disproportionately. Also in the original units (Module 05).
*The error measure to watch when one wildly misweighed penguin matters more than ten slightly-off ones.*

**ROC curve** — A plot of sensitivity vs false-positive rate at every possible decision threshold, summarizing the whole screening trade-off in one curve. Area under it (AUC) near 1.0 is excellent, 0.5 is coin-flipping (Module 06).
*The standard picture for comparing diagnostic tests: how much specificity must you give up to catch more true cases?*

## S

**Sensitivity** — The medical name for recall: the fraction of true positives the test catches. A PCR test with 98% sensitivity misses 2 of every 100 infected patients (Module 06).
*Same word, same meaning as in your diagnostics coursework — ML borrowed it from medicine.*

**Sigmoid** — The S-shaped function that squashes any number into the range 0–1, turning a raw score into a probability. The heart of logistic regression and an early activation function (Modules 06, 09).
*The shape of a dose-response or enzyme-saturation curve: flat, then steep, then flat again.*

**Specificity** — The fraction of true *negatives* correctly cleared by the test. High specificity = few false alarms among healthy patients (Module 06).
*A highly specific antibody binds only its target — a highly specific test flags only the truly sick.*

**Supervised learning** — Learning from examples that come with labels (answers), to predict labels for new examples. Includes classification and regression (Modules 04–07, 09–11).
*Studying a field guide with named photos, then identifying birds the guide never showed you.*

## T

**Tensor** — PyTorch's array type: like a NumPy array, but it can live on a GPU and remembers the operations done to it so gradients come free (Module 10).
*A NumPy array that keeps a lab notebook of everything done to it.*

**Test set** — The portion of data (often 20%) locked away before training and used exactly once, at the end, for the honest final score (Module 04).
*Samples sealed in the −80 °C freezer until the day of the final assay.*

**Threshold (decision threshold)** — The probability cutoff for calling a prediction positive — 0.5 by default, but movable. Lowering it catches more true cases at the cost of more false alarms (Module 06).
*The Ct cutoff on a qPCR test: where you set it decides the balance between missed infections and false positives.*

**Training** — The process of fitting a model: adjusting its internal parameters so its predictions on the training examples improve. In scikit-learn this is the `.fit()` call.
*The curve-fitting step: sliding the growth curve's parameters around until it hugs the measured points.*

**Training set** — The portion of data (often 80%) the model is allowed to learn from (Module 04).
*The labeled photos you study — as opposed to the exam questions you'll be tested on.*

## U

**Underfitting** — When a model is too simple to capture the real pattern: it does badly on training *and* test data. The opposite failure from overfitting (Modules 05, 07).
*Fitting a straight line to an obviously sigmoid growth curve — wrong everywhere, including on the data it was fitted to.*

**Unsupervised learning** — Learning from data *without* labels: finding clusters, or compressing many features into few (PCA). The data must reveal its own structure (Module 08).
*Sorting unlabeled specimens into piles by similarity, without a field guide.*

## V

**Validation set / validation curve** — Data used *during* development to compare models and tune hyperparameters — so the test set can stay sealed until the end. A validation curve plots score vs one hyperparameter and makes under- and overfitting visible as a picture (Module 07).
*The pilot study you optimize the protocol on, kept separate from the confirmatory experiment.*

## W

**Weights** — The numbers inside a model that training adjusts: the slope of a regression line, the millions of connection strengths in a neural network. (Together with biases, also called *parameters*.)
*Synaptic strengths: learning, in brains and in networks, is the adjustment of connection weights.*
