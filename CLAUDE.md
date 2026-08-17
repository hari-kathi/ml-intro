# Project context

This is a teaching repo: a parent teaching data science and machine learning to their daughter. All modules are AI-generated.

## The learner

- College student with basic Python knowledge — no need to teach syntax basics, but avoid advanced idioms without explanation.
- Very knowledgeable in biology and genetics — lean on that: use biology/genetics datasets, examples, and analogies wherever possible (gene expression, DNA sequences, species traits, epidemiology).

## Guidelines for generating content

- Every lesson must include good, concrete examples and visual plots — show concepts graphically before (or alongside) equations.
- Explain concepts simply and define DS/ML jargon on first use; prefer intuition and visuals over math notation.
- Keep lessons hands-on; every concept should come with runnable code the learner can tweak.
- Use scikit-learn, pandas, numpy, matplotlib, and seaborn.
- Each module goes in `modules/NN-topic-name/` with `README.md` (lesson), `notebook.ipynb` (runnable code), and `exercises.md` (practice problems, easiest first).
- Use small, relatable datasets (place them in `datasets/`) — biology and genetics themes are ideal (e.g. iris species, gene expression, heart disease, penguin measurements).
- Modules should build on earlier ones; state prerequisites at the top of each lesson.
