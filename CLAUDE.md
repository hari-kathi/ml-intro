# Project context

This is a teaching repo: a parent teaching data science and machine learning to their daughter. All modules are AI-generated.

## The learner

- College student with basic Python knowledge but no computer-science background. Do not assume Python or CS expertise: comment code liberally (explain what each non-obvious line does, right next to it), introduce every new function or idiom on first use, and avoid advanced idioms (comprehensions, format specs, lambdas) unless the lesson is about them.
- Very knowledgeable in biology and genetics — lean on that: use biology/genetics datasets, examples, and analogies wherever possible (gene expression, DNA sequences, species traits, epidemiology).

## Guidelines for generating content

- Every lesson must include good, concrete examples and visual plots — show concepts graphically before (or alongside) equations.
- Explain concepts simply and define DS/ML jargon on first use; prefer intuition and visuals over math notation.
- Keep lessons hands-on; every concept should come with runnable code the learner can tweak.
- Use scikit-learn, pandas, numpy, matplotlib, and seaborn.
- Each module goes in `modules/NN-topic-name/` with `README.md` (lesson), `notebook.ipynb` (runnable code), and `exercises.md` (practice problems, easiest first).
- Source example datasets from HuggingFace via the `datasets` library (`load_dataset(...).to_pandas()`), verifying the dataset ID exists before using it. Biology and genetics themes are ideal (e.g. `scikit-learn/iris`, `SIH/palmer-penguins`, gene expression, heart disease). Use `datasets/` only for small files that have no good HuggingFace equivalent.
- Modules should build on earlier ones; state prerequisites at the top of each lesson.
