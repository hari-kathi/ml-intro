# Learning Data Science & Machine Learning

A father–daughter learning project. All lesson modules are generated with AI assistance and worked through together.

## How this repo is organized

- `modules/` — numbered lesson modules (e.g. `01-intro-to-data/`), each with a short lesson, a notebook, and exercises
- `docs/` — the [curriculum roadmap](docs/curriculum.md), a [big-picture map of ML](docs/the-big-picture.md), a [glossary](docs/glossary.md), and cheat sheets
- `datasets/` — small datasets used by the modules
- `projects/` — bigger hands-on projects that combine skills from several modules

**Start here:** read [docs/the-big-picture.md](docs/the-big-picture.md), then follow the
module order in [docs/curriculum.md](docs/curriculum.md) — NumPy → pandas → scikit-learn →
neural networks → PyTorch → a malaria-detecting capstone.

## Getting started

1. Create a virtual environment and install the basics:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Launch Jupyter and open a module's notebook:

   ```bash
   jupyter lab
   ```

## Module format

Each module lives in its own folder under `modules/` and typically contains:

- `README.md` — the lesson: concepts explained simply, with examples
- `notebook.ipynb` — code to run and experiment with
- `exercises.md` — practice problems to try on your own
