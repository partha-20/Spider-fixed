# Contributing

This started as a hackathon build (see `README.md` for the honest list of
what's simulated vs. real). Contributions that move it toward production are
welcome, especially:

- Swapping `data/generate_dataset.py` for real scraped/licensed corpora and
  real multi-LLM generations (see the "Important limitation" section of the
  README for why this matters).
- Replacing `models/perplexity.py`'s bigram model with a downloaded
  GPT-2/Pythia checkpoint once running somewhere with internet/GPU access.
- Replacing the TF-IDF + Logistic Regression classifier in
  `train_ensemble.py` with a fine-tuned DistilBERT/RoBERTa.
- Adding Redis caching in place of the in-memory dict cache in `api/main.py`.

## Workflow

1. Fork the repo and create a branch off `main`.
2. `pip install -r requirements-dev.txt`
3. Run `python data/generate_dataset.py && python models/train_ensemble.py`
   once to produce local model artifacts.
4. Run `pytest tests/ -v` before opening a PR — CI runs the same suite.
5. Open a PR describing what changed and why.

## Code style

Keep new signal engines behind the same interface pattern already used
(`.fit(texts)` / `.predict_proba(text)` or `.perplexity(text)`), so the
fusion layer in `train_ensemble.py` doesn't need to change when a component
is swapped out.
