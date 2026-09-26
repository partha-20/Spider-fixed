# The Daily Bugle Truth Serum — Spider-Sense for AI Text

A working detection pipeline that scores text as human-written or AI-generated,
combining three independent signal engines through a fusion (ensemble) layer,
per the project roadmap.

## What's actually implemented here

| Roadmap phase | Status | File |
|---|---|---|
| Phase 2 — Data pipeline | Synthetic, topic-matched human/AI/humanized-AI dataset | `data/generate_dataset.py` → `data/dataset.csv` |
| Phase 3 — Feature engineering | Stylometric features (regex/statistics, no model downloads) | `models/features.py` |
| Phase 4 — Signal Engine A (perplexity) | Trainable-from-scratch bigram LM | `models/perplexity.py` |
| Phase 4 — Signal Engine B (stylometric classifier) | Logistic regression on 9 stylometric features | trained inside `train_ensemble.py` |
| Phase 4 — Signal Engine C (learned classifier) | TF-IDF + calibrated Logistic Regression | trained inside `train_ensemble.py` |
| Phase 4 — Fusion layer | Meta logistic-regression stacking the 3 signals | `models/train_ensemble.py` |
| Phase 5 — Evaluation | Accuracy/F1/AUC, confusion matrix, adversarial + false-positive breakdown | printed by `train_ensemble.py` |
| Phase 6 — Backend API | FastAPI `/analyze` endpoint wired to the trained artifacts | `api/main.py` |
| Phase 7 — Frontend | Lite client-side demo (stylometric engine only) | published separately as an HTML artifact |

## Important limitation — please read

This sandbox has **no internet access to HuggingFace, OpenAI, or any model
hub**, so two substitutions were made from the original roadmap:

1. **Perplexity engine**: uses a bigram language model trained from scratch
   on the human-text training split, instead of a downloaded GPT-2/Pythia
   checkpoint. Swap-in path: replace `NgramLM` in `models/perplexity.py`
   with `transformers.AutoModelForCausalLM.from_pretrained("gpt2")` and
   compute true token log-probabilities — the `.perplexity(text)` interface
   used everywhere else does not need to change.

2. **Learned classifier**: uses TF-IDF + Logistic Regression instead of a
   fine-tuned DistilBERT/RoBERTa. Swap-in path: replace the block in
   `train_ensemble.py` marked "Signal Engine C" with a HuggingFace
   `Trainer` fine-tune; keep exposing a `.predict_proba(text) -> [p0, p1]`
   interface so the fusion layer is unaffected.

3. **The dataset is synthetic** (rule-based templates), not real scraped
   news/social data or real LLM API output, for the same reason. This is
   why the reported test accuracy is 100% — the synthetic classes are, by
   construction, more cleanly separable than real human vs. real AI text
   will be. Treat all metrics as a pipeline correctness check, not a
   real-world performance claim. Swap `data/generate_dataset.py` for
   real licensed news corpora + actual GPT/Claude/Gemini API calls on the
   same topics before trusting the numbers.

## Repo contents at a glance

```
.
├── data/generate_dataset.py     # Phase 2
├── models/features.py           # Phase 3
├── models/perplexity.py         # Phase 4, Signal Engine A
├── models/train_ensemble.py     # Phase 4 (Engines B, C, Fusion) + Phase 5 (Eval)
├── models/export_web_weights.py # exports weights for the browser demo
├── api/main.py                  # Phase 6
├── web/stylometric_weights.json # trained weights consumed by the live demo
├── tests/                       # pytest suite (features, perplexity, API)
├── .github/workflows/ci.yml     # runs the test suite on every push/PR
├── Dockerfile / .dockerignore   # Phase 8 containerization
├── requirements.txt             # runtime deps
├── requirements-dev.txt         # + pytest/httpx for testing
├── LICENSE                      # MIT
└── CONTRIBUTING.md
```

## Running it

```bash
pip install -r requirements.txt

# 1. Generate the dataset
python data/generate_dataset.py

# 2. Train all three engines + the fusion layer, print evaluation report
python models/train_ensemble.py

# 3. Run the API
uvicorn api.main:app --reload --port 8000

# 4. Query it
curl -X POST localhost:8000/analyze \
     -H "Content-Type: application/json" \
     -d '{"text": "Paste a suspicious article or post here..."}'
```

### Running tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

### Running with Docker

```bash
docker build -t spider-sense .
docker run -p 8000:8000 spider-sense
```

Response shape:
```json
{
  "verdict": "AI-Generated",
  "ai_probability": 0.98,
  "confidence_label": "AI-Generated",
  "signal_breakdown": {
    "perplexity_engine": 0.94,
    "stylometric_engine": 0.99,
    "learned_classifier": 0.97
  },
  "evidence": [{"sentence": "...", "ai_likelihood": 0.98}]
}
```

## Next steps toward production (Phases 8–9 of the roadmap)

- Replace synthetic data with real corpora + real multi-LLM generations.
- Swap in a downloaded reference LM and fine-tuned transformer once deployed
  with internet/GPU access.
- Add Redis caching (a stub in-memory dict cache is already wired in `api/main.py`).
- Containerize with Docker; deploy behind Vercel/Render as described in the
  project document.
- Add a feedback endpoint to collect mislabeled cases for retraining.
