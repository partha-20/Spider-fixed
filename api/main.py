import sys
import os
import hashlib
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from features import extract_features, FEATURE_FUNCS, sentence_evidence_scores
from perplexity import NgramLM

ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "models", "artifacts")
FEATURE_NAMES = list(FEATURE_FUNCS.keys())

app = FastAPI(title="Spider-Sense — Daily Bugle Truth Serum API", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_lm = joblib.load(f"{ARTIFACTS}/ngram_lm.joblib")
_lr_ppl = joblib.load(f"{ARTIFACTS}/lr_perplexity.joblib")
_lr_style = joblib.load(f"{ARTIFACTS}/lr_stylometric.joblib")
_tfidf = joblib.load(f"{ARTIFACTS}/tfidf_vectorizer.joblib")
_clf_learned = joblib.load(f"{ARTIFACTS}/clf_learned.joblib")
_meta_clf = joblib.load(f"{ARTIFACTS}/meta_classifier.joblib")
_ppl_mean = np.load(f"{ARTIFACTS}/ppl_norm_mean.npy")
_ppl_std = np.load(f"{ARTIFACTS}/ppl_norm_std.npy")
_style_mean = np.load(f"{ARTIFACTS}/style_norm_mean.npy")
_style_std = np.load(f"{ARTIFACTS}/style_norm_std.npy")

_cache = {}

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=20, description="Article, post, or transcript text to analyze")

class AnalyzeResponse(BaseModel):
    verdict: str
    ai_probability: float
    confidence_label: str
    signal_breakdown: dict
    evidence: list

def _verdict_label(p):
    if p < 0.2: return "Human"
    if p < 0.4: return "Likely Human"
    if p < 0.6: return "Uncertain"
    if p < 0.8: return "Likely AI"
    return "AI-Generated"

def analyze_text(text: str) -> dict:
    key = hashlib.sha256(text.encode()).hexdigest()
    if key in _cache: return _cache[key]

    ppl = _lm.perplexity(text)
    burst = _lm.burstiness_of_surprisal(text)
    ppl_vec = (np.array([[np.log1p(ppl), burst]]) - _ppl_mean) / _ppl_std
    p_ppl = float(_lr_ppl.predict_proba(ppl_vec)[0, 1])

    feats = extract_features(text)
    style_vec = (np.array([[feats[f] for f in FEATURE_NAMES]]) - _style_mean) / _style_std
    p_style = float(_lr_style.predict_proba(style_vec)[0, 1])

    tfidf_vec = _tfidf.transform([text])
    p_learned = float(_clf_learned.predict_proba(tfidf_vec)[0, 1])

    meta_vec = np.array([[p_ppl, p_style, p_learned]])
    p_final = float(_meta_clf.predict_proba(meta_vec)[0, 1])

    evidence = [{"sentence": s, "ai_likelihood": score} for s, score in sentence_evidence_scores(text)]
    evidence.sort(key=lambda e: -e["ai_likelihood"])

    result = {
        "verdict": _verdict_label(p_final),
        "ai_probability": round(p_final, 4),
        "confidence_label": _verdict_label(p_final),
        "signal_breakdown": {
            "perplexity_engine": round(p_ppl, 4),
            "stylometric_engine": round(p_style, 4),
            "learned_classifier": round(p_learned, 4),
        },
        "evidence": evidence[:5],
    }
    _cache[key] = result
    return result

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    return analyze_text(req.text)

@app.get("/health")
def health():
    return {"status": "ok"}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(BASE_DIR, "web")

# Mounted last and matched last: FastAPI checks routes in the order they were
# added, so /analyze and /health above are always matched first, and anything
# else (including "/") falls through to the static files here. Using the
# absolute WEB_DIR (instead of a relative "web" string) means this works
# regardless of the working directory the server was started from.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")
