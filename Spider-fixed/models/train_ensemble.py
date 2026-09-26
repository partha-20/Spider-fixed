"""
Phase 4 (Fusion) + Phase 5 (Evaluation)
=========================================
Trains all three signal engines on the TRAIN split only (no leakage),
scores the TEST split, fuses signals with a meta-classifier, calibrates
probabilities, and reports metrics — including a breakdown on the
adversarial ("humanized AI") subset, which is the hardest case per the brief.
"""
import csv
import json
import os
import sys
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report,
)

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(MODELS_DIR)

sys.path.insert(0, MODELS_DIR)
from features import extract_features, FEATURE_FUNCS
from perplexity import NgramLM

DATA_PATH = os.path.join(ROOT_DIR, "data", "dataset.csv")
OUT_DIR = os.path.join(MODELS_DIR, "artifacts")

os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_NAMES = list(FEATURE_FUNCS.keys())


def load_data():
    rows = list(csv.DictReader(open(DATA_PATH, encoding="utf-8")))
    texts = [r["text"] for r in rows]
    labels = [1 if r["label"] == "ai" else 0 for r in rows]  # 1 = AI
    adversarial = [r.get("adversarial") == "True" for r in rows]
    return texts, labels, adversarial


def stylometric_matrix(texts):
    return np.array([[extract_features(t)[f] for f in FEATURE_NAMES] for t in texts])


def main():
    texts, labels, adversarial = load_data()
    idx = np.arange(len(texts))
    idx_train, idx_test = train_test_split(idx, test_size=0.25, stratify=labels, random_state=42)

    texts_train = [texts[i] for i in idx_train]
    texts_test = [texts[i] for i in idx_test]
    y_train = np.array([labels[i] for i in idx_train])
    y_test = np.array([labels[i] for i in idx_test])
    adv_test = np.array([adversarial[i] for i in idx_test])

    # ---------- Signal Engine A: Perplexity (n-gram LM trained on TRAIN human text only) ----------
    human_train_texts = [t for t, y in zip(texts_train, y_train) if y == 0]
    lm = NgramLM().fit(human_train_texts)

    def perplexity_features(texts):
        ppl = np.array([lm.perplexity(t) for t in texts])
        burst = np.array([lm.burstiness_of_surprisal(t) for t in texts])
        # log-scale perplexity (heavy-tailed) then z-normalize using TRAIN stats
        return np.column_stack([np.log1p(ppl), burst])

    ppl_train_raw = perplexity_features(texts_train)
    ppl_mean, ppl_std = ppl_train_raw.mean(axis=0), ppl_train_raw.std(axis=0) + 1e-6
    ppl_train = (ppl_train_raw - ppl_mean) / ppl_std
    ppl_test = (perplexity_features(texts_test) - ppl_mean) / ppl_std

    lr_ppl = LogisticRegression(max_iter=1000).fit(ppl_train, y_train)
    p_ppl_train = lr_ppl.predict_proba(ppl_train)[:, 1]
    p_ppl_test = lr_ppl.predict_proba(ppl_test)[:, 1]

    # ---------- Signal Engine B: Stylometric classifier ----------
    X_style_train_raw = stylometric_matrix(texts_train)
    style_mean, style_std = X_style_train_raw.mean(axis=0), X_style_train_raw.std(axis=0) + 1e-6
    X_style_train = (X_style_train_raw - style_mean) / style_std
    X_style_test = (stylometric_matrix(texts_test) - style_mean) / style_std

    lr_style = LogisticRegression(max_iter=1000).fit(X_style_train, y_train)
    p_style_train = lr_style.predict_proba(X_style_train)[:, 1]
    p_style_test = lr_style.predict_proba(X_style_test)[:, 1]

    # ---------- Signal Engine C: Learned classifier (TF-IDF + Logistic Regression) ----------
    # Stand-in for the fine-tuned transformer (DistilBERT/RoBERTa) specified in the
    # roadmap; swap this block for a HuggingFace Trainer fine-tune once deployed
    # somewhere with internet/GPU access. Interface (`predict_proba`) stays identical.
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=3000, sublinear_tf=True)
    X_tfidf_train = tfidf.fit_transform(texts_train)
    X_tfidf_test = tfidf.transform(texts_test)

    base_clf = LogisticRegression(max_iter=2000, C=1.0)
    clf_learned = CalibratedClassifierCV(base_clf, method="sigmoid", cv=3)
    clf_learned.fit(X_tfidf_train, y_train)
    p_learned_train = clf_learned.predict_proba(X_tfidf_train)[:, 1]
    p_learned_test = clf_learned.predict_proba(X_tfidf_test)[:, 1]

    # ---------- Fusion Layer: meta-classifier (stacking) ----------
    meta_train = np.column_stack([p_ppl_train, p_style_train, p_learned_train])
    meta_test = np.column_stack([p_ppl_test, p_style_test, p_learned_test])

    meta_clf = LogisticRegression(max_iter=1000).fit(meta_train, y_train)
    final_proba_test = meta_clf.predict_proba(meta_test)[:, 1]
    final_pred_test = (final_proba_test >= 0.5).astype(int)

    # ---------- Evaluation ----------
    def report(name, y_true, proba):
        pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(y_true, pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(y_true, proba)
        except ValueError:
            auc = float("nan")
        print(f"{name:28s} acc={acc:.3f}  prec={p:.3f}  rec={r:.3f}  f1={f1:.3f}  auc={auc:.3f}")
        return acc, p, r, f1, auc

    print("\n=== Individual Signal Performance (test set) ===")
    report("Perplexity engine alone", y_test, p_ppl_test)
    report("Stylometric engine alone", y_test, p_style_test)
    report("Learned (TF-IDF+LR) engine alone", y_test, p_learned_test)

    print("\n=== Ensemble (fused) Performance (test set) ===")
    report("ENSEMBLE (final)", y_test, final_proba_test)

    print("\nConfusion matrix (rows=true[human,ai], cols=pred[human,ai]):")
    print(confusion_matrix(y_test, final_pred_test))

    print("\n=== Robustness: adversarial (humanized AI) vs raw AI vs human ===")
    ai_mask = y_test == 1
    raw_ai_mask = ai_mask & (~adv_test)
    adv_mask = ai_mask & adv_test
    human_mask = y_test == 0
    if raw_ai_mask.sum() > 0:
        report("Raw AI samples only", y_test[raw_ai_mask], final_proba_test[raw_ai_mask])
    if adv_mask.sum() > 0:
        report("Humanized/adversarial AI only", y_test[adv_mask], final_proba_test[adv_mask])
    if human_mask.sum() > 0:
        # false positive rate on humans = 1 - "recall of class 0"
        human_pred = final_pred_test[human_mask]
        fpr = (human_pred == 1).mean()
        print(f"{'False positive rate (real humans flagged as AI)':45s} = {fpr:.3f}")

    print("\n" + classification_report(y_test, final_pred_test, target_names=["human", "ai"]))

    # ---------- Save artifacts for the FastAPI backend ----------
    joblib.dump(lm, f"{OUT_DIR}/ngram_lm.joblib")
    joblib.dump(lr_ppl, f"{OUT_DIR}/lr_perplexity.joblib")
    joblib.dump(lr_style, f"{OUT_DIR}/lr_stylometric.joblib")
    joblib.dump(tfidf, f"{OUT_DIR}/tfidf_vectorizer.joblib")
    joblib.dump(clf_learned, f"{OUT_DIR}/clf_learned.joblib")
    joblib.dump(meta_clf, f"{OUT_DIR}/meta_classifier.joblib")
    np.save(f"{OUT_DIR}/ppl_norm_mean.npy", ppl_mean)
    np.save(f"{OUT_DIR}/ppl_norm_std.npy", ppl_std)
    np.save(f"{OUT_DIR}/style_norm_mean.npy", style_mean)
    np.save(f"{OUT_DIR}/style_norm_std.npy", style_std)

    metrics = {
        "test_size": len(y_test),
        "ensemble_accuracy": float(accuracy_score(y_test, final_pred_test)),
        "ensemble_auc": float(roc_auc_score(y_test, final_proba_test)),
        "false_positive_rate_humans": float((final_pred_test[human_mask] == 1).mean()),
    }
    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved model artifacts to {OUT_DIR}/")


if __name__ == "__main__":
    main()
