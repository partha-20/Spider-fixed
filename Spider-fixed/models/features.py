"""
Phase 3 — Feature Engineering
==============================
Stylometric feature extractor. Pure Python/regex — no spaCy/NLTK model
downloads required, so it runs anywhere including offline sandboxes.

Each function returns a plain float so features are easy to serialize
to JSON for the client-side demo (see export_web_weights.py).
"""
import re
import statistics as stats

FILLER_PHRASES = [
    "it is important to note", "furthermore", "moreover", "in conclusion",
    "in summary", "as a result", "consequently", "on the other hand",
    "it should be emphasized", "in addition", "notably", "overall",
    "in today's", "rapidly evolving", "plays a significant role",
    "it is worth", "remains to be seen", "widely regarded",
]

HEDGE_WORDS = [
    "may", "might", "could", "potentially", "possibly", "generally",
    "often", "typically", "tends to", "appears to", "seems to",
]


def _sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p.strip()]


def _words(text):
    return re.findall(r"[A-Za-z']+", text)


def sentence_length_variance(text):
    sents = _sentences(text)
    lengths = [len(_words(s)) for s in sents if _words(s)]
    if len(lengths) < 2:
        return 0.0
    return float(stats.pvariance(lengths))


def avg_sentence_length(text):
    sents = _sentences(text)
    lengths = [len(_words(s)) for s in sents if _words(s)]
    return float(sum(lengths) / len(lengths)) if lengths else 0.0


def type_token_ratio(text):
    words = [w.lower() for w in _words(text)]
    if not words:
        return 0.0
    return float(len(set(words)) / len(words))


def avg_word_length(text):
    words = _words(text)
    if not words:
        return 0.0
    return float(sum(len(w) for w in words) / len(words))


def punctuation_density(text):
    if not text:
        return 0.0
    punct = len(re.findall(r"[,;:\-—()]", text))
    return float(punct / max(len(text), 1) * 100)


def filler_phrase_rate(text):
    low = text.lower()
    count = sum(low.count(p) for p in FILLER_PHRASES)
    n_sents = max(len(_sentences(text)), 1)
    return float(count / n_sents)


def hedge_word_rate(text):
    words = [w.lower() for w in _words(text)]
    if not words:
        return 0.0
    count = sum(1 for w in words if w in HEDGE_WORDS)
    return float(count / len(words) * 100)


def contraction_rate(text):
    contractions = len(re.findall(r"\b\w+'(t|s|re|ve|ll|d|m)\b", text.lower()))
    n_sents = max(len(_sentences(text)), 1)
    return float(contractions / n_sents)


def burstiness(text):
    """Human text alternates short/long sentences more; AI text is uniform.
    Burstiness = (std - mean) / (std + mean) of sentence lengths (Ortuno et al.)."""
    sents = _sentences(text)
    lengths = [len(_words(s)) for s in sents if _words(s)]
    if len(lengths) < 2:
        return 0.0
    mean = sum(lengths) / len(lengths)
    sd = stats.pstdev(lengths)
    if sd + mean == 0:
        return 0.0
    return float((sd - mean) / (sd + mean))


FEATURE_FUNCS = {
    "sentence_length_variance": sentence_length_variance,
    "avg_sentence_length": avg_sentence_length,
    "type_token_ratio": type_token_ratio,
    "avg_word_length": avg_word_length,
    "punctuation_density": punctuation_density,
    "filler_phrase_rate": filler_phrase_rate,
    "hedge_word_rate": hedge_word_rate,
    "contraction_rate": contraction_rate,
    "burstiness": burstiness,
}


def extract_features(text):
    """Returns an ordered dict of stylometric features for one document."""
    return {name: fn(text) for name, fn in FEATURE_FUNCS.items()}


def sentence_evidence_scores(text):
    """Per-sentence 'AI-ness' heuristic used for evidence highlighting in the UI:
    combines filler-phrase presence and closeness to the doc's mean sentence
    length (uniformity is itself a signal)."""
    sents = _sentences(text)
    lengths = [len(_words(s)) for s in sents]
    mean_len = sum(lengths) / len(lengths) if lengths else 0
    scores = []
    for s, l in zip(sents, lengths):
        low = s.lower()
        filler_hit = any(p in low for p in FILLER_PHRASES)
        uniformity = 1 - min(abs(l - mean_len) / (mean_len + 1e-6), 1)
        score = 0.6 * float(filler_hit) + 0.4 * uniformity
        scores.append((s, round(score, 3)))
    return scores


if __name__ == "__main__":
    sample_ai = "In recent developments regarding the situation, it is important to note that several factors must be considered. Furthermore, experts suggest that the outcome may potentially affect residents."
    sample_human = "Okay so here's the thing, nobody actually saw it coming. I got there late and honestly it was already chaos on the block."
    print("AI sample features:", extract_features(sample_ai))
    print("Human sample features:", extract_features(sample_human))
