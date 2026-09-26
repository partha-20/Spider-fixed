"""
Phase 4 — Signal Engine A: Perplexity / Predictability
========================================================
A trainable-from-scratch bigram language model, used as a lightweight,
fully-offline stand-in for the "reference LM" (GPT-2/Pythia) specified
in the roadmap. GPT-2-class checkpoints require downloading from
huggingface.co, which this sandbox cannot reach.

Swap-in path for production: replace `NgramLM` with
`transformers.AutoModelForCausalLM.from_pretrained("gpt2")` and compute
true token-level log-probabilities the same way DetectGPT/GPTZero do —
the rest of the pipeline (ensemble, API contract) does not need to change,
since this module's public interface (`score(text) -> float`) stays the same.

Concept: text that is *unsurprising* to a language model (low perplexity)
is more likely machine-generated, because LLMs are trained to produce
likely continuations. This bigram model is trained on the human-written
portion of the training set, so "surprising to this model" approximates
"surprising relative to natural human phrasing."
"""
import re
import math
from collections import defaultdict, Counter


def _tokenize(text):
    return re.findall(r"[a-z']+", text.lower())


class NgramLM:
    def __init__(self):
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.vocab = set()

    def fit(self, texts):
        for text in texts:
            tokens = ["<s>"] + _tokenize(text) + ["</s>"]
            for w in tokens:
                self.vocab.add(w)
            for i in range(len(tokens) - 1):
                self.unigram_counts[tokens[i]] += 1
                self.bigram_counts[tokens[i]][tokens[i + 1]] += 1
        return self

    def _bigram_logprob(self, w1, w2):
        vocab_size = max(len(self.vocab), 1)
        count_bigram = self.bigram_counts[w1][w2]
        count_unigram = self.unigram_counts[w1]
        # add-1 (Laplace) smoothing
        prob = (count_bigram + 1) / (count_unigram + vocab_size)
        return math.log(prob)

    def perplexity(self, text):
        tokens = ["<s>"] + _tokenize(text) + ["</s>"]
        if len(tokens) < 2:
            return 0.0
        logprob_sum = 0.0
        for i in range(len(tokens) - 1):
            logprob_sum += self._bigram_logprob(tokens[i], tokens[i + 1])
        avg_neg_logprob = -logprob_sum / (len(tokens) - 1)
        return math.exp(avg_neg_logprob)

    def burstiness_of_surprisal(self, text):
        """Variance of per-token surprisal — human text has 'bursty', uneven
        surprisal; AI text is more uniformly predictable token-to-token."""
        tokens = ["<s>"] + _tokenize(text) + ["</s>"]
        if len(tokens) < 3:
            return 0.0
        surprisals = [-self._bigram_logprob(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
        mean = sum(surprisals) / len(surprisals)
        var = sum((s - mean) ** 2 for s in surprisals) / len(surprisals)
        return float(var)


if __name__ == "__main__":
    import csv
    import os
    _data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset.csv"
    )
    rows = list(csv.DictReader(open(_data_path, encoding="utf-8")))
    human_texts = [r["text"] for r in rows if r["label"] == "human"]
    lm = NgramLM().fit(human_texts)

    ai_texts = [r["text"] for r in rows if r["label"] == "ai"][:5]
    print("--- Perplexity under human-trained bigram LM ---")
    for t in human_texts[:5]:
        print("HUMAN  ppl=%.1f" % lm.perplexity(t))
    for t in ai_texts:
        print("AI     ppl=%.1f" % lm.perplexity(t))
