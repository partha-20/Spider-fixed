import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from perplexity import NgramLM  # noqa: E402

HUMAN_TRAIN_TEXTS = [
    "Okay so here's the thing, nobody actually saw it coming.",
    "I got there late and honestly it was already chaos on the block.",
    "My editor didn't want this angle, but I'm running it anyway.",
    "We'll see what happens next, no joke.",
]

AI_TEXT = (
    "In recent developments regarding the situation, it is important to note that "
    "several factors must be considered for the outcome going forward."
)


def test_lm_fits_without_error():
    lm = NgramLM().fit(HUMAN_TRAIN_TEXTS)
    assert len(lm.vocab) > 0


def test_perplexity_is_lower_for_similar_style_text():
    lm = NgramLM().fit(HUMAN_TRAIN_TEXTS)
    human_like = "Okay so here's the thing, I got there late."
    p_human = lm.perplexity(human_like)
    p_ai = lm.perplexity(AI_TEXT)
    assert p_human < p_ai


def test_perplexity_handles_short_text():
    lm = NgramLM().fit(HUMAN_TRAIN_TEXTS)
    assert lm.perplexity("Hi.") >= 0


def test_burstiness_of_surprisal_is_non_negative():
    lm = NgramLM().fit(HUMAN_TRAIN_TEXTS)
    assert lm.burstiness_of_surprisal(AI_TEXT) >= 0
