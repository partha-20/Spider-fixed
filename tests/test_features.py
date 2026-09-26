import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from features import extract_features, sentence_evidence_scores  # noqa: E402

AI_SAMPLE = (
    "In recent developments regarding the situation, it is important to note that "
    "several factors must be considered. Furthermore, experts suggest that the outcome "
    "may potentially affect residents."
)
HUMAN_SAMPLE = (
    "Okay so here's the thing, nobody actually saw it coming. I got there late and "
    "honestly it was already chaos on the block."
)


def test_extract_features_returns_all_expected_keys():
    feats = extract_features(HUMAN_SAMPLE)
    expected = {
        "sentence_length_variance", "avg_sentence_length", "type_token_ratio",
        "avg_word_length", "punctuation_density", "filler_phrase_rate",
        "hedge_word_rate", "contraction_rate", "burstiness",
    }
    assert set(feats.keys()) == expected


def test_ai_sample_has_higher_filler_and_hedge_rates_than_human():
    ai_feats = extract_features(AI_SAMPLE)
    human_feats = extract_features(HUMAN_SAMPLE)
    assert ai_feats["filler_phrase_rate"] > human_feats["filler_phrase_rate"]
    assert ai_feats["hedge_word_rate"] > human_feats["hedge_word_rate"]


def test_human_sample_has_more_contractions_than_ai():
    ai_feats = extract_features(AI_SAMPLE)
    human_feats = extract_features(HUMAN_SAMPLE)
    assert human_feats["contraction_rate"] > ai_feats["contraction_rate"]


def test_empty_text_does_not_crash():
    feats = extract_features("")
    assert all(v == 0.0 for v in feats.values())


def test_sentence_evidence_scores_returns_pairs():
    result = sentence_evidence_scores(AI_SAMPLE)
    assert len(result) > 0
    for sentence, score in result:
        assert isinstance(sentence, str)
        assert 0.0 <= score <= 1.0
