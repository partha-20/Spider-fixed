"""
Phase 2 — Data Pipeline
========================
Builds a labeled human vs. AI-generated text dataset.

IMPORTANT (sandbox note): this environment has no internet access to
real news corpora or LLM APIs, so this script *simulates* the data
collection step described in the roadmap:
  - "human" samples are written with deliberately human traits: variable
    sentence length, contractions, asides, opinion, minor imperfection.
  - "ai_raw" samples are generated with deliberate LLM traits: uniform
    sentence length, hedging, stock transitions, listy structure.
  - "ai_humanized" samples take ai_raw text and lightly perturb it
    (contractions, filler removal, punctuation variance) to simulate
    the adversarial "paraphrased AI" case the brief describes.

In production, replace `build_human_samples()` with real scraped/licensed
corpora and `build_ai_samples()` with actual API calls to GPT/Claude/Gemini
on the SAME topics (topic-matched, to avoid the classifier learning
topic bias instead of authorship style).
"""
import os
import random
import csv
import re

random.seed(42)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(DATA_DIR, "dataset.csv")

TOPICS = [
    "Spider-Man's rescue during the bridge collapse",
    "the new city council transit funding bill",
    "the ongoing heatwave affecting Queens",
    "the Daily Bugle's latest circulation numbers",
    "the ESU basketball team's playoff run",
    "a break-in at an Oscorp research facility",
    "the mayor's press conference on crime rates",
    "a community garden opening in Harlem",
    "the subway delays on the 4 train line",
    "a viral video showing a costumed vigilante",
    "the annual Halloween parade in the Village",
    "a labor strike at the Brooklyn shipping docks",
    "a gallery opening downtown",
    "rising rent prices in Manhattan",
    "a school fundraiser at Midtown High",
]

HUMAN_OPENERS = [
    "Look, I wasn't expecting much when I showed up, but",
    "Okay so here's the thing about",
    "I've covered a lot of stories, and honestly",
    "You'd think by now people would stop being surprised by",
    "Three witnesses told me pretty much the same thing about",
    "It's hard to explain how chaotic it got when",
    "My editor didn't want this angle, but",
    "Half the block showed up to see",
    "I'll be straight with you about",
    "Nobody saw this coming, least of all",
]

HUMAN_ASIDES = [
    "(don't ask how long that took)",
    "— which, sure, tracks —",
    "no joke",
    "I still don't buy it, honestly",
    "and yeah, it was as messy as it sounds",
    "which nobody bothered to fact-check",
    "for what it's worth",
    "go figure",
    "which is saying something",
    "still can't believe I'm writing this",
]

AI_OPENERS = [
    "In recent developments regarding",
    "It is important to note that",
    "As reports continue to emerge about",
    "Furthermore, sources indicate that",
    "In light of recent events surrounding",
    "It has become increasingly evident that",
    "Notably, the situation involving",
    "According to available information about",
    "In today's rapidly evolving landscape,",
    "Overall, it is worth highlighting that",
]

AI_TRANSITIONS = [
    "Moreover,", "In addition,", "Furthermore,", "It is also worth noting that",
    "Additionally,", "As a result,", "Consequently,", "In summary,",
    "On the other hand,", "It should be emphasized that",
]

AI_HEDGES = [
    "may potentially", "could possibly", "is generally considered",
    "is widely regarded as", "tends to suggest", "appears to indicate",
    "is often seen as", "remains to be seen", "could be argued",
]


def human_sentence(topic, i):
    frag = random.choice([
        f"honestly {topic} was wilder than anyone expected",
        f"three people I talked to about {topic} couldn't even agree on what happened",
        f"I got there late and {topic} was already old news to everyone on the block",
        f"my gut says {topic} isn't going away anytime soon",
        f"you can't make this stuff up, {topic}, of all things",
        f"{topic}? sure, why not, this city never stops",
        f"nobody warned me {topic} would take this long to sort out",
        f"I'm still annoyed about {topic}, if I'm honest",
    ])
    return frag[0].upper() + frag[1:] + "."


def build_human_sample(topic):
    n_sentences = random.randint(3, 9)
    sentences = [random.choice(HUMAN_OPENERS) + " " + topic + "."]
    for i in range(n_sentences):
        s = human_sentence(topic, i)
        if random.random() < 0.4:
            s += " " + random.choice(HUMAN_ASIDES) + "."
        sentences.append(s)
    if random.random() < 0.5:
        sentences.append(random.choice([
            "Anyway, that's where things stand.",
            "We'll see what happens next.",
            "More on this if it actually goes anywhere.",
            "Make of that what you will.",
        ]))
    if len(sentences) > 2:
        middle = sentences[1:-1]
        random.shuffle(middle)
        sentences[1:-1] = middle
    else:
        random.shuffle(sentences)
    return " ".join(sentences)


def ai_sentence(topic):
    hedge = random.choice(AI_HEDGES)
    frag = random.choice([
        f"the situation surrounding {topic} {hedge} reflect broader trends in the community",
        f"analysts note that {topic} {hedge} have significant implications going forward",
        f"it is important to consider that {topic} {hedge} affect residents in the area",
        f"the developments related to {topic} {hedge} warrant further monitoring",
        f"experts suggest that {topic} {hedge} require continued attention from officials",
    ])
    return frag[0].upper() + frag[1:] + "."


def build_ai_raw_sample(topic):
    n_sentences = random.randint(5, 7)
    sentences = [random.choice(AI_OPENERS) + " " + topic + ", several key factors must be considered."]
    for i in range(n_sentences):
        s = (random.choice(AI_TRANSITIONS) + " " + ai_sentence(topic)) if i > 0 else ai_sentence(topic)
        sentences.append(s)
    sentences.append(
        f"In conclusion, {topic} highlights the importance of continued vigilance and community engagement."
    )
    return " ".join(sentences)


def humanize(text):
    """Simulate a villain 'humanizing' AI output to evade detection."""
    replacements = {
        "It is important to note that": "Worth noting,",
        "Furthermore,": "Also,",
        "Moreover,": "Plus,",
        "In addition,": "Also,",
        "In conclusion,": "Bottom line,",
        "In summary,": "So basically,",
        "is generally considered": "is kind of seen as",
        "is widely regarded as": "is basically seen as",
        "may potentially": "might",
        "could possibly": "could",
        "As a result,": "So,",
        "Consequently,": "So,",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # break up a couple of long sentences to reduce uniformity
    sentences = re.split(r'(?<=[.!?]) ', text)
    if len(sentences) > 3:
        idx = random.randint(1, len(sentences) - 2)
        parts = sentences[idx].split(",", 1)
        if len(parts) == 2:
            sentences[idx] = parts[0].strip() + "."
            sentences.insert(idx + 1, parts[1].strip().capitalize())
    return " ".join(sentences)


def build_dataset(n_human=220, n_ai_raw=110, n_ai_humanized=110):
    rows = []
    for _ in range(n_human):
        topic = random.choice(TOPICS)
        rows.append({"text": build_human_sample(topic), "label": "human", "source_type": random.choice(["article", "social_post", "interview"])})
    for _ in range(n_ai_raw):
        topic = random.choice(TOPICS)
        rows.append({"text": build_ai_raw_sample(topic), "label": "ai", "source_type": random.choice(["article", "social_post", "interview"])})
    for _ in range(n_ai_humanized):
        topic = random.choice(TOPICS)
        raw = build_ai_raw_sample(topic)
        rows.append({"text": humanize(raw), "label": "ai", "source_type": random.choice(["article", "social_post", "interview"]), "adversarial": True})
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    rows = build_dataset(n_human=220, n_ai_raw=110, n_ai_humanized=110)
    fieldnames = ["text", "label", "source_type", "adversarial"]
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            r.setdefault("adversarial", False)
            writer.writerow(r)
    print(f"Wrote {len(rows)} rows to dataset.csv")
    n_ai = sum(1 for r in rows if r["label"] == "ai")
    n_adv = sum(1 for r in rows if r.get("adversarial"))
    print(f"human={len(rows)-n_ai}  ai={n_ai}  (of which humanized/adversarial={n_adv})")
