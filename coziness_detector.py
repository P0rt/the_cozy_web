"""
coziness_detector.py
====================

A small, transparent, *heuristic* detector for what this project calls
"AI coziness" in blog comments — the smooth, uniformly-supportive, low-surprise
register that appears when a comment was produced (or heavily assisted) by an LLM
rather than written by an irritated, distracted, opinionated human.

This is NOT a production AI-text classifier. Real detectors (GPTZero, DetectGPT)
estimate token-level perplexity and burstiness with an actual language model, and
even they are unreliable enough that schools and journals have walked back their
use. This file deliberately uses cheap, *interpretable* surface features so you
can read every number and disagree with it. The point of the essay is the
mechanism, not a magic classifier.

The features, and why each one points at "cozy":

  burstiness          Humans write in lumps: a three-word sentence, then a
                      run-on. LLMs regress to a comfortable mean sentence length.
                      Low burstiness -> cozy. (This is the one idea GPTZero is
                      actually built on.)
  cliche_density      "Great post!", "Thanks for sharing", "well written".
                      Phatic filler that says *I am being supportive* and nothing
                      about the post. High -> cozy.
  marker_words        The LLM house style: delve, tapestry, moreover, leverage,
                      underscore, pivotal, realm, landscape, testament. High -> cozy.
  politeness_hedging  "I think", "perhaps", "it's worth noting", "great question".
                      Uniform deference. High -> cozy.
  em_dash_rate        LLMs love an em dash—like this. Per 100 words. High -> cozy.
  sentiment_uniformity How one-note the affect is. Real threads contain a grump.
                      All-positive, zero-variance -> cozy.
  specificity         The ANTI-cozy feature. Numbers, code tokens, named tools,
                      version strings, concrete disagreement. Humans cite the
                      thing. Generic praise doesn't. High specificity -> human.

Each feature is mapped to [0,1] and combined with hand-set weights into a single
coziness score in [0,1]. Calibrate the weights to taste; they are right there.

Run:  python3 coziness_detector.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import mean, pstdev


# --------------------------------------------------------------------------- #
# Lexicons. Small on purpose — read them, argue with them.
# --------------------------------------------------------------------------- #

CLICHE_PHRASES = [
    "great post", "great article", "great write", "great read", "nice post",
    "nice article", "thanks for sharing", "thank you for sharing",
    "well written", "well explained", "really insightful", "very insightful",
    "this is gold", "this is so helpful", "love this", "couldn't agree more",
    "could not agree more", "keep up the good work", "keep up the great work",
    "looking forward to", "spot on", "this resonates", "well said",
    "amazing work", "fantastic post", "very informative", "super helpful",
    "exactly what i needed", "you nailed it",
]

MARKER_WORDS = {
    "delve", "delved", "delving", "tapestry", "moreover", "furthermore",
    "leverage", "leveraging", "underscore", "underscores", "pivotal", "realm",
    "landscape", "testament", "intricate", "intricacies", "seamless",
    "seamlessly", "robust", "holistic", "nuanced", "facet", "facets",
    "paradigm", "myriad", "plethora", "elevate", "elevating", "navigate",
    "navigating", "embark", "unlock", "unlocking", "foster", "fostering",
}

HEDGE_POLITE = [
    "i think", "i believe", "in my opinion", "it's worth noting",
    "it is worth noting", "it's important to note", "it is important to note",
    "perhaps", "arguably", "to be fair", "great question", "good question",
    "that said", "if i may", "just my two cents", "correct me if i'm wrong",
]

POSITIVE_WORDS = {
    "great", "love", "loved", "amazing", "awesome", "excellent", "fantastic",
    "wonderful", "helpful", "insightful", "brilliant", "perfect", "nice",
    "good", "best", "clear", "clean", "elegant", "beautiful", "appreciate",
    "thanks", "thank", "useful", "valuable", "inspiring", "solid",
}
NEGATIVE_WORDS = {
    "wrong", "bad", "broken", "buggy", "slow", "confusing", "misleading",
    "disagree", "actually", "but", "however", "no", "not", "doesn't",
    "won't", "fails", "fail", "hate", "terrible", "useless", "nonsense",
    "incorrect", "flawed", "footgun", "antipattern", "smells", "leak",
}

# Things only a human-with-a-specific-grievance tends to drop in.
CODE_TOKEN_RE = re.compile(r"`[^`]+`|\b\w+\(\)|\b\w+\.\w+\b|--\w+|\bv?\d+\.\d+")
NUMBER_RE = re.compile(r"\b\d+(\.\d+)?\b")
URL_RE = re.compile(r"https?://\S+")
NAMED_TOOL_RE = re.compile(
    r"\b(react|vue|svelte|rust|golang|postgres|redis|kafka|docker|k8s|"
    r"webpack|vite|eslint|tsc|pytest|numpy|pandas|nginx|grpc|wasm|"
    r"tailwind|nextjs|deno|bun)\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Tokenisation helpers
# --------------------------------------------------------------------------- #

def _sentences(text: str) -> list[str]:
    parts = re.split(r"[.!?]+(?:\s+|$)", text.strip())
    return [p for p in parts if p.strip()]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _clip01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #

@dataclass
class Features:
    burstiness: float
    cliche_density: float
    marker_words: float
    politeness_hedging: float
    em_dash_rate: float
    sentiment_uniformity: float
    specificity: float
    raw: dict = field(default_factory=dict)


def extract_features(text: str) -> Features:
    words = _words(text)
    n_words = max(len(words), 1)
    sents = _sentences(text)
    low = text.lower()

    # --- burstiness: 1 - normalised variation of sentence length -----------
    # Human writing varies wildly; LLM writing clusters near a mean.
    # We map "uniform" -> high cozy. Coefficient of variation, inverted.
    if len(sents) >= 2:
        lengths = [len(_words(s)) for s in sents]
        m = mean(lengths) or 1.0
        cv = pstdev(lengths) / m            # 0 = perfectly uniform
        burst_human = _clip01(cv / 0.7)     # ~0.7 CV reads as very human
    else:
        burst_human = 0.15                  # a single sentence is mildly cozy
    burstiness = 1.0 - burst_human          # high = cozy/uniform

    # --- cliche density: phatic filler per comment -------------------------
    cliche_hits = sum(low.count(p) for p in CLICHE_PHRASES)
    cliche_density = _clip01(cliche_hits / 1.0 * 0.6)   # one cliche already loud

    # --- marker words: the LLM house style ---------------------------------
    marker_hits = sum(1 for w in words if w in MARKER_WORDS)
    marker_words = _clip01(marker_hits / n_words * 25.0)

    # --- politeness / hedging ----------------------------------------------
    hedge_hits = sum(low.count(p) for p in HEDGE_POLITE)
    politeness_hedging = _clip01(hedge_hits / 2.0)

    # --- em dash rate ------------------------------------------------------
    em_dashes = text.count("—") + len(re.findall(r"\s--\s|\w--\w", text))
    em_dash_rate = _clip01(em_dashes / n_words * 120.0)

    # --- sentiment uniformity ----------------------------------------------
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    total_sent = pos + neg
    if total_sent == 0:
        sentiment_uniformity = 0.5          # neutral text: ambiguous
    else:
        # all-positive with zero friction -> maximally cozy
        polarity = (pos - neg) / total_sent          # +1 .. -1
        friction = neg / total_sent                  # 0 .. 1
        sentiment_uniformity = _clip01(polarity * (1 - friction))

    # --- specificity (anti-cozy) -------------------------------------------
    spec_hits = (
        len(CODE_TOKEN_RE.findall(text))
        + len(NUMBER_RE.findall(text))
        + len(URL_RE.findall(text))
        + len(NAMED_TOOL_RE.findall(text))
    )
    specificity = _clip01(spec_hits / n_words * 18.0)

    return Features(
        burstiness=burstiness,
        cliche_density=cliche_density,
        marker_words=marker_words,
        politeness_hedging=politeness_hedging,
        em_dash_rate=em_dash_rate,
        sentiment_uniformity=sentiment_uniformity,
        specificity=specificity,
        raw=dict(
            n_words=n_words, n_sents=len(sents), cliche_hits=cliche_hits,
            marker_hits=marker_hits, hedge_hits=hedge_hits, em_dashes=em_dashes,
            pos=pos, neg=neg, spec_hits=spec_hits,
        ),
    )


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

# Hand-set weights. Positive = pushes toward "cozy/AI"; specificity is negative.
WEIGHTS = {
    "burstiness":           0.22,
    "cliche_density":       0.24,
    "marker_words":         0.18,
    "politeness_hedging":   0.12,
    "em_dash_rate":         0.10,
    "sentiment_uniformity": 0.20,
    "specificity":         -0.34,   # the human's escape hatch
}
BIAS = 0.04


def coziness_score(text: str) -> float:
    """Return a coziness score in [0,1]. 1 = textbook AI-cozy, 0 = cranky human."""
    f = extract_features(text)
    s = BIAS
    for name, w in WEIGHTS.items():
        s += w * getattr(f, name)
    return _clip01(s)


def explain(text: str) -> str:
    f = extract_features(text)
    lines = [f"  score = {coziness_score(text):.2f}"]
    for name in WEIGHTS:
        lines.append(f"    {name:<22} {getattr(f, name):.2f}  (w={WEIGHTS[name]:+.2f})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Demo / mini-evaluation on labelled synthetic comments
# --------------------------------------------------------------------------- #

# These are HAND-WRITTEN, clearly-labelled synthetic examples. They are caricatures
# meant to show the detector's axis of variation, not scraped ground truth.
SAMPLES = {
    "human": [
        "wait, doesn't useMemo here just recompute every render because the dep "
        "array has an object literal in it? line 14. that's the bug, not the cache.",
        "nah. tried this exact setup with postgres 14 and the LATERAL join was "
        "3x slower than the subquery. benchmark or it didn't happen.",
        "ok but you never mention what happens when the websocket drops mid-stream. "
        "we lost 4 hours to this last sprint. reconnect logic is the whole game.",
        "honestly this reads like the docs. what's the actual gotcha you hit?",
        "i mean it works but `Array.from({length:n})` allocates twice. minor.",
        "disagree. monorepos are great until your CI is 40 min and nobody can "
        "tell which package broke. been there.",
    ],
    "ai_cozy": [
        "Great post! This is really insightful and well written. Thanks for sharing "
        "your knowledge with the community — looking forward to your next article.",
        "What a fantastic write-up. You've done an excellent job delving into the "
        "intricacies of this topic. It's important to note how robust and seamless "
        "your approach is. Keep up the great work!",
        "This resonates with me deeply. Moreover, your explanation underscores the "
        "pivotal role of clean architecture in the modern development landscape. "
        "Truly a testament to thoughtful engineering.",
        "Amazing work! I couldn't agree more with your points. This is so helpful "
        "and exactly what I needed today. Well said and beautifully explained.",
        "Thank you for this valuable contribution. Your insights navigate the "
        "complex realm of software design with remarkable clarity and elegance.",
        "Spot on! Such a clear and elegant breakdown. I appreciate how you foster "
        "understanding for developers at every level. Inspiring and informative.",
    ],
}


def _run_demo() -> None:
    print("=" * 72)
    print("AI-COZINESS DETECTOR — labelled synthetic demo")
    print("=" * 72)

    scores = {"human": [], "ai_cozy": []}
    for label, texts in SAMPLES.items():
        print(f"\n[{label}]")
        for t in texts:
            s = coziness_score(t)
            scores[label].append(s)
            bar = "#" * int(s * 30)
            print(f"  {s:.2f} |{bar:<30}| {t[:54]}...")

    h, a = scores["human"], scores["ai_cozy"]
    print("\n" + "-" * 72)
    print(f"human   mean coziness: {mean(h):.2f}   (max {max(h):.2f})")
    print(f"ai_cozy mean coziness: {mean(a):.2f}   (min {min(a):.2f})")
    thr = (mean(h) + mean(a)) / 2
    correct = sum(s < thr for s in h) + sum(s >= thr for s in a)
    print(f"threshold {thr:.2f} -> {correct}/{len(h)+len(a)} separated")
    print("\nNote: separation on caricatures is easy. On real mixed-autonomy")
    print("comments the distributions overlap badly — which is exactly the point.")

    # Optional histogram if matplotlib is present.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import os

        os.makedirs("figures", exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4.2))
        bins = [i / 20 for i in range(21)]
        ax.hist(h, bins=bins, alpha=0.7, label="human (cranky)", color="#2a9d8f")
        ax.hist(a, bins=bins, alpha=0.7, label="AI-cozy", color="#e76f51")
        ax.axvline(thr, ls="--", c="#264653", label=f"threshold {thr:.2f}")
        ax.set_xlabel("coziness score")
        ax.set_ylabel("comments")
        ax.set_title("Coziness score: human vs AI-cozy (synthetic caricatures)")
        ax.legend()
        fig.tight_layout()
        fig.savefig("figures/coziness_hist.png", dpi=120)
        print("\nsaved figures/coziness_hist.png")
    except Exception as e:  # pragma: no cover
        print(f"\n(skipped plot: {e})")


if __name__ == "__main__":
    _run_demo()
