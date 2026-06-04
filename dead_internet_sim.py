"""
dead_internet_sim.py
====================

An agent-based toy model of what happens to a comment thread as participants
move along the *autonomy spectrum* — from "I typed this myself, annoyed" to
"my assistant drafted it" to "an agent posts on my behalf and I never read the
thread."

The user's thesis, restated operationally: a blog feels "cozy" not because the
community got nicer, but because the *high-entropy parts of human conversation*
— disagreement, tangents, typos, oddly specific war stories — are exactly the
parts an LLM smooths away. Crank up autonomy and you don't get a worse
conversation; you get a *flatter* one. The friction that made it feel alive is
the first thing to go.

We don't simulate language. We simulate the STATISTICS of language, because the
thesis is statistical:

  * Every comment is a bag of K tokens.
  * Tokens come from two pools:
      - HUMAN pool: a big Zipfian vocabulary with a fat tail. The tail is where
        the topic-specific terms, the typos, the tangents live. High entropy.
      - COZY pool: a tiny near-uniform vocabulary of phatic/positive tokens
        ("great", "thanks", "insightful", ...). Low entropy.
  * Each comment has an *assist level* alpha in [0,1]. With probability alpha a
    given token is drawn from the COZY pool instead of the HUMAN pool, and the
    comment's stance is pulled toward agreement. alpha is the autonomy dial,
    applied per-comment with noise around a community mean.

Then we sweep the community-mean autonomy from 0 -> 1 and watch four "liveness"
metrics, plus a composite Liveness Index, collapse. The collapse is not linear:
there is a knee — a point past which a thread is statistically a smooth surface.

Run:  python3 dead_internet_sim.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


RNG = np.random.default_rng(7)  # fixed seed: the charts are reproducible.


# --------------------------------------------------------------------------- #
# Vocabulary pools
# --------------------------------------------------------------------------- #

HUMAN_VOCAB = 6000          # big, fat-tailed: room for specifics + tangents
COZY_VOCAB = 60             # small, near-uniform: phatic praise
HUMAN_ZIPF_S = 1.07         # Zipf exponent; ~natural language
COMMENT_LEN = 28            # tokens per comment

# Precompute Zipf weights for the human pool once.
_ranks = np.arange(1, HUMAN_VOCAB + 1)
_human_weights = 1.0 / np.power(_ranks, HUMAN_ZIPF_S)
_human_weights /= _human_weights.sum()

# Cozy pool: gently skewed but basically flat -> very low entropy as a *pool*,
# and crucially a TINY set of types, so reuse across comments is near-total.
_cozy_weights = 1.0 / np.power(np.arange(1, COZY_VOCAB + 1), 0.4)
_cozy_weights /= _cozy_weights.sum()

# Token id spaces are kept disjoint so we can tell pools apart downstream.
COZY_OFFSET = HUMAN_VOCAB


@dataclass
class Comment:
    tokens: np.ndarray      # token ids
    stance: int             # -1 disagree, 0 neutral, +1 agree
    alpha: float            # this comment's realised autonomy


# --------------------------------------------------------------------------- #
# Generating comments
# --------------------------------------------------------------------------- #

def make_comment(alpha: float) -> Comment:
    """One comment at autonomy level alpha in [0,1]."""
    alpha = float(np.clip(alpha, 0.0, 1.0))

    # How many of this comment's tokens come from the cozy pool?
    n_cozy = RNG.binomial(COMMENT_LEN, alpha)
    n_human = COMMENT_LEN - n_cozy

    human_tok = RNG.choice(HUMAN_VOCAB, size=n_human, p=_human_weights)
    cozy_tok = RNG.choice(COZY_VOCAB, size=n_cozy, p=_cozy_weights) + COZY_OFFSET
    tokens = np.concatenate([human_tok, cozy_tok])

    # Stance: real humans disagree ~30% of the time; assistance sands that down.
    # P(disagree) shrinks with alpha; P(agree) grows toward the cozy ceiling.
    p_disagree = 0.30 * (1 - alpha)
    p_agree = 0.45 + 0.50 * alpha
    p_neutral = max(0.0, 1.0 - p_disagree - p_agree)
    stance = RNG.choice([-1, 0, 1], p=_norm([p_disagree, p_neutral, p_agree]))

    return Comment(tokens=tokens, stance=int(stance), alpha=alpha)


def _norm(xs: list[float]) -> np.ndarray:
    a = np.asarray(xs, dtype=float)
    a[a < 0] = 0
    return a / a.sum()


def make_thread(mean_autonomy: float, n_comments: int = 60) -> list[Comment]:
    """A thread whose comments scatter around a community-mean autonomy."""
    # Per-comment autonomy: Beta distribution centred near mean_autonomy so the
    # community is a MIX of fully-human, assisted, and fully-autonomous posters.
    m = np.clip(mean_autonomy, 0.02, 0.98)
    conc = 6.0                      # spread; lower = more polarised population
    a_param = m * conc
    b_param = (1 - m) * conc
    alphas = RNG.beta(a_param, b_param, size=n_comments)
    return [make_comment(a) for a in alphas]


# --------------------------------------------------------------------------- #
# Liveness metrics
# --------------------------------------------------------------------------- #

def lexical_diversity(thread: list[Comment]) -> float:
    """Type-token ratio across the whole thread. Reuse of phatic tokens tanks it."""
    all_tok = np.concatenate([c.tokens for c in thread])
    return len(np.unique(all_tok)) / len(all_tok)


def effective_vocabulary(thread: list[Comment]) -> float:
    """exp(Shannon entropy) of the token distribution: the 'how many words is
    this thread really using' number. A Hill number of order 1."""
    all_tok = np.concatenate([c.tokens for c in thread])
    _, counts = np.unique(all_tok, return_counts=True)
    p = counts / counts.sum()
    h = -np.sum(p * np.log(p))
    return float(np.exp(h))


def disagreement_rate(thread: list[Comment]) -> float:
    return float(np.mean([c.stance == -1 for c in thread]))


def mean_surprise(thread: list[Comment]) -> float:
    """Average information each comment ADDS: fraction of its tokens never seen
    earlier in the thread. This is the 'did anyone say anything new' metric."""
    seen: set[int] = set()
    surprises = []
    for c in thread:
        toks = c.tokens
        novel = np.fromiter((t not in seen for t in toks), dtype=bool, count=len(toks))
        surprises.append(float(novel.mean()))
        seen.update(int(t) for t in toks)
    return float(np.mean(surprises))


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #

@dataclass
class Point:
    autonomy: float
    diversity: float
    eff_vocab: float
    disagreement: float
    surprise: float
    liveness: float


def sweep(levels: int = 21, threads_per: int = 40) -> list[Point]:
    out = []
    for a in np.linspace(0.0, 1.0, levels):
        div, ev, dis, sur = [], [], [], []
        for _ in range(threads_per):
            th = make_thread(a)
            div.append(lexical_diversity(th))
            ev.append(effective_vocabulary(th))
            dis.append(disagreement_rate(th))
            sur.append(mean_surprise(th))
        d, e, g, s = np.mean(div), np.mean(ev), np.mean(dis), np.mean(sur)
        out.append(Point(a, d, e, g, s, liveness=0.0))

    # Liveness Index: geometric mean of the three normalised "human-ness"
    # channels (diversity, disagreement, surprise), each scaled to its value at
    # autonomy=0. Geometric mean so that zeroing ANY channel kills liveness —
    # a thread with no disagreement is dead even if it's lexically diverse.
    base = out[0]
    for p in out:
        nd = p.diversity / base.diversity
        ng = p.disagreement / base.disagreement if base.disagreement else 0.0
        ns = p.surprise / base.surprise
        p.liveness = float((max(nd, 1e-9) * max(ng, 1e-9) * max(ns, 1e-9)) ** (1 / 3))
    return out


def find_knee(points: list[Point]) -> float:
    """Autonomy at which Liveness Index first drops below half its starting value."""
    start = points[0].liveness
    for p in points:
        if p.liveness < 0.5 * start:
            return p.autonomy
    return 1.0


# --------------------------------------------------------------------------- #
# Report + plots
# --------------------------------------------------------------------------- #

def _print_table(points: list[Point]) -> None:
    print("=" * 72)
    print("DEAD INTERNET SIM — liveness vs community autonomy")
    print("=" * 72)
    print(f"{'autonomy':>9} {'diversity':>10} {'eff_vocab':>10} "
          f"{'disagree':>9} {'surprise':>9} {'LIVENESS':>9}")
    for p in points:
        print(f"{p.autonomy:9.2f} {p.diversity:10.3f} {p.eff_vocab:10.0f} "
              f"{p.disagreement:9.3f} {p.surprise:9.3f} {p.liveness:9.3f}")
    knee = find_knee(points)
    print("-" * 72)
    print(f"Liveness halves at autonomy ~= {knee:.2f}.")
    print("Below that knee the thread is statistically a smooth surface: still")
    print("polite, still 'engaged', but contributing almost no new information.")


def _plot(points: list[Point]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(skipped plots: {e})")
        return

    os.makedirs("figures", exist_ok=True)
    a = [p.autonomy for p in points]

    # Chart 1: normalised liveness channels + composite index.
    base = points[0]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(a, [p.diversity / base.diversity for p in points],
            label="lexical diversity", color="#2a9d8f", lw=2)
    ax.plot(a, [(p.disagreement / base.disagreement) if base.disagreement else 0
                for p in points], label="disagreement", color="#e76f51", lw=2)
    ax.plot(a, [p.surprise / base.surprise for p in points],
            label="surprise / novelty", color="#e9c46a", lw=2)
    ax.plot(a, [p.liveness for p in points],
            label="LIVENESS INDEX", color="#264653", lw=3.2)
    knee = find_knee(points)
    ax.axvline(knee, ls="--", c="#7d7d7d")
    ax.axhline(0.5 * points[0].liveness, ls=":", c="#7d7d7d")
    ax.annotate(f"knee ≈ {knee:.2f}", xy=(knee, 0.52), xytext=(knee + 0.04, 0.72),
                arrowprops=dict(arrowstyle="->", color="#7d7d7d"))
    ax.set_xlabel("community mean autonomy  (human-typed → agent-posted)")
    ax.set_ylabel("fraction of all-human baseline")
    ax.set_title("As autonomy rises, the high-entropy parts of conversation die first")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig("figures/liveness_vs_autonomy.png", dpi=120)
    print("saved figures/liveness_vs_autonomy.png")

    # Chart 2: effective vocabulary collapse (absolute, the 'how many words').
    fig2, ax2 = plt.subplots(figsize=(9, 4.6))
    ax2.fill_between(a, [p.eff_vocab for p in points], color="#2a9d8f", alpha=0.25)
    ax2.plot(a, [p.eff_vocab for p in points], color="#2a9d8f", lw=2.5)
    ax2.set_xlabel("community mean autonomy")
    ax2.set_ylabel("effective vocabulary  exp(H)")
    ax2.set_title("A 'cozy' thread literally uses fewer distinct words")
    fig2.tight_layout()
    fig2.savefig("figures/effective_vocab.png", dpi=120)
    print("saved figures/effective_vocab.png")


def main() -> None:
    points = sweep()
    _print_table(points)
    _plot(points)


if __name__ == "__main__":
    main()
