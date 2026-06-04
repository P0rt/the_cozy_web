# The Comments Got Good. That's How I Knew.

A first-person investigation: I wrote a post about model distillation, the
comments were suspiciously thoughtful, and I went looking for whether any of
them were written by people. The trail runs from one dev.to thread, through a
38-article sweep, into "dead internet theory" and the peer-reviewed research on
why you can't tell LLM comments from human ones anymore.

It's a **hybrid**: an essay with real, cited sources, plus working code that (a)
tears apart real dev.to threads via the public API, and (b) models what rising
automation does to a conversation.

👉 **The essay:** [`the_cozy_web.md`](the_cozy_web.md)

## Contents

| file | what it is |
|---|---|
| [`the_cozy_web.md`](the_cozy_web.md) | The essay. Cited inline. Written to be publishable on dev.to. |
| [`analyze_devto.py`](analyze_devto.py) | Pull a real dev.to thread via the API; score it with the v1 detector (watch it fail) and compute v2 "eco-astroturf" signals (product plugs, validation-openers, throwaway usernames). |
| [`sweep_devto.py`](sweep_devto.py) | Cross-post sweep: is the eco-comment template platform-wide? Pulls ~38 articles, finds accounts spraying it across dozens of threads + 4-grams reused across distinct accounts. |
| [`dead_internet_sim.py`](dead_internet_sim.py) | Agent-based sim: sweep "community autonomy" 0→1, watch a thread's *liveness* collapse. Finds a knee ≈ 0.65; disagreement dies first. |
| [`coziness_detector.py`](coziness_detector.py) | A transparent heuristic scorer for the *old* "Great post!" style — kept around to demonstrate why it no longer works on substantive AI comments. |
| `figures/` | Generated charts (created by running the sim + detector). |

## Run it

```bash
pip install -r requirements.txt

python3 dead_internet_sim.py     # -> figures/liveness_vs_autonomy.png, figures/effective_vocab.png
python3 coziness_detector.py     # -> figures/coziness_hist.png
python3 analyze_devto.py         # tears apart a real thread (defaults to the distillation post)
python3 sweep_devto.py           # the cross-platform template sweep (hits the live dev.to API)
```

The two simulation scripts are seeded (`numpy.default_rng(7)`), so their figures
are reproducible. The two `*_devto.py` scripts hit the **live** dev.to API, so
their exact numbers drift as new comments arrive.

## The headline findings

- **On my own post:** my old detector scored the 8 substantive comments at a mean
  "coziness" of **0.25** — it waved them through as human. The real tells were
  structural: 4/8 were product plugs, 5/8 opened by validating a phrase, and only
  2/8 ever pushed back (both of which I conceded instantly).
- **Across 38 posts / 1,366 comments / 346 accounts:** the same accounts spray the
  same "validate → nuance → we-at-Product → number" skeleton across **14–22
  unrelated threads each**, and distinct accounts reuse identical 4-grams
  (`"exactly the kind of"` across 13 accounts). Humans don't converge like that.
- **The detector fails *because* of the specifics.** Light paraphrasing collapses
  perplexity-based detection ([Krishna et al., NeurIPS 2023](https://arxiv.org/abs/2303.13408));
  fake-but-substantive content is now indistinguishable to people
  ([Meng et al., 2025](https://arxiv.org/abs/2506.13313)). Specificity is camouflage, not proof of a human.
- **In simulation, liveness collapses non-linearly** — it halves once the *average*
  poster is ~0.65 on the human→autonomous dial, and disagreement is the first thing to die.

## Honesty notes

The simulation is a toy and the detector is a strawman shown failing on purpose —
neither is evidence, and neither should gate real people. I can't prove any single
named account is a bot; the claim is about aggregate texture. The Zurich
r/changemyview study is cited as a *withdrawn* preprint, and "% of the web is
bots" figures are detector-dependent. Every load-bearing claim links to
peer-reviewed work.
