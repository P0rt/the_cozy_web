# The Cozy Web Is a Dead Internet With Good Manners

A small research project investigating a specific, falsifiable version of "dead
internet theory": **dev blogs (including [DEV/dev.to](https://dev.to)) feel
cozy not because the community got kinder, but because real reading-and-replying
is being replaced — across a spectrum of autonomy — by AI that produces
pleasant, low-surprise text.** The high-entropy parts of human conversation
(disagreement, specificity, surprise) are exactly the parts an LLM smooths away.

It's a **hybrid**: an essay with real, cited sources, plus two working code
demos that make the argument precise (and show where it's just a caricature).

## Contents

| file | what it is |
|---|---|
| [`the_cozy_web.md`](the_cozy_web.md) | The essay. Cited inline. Written to be publishable on dev.to. |
| [`dead_internet_sim.py`](dead_internet_sim.py) | Agent-based sim: sweep "community autonomy" 0→1, watch a thread's *liveness* collapse. Finds a knee ≈ 0.65. |
| [`coziness_detector.py`](coziness_detector.py) | A transparent heuristic scorer for "AI coziness" in a comment — built partly to demonstrate why such detectors fail in the wild. |
| `figures/` | Generated charts (created by running the two scripts). |

## Run it

```bash
pip install -r requirements.txt

python3 dead_internet_sim.py     # -> figures/liveness_vs_autonomy.png, figures/effective_vocab.png
python3 coziness_detector.py     # -> figures/coziness_hist.png
```

Both scripts are seeded (`numpy.default_rng(7)`), so the figures are reproducible.

## The headline results

- **Liveness collapses non-linearly.** A thread's composite "liveness" (lexical
  diversity × disagreement × surprise) halves once the *average* poster sits at
  ~0.65 on the human→autonomous dial. You don't need a botnet; you need the
  average comment to be two-thirds assisted.
- **Disagreement dies first.** It's the steepest curve. The first thing AI sands
  off a conversation is friction — which we then misread as "kindness."
- **A cozy thread literally uses fewer words.** Effective vocabulary `exp(H)`
  falls ~175 → ~60 as autonomy maxes out (with a small honest *bump* at low
  autonomy — a little assistance adds a register before saturation homogenizes
  everything).
- **You can't just "detect the AI and ban it."** The same burstiness/perplexity
  signals real detectors use ([GPTZero](https://gptzero.me/news/perplexity-and-burstiness-what-is-it/),
  [DetectGPT](https://arxiv.org/abs/2301.11305)) are biased against non-native
  English writers ([Liang et al., 2023](https://arxiv.org/pdf/2304.02819)). A
  coziness detector is partly a *fluency* detector, and fluency ≠ AI.

## Honesty notes

These are **toy models**, not evidence. The simulation is a cartoon of language
(two token pools + a stance variable); the detector is a strawman built to show
its own failure mode — do not deploy it as a gate on real people. The essay's
"Limitations" section spells out what's caricature and which stats were dropped
for failing fact-checking. Every factual claim in the essay links to a
traceable source.
