"""
analyze_devto.py
================

Take REAL comments from a dev.to article (via the public API) and:
  1. score them with the v1 coziness detector — and watch it FAIL, because the
     new generation of LLM comments defeats it on purpose by injecting
     specificity (numbers, tool names, real technical nuance);
  2. compute a set of v2 "eco-astroturf" signals that the v1 detector misses:
     cross-account template reuse, company/product injection, relentless
     agreement, auto-generated usernames, apostrophe-drop humanizer tells.

Usage:
  python3 analyze_devto.py <article_api_path>
  e.g. python3 analyze_devto.py p0rt/how-model-distillation-actually-works-...-3o0o
"""
from __future__ import annotations

import html
import json
import re
import sys
import urllib.request
from collections import Counter

from coziness_detector import coziness_score


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.load(r)


def _strip(h: str) -> str:
    t = re.sub(r"<[^>]+>", " ", h or "")
    return re.sub(r"\s+", " ", html.unescape(t)).strip()


def fetch(article_path: str):
    art = _get(f"https://dev.to/api/articles/{article_path}")
    comments = _get(f"https://dev.to/api/comments?a_id={art['id']}")
    out = []

    def walk(node, depth=0):
        u = node.get("user", {})
        out.append({
            "depth": depth,
            "user": u.get("username", ""),
            "name": u.get("name", ""),
            "body": _strip(node.get("body_html", "")),
        })
        for c in node.get("children", []) or []:
            walk(c, depth + 1)

    for top in comments:
        walk(top)
    return art, out


# --------------------------------------------------------------------------- #
# v2 signals — the things the new eco-style comments share
# --------------------------------------------------------------------------- #

# A product/company plug: first-person-plural usage anecdote naming a tool.
PLUG_RE = re.compile(
    r"\b(we use|we've seen|we've been|we ran|we are|we're|our (team|setup|product|"
    r"stack|daily driver))\b", re.I)
BRAND_NEAR_PLUG_RE = re.compile(
    r"\b(we|our|using|with)\b[^.]{0,40}\b([A-Z][a-zA-Z]*[A-Z][a-zA-Z]*|"
    r"DeepSeek|VoltageGPU|MemBridge|GPU)\b")

# The opening-validation skeleton: "the X framing is spot on / clicks / is gold".
VALIDATION_OPENERS = [
    "spot on", "clicks immediately", "clicked immediately", "is gold",
    "is the public service here", "is right", "is striking", "is reassuring",
    "the part worth", "one thing i would add", "one thing i'd add",
    "the part most", "the bit about", "framing is", "framing clicks",
]

# Auto-generated-looking username: trailing hex blob.
HEX_SUFFIX_RE = re.compile(r"_[0-9a-f]{6,}$")

NUMBER_RE = re.compile(r"\b\d+(\.\d+)?x?%?\b")
APOSTROPHE_DROP = ["dont", "doesnt", "didnt", "wont", "cant", "its ", "im ",
                   "ive ", "youre", "thats", "isnt", "wasnt"]


def v2_signals(c: dict) -> dict:
    b = c["body"]
    low = b.lower()
    return {
        "product_plug": bool(PLUG_RE.search(b)) and bool(BRAND_NEAR_PLUG_RE.search(b)),
        "validation_opener": any(v in low for v in VALIDATION_OPENERS),
        "number_props": len(NUMBER_RE.findall(b)),
        "apostrophe_drop": sum(low.count(a) for a in APOSTROPHE_DROP),
        "hex_username": bool(HEX_SUFFIX_RE.search(c["user"])),
        "len_words": len(re.findall(r"\w+", b)),
    }


def cross_account_phrase_reuse(comments: list[dict]) -> list[tuple[str, int, set]]:
    """Find 4-grams that recur across comments from DIFFERENT accounts.
    Bot fleets reuse skeletons; humans rarely echo each other's phrasing."""
    grams = {}
    for c in comments:
        toks = re.findall(r"[a-z']+", c["body"].lower())
        seen = set()
        for i in range(len(toks) - 3):
            g = " ".join(toks[i:i + 4])
            if g in seen:
                continue
            seen.add(g)
            grams.setdefault(g, set()).add(c["user"])
    reuse = [(g, len(users), users) for g, users in grams.items() if len(users) >= 2]
    # drop generic glue
    reuse = [r for r in reuse if not all(w in {"the", "a", "to", "of", "is",
             "that", "it", "and", "you", "for", "in", "on", "this"}
             for w in r[0].split())]
    return sorted(reuse, key=lambda r: -r[1])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "p0rt/how-model-distillation-actually-works-and-what-the-china-"
        "distilled-our-model-headlines-really-3o0o")
    art, comments = fetch(path)
    others = [c for c in comments if c["user"] != "p0rt"]  # exclude OP replies

    print("=" * 78)
    print(art["title"])
    print(f"{art['comments_count']} comments · {art['public_reactions_count']} reactions")
    print("=" * 78)

    print("\n--- v1 coziness detector on the real (non-OP) comments ---")
    print("(built for the OLD 'Great post!' style — watch it score these LOW)\n")
    v1 = []
    for c in others:
        s = coziness_score(c["body"])
        v1.append(s)
        sig = v2_signals(c)
        flags = []
        if sig["product_plug"]:        flags.append("PLUG")
        if sig["validation_opener"]:   flags.append("VALIDATE-OPEN")
        if sig["hex_username"]:        flags.append("HEX-USER")
        if sig["apostrophe_drop"] and sig["len_words"] > 30: flags.append("APOS-DROP")
        print(f"  v1={s:.2f}  nums={sig['number_props']:>2}  "
              f"@{c['user'][:22]:<22} {' '.join(flags)}")
    print(f"\n  v1 mean on these = {sum(v1)/len(v1):.2f}  "
          f"(the old detector basically shrugs — that's the point)")

    print("\n--- v2 eco-astroturf signals (what v1 misses) ---")
    plugs = [c for c in others if v2_signals(c)["product_plug"]]
    opens = [c for c in others if v2_signals(c)["validation_opener"]]
    hexu = sorted({c["user"] for c in comments if v2_signals(c)["hex_username"]})
    print(f"  product/company plug:        {len(plugs)}/{len(others)} comments")
    for c in plugs:
        brand = BRAND_NEAR_PLUG_RE.search(c["body"])
        print(f"      @{c['user']:<24} -> '{c['body'][:70]}...'")
    print(f"  opens by validating a phrase: {len(opens)}/{len(others)} comments")
    print(f"  auto-gen-looking usernames:   {hexu}")

    print("\n--- cross-account phrase reuse (>=2 distinct users) ---")
    print("  (different 'people' echoing the same 4-grams = shared skeleton)\n")
    reuse = cross_account_phrase_reuse(others)
    for g, n, users in reuse[:12]:
        print(f"    x{n}  \"{g}\"  <- {', '.join(sorted(users))}")
    if not reuse:
        print("    (none)")

    print("\n--- disagreement audit ---")
    # how many non-OP comments actually push back vs. validate-then-add
    pushback = sum(1 for c in others if re.search(
        r"\b(disagree|wrong|that's not|actually no|i don't think|doesn't hold|"
        r"overstat|undersell)\b", c["body"], re.I))
    print(f"  comments that genuinely push back: {pushback}/{len(others)}")
    print("  (everything else is validate -> gentle additive nuance -> we-at-X plug)")


if __name__ == "__main__":
    main()
