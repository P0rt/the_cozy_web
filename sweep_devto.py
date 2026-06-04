"""
sweep_devto.py — is the eco-comment template platform-wide or just one post?

Pulls comments across many recent dev.to articles and measures:
  * how prevalent the "validate a phrase -> additive nuance -> we-at-<Product>
    plug" skeleton is, across DISTINCT accounts;
  * 4-grams reused across different accounts (shared skeleton signal);
  * repeat appearances of specific suspect accounts on unrelated posts.

Read-only, polite (sequential, capped). Run:  python3 sweep_devto.py
"""
from __future__ import annotations

import html
import json
import re
import urllib.request
from collections import Counter, defaultdict

TAGS = ["ai", "machinelearning", "webdev", "programming"]
PER_TAG = 12

PLUG_RE = re.compile(r"\b(we use|we've seen|we've been|we ran|we're using|"
                     r"our (team|setup|product|stack|daily driver))\b", re.I)
BRAND_RE = re.compile(r"\b([A-Z][a-z]+[A-Z][A-Za-z]+|[A-Z]{2,}[a-z]+)\b")
VALIDATE = ["spot on", "clicks immediately", "clicked immediately", "is gold",
            "framing is", "framing clicks", "the part worth", "one thing i",
            "the bit about", "is the public service", "incredibly helpful",
            "this is great", "really resonates", "well put"]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)
    except Exception:
        return None


def strip(h):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", h or ""))).strip()


def flatten(comments):
    out = []
    def walk(n, d=0):
        u = n.get("user", {})
        out.append({"user": u.get("username", ""), "name": u.get("name", ""),
                    "body": strip(n.get("body_html", "")), "depth": d})
        for c in n.get("children", []) or []:
            walk(c, d + 1)
    for t in comments or []:
        walk(t)
    return out


def main():
    seen_ids = set()
    articles = []
    for tag in TAGS:
        lst = get(f"https://dev.to/api/articles?tag={tag}&top=30&per_page={PER_TAG}") or []
        for a in lst:
            if a["id"] not in seen_ids and a.get("comments_count", 0) >= 2:
                seen_ids.add(a["id"])
                articles.append(a)
    print(f"sampled {len(articles)} articles with >=2 comments across {TAGS}\n")

    all_comments = []
    plug_authors = set()
    validate_authors = set()
    author_posts = defaultdict(set)         # author -> set of article ids
    gram_authors = defaultdict(set)         # 4-gram -> set of authors

    for a in articles:
        cs = flatten(get(f"https://dev.to/api/comments?a_id={a['id']}"))
        for c in cs:
            if not c["body"]:
                continue
            all_comments.append(c)
            author_posts[c["user"]].add(a["id"])
            low = c["body"].lower()
            if PLUG_RE.search(c["body"]) and len(c["body"].split()) > 25:
                plug_authors.add(c["user"])
            if any(v in low for v in VALIDATE) and len(c["body"].split()) > 25:
                validate_authors.add(c["user"])
            toks = re.findall(r"[a-z']+", low)
            for i in range(len(toks) - 3):
                g = " ".join(toks[i:i + 4])
                gram_authors[g].add(c["user"])

    n = len(all_comments)
    print(f"collected {n} comments from {len(author_posts)} distinct accounts\n")

    print("--- prevalence (accounts, not comments) ---")
    print(f"  accounts posting a long product/company plug: {len(plug_authors)}")
    print(f"  accounts opening with phrase-validation:      {len(validate_authors)}")
    print(f"  accounts doing BOTH (the full skeleton):      "
          f"{len(plug_authors & validate_authors)}")

    print("\n--- accounts commenting on the MOST distinct posts (spray pattern) ---")
    top = sorted(author_posts.items(), key=lambda kv: -len(kv[1]))[:12]
    for user, posts in top:
        if len(posts) >= 2:
            tag = " <- PLUG" if user in plug_authors else ""
            print(f"  {len(posts):>2} posts  @{user}{tag}")

    print("\n--- distinctive 4-grams reused across >=3 different accounts ---")
    glue = {"the", "a", "to", "of", "is", "that", "it", "and", "you", "for",
            "in", "on", "this", "with", "as", "but", "i", "we", "are", "be",
            "your", "have", "not", "so", "if", "what", "how", "they", "an"}
    reuse = [(g, len(us)) for g, us in gram_authors.items()
             if len(us) >= 3 and not all(w in glue for w in g.split())]
    for g, k in sorted(reuse, key=lambda x: -x[1])[:20]:
        print(f"    x{k:>2} distinct accounts:  \"{g}\"")
    if not reuse:
        print("    (none — template is structural, not lexical)")


if __name__ == "__main__":
    main()
