"""
Rediscover the construct's linguistic signature from the ensemble's high-agreement labels.

Weak supervision: silver_consensus.csv gives unanimous positive / negative sets. We mine the
terms and collocations that DISTINGUISH positives from negatives -- a data-driven lexicon to
replace the a priori seed n-grams (which had the false-negative blind spot).

Methods:
  * Monroe, Colaresi & Quinn (2008) "Fightin' Words": log-odds-ratio with an informative
    Dirichlet prior, z-scored. Gold standard for distinguishing words between two corpora;
    robust to rare terms and class imbalance (vs raw frequency / plain n-grams).
  * NLTK bigram collocations within the positive set, scored by Dunning log-likelihood.
  * WordNet lemmatization + stopword removal so morphology doesn't fragment the signal.

CAVEAT: positives are few (unanimous set is small) and these are SILVER labels (model
consensus, not hand gold). Treat the output as hypotheses to validate against the hand gold,
not as established signatures. Re-run after the gold correction.

Usage:  python signature_mining.py
"""
from __future__ import annotations
import csv, math, os, re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))

# --- light NLP with graceful fallback ---
try:
    import nltk
    # nltk 3.9 path-security rejects the virtualized AppData\Roaming path the Claude app
    # redirects to; download to a plain, non-virtualized dir under the user home instead.
    _NLTK_DIR = os.path.join(os.path.expanduser("~"), "nltk_data")
    os.makedirs(_NLTK_DIR, exist_ok=True)
    if _NLTK_DIR not in nltk.data.path:
        nltk.data.path.insert(0, _NLTK_DIR)
    for _pkg in ("stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.download(_pkg, download_dir=_NLTK_DIR, quiet=True)
        except Exception:
            pass
    from nltk.corpus import stopwords as _sw
    from nltk.stem import WordNetLemmatizer
    from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
    STOP = set(_sw.words("english"))
    _LEM = WordNetLemmatizer()
    lemma = lambda w: _LEM.lemmatize(w)
    HAVE_NLTK = True
except Exception:
    STOP = set("a an the of to and is in it you i we they he she that this for on with as be are "
               "was were do does did have has had not no so if then than too very can will just "
               "me my your our their them his her its at or but by from up out".split())
    lemma = lambda w: w
    HAVE_NLTK = False

TOKEN = re.compile(r"[a-z][a-z'-]{1,}")


def toks(text: str) -> list[str]:
    return [lemma(t) for t in TOKEN.findall(text.lower()) if t not in STOP and len(t) > 2]


def fightin_words(pos_docs, neg_docs, top=25):
    """Monroe et al. log-odds with informative Dirichlet prior (prior = combined counts)."""
    cp, cn = Counter(), Counter()
    for d in pos_docs: cp.update(toks(d))
    for d in neg_docs: cn.update(toks(d))
    vocab = set(cp) | set(cn)
    prior = {w: cp[w] + cn[w] for w in vocab}          # informative prior from the corpus
    a0 = sum(prior.values())
    np_, nn = sum(cp.values()), sum(cn.values())
    z = {}
    for w in vocab:
        aw = prior[w]
        lo_p = math.log((cp[w] + aw) / (np_ + a0 - cp[w] - aw))
        lo_n = math.log((cn[w] + aw) / (nn + a0 - cn[w] - aw))
        delta = lo_p - lo_n
        var = 1.0 / (cp[w] + aw) + 1.0 / (cn[w] + aw)
        z[w] = delta / math.sqrt(var)
    ranked = sorted(z.items(), key=lambda kv: kv[1], reverse=True)
    return ranked, cp, cn


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "silver_consensus.csv"), encoding="utf-8")))
    pos = [r["body"] for r in rows if r["consensus"] == "positive"]
    neg = [r["body"] for r in rows if r["consensus"] == "negative"]
    print(f"NLTK available: {HAVE_NLTK}")
    print(f"silver positives: {len(pos)}   silver negatives: {len(neg)}")
    if len(pos) < 3:
        print("Too few positives to mine; re-run after more consensus/gold."); return

    ranked, cp, cn = fightin_words(pos, neg)
    print("\n== Terms most DISTINCTIVE of the target-construct-positive set (Fightin' Words z) ==")
    for w, zv in ranked[:25]:
        print(f"  {w:18} z={zv:+.2f}  pos={cp[w]:>3} neg={cn[w]:>3}")
    print("\n== Terms most distinctive of NEGATIVES (for the discriminative screen) ==")
    for w, zv in ranked[-12:]:
        print(f"  {w:18} z={zv:+.2f}  pos={cp[w]:>3} neg={cn[w]:>3}")

    # collocations within the positive set
    if HAVE_NLTK:
        all_pos_tokens = [t for d in pos for t in toks(d)]
        if len(all_pos_tokens) > 5:
            finder = BigramCollocationFinder.from_words(all_pos_tokens)
            finder.apply_freq_filter(2)
            bam = BigramAssocMeasures()
            print("\n== Top positive-set collocations (Dunning log-likelihood, freq>=2) ==")
            for (a, b), s in finder.score_ngrams(bam.likelihood_ratio)[:15]:
                print(f"  {a} {b:22} LL={s:.1f}")

    # write the rediscovered lexicon (positive-leaning terms) for the next-gen screen
    out = os.path.join(HERE, "rediscovered_lexicon.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["term", "z_score", "pos_count", "neg_count"])
        for term, zv in ranked:
            if zv > 1.0:
                w.writerow([term, f"{zv:.3f}", cp[term], cn[term]])
    print(f"\nWrote {out} (data-driven seed candidates; z>1.0). Validate against hand gold before use.")


if __name__ == "__main__":
    main()
