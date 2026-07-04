"""
No-gold-standard prevalence + per-judge accuracy via Dawid-Skene / Hui-Walter latent class.

Answers "can we estimate the true rate and each classifier's sensitivity/specificity WITHOUT
hand labels?" -- yes, when the raters are conditionally independent given the latent truth
(which cross-FAMILY judges buy; correlated same-family tiers do not). Hui & Walter (1980)
formalized this for diagnostic tests with no gold standard; Dawid & Skene (1979) is the EM.

Model: latent true label z_i in {0,1}; prevalence p = P(z=1). Each judge j has
sensitivity se_j = P(obs=1 | z=1) and specificity sp_j = P(obs=0 | z=0). 'indeterminate'
is treated as ABSTENTION (missing), handled naturally by the likelihood.

Outputs: prevalence (+ bootstrap CI), each judge's se/sp, posterior labels, an
uncertain-doc count, and a conditional-independence diagnostic.

Usage:  python latent_class.py [--judges flash,haiku,qwen_fs]
"""
from __future__ import annotations
import argparse, csv, glob, os, random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-6

# Column carrying the yes/no/indeterminate judge verdict. Construct-specific runs name it
# is_<construct>_instance; override via env var for your own prediction CSVs.
LABEL_COLUMN = os.environ.get("LC_LABEL_COLUMN", "is_target_construct_instance")


def load(judge_names):
    mats = {}
    for n in judge_names:
        p = os.path.join(HERE, f"predictions_{n}.csv")
        if not os.path.exists(p):
            continue
        R = {r["doc_id"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}
        if len(R) >= 120:
            mats[n] = R
    names = sorted(mats)
    ids = sorted(set.intersection(*[set(mats[n]) for n in names]))
    def lab(n, d):
        v = mats[n][d].get(LABEL_COLUMN)
        return 1 if v == "yes" else (0 if v == "no" else None)  # indeterminate -> abstain
    obs = {d: {n: lab(n, d) for n in names} for d in ids}
    strat = {d: mats[names[0]][d].get("source_stratum", "?") for d in ids}
    return names, ids, obs, strat


def em(names, ids, obs, iters=200):
    # init posterior from observed mean
    q = {}
    for d in ids:
        vals = [obs[d][n] for n in names if obs[d][n] is not None]
        q[d] = (sum(vals) / len(vals)) if vals else 0.5
    p = sum(q.values()) / len(q)
    se = {n: 0.8 for n in names}; sp = {n: 0.8 for n in names}
    for _ in range(iters):
        # M-step
        p = min(1 - EPS, max(EPS, sum(q[d] for d in ids) / len(ids)))
        for n in names:
            num_se = den_se = num_sp = den_sp = 0.0
            for d in ids:
                o = obs[d][n]
                if o is None: continue
                num_se += q[d] * (o == 1); den_se += q[d]
                num_sp += (1 - q[d]) * (o == 0); den_sp += (1 - q[d])
            se[n] = min(1 - EPS, max(EPS, num_se / den_se)) if den_se else 0.5
            sp[n] = min(1 - EPS, max(EPS, num_sp / den_sp)) if den_sp else 0.5
        # E-step
        new_q = {}
        for d in ids:
            a, b = p, 1 - p
            for n in names:
                o = obs[d][n]
                if o is None: continue
                a *= se[n] if o == 1 else (1 - se[n])
                b *= (1 - sp[n]) if o == 1 else sp[n]
            new_q[d] = a / (a + b) if (a + b) > 0 else 0.5
        if max(abs(new_q[d] - q[d]) for d in ids) < 1e-9:
            q = new_q; break
        q = new_q
    return p, se, sp, q


def bootstrap_ci(names, ids, obs, B=300, seed=20260623):
    rng = random.Random(seed); ps = []
    for _ in range(B):
        samp = [rng.choice(ids) for _ in ids]
        # relabel duplicate ids uniquely so obs lookups still work
        sub_ids = list(range(len(samp)))
        sub_obs = {i: obs[samp[i]] for i in sub_ids}
        p, *_ = em(names, sub_ids, sub_obs, iters=80)
        ps.append(p)
    ps.sort()
    return ps[int(0.025 * B)], ps[int(0.975 * B)]


def independence_diag(names, ids, obs, q):
    """Residual co-error: do two judges err together more than independence predicts?"""
    print("\nConditional-independence check (pairwise error correlation given latent MAP):")
    z = {d: 1 if q[d] >= 0.5 else 0 for d in ids}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            err_a, err_b, both = 0, 0, 0; n = 0
            for d in ids:
                oa, ob = obs[d][a], obs[d][b]
                if oa is None or ob is None: continue
                ea = int(oa != z[d]); eb = int(ob != z[d])
                err_a += ea; err_b += eb; both += ea * eb; n += 1
            if n:
                exp_both = (err_a / n) * (err_b / n) * n
                print(f"  {a:10} x {b:10}  co-errors obs={both:>3} exp={exp_both:5.1f}  "
                      f"{'(correlated!)' if both > exp_both + 2 else '(ok)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default="flash,haiku,qwen_fs",
                    help="comma list of conditionally-independent families")
    args = ap.parse_args()
    want = [s.strip() for s in args.judges.split(",")]
    names, ids, obs, strat = load(want)
    print(f"Latent-class (Dawid-Skene / Hui-Walter) on judges: {names}")
    print(f"Docs: {len(ids)}  (no hand labels used)\n")

    p, se, sp, q = em(names, ids, obs)
    lo, hi = bootstrap_ci(names, ids, obs)
    print(f"ESTIMATED PREVALENCE p = {p:.3%}   95% bootstrap CI [{lo:.3%}, {hi:.3%}]")
    print("\nPer-judge accuracy (estimated WITHOUT gold):")
    print(f"  {'judge':12} {'sensitivity':>12} {'specificity':>12}")
    for n in names:
        print(f"  {n:12} {se[n]:>12.3f} {sp[n]:>12.3f}")

    npos = sum(1 for d in ids if q[d] >= 0.5)
    unc = sum(1 for d in ids if 0.1 < q[d] < 0.9)
    print(f"\nPosterior: {npos} docs MAP-positive of {len(ids)}; {unc} uncertain (0.1<q<0.9).")

    # per-stratum prevalence (the Hui-Walter multi-population angle)
    from collections import Counter
    by = defaultdict(list)
    for d in ids: by[strat[d]].append(q[d])
    print("\nLatent prevalence by stratum (multi-population identifiability):")
    for s, qs in by.items():
        print(f"  {s:18} mean posterior = {sum(qs)/len(qs):.3%}  (n={len(qs)})")

    independence_diag(names, ids, obs, q)

    out = os.path.join(HERE, "latent_labels.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["doc_id", "source_stratum", "posterior_positive", "map_label"])
        for d in ids:
            w.writerow([d, strat[d], f"{q[d]:.4f}", 1 if q[d] >= 0.5 else 0])
    print(f"\nWrote {out} (latent 'truth' estimate, no gold). Validate assumptions against "
          "your existing hand-coded LCR cases.")


if __name__ == "__main__":
    main()
