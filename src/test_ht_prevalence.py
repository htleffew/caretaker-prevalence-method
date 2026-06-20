"""
Deterministic validation of the rare-event estimator toolkit.

Each test pins an estimator to an analytic special case where the right answer is known in
closed form, so a regression is unambiguous. Run:  python -m pytest test_ht_prevalence.py -q
(or just `python test_ht_prevalence.py` for a no-pytest summary).
"""
from __future__ import annotations

import math
import random

from scipy.stats import beta as _beta_dist

from ht_prevalence import (
    Stratum,
    horvitz_thompson_prevalence,
    neyman_allocation,
    beta_binomial_upper,
    clopper_pearson_upper,
)

TOL = 1e-9


# --- Horvitz-Thompson ------------------------------------------------------- #
def test_ht_single_stratum_is_sample_proportion():
    est = horvitz_thompson_prevalence([Stratum("only", N=10000, n=200, x=10)])
    assert abs(est.p_hat - 0.05) < TOL


def test_ht_is_weighted_mean_of_stratum_proportions():
    strata = [Stratum("A", 8000, 200, 4), Stratum("B", 2000, 200, 20)]
    # W_A=0.8 p=0.02 ; W_B=0.2 p=0.10 -> 0.8*0.02 + 0.2*0.10 = 0.036
    est = horvitz_thompson_prevalence(strata)
    assert abs(est.p_hat - 0.036) < TOL


def test_ht_census_stratum_has_zero_variance():
    # Reviewing every unit (n == N) => no sampling uncertainty from that stratum.
    est = horvitz_thompson_prevalence([Stratum("census", N=500, n=500, x=20)])
    assert est.variance == 0.0
    assert abs(est.p_hat - 0.04) < TOL


def test_ht_unbiased_by_monte_carlo():
    """Empirically confirm E[p_hat] = true prevalence over many SRSWOR draws."""
    rng = random.Random(20260619)
    # Finite population: stratum A 9000 units 1% positive; stratum B 1000 units 20% positive.
    popA = [1] * 90 + [0] * 8910            # 9000 units, 90 positive
    popB = [1] * 200 + [0] * 800            # 1000 units, 200 positive
    true_p = (90 + 200) / 10000             # 0.029
    nA, nB = 300, 300
    draws = 4000
    acc = 0.0
    for _ in range(draws):
        sA = rng.sample(popA, nA)
        sB = rng.sample(popB, nB)
        est = horvitz_thompson_prevalence(
            [Stratum("A", len(popA), nA, sum(sA)),
             Stratum("B", len(popB), nB, sum(sB))]
        )
        acc += est.p_hat
    mean_p = acc / draws
    # Monte Carlo SE of the mean is tiny here; 5e-4 is comfortable headroom.
    assert abs(mean_p - true_p) < 5e-4, (mean_p, true_p)


# --- Neyman allocation ------------------------------------------------------ #
def test_neyman_matches_closed_form():
    alloc = neyman_allocation(N_h=[1000, 1000], S_h=[0.5, 0.1], n_total=300)
    # weights 500:100 -> 250:50
    assert alloc == [250, 50]
    assert sum(alloc) == 300


def test_neyman_sums_exactly_and_is_positive():
    alloc = neyman_allocation(N_h=[14721, 30727, 29939, 14009],
                              S_h=[0.02, 0.05, 0.03, 0.04], n_total=2000)
    assert sum(alloc) == 2000
    assert all(a >= 1 for a in alloc)


def test_neyman_degenerate_falls_back_to_proportional():
    # All assumed SDs zero -> proportional to N_h.
    alloc = neyman_allocation(N_h=[3000, 1000], S_h=[0.0, 0.0], n_total=400)
    assert alloc == [300, 100]


# --- Beta-Binomial ---------------------------------------------------------- #
def test_beta_binomial_uniform_prior_equals_beta_quantile():
    # Beta(1,1) posterior on x=3,n=100 is Beta(4,98); upper 95% is its 0.95 quantile.
    got = beta_binomial_upper(3, 100, alpha=0.05, prior_a=1, prior_b=1)
    want = float(_beta_dist.ppf(0.95, 4, 98))
    assert abs(got - want) < TOL


def test_beta_binomial_zero_event_is_below_one_and_positive():
    ub = beta_binomial_upper(0, 2000, alpha=0.05)  # Jeffreys
    assert 0.0 < ub < 0.01   # ~ a few per thousand


# --- Clopper-Pearson -------------------------------------------------------- #
def test_clopper_pearson_zero_event_closed_form():
    n = 2000
    got = clopper_pearson_upper(0, n, alpha=0.05)
    want = 1.0 - 0.05 ** (1.0 / n)
    assert abs(got - want) < TOL


def test_clopper_pearson_zero_event_rule_of_three():
    # For x=0, the 95% upper bound is ~ 3/n for large n ("rule of three").
    n = 1000
    got = clopper_pearson_upper(0, n, alpha=0.05)
    assert abs(got - 3.0 / n) < 5e-4


def test_clopper_pearson_equals_beta_identity():
    # Exact CP upper bound == qbeta(1-alpha, x+1, n-x).
    x, n = 4, 250
    got = clopper_pearson_upper(x, n, alpha=0.05)
    want = float(_beta_dist.ppf(0.95, x + 1, n - x))
    assert abs(got - want) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {fn.__name__}: {e!r}")
    print(f"\n{passed}/{len(fns)} passed")
