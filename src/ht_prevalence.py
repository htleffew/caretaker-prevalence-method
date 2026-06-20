"""
Rare-event prevalence estimation toolkit.

Implements the four estimators an a rare-event prevalence estimation task
asks for, in the form they ask for them:

    1. Horvitz-Thompson stratified prevalence  (point estimate + analytical variance)
    2. Neyman optimal allocation               (minimum-variance sample split)
    3. Beta-Binomial posterior upper bound      (zero-event strata)
    4. Clopper-Pearson exact upper bound        (zero-event strata, frequentist)

Psychometric lineage (say it in the room):
    - Inclusion probability pi_i and the 1/pi weighting IS the selection-ratio logic of
      Taylor & Russell (1939): we oversample the high-yield stratum and reweight.
    - Estimating a base rate from a noisy screen over a population where a missed case is
      catastrophic IS Meehl & Rosen (1955).
    - The zero-event upper bound is the honest answer to "I observed nothing -- how high
      could the true rate still be?" -- a base-rate question, not a point estimate.

Pure standard library + a single scipy call for the Beta quantile / F quantile.
Everything is typed and unit-tested (see test_ht_prevalence.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import beta as _beta_dist  # quantile of Beta posterior
from scipy.stats import f as _f_dist        # exact Clopper-Pearson via F distribution


# --------------------------------------------------------------------------- #
# 1. Horvitz-Thompson stratified prevalence
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Stratum:
    """One stratum in a stratified prevalence design.

    N : int   population size of the stratum (number of documents in it)
    n : int   number of units we drew for adjudication (review sample size)
    x : int   number of adjudicated positives in that sample (0 <= x <= n)

    A census stratum has n == N; then the stratum is observed exactly (no variance).
    """
    name: str
    N: int
    n: int
    x: int

    def __post_init__(self) -> None:
        if self.N <= 0:
            raise ValueError(f"{self.name}: N must be > 0")
        if not (0 <= self.n <= self.N):
            raise ValueError(f"{self.name}: need 0 <= n ({self.n}) <= N ({self.N})")
        if not (0 <= self.x <= self.n):
            raise ValueError(f"{self.name}: need 0 <= x ({self.x}) <= n ({self.n})")

    @property
    def p_hat(self) -> float:
        """Within-stratum sample proportion."""
        return self.x / self.n if self.n else 0.0


@dataclass(frozen=True)
class PrevalenceEstimate:
    p_hat: float          # population prevalence point estimate
    variance: float       # analytical variance of p_hat
    se: float             # standard error = sqrt(variance)
    ci_low: float
    ci_high: float
    z: float
    total_positives_hat: float   # tau_hat = p_hat * N_total
    N_total: int


def horvitz_thompson_prevalence(
    strata: list[Stratum],
    z: float = 1.959963984540054,  # 95% normal
) -> PrevalenceEstimate:
    r"""Stratified Horvitz-Thompson prevalence estimator.

    Each unit in stratum h is sampled with inclusion probability pi_h = n_h / N_h, so its
    HT weight is 1/pi_h = N_h / n_h. Summing the weighted positives in stratum h:

        sum_{i in s_h} y_i / pi_h = (N_h / n_h) * x_h = N_h * p_hat_h

    so the HT estimate of the population total is tau_hat = sum_h N_h * p_hat_h and the
    prevalence is

        p_hat = tau_hat / N = sum_h W_h * p_hat_h,     W_h = N_h / N.

    Analytical variance under SRS-without-replacement within each stratum (independent
    strata), using the finite-population correction (1 - n_h/N_h):

        Var(p_hat) = sum_h W_h^2 * (1 - n_h/N_h) * S_h^2 / n_h,

    with the unbiased stratum variance for a 0/1 variable

        S_h^2 = n_h / (n_h - 1) * p_hat_h * (1 - p_hat_h).

    Census strata (n_h == N_h) contribute exactly zero variance (FPC = 0), which is the
    right behaviour: if you reviewed every flagged document, that stratum carries no
    sampling uncertainty.
    """
    if not strata:
        raise ValueError("need at least one stratum")
    N_total = sum(s.N for s in strata)

    tau_hat = 0.0
    var = 0.0
    for s in strata:
        W = s.N / N_total
        p = s.p_hat
        tau_hat += s.N * p
        if s.n >= 2 and s.n < s.N:
            S2 = (s.n / (s.n - 1)) * p * (1 - p)         # unbiased within-stratum var
            fpc = 1.0 - s.n / s.N
            var += (W ** 2) * fpc * S2 / s.n
        # n == N (census): FPC = 0 -> no contribution.
        # n == 1: variance undefined; treated as 0 and flagged by caller if needed.

    p_hat = tau_hat / N_total
    se = math.sqrt(var)
    return PrevalenceEstimate(
        p_hat=p_hat,
        variance=var,
        se=se,
        ci_low=max(0.0, p_hat - z * se),
        ci_high=min(1.0, p_hat + z * se),
        z=z,
        total_positives_hat=tau_hat,
        N_total=N_total,
    )


# --------------------------------------------------------------------------- #
# 2. Neyman optimal allocation
# --------------------------------------------------------------------------- #
def neyman_allocation(
    N_h: list[int],
    S_h: list[float],
    n_total: int,
) -> list[int]:
    r"""Optimal (Neyman) allocation of a fixed total sample across strata.

    Minimising Var(p_hat) = sum_h W_h^2 S_h^2 / n_h subject to sum_h n_h = n yields,
    by Lagrange multipliers,

        n_h = n * (N_h S_h) / sum_k (N_k S_k).

    For a 0/1 (prevalence) variable, S_h = sqrt(p_h (1 - p_h)) from a pilot or prior.
    Returns integer allocations that sum exactly to n_total (largest-remainder rounding),
    with every stratum that has positive N_h S_h getting at least 1 unit.
    """
    if n_total <= 0:
        raise ValueError("n_total must be > 0")
    if len(N_h) != len(S_h):
        raise ValueError("N_h and S_h must be the same length")
    weights = [n * s for n, s in zip(N_h, S_h)]
    total_w = sum(weights)
    if total_w <= 0:
        # Degenerate: no stratum has any assumed variance -> fall back to proportional.
        weights = [float(n) for n in N_h]
        total_w = sum(weights)

    raw = [n_total * w / total_w for w in weights]
    floor = [int(math.floor(r)) for r in raw]
    # guarantee >=1 where the stratum carries weight
    for i, w in enumerate(weights):
        if w > 0 and floor[i] == 0:
            floor[i] = 1
    remaining = n_total - sum(floor)
    if remaining > 0:
        frac = sorted(range(len(raw)), key=lambda i: raw[i] - math.floor(raw[i]),
                      reverse=True)
        for i in range(remaining):
            floor[frac[i % len(frac)]] += 1
    elif remaining < 0:
        # over-allocated by the >=1 guarantees; trim from the largest allocations
        order = sorted(range(len(floor)), key=lambda i: floor[i], reverse=True)
        k = 0
        while remaining < 0:
            i = order[k % len(order)]
            if floor[i] > 1:
                floor[i] -= 1
                remaining += 1
            k += 1
    return floor


# --------------------------------------------------------------------------- #
# 3. Beta-Binomial posterior upper bound (zero-event friendly)
# --------------------------------------------------------------------------- #
def beta_binomial_upper(
    x: int,
    n: int,
    alpha: float = 0.05,
    prior_a: float = 0.5,
    prior_b: float = 0.5,
) -> float:
    r"""Upper (1 - alpha) credible bound on a proportion via the Beta-Binomial conjugate.

    Prior  p ~ Beta(a, b);  likelihood x | p ~ Binomial(n, p);  posterior

        p | x ~ Beta(a + x, b + n - x).

    The upper bound is the (1 - alpha) quantile of that posterior. Defaults to the
    Jeffreys prior Beta(0.5, 0.5) -- the standard weakly-informative default.

    This is the right tool when x == 0: the frequentist point estimate is 0 (useless for a
    risk bound), but the posterior still says "given a review of n with zero hits, the true
    rate is below this with probability 1 - alpha." Choosing an *informative* low prior on a
    category you have not measured is the documented evaluation task failure -- it manufactures a
    low bound. Keep the prior weak and justify any deviation with historical data.
    """
    if x < 0 or n < 0 or x > n:
        raise ValueError("need 0 <= x <= n")
    a_post = prior_a + x
    b_post = prior_b + (n - x)
    return float(_beta_dist.ppf(1.0 - alpha, a_post, b_post))


# --------------------------------------------------------------------------- #
# 4. Clopper-Pearson exact upper bound (frequentist)
# --------------------------------------------------------------------------- #
def clopper_pearson_upper(x: int, n: int, alpha: float = 0.05) -> float:
    r"""Exact (Clopper-Pearson) one-sided upper confidence bound on a binomial proportion.

    General form via the F distribution:

        p_U = 1 / (1 + (n - x) / ((x + 1) * F_{1-alpha; 2(x+1), 2(n-x)})).

    For the zero-event case x == 0 this collapses to the closed form

        p_U = 1 - alpha^(1/n),

    the "rule of 3"-style exact bound (at alpha = 0.05, p_U ~ 3/n for large n).
    """
    if x < 0 or n <= 0 or x > n:
        raise ValueError("need 0 <= x <= n and n > 0")
    if x == n:
        return 1.0
    if x == 0:
        return 1.0 - alpha ** (1.0 / n)
    d1 = 2 * (x + 1)
    d2 = 2 * (n - x)
    f_crit = _f_dist.ppf(1.0 - alpha, d1, d2)
    return float((x + 1) * f_crit / ((n - x) + (x + 1) * f_crit))
