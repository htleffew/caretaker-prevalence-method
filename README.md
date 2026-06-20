# No-gold prevalence estimation for rare latent constructs

A construct-agnostic toolkit for estimating the prevalence of a rare, latent behavioral construct in a large corpus **without a gold standard**, by combining a cheap noisy screen with a cross-family ensemble of LLM judges.

## Methods
- **Horvitz-Thompson** stratified prevalence + analytical variance; Neyman allocation; Beta-Binomial and Clopper-Pearson zero-event upper bounds (`src/ht_prevalence.py`).
- **Hui-Walter / Dawid-Skene latent class** — recover each judge's sensitivity/specificity AND prevalence from agreement alone, no gold standard, given conditionally-independent judges across populations with different base rates (`src/latent_class.py`).
- **Rogan-Gladen** correction of a single cheap judge's observed rate using the no-gold se/sp estimates.
- **Fightin' Words** (Monroe et al. 2008) log-odds + log-likelihood collocations to rediscover a construct's lexical signature from ensemble-consensus labels (`src/signature_mining.py`).

## Why no gold standard
At frontier scale there is often no ground truth for a rare construct. The lineage is measurement science: estimating test accuracy and prevalence with no gold standard is Hui & Walter (1980); the base-rate problem is Meehl & Rosen (1955). See `METHODS_CROSSWALK.md` for the full measurement-science / ML mapping and references.

Tests: `python src/test_ht_prevalence.py` (12 closed-form checks).

_The construct-specific judge rubric, codebook, corpus data, and application results are maintained in a separate private repository._
