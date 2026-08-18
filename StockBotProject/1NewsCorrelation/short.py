"""
stat_significance_test.py
--------------------------
Tests whether the HIGH_SF vs LOW_SF differences from screener_squeeze_compare.py
are statistically significant, at two levels:

  1. TICKER-LEVEL (the valid comparison, n=30 vs n=30): one independent
     observation per ticker (its own average). This is what the tests
     should be run on -- each ticker is one independent draw.

  2. DAY-LEVEL (pooled, n=283 vs n=184): included for reference only,
     clearly flagged as exploratory. Treating each explosive day as an
     independent observation is pseudoreplication -- a ticker with 19
     explosive days contributes 19 correlated data points (same
     float/short-interest regime, often clustered in the same run), which
     inflates apparent sample size and can manufacture significance that
     isn't really there. The ticker-level test is the one to trust.

Tests used:
  - Mann-Whitney U (primary): rank-based, doesn't assume normality. Used
    for everything here because both GainPct and VolFloatRatio are
    heavily right-skewed (means sit way above medians in your summary --
    a few outlier tickers/days are dragging the mean up), which violates
    the equal-variance/normality assumptions a t-test relies on.
  - Welch's t-test (secondary): reported alongside for comparison, since
    it's what most people expect to see, but don't lean on it alone given
    the skew.
  - Fisher's exact test: for the "tickers with 0 explosive days" count
    (1/30 vs 8/30), since that's a proportion/count comparison, not a
    continuous one.
  - Effect sizes: Cliff's delta (rank-based, pairs with Mann-Whitney) and
    Cohen's d (pairs with the t-test), so a significant p-value can be
    read alongside how *large* the difference actually is.

Multiple comparisons: 4 metrics are tested per level below. A Bonferroni-
corrected alpha (0.05/4 = 0.0125) is shown alongside the raw p-value so
you can judge significance either way.

Requires: pip install pandas scipy
Reads: squeeze_compare_results.csv, squeeze_compare_explosive_days.csv
       (the filenames screener_squeeze_compare.py already writes -- run
       that script first if these aren't present)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TICKER_CSV = "squeeze_compare_results.csv"
DAYS_CSV   = "squeeze_compare_explosive_days.csv"
GROUP_A    = "HIGH_SF"
GROUP_B    = "LOW_SF"
N_COMPARISONS = 4          # ExplosiveDays, zero-day proportion, GainPct, VolFloatRatio
ALPHA         = 0.05
ALPHA_BONFERRONI = ALPHA / N_COMPARISONS


# ── Effect sizes ─────────────────────────────────────────────────────────
def cliffs_delta(x, y) -> float:
    """
    Rank-based effect size for Mann-Whitney comparisons, range [-1, 1].
    0 = complete overlap, +/-1 = complete separation.
    Rough interpretation (Romano et al.): |d|<0.147 negligible,
    <0.33 small, <0.474 medium, else large.
    """
    x, y = np.asarray(x), np.asarray(y)
    more = sum((xi > y).sum() for xi in x)
    less = sum((xi < y).sum() for xi in x)
    return (more - less) / (len(x) * len(y))


def cohens_d(x, y) -> float:
    """Standardized mean difference for the t-test comparison."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * x.std(ddof=1) ** 2 + (ny - 1) * y.std(ddof=1) ** 2) / (nx + ny - 2))
    return (x.mean() - y.mean()) / pooled_std if pooled_std else float("nan")


def interpret_cliffs(d: float) -> str:
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


# ── Reporting helper ─────────────────────────────────────────────────────
def compare(label: str, a, b, alpha_used: float):
    a = pd.Series(a).dropna().values
    b = pd.Series(b).dropna().values
    if len(a) < 2 or len(b) < 2:
        print(f"\n  {label}: insufficient data (n_a={len(a)}, n_b={len(b)}) -- skipped")
        return

    u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
    t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)  # Welch's, doesn't assume equal variance
    delta = cliffs_delta(a, b)
    d = cohens_d(a, b)

    sig_raw   = "YES" if u_p < ALPHA else "no"
    sig_bonf  = "YES" if u_p < alpha_used else "no"

    print(f"\n  {label}")
    print(f"    n: {GROUP_A}={len(a)}, {GROUP_B}={len(b)}")
    print(f"    Means: {GROUP_A}={a.mean():.3f}  {GROUP_B}={b.mean():.3f}  "
          f"Medians: {GROUP_A}={np.median(a):.3f}  {GROUP_B}={np.median(b):.3f}")
    print(f"    Mann-Whitney U: p={u_p:.4f}  -> significant at alpha={ALPHA}? {sig_raw}  "
          f"| at Bonferroni alpha={alpha_used:.4f}? {sig_bonf}")
    print(f"    Cliff's delta: {delta:+.3f} ({interpret_cliffs(delta)})")
    print(f"    Welch's t-test (secondary): t={t_stat:.3f}, p={t_p:.4f}  |  Cohen's d={d:+.3f}")


def compare_proportions(label: str, count_a: int, n_a: int, count_b: int, n_b: int, alpha_used: float):
    table = [[count_a, n_a - count_a], [count_b, n_b - count_b]]
    odds_ratio, p = stats.fisher_exact(table)
    sig_raw  = "YES" if p < ALPHA else "no"
    sig_bonf = "YES" if p < alpha_used else "no"
    print(f"\n  {label}")
    print(f"    {GROUP_A}: {count_a}/{n_a} ({100*count_a/n_a:.1f}%)   "
          f"{GROUP_B}: {count_b}/{n_b} ({100*count_b/n_b:.1f}%)")
    print(f"    Fisher's exact test: odds ratio={odds_ratio:.3f}, p={p:.4f}  "
          f"-> significant at alpha={ALPHA}? {sig_raw}  | at Bonferroni alpha={alpha_used:.4f}? {sig_bonf}")


def main():
    if not Path(TICKER_CSV).exists() or not Path(DAYS_CSV).exists():
        print(f"[ERROR] Expected {TICKER_CSV} and {DAYS_CSV} in the current directory -- "
              f"run screener_squeeze_compare.py first.")
        sys.exit(1)

    tdf = pd.read_csv(TICKER_CSV)
    ddf = pd.read_csv(DAYS_CSV)

    a_t = tdf[tdf["Group"] == GROUP_A]
    b_t = tdf[tdf["Group"] == GROUP_B]
    a_d = ddf[ddf["Group"] == GROUP_A]
    b_d = ddf[ddf["Group"] == GROUP_B]

    print("=" * 70)
    print(f"  TICKER-LEVEL TESTS (n={len(a_t)} vs n={len(b_t)}) -- the valid comparison")
    print("=" * 70)

    compare("Explosive days per ticker", a_t["ExplosiveDays"], b_t["ExplosiveDays"], ALPHA_BONFERRONI)

    compare_proportions(
        "Tickers with zero explosive days",
        count_a=int((a_t["ExplosiveDays"] == 0).sum()), n_a=len(a_t),
        count_b=int((b_t["ExplosiveDays"] == 0).sum()), n_b=len(b_t),
        alpha_used=ALPHA_BONFERRONI,
    )

    compare("Per-ticker avg % gain on explosive days",
            a_t["AvgExplosiveGainPct"], b_t["AvgExplosiveGainPct"], ALPHA_BONFERRONI)

    compare("Per-ticker avg volume/float ratio on explosive days",
            a_t["AvgExplosiveVolFloatRatio"], b_t["AvgExplosiveVolFloatRatio"], ALPHA_BONFERRONI)

    print("\n" + "=" * 70)
    print(f"  DAY-LEVEL TESTS (n={len(a_d)} vs n={len(b_d)}) -- EXPLORATORY ONLY")
    print("  (pseudoreplication: days from the same ticker aren't independent --")
    print("   treat these as descriptive, not confirmatory)")
    print("=" * 70)

    compare("% gain, pooled across all explosive-day instances",
            a_d["GainPct"], b_d["GainPct"], ALPHA_BONFERRONI)

    compare("Volume/float ratio, pooled across all explosive-day instances",
            a_d["VolFloatRatio"], b_d["VolFloatRatio"], ALPHA_BONFERRONI)

    print("\n" + "=" * 70)
    print("  Notes")
    print("=" * 70)
    print(f"  - Raw alpha = {ALPHA}, Bonferroni-corrected alpha (4 comparisons) = {ALPHA_BONFERRONI:.4f}.")
    print("    A p-value that clears the raw alpha but not the Bonferroni one is 'suggestive,'")
    print("    not confirmed -- worth another sample before treating it as solid.")
    print("  - Sample size caveat: n=30 tickers per group is workable for Mann-Whitney but")
    print("    still fairly small for small-cap names with high dispersion -- a single")
    print("    outlier ticker (e.g. the 19-explosive-day or 25-explosive-day max in your")
    print("    summary) can move the ticker-level mean/median noticeably. Consider re-running")
    print("    with a larger SAMPLE_SIZE, or a fresh random sample, before treating any single")
    print("    run's p-value as final.")
    print("  - This tests whether the two SCREENED POPULATIONS differ, not causation -- it")
    print("    doesn't establish that short float above 30% *causes* more explosive days.")


if __name__ == "__main__":
    main()