"""Non-parametric comparison pipeline.

    per-run raw results
            |
            +-- aggregate per (algorithm, problem) -> median
            v
      [1] Friedman across algorithms, problems as blocks  -> average ranks
            |   if not significant: STOP. There is no post-hoc.
            v
      [2] post-hoc pairwise Wilcoxon signed-rank + Holm correction
            v
      [3] Cliff's delta -- PER PROBLEM, on RAW RUNS, never on the aggregates

THE DISTINCTION THAT MATTERS
----------------------------
Steps 1-2 consume the algorithms x problems matrix of aggregates.
Step 3 does not. Cliff's delta measures stochastic dominance between two
SAMPLES. Computed over one median per problem it merely restates the Friedman
ranks; computed over the raw runs within one problem it answers "when both run
here, how often does A win" -- which is what an effect size is for.

The API enforces this: `friedman_ranks` and `posthoc_holm` take a DataFrame of
aggregates; `cliffs_delta` takes two 1-D arrays of raw run values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp

# Romano et al. magnitude cut-points.
_DELTA_BINS = [(0.147, "negligible"), (0.33, "small"), (0.474, "medium")]


def friedman_ranks(matrix: pd.DataFrame):
    """Friedman test over an aggregates matrix.

    Parameters
    ----------
    matrix : rows = problems (blocks), cols = algorithms, values = per-problem
             aggregate (median target error). Lower is better.

    Returns
    -------
    stat, p, ranks : ranks is a Series, rank 1 = best, sorted ascending.
    """
    if matrix.shape[1] < 3:
        raise ValueError("Friedman needs >= 3 groups; use Wilcoxon directly for 2")
    stat, p = friedmanchisquare(*[matrix[c].to_numpy() for c in matrix.columns])
    ranks = matrix.rank(axis=1, ascending=True).mean(axis=0)
    return float(stat), float(p), ranks.sort_values()


def posthoc_holm(matrix: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Wilcoxon SIGNED-RANK (paired: problems are the pairs) with Holm.

    Use when asking "is A better than B ACROSS THE SUITE".
    """
    cols = list(matrix.columns)
    out = sp.posthoc_wilcoxon(
        [matrix[c].to_numpy() for c in cols], p_adjust="holm", zero_method="zsplit"
    )
    out.index = cols
    out.columns = cols
    return out


def pairwise_within_problem(runs_a, runs_b):
    """Wilcoxon RANK-SUM (unpaired) for "is A better than B ON THIS PROBLEM".

    A different test from `posthoc_holm`, answering a different question, on
    raw runs rather than aggregates. Named separately so the two cannot be
    confused in a caption.
    """
    from scipy.stats import mannwhitneyu

    a, b = np.asarray(runs_a, float), np.asarray(runs_b, float)
    if np.allclose(a, b):
        return 1.0
    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)


def cliffs_delta(a, b) -> float:
    """Stochastic dominance of `a` over `b`. RAW RUNS, one problem.

    Returns delta in [-1, 1]. Sign convention: these are errors (lower better),
    so NEGATIVE delta means `a` is better than `b`.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = int((a[:, None] > b[None, :]).sum())
    lt = int((a[:, None] < b[None, :]).sum())
    return (gt - lt) / (a.size * b.size)


def delta_magnitude(delta: float) -> str:
    """Romano et al. label. Reported alongside the number so the reader does
    not have to look up the scale."""
    if np.isnan(delta):
        return "undefined"
    d = abs(delta)
    for cut, label in _DELTA_BINS:
        if d < cut:
            return label
    return "large"


def full_comparison(long_df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Run the whole pipeline on a tidy frame.

    long_df columns: problem, algorithm, run, error
    """
    agg = long_df.pivot_table(
        index="problem", columns="algorithm", values="error", aggfunc="median"
    )
    stat, p, ranks = friedman_ranks(agg)
    result = {
        "aggregates": agg,
        "friedman_stat": stat,
        "friedman_p": p,
        "ranks": ranks,
        "significant": p < alpha,
        "posthoc": None,
        "effects": None,
    }
    if p >= alpha:
        return result  # no post-hoc after a non-significant omnibus test

    result["posthoc"] = posthoc_holm(agg)

    rows = []
    algos = list(agg.columns)
    for prob, grp in long_df.groupby("problem"):
        by_algo = {a: g["error"].to_numpy() for a, g in grp.groupby("algorithm")}
        for i, a in enumerate(algos):
            for b in algos[i + 1:]:
                if a not in by_algo or b not in by_algo:
                    continue
                d = cliffs_delta(by_algo[a], by_algo[b])
                rows.append({
                    "problem": prob, "a": a, "b": b, "cliffs_delta": d,
                    "magnitude": delta_magnitude(d),
                    "p_within": pairwise_within_problem(by_algo[a], by_algo[b]),
                })
    result["effects"] = pd.DataFrame(rows)
    return result
