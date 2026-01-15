"""Tables. Median and IQR throughout -- never mean +/- std.

These distributions are bounded below and heavily skewed; mean +/- std asserts a
normality they do not have (12-build-guides.md section 1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..stats import full_comparison, delta_magnitude


def ok_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Analysis-eligible runs only.

    A failed run is a fact about the algorithm and is kept in runs.csv, but it
    has no target error and must not enter a median or a rank. Excluding it
    here -- in one place, explicitly -- is the alternative to letting NaN
    propagate silently through pandas aggregations.
    """
    if "status" not in df.columns:
        return df                      # results predating failure tracking
    return df[df["status"] == "ok"].copy()


def failure_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-algorithm failure counts. Empty frame when nothing failed."""
    if "status" not in df.columns:
        return pd.DataFrame()
    bad = df[df["status"] != "ok"]
    if bad.empty:
        return pd.DataFrame()
    return (bad.groupby(["algorithm", "error_msg"])
               .size().rename("failed_runs").reset_index()
               .sort_values("failed_runs", ascending=False))


def final_error_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per (problem, algorithm): median, IQR, min, and success rate."""
    df = ok_runs(df)
    g = df.groupby(["problem", "algorithm"])["error"]
    out = pd.DataFrame({
        "median": g.median(),
        "iqr": g.quantile(0.75) - g.quantile(0.25),
        "min": g.min(),
        "max": g.max(),
        "n_runs": g.size(),
    })
    out["solved_1e-8"] = g.apply(lambda s: float((s <= 1e-8).mean()))
    return out.reset_index()


def markdown_error_table(df: pd.DataFrame, max_problems: int = 24) -> str:
    tab = final_error_table(df)   # already filtered
    piv = tab.pivot(index="problem", columns="algorithm", values="median")
    piv = piv.reindex(sorted(piv.index, key=_natural_key))[:max_problems]
    lines = ["| Problem | " + " | ".join(piv.columns) + " |",
             "|---" * (len(piv.columns) + 1) + "|"]
    for prob, row in piv.iterrows():
        best = row.min()
        cells = []
        for v in row:
            s = f"{v:.3e}"
            cells.append(f"**{s}**" if np.isclose(v, best, rtol=1e-12) else s)
        lines.append(f"| {prob} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _natural_key(s):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def ranks_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Friedman ranks plus the full post-hoc comparison. Failed runs excluded."""
    df = ok_runs(df)
    res = full_comparison(df[["problem", "algorithm", "run", "error"]])
    ranks = res["ranks"].rename("avg_rank").to_frame().reset_index()
    ranks.columns = ["algorithm", "avg_rank"]
    return ranks, res


def effects_summary(res: dict, top: int = 25) -> pd.DataFrame:
    """Largest effects first -- the pairs that actually differ.

    Takes the comparison result, not a run frame: failed runs were already
    excluded upstream in `ranks_table`.
    """
    if res.get("effects") is None:
        return pd.DataFrame()
    e = res["effects"].copy()
    e["abs_delta"] = e["cliffs_delta"].abs()
    return e.sort_values("abs_delta", ascending=False).head(top).drop(columns="abs_delta")


def per_problem_winners(df: pd.DataFrame) -> pd.DataFrame:
    """Which algorithm wins each problem, and whether the margin is significant."""
    df = ok_runs(df)
    from ..stats import cliffs_delta, pairwise_within_problem

    rows = []
    for prob, grp in df.groupby("problem"):
        med = grp.groupby("algorithm")["error"].median().sort_values()
        win, second = med.index[0], med.index[1] if len(med) > 1 else None
        a = grp[grp.algorithm == win]["error"].to_numpy()
        b = grp[grp.algorithm == second]["error"].to_numpy() if second else a
        d = cliffs_delta(a, b)
        rows.append({
            "problem": prob, "winner": win, "runner_up": second,
            "median_winner": med.iloc[0],
            "median_runner_up": med.iloc[1] if len(med) > 1 else np.nan,
            "cliffs_delta": d, "magnitude": delta_magnitude(d),
            "p_within": pairwise_within_problem(a, b),
        })
    return pd.DataFrame(rows).sort_values("problem", key=lambda s: s.map(_natural_key))
