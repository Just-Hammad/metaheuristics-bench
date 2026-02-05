"""Structural bias detection.

METHOD
------
Optimise f(x) ~ U(0,1), drawn independently of x. The objective carries zero
information about the search space, so the location an algorithm converges to
cannot be justified by the problem. Over many independent runs, an unbiased
optimizer's final best position must be uniformly distributed over the domain.

Any systematic concentration -- classically toward the domain centre -- is a
property of the ALGORITHM, not the problem. Because most benchmark suites place
optima at or near the centre of the domain, a centre-biased algorithm scores
well on them for a reason unrelated to search quality.

This follows the structural-bias methodology of Kononova, Caraffini, Back et al.

WHY THIS IS THE NEGATIVE RESULT
-------------------------------
It is cheap (no benchmark suite needed), it is visual, and it produces a claim
about an algorithm that a performance table cannot: that part of its measured
advantage is an artifact of where benchmark optima happen to sit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kstest

from .runner import ProblemSpec, run_experiment

ALGORITHMS = [
    "random",
    {"name": "de", "params": {"F": 0.5, "CR": 0.9, "NP": 50}},
    {"name": "pso", "params": {"swarm": 40}},
    {"name": "ga", "params": {"NP": 50}},
    {"name": "cmaes", "params": {"sigma0_frac": 0.3}},
]


def _holm(pvals):
    """Holm step-down correction. Implemented inline rather than pulling in a
    whole statistics package for six lines of arithmetic."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def analyse(positions: dict, lb=-5.0, ub=5.0, alpha=0.05) -> pd.DataFrame:
    """Score each algorithm's final-position distribution against uniformity.

    Separated from the running so the verdict can be recomputed from archived
    positions without re-running the experiment.

    VERDICT TIERS
    -------------
    The per-dimension test is the conservative one but is underpowered at a few
    hundred runs; the pooled test has more samples but its samples are not
    independent (coordinates within a run come from one point). Reporting a
    single binary flag would hide the disagreement, so the evidence is graded:

      biased      per-dimension KS rejects for >=1 dimension after Holm
      suggestive  pooled KS rejects but no single dimension survives Holm
      no evidence neither rejects
    """
    rows = []
    for algo, X in positions.items():
        X = np.asarray(X, float)
        dim = X.shape[1]
        u = (X - lb) / (ub - lb)

        per_dim_p = [kstest(u[:, j], "uniform").pvalue for j in range(dim)]
        adj = _holm(per_dim_p)
        pooled = kstest(u.ravel(), "uniform")

        centre_dist = np.linalg.norm(X, axis=1)
        rng = np.random.default_rng(0)
        ref = np.linalg.norm(rng.uniform(lb, ub, size=(20000, dim)), axis=1)
        ratio = float(np.median(centre_dist) / np.median(ref))

        n_biased = int((adj < alpha).sum())
        if n_biased:
            verdict = "biased"
        elif pooled.pvalue < alpha:
            verdict = "suggestive"
        else:
            verdict = "no evidence"

        rows.append({
            "algorithm": algo,
            "n_runs": len(X),
            "ks_pooled_stat": pooled.statistic,
            "ks_pooled_p": pooled.pvalue,
            "n_dims_biased_holm": n_biased,
            "min_dim_p_holm": float(adj.min()),
            "median_dist_to_centre": float(np.median(centre_dist)),
            "expected_dist_uniform": float(np.median(ref)),
            "centre_ratio": ratio,
            "direction": "outward" if ratio > 1 else "inward",
            "verdict": verdict,
        })
    return pd.DataFrame(rows).sort_values("ks_pooled_p")


def run_bias_study(out_dir: Path, n_runs=200, dim=10, budget=10000,
                   workers=None, seed=20260914):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    spec = ProblemSpec("random", dim=dim, budget=budget)
    print(f"[bias] {len(ALGORITHMS)} algorithms x {n_runs} runs, "
          f"d={dim}, budget={budget} on a random objective")

    df = run_experiment(
        problems=[spec], algorithms=ALGORITHMS, n_runs=n_runs, master_seed=seed,
        out_dir=out, workers=workers, keep_traces=False, keep_best_x=True,
    )
    positions = {
        algo: np.array([json.loads(s) if isinstance(s, str) else s
                        for s in grp["best_x"]], dtype=float)
        for algo, grp in df.groupby("algorithm")
    }
    return finalise(positions, out)


def finalise(positions: dict, out: Path) -> pd.DataFrame:
    """Analyse, write the summary and the figure. Reusable on archived data."""
    out = Path(out)
    res = analyse(positions)
    res.to_csv(out / "bias_summary.csv", index=False)
    np.savez_compressed(
        out / "positions.npz",
        **{k.replace("/", "_"): v for k, v in positions.items()},
    )
    _plot(positions, res, out / "structural_bias.png", -5.0, 5.0)
    _inject_readme(res)
    print(res.to_string(index=False))
    return res


BIAS_START = "<!-- BIAS:START -->"
BIAS_END = "<!-- BIAS:END -->"


def _inject_readme(res: pd.DataFrame, readme: Path = Path("README.md")):
    """Write the bias verdict table into the README from the data.

    Same rule as the benchmark results: generated, never typed. A hand-written
    table here already drifted once, when CMA-ES gained IPOP restarts and the
    README kept describing the previous implementation.
    """
    if not readme.exists():
        return
    text = readme.read_text()
    if BIAS_START not in text or BIAS_END not in text:
        return

    _SUP = str.maketrans("0123456789", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")

    def fmt(x):
        mant, exp = f"{x:.1e}".split("e")
        return f"{mant} × 10{str(int(exp)).translate(_SUP).replace('-', '\u207b')}"

    rows = ["| Algorithm | Verdict | Dims biased (Holm) | Pooled KS *p* | Distance to centre vs. uniform |",
            "|---|---|---|---|---|"]
    label = {"biased": "**biased**", "suggestive": "suggestive",
             "no evidence": "no evidence"}
    for _, r in res.iterrows():
        name = r["algorithm"].split()[0]
        dims = f"**{r['n_dims_biased_holm']} / 10**" if r["verdict"] == "biased" \
            else f"{r['n_dims_biased_holm']} / 10"
        rows.append(
            f"| `{name}` | {label[r['verdict']]} | {dims} | {fmt(r['ks_pooled_p'])} | "
            f"{r['centre_ratio']:.3f}× ({r['direction']}) |"
        )

    n = int(res["n_runs"].iloc[0])
    block = [BIAS_START, "",
             f"*Generated by `make exp02` from {n} independent runs per algorithm "
             f"on a random objective — do not edit by hand.*", "",
             *rows, "", BIAS_END]
    head, rest = text.split(BIAS_START, 1)
    _, tail = rest.split(BIAS_END, 1)
    readme.write_text(head + "\n".join(block) + tail)


def _plot(positions, res, path, lb, ub):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    algos = list(res["algorithm"])
    fig, axes = plt.subplots(1, len(algos), figsize=(2.7 * len(algos), 2.9), sharey=True)
    for ax, algo in zip(np.atleast_1d(axes), algos):
        X = positions[algo]
        ax.hist(X.ravel(), bins=40, range=(lb, ub), density=True,
                color="#3b6ea5", alpha=0.85)
        ax.axhline(1.0 / (ub - lb), color="crimson", ls="--", lw=1.2,
                   label="uniform")
        row = res[res.algorithm == algo].iloc[0]
        v = row["verdict"]
        flag = {"biased": "BIASED", "suggestive": "suggestive",
                "no evidence": "no evidence"}[v]
        ax.set_title(
            f"{algo.split()[0]}\n{flag}  "
            f"({row['n_dims_biased_holm']}/{positions[algo].shape[1]} dims, "
            f"pooled p={row['ks_pooled_p']:.1e})",
            fontsize=7.5,
            color={"biased": "#b3261e", "suggestive": "#8a6d00",
                   "no evidence": "#33691e"}[v],
        )
        ax.set_xlabel("coordinate of best-found point")
    np.atleast_1d(axes)[0].set_ylabel("density")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle("Structural bias: where each algorithm converges on a random objective",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
