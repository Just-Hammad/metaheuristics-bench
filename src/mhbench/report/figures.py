"""Figures. Three artifacts, per 12-build-guides.md section 3.

The ECDF is the one that matters most to this audience: an algorithm that wins
at full budget can lose badly at 10% of it, and a final-error table cannot show
that.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TARGETS = np.logspace(2, -8, 41)  # COCO-style target levels


def _style():
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
        "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
    })


def convergence(traces_path, out_path, problems=None, ncols=4):
    """Median best-so-far with IQR band, log-log. One panel per problem."""
    _style()
    z = np.load(traces_path)
    grid = z["grid"]
    keys = [k for k in z.files if k != "grid"]
    probs = sorted({k.split("|")[0] for k in keys}, key=_nat)
    if problems:
        probs = [p for p in probs if p in problems]
    algos = sorted({k.split("|")[1] for k in keys})

    nrows = int(np.ceil(len(probs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.5 * nrows),
                             squeeze=False, sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    handles, labels = [], []
    for ax, prob in zip(axes.ravel(), probs):
        for i, algo in enumerate(algos):
            key = f"{prob}|{algo}"
            if key not in z.files:
                continue
            M = np.where(np.isfinite(z[key]), z[key], np.nan)
            M = np.maximum(M, 1e-12)
            med = np.nanmedian(M, axis=0)
            lo, hi = np.nanpercentile(M, [25, 75], axis=0)
            ln, = ax.plot(grid, med, color=colors[i % 10], lw=1.4,
                          label=algo.split()[0])
            ax.fill_between(grid, lo, hi, color=colors[i % 10], alpha=0.13, lw=0)
            if algo.split()[0] not in labels:
                handles.append(ln)
                labels.append(algo.split()[0])
        ax.set(xscale="log", yscale="log", title=prob)
        ax.tick_params(labelsize=7)
        ax.title.set_size(8)
    for ax in axes.ravel()[len(probs):]:
        ax.axis("off")

    # Figure-level legend: a per-axes legend covers data in whichever panel
    # hosts it, and the series are identical across all panels anyway.
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               fontsize=8.5, bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.supxlabel("function evaluations", y=0.005, fontsize=9)
    fig.supylabel("target error (median, IQR band)", x=0.005, fontsize=9)
    fig.tight_layout(rect=[0.01, 0.01, 1, 0.975])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def ecdf(traces_path, out_path, budget_per_dim=None):
    """COCO-style empirical runtime distribution.

    For every (problem, run, target) the first budget reaching that target is a
    'task'. The curve is the fraction of tasks solved within budget b. This is
    the anytime view a single final-error number discards.
    """
    _style()
    z = np.load(traces_path)
    grid = z["grid"]
    keys = [k for k in z.files if k != "grid"]
    algos = sorted({k.split("|")[1] for k in keys})
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for i, algo in enumerate(algos):
        mats = [z[k] for k in keys if k.split("|")[1] == algo]
        if not mats:
            continue
        M = np.vstack(mats)                       # (runs*problems, len(grid))
        M = np.where(np.isfinite(M), M, np.inf)
        # hit[t, r, g] : run r reached target t by budget g
        hit = M[None, :, :] <= TARGETS[:, None, None]
        frac = hit.reshape(-1, len(grid)).mean(axis=0)
        ax.plot(grid, frac, lw=1.8, color=colors[i % 10], label=algo.split()[0])

    ax.set(xscale="log", xlabel="function evaluations", ylim=(0, 1),
           ylabel="fraction of (problem, run, target) pairs solved",
           title="Empirical runtime distribution — 41 targets, 1e2 to 1e-8")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def rank_plot(ranks_df, out_path, n_problems=None):
    """Average Friedman rank, lower is better."""
    _style()
    d = ranks_df.sort_values("avg_rank")
    fig, ax = plt.subplots(figsize=(6.0, 0.45 * len(d) + 1.2))
    ax.barh(d["algorithm"], d["avg_rank"], color="#3b6ea5", height=0.6)
    for y, v in enumerate(d["avg_rank"]):
        ax.text(v + 0.04, y, f"{v:.2f}", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set(xlabel="average Friedman rank (1 = best)",
           title=f"Average rank over {n_problems} problems" if n_problems else "Average rank")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _nat(s):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]
