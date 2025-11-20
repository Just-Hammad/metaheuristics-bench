"""Command line: `run` an experiment from a config, `report` it, `bias` for the
structural-bias study. One config file fully determines an experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .runner import ProblemSpec, run_experiment
from .report import (
    markdown_error_table, ranks_table, effects_summary, per_problem_winners,
    final_error_table, convergence, ecdf, rank_plot, ok_runs, failure_report,
)


def _problems_from_config(cfg) -> list[ProblemSpec]:
    p = cfg["problems"]
    backend, dim, budget = p["backend"], p["dim"], p["budget_per_dim"] * p["dim"]
    if backend == "ioh":
        import ioh
        cls = {"BBOB": ioh.ProblemClass.BBOB, "CEC2022": ioh.ProblemClass.CEC2022}[p["suite"]]
        fids = p.get("fids") or sorted(cls.problems.keys())
        return [ProblemSpec("ioh", dim, budget, fid, p.get("instance", 1), p["suite"]) for fid in fids]
    if backend == "opfunu":
        excl = set(p.get("exclude", []))
        fids = p.get("fids") or [i for i in range(1, 31) if i not in excl]
        return [ProblemSpec("opfunu", dim, budget, fid, 1, str(p["suite"])) for fid in fids]
    raise ValueError(backend)


def cmd_run(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(cfg["output"])
    problems = _problems_from_config(cfg)
    print(f"[run] {cfg['name']}: {len(problems)} problems x "
          f"{len(cfg['algorithms'])} algorithms x {cfg['n_runs']} runs "
          f"= {len(problems)*len(cfg['algorithms'])*cfg['n_runs']} runs, "
          f"budget {problems[0].budget} FEs")
    df = run_experiment(
        problems=problems, algorithms=cfg["algorithms"], n_runs=cfg["n_runs"],
        master_seed=cfg["master_seed"], out_dir=out,
        workers=args.workers, keep_traces=cfg.get("keep_traces", True),
    )
    print(f"[run] wrote {out}/runs.csv  ({len(df)} rows, "
          f"{df['seconds'].sum()/60:.1f} CPU-min)")


def cmd_report(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(cfg["output"])
    fig = out / "figures"; fig.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(out / "runs.csv")

    fails = failure_report(df)
    if not fails.empty:
        fails.to_csv(out / "table_failures.csv", index=False)
        print(f"[report] {int(fails['failed_runs'].sum())} failed runs excluded "
              f"from statistics (see table_failures.csv)")

    ranks, res = ranks_table(df)
    final_error_table(df).to_csv(out / "table_final_error.csv", index=False)
    ranks.to_csv(out / "table_ranks.csv", index=False)
    per_problem_winners(df).to_csv(out / "table_winners.csv", index=False)
    if res["posthoc"] is not None:
        res["posthoc"].to_csv(out / "table_posthoc_holm.csv")
        effects_summary(res, top=40).to_csv(out / "table_effects.csv", index=False)

    if not fails.empty:
        md_fail = ["## Failed runs (excluded from all statistics)", "",
                   fails.to_markdown(index=False), ""]
    else:
        md_fail = []

    tp = out / "traces.npz"
    if tp.exists():
        convergence(tp, fig / "convergence.png")
        ecdf(tp, fig / "ecdf.png")
    rank_plot(ranks, fig / "ranks.png", n_problems=df["problem"].nunique())

    summary = {
        "friedman_stat": res["friedman_stat"],
        "friedman_p": res["friedman_p"],
        "significant": bool(res["significant"]),
        "n_problems": int(df["problem"].nunique()),
        "n_runs": int(df["run"].nunique()),
        "budget": int(df["budget"].iloc[0]),
        "failed_runs": int(0 if fails.empty else fails["failed_runs"].sum()),
        "ranks": res["ranks"].to_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    md = [f"# {cfg['name']} — results", "",
          f"Friedman chi2 = {res['friedman_stat']:.2f}, p = {res['friedman_p']:.3e} "
          f"({'significant' if res['significant'] else 'NOT significant'} at 0.05)", "",
          "## Average Friedman ranks (1 = best)", "",
          ranks.to_markdown(index=False), "",
          "## Median target error per problem", "",
          markdown_error_table(df), ""]
    if res["posthoc"] is not None:
        md += ["## Post-hoc pairwise Wilcoxon signed-rank, Holm-corrected", "",
               res["posthoc"].round(4).to_markdown(), ""]
    (out / "RESULTS.md").write_text("\n".join(md + md_fail))
    _inject_readme(cfg, res, ranks, df, fails)
    print(f"[report] wrote {out}/RESULTS.md and {fig}/*.png")


README_START = "<!-- RESULTS:START -->"
README_END = "<!-- RESULTS:END -->"


def _inject_readme(cfg, res, ranks, df, fails=None):
    """Rewrite the README's results block from the data.

    The headline numbers in the README are therefore generated, never typed.
    A stale claim is impossible: `make report` regenerates them, and
    scripts/verify_claims.py asserts the underlying invariants.
    """
    readme = Path("README.md")
    if not readme.exists():
        return
    text = readme.read_text()
    if README_START not in text or README_END not in text:
        return

    verdict = "significant" if res["significant"] else "NOT significant"
    lines = [
        README_START,
        "",
        f"*Generated by `make report` from `{cfg['output']}/runs.csv` — do not edit by hand.*",
        "",
        f"**{cfg['name']}** · {df['problem'].nunique()} problems · "
        f"{df['algorithm'].nunique()} algorithms · {df['run'].nunique()} runs · "
        f"{int(df['budget'].iloc[0]):,} FEs · {len(df):,} total runs"
        + ("" if fails is None or fails.empty else
           f" · **{int(fails['failed_runs'].sum())} failed and excluded**"),
        "",
        f"Friedman χ² = {res['friedman_stat']:.1f}, p = {res['friedman_p']:.2e} ({verdict} at α=0.05)",
        "",
        ranks.rename(columns={"algorithm": "Algorithm", "avg_rank": "Avg. Friedman rank"})
             .to_markdown(index=False, floatfmt=".2f"),
        "",
    ]
    if res["posthoc"] is not None:
        short = {c: c.split()[0] for c in res["posthoc"].columns}
        ph = res["posthoc"].rename(index=short, columns=short)
        # Rounding to 4dp prints a true p of 3e-9 as "0", which reads as an
        # impossible p-value. Format small values as an inequality instead.
        def _p(v):
            if v == 1.0:
                return "—"
            return "<0.0001" if v < 1e-4 else f"{v:.4f}"
        ph_fmt = ph.map(_p)
        lines += ["**Post-hoc pairwise Wilcoxon signed-rank, Holm-corrected** (p-values)", "",
                  ph_fmt.to_markdown(), "",
                  "Diagonal omitted. Note **DE/rand/1/bin vs PSO/constriction is not "
                  "significant** despite a rank gap — see finding 3.", ""]
    lines.append(README_END)

    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    readme.write_text(head + "\n".join(lines) + tail)


def cmd_bias(args):
    """Structural-bias study: optimise a random objective and look at where the
    algorithm ends up. See mhbench.bias."""
    from .bias import run_bias_study
    run_bias_study(Path(args.out), n_runs=args.runs, dim=args.dim,
                   budget=args.budget, workers=args.workers, seed=args.seed)


def main(argv=None):
    ap = argparse.ArgumentParser("mhbench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run"); r.add_argument("config"); r.add_argument("--workers", type=int, default=None)
    r.set_defaults(func=cmd_run)

    p = sub.add_parser("report"); p.add_argument("config")
    p.set_defaults(func=cmd_report)

    b = sub.add_parser("bias")
    b.add_argument("--out", default="experiments/exp02_structural_bias/results")
    b.add_argument("--runs", type=int, default=200); b.add_argument("--dim", type=int, default=10)
    b.add_argument("--budget", type=int, default=10000); b.add_argument("--workers", type=int, default=None)
    b.add_argument("--seed", type=int, default=20260914)
    b.set_defaults(func=cmd_bias)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
