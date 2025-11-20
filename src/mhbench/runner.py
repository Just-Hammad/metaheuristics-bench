"""Experiment runner: seeding, budget enforcement, parallelism, persistence.

Seeding
-------
Per-run seeds are SPAWNED from one master seed via numpy's SeedSequence, which
guarantees independent streams (master_seed + i does not). Every spawned seed's
entropy is written into the results file, so "reproducible" means a specific
run can be replayed, not merely that a seed was set somewhere.

Budget
------
Enforced by TrackedProblem, asserted here after every run. An algorithm cannot
overspend even by accident, and if one ever did the run would fail loudly
rather than silently winning.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .algorithms import build as build_algorithm
from .problems import ioh_problem, opfunu_problem, make_classic, make_random_objective

TRACE_POINTS = 60  # log-spaced budget grid for convergence curves


@dataclass(frozen=True)
class ProblemSpec:
    """Serialisable description of a problem; rebuilt inside each worker."""

    backend: str            # 'ioh' | 'opfunu' | 'classic' | 'random'
    dim: int
    budget: int
    fid: int = 1
    instance: int = 1
    suite: str = "BBOB"     # ioh: BBOB|CEC2022 ; opfunu: year string
    name: str = ""
    shift: tuple = ()

    def build(self, run_seed: int = 0):
        if self.backend == "ioh":
            return ioh_problem(self.fid, self.instance, self.dim, self.budget, self.suite)
        if self.backend == "opfunu":
            return opfunu_problem(self.fid, self.dim, self.budget, self.suite)
        if self.backend == "classic":
            shift = np.asarray(self.shift, float) if self.shift else None
            return make_classic(self.name, self.dim, self.budget, shift)
        if self.backend == "random":
            return make_random_objective(self.dim, self.budget, seed=run_seed)
        raise ValueError(f"unknown backend {self.backend!r}")

    def label(self) -> str:
        if self.backend == "ioh":
            return f"{self.suite}_f{self.fid}"
        if self.backend == "opfunu":
            return f"CEC{self.suite}_f{self.fid}"
        if self.backend == "classic":
            return self.name + ("_shifted" if self.shift else "")
        return "random_objective"


def _one_run(args):
    """Single (algorithm, problem, run). Runs in a worker process.

    A run that raises is RECORDED AS A FAILURE, not propagated. One numerically
    diverging optimizer must not destroy an entire multi-hour experiment -- an
    earlier version did exactly that. Failures are excluded from the statistics
    and reported explicitly, never silently dropped: a run that fails is a fact
    about the algorithm and belongs in the results.
    """
    algo_spec, prob_spec, run_idx, seed_entropy, keep_trace = args
    rng = np.random.default_rng(seed_entropy)

    t0 = time.perf_counter()
    status, error_msg = "ok", ""
    problem = algo = None
    try:
        # Construction is inside the guard too: a bad spec or an unregistered
        # algorithm is a recordable failure, not a reason to lose the run.
        problem = prob_spec.build(run_seed=int(seed_entropy % (2**31)))
        algo = build_algorithm(algo_spec)
        assert problem.evaluations == 0, "problem not fresh at run start"
        algo(problem, rng)
    except Exception as exc:                       # noqa: BLE001 - deliberate
        status = "failed"
        error_msg = f"{type(exc).__name__}: {exc}"[:200]
        if problem is None:
            problem = prob_spec.build(run_seed=0)   # empty shell for the record
    elapsed = time.perf_counter() - t0

    algo_label = (algo.label() if algo is not None
                  else (algo_spec if isinstance(algo_spec, str)
                        else algo_spec.get("name", "unknown")))

    assert problem.evaluations <= prob_spec.budget, (
        f"budget violated: {problem.evaluations} > {prob_spec.budget}"
    )

    trace = None
    if keep_trace:
        grid = np.unique(
            np.geomspace(1, prob_spec.budget, TRACE_POINTS).astype(int)
        )
        trace = problem.trace.resample(grid).astype(np.float64) - problem.optimum

    return {
        "problem": prob_spec.label(),
        "algorithm": algo_label,
        "status": status,
        "error_msg": error_msg,
        "algorithm_key": algo_spec if isinstance(algo_spec, str) else algo_spec["name"],
        "dim": prob_spec.dim,
        "run": run_idx,
        "seed": int(seed_entropy),
        "error": float(problem.target_error) if status == "ok" else float("nan"),
        "best_value": float(problem.best_value),
        "evaluations": int(problem.evaluations),
        "budget": int(prob_spec.budget),
        "seconds": round(elapsed, 4),
        "best_x": None if problem.best_x is None else problem.best_x.tolist(),
    }, trace


def run_experiment(
    problems: list[ProblemSpec],
    algorithms: list,
    n_runs: int,
    master_seed: int,
    out_dir: str | Path,
    workers: int | None = None,
    keep_traces: bool = True,
    keep_best_x: bool = False,
    checkpoint_every: int = 250,
) -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Independent per-run streams, derived once and logged.
    ss = np.random.SeedSequence(master_seed)
    children = ss.spawn(len(problems) * len(algorithms) * n_runs)

    jobs, k = [], 0
    for prob in problems:
        for algo in algorithms:
            for r in range(n_runs):
                entropy = int(children[k].generate_state(1, dtype=np.uint32)[0])
                jobs.append((algo, prob, r, entropy, keep_traces))
                k += 1

    records, traces = [], {}
    ckpt = out / "runs.partial.csv"
    done = 0
    t_start = time.perf_counter()
    # as_completed, NOT map: map yields strictly in submission order, so a slow
    # early job buffers every later result and progress reporting becomes a
    # function of ordering rather than of work actually finished.
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_one_run, j) for j in jobs]
        for fut in as_completed(futures):
            rec, tr = fut.result()
            if not keep_best_x:
                rec.pop("best_x", None)
            records.append(rec)
            if tr is not None:
                traces.setdefault(f"{rec['problem']}|{rec['algorithm']}", []).append(
                    (rec["run"], tr))
            done += 1
            # Checkpoint so a long experiment is never lost to a late failure.
            if done % checkpoint_every == 0:
                pd.DataFrame(records).to_csv(ckpt, index=False)
                el = time.perf_counter() - t_start
                eta = el / done * (len(jobs) - done)
                print(f"  [{done}/{len(jobs)}] {100*done/len(jobs):.0f}%  "
                      f"elapsed {el/60:.1f}m  eta {eta/60:.1f}m", flush=True)

    # as_completed yields in completion order, which varies run to run. Sort on
    # a deterministic key so runs.csv is byte-identical for a given master seed;
    # otherwise "reproducible" would only mean "the same numbers in some order".
    df = (pd.DataFrame(records)
          .sort_values(["problem", "algorithm", "run"], kind="stable")
          .reset_index(drop=True))
    df.to_csv(out / "runs.csv", index=False)
    ckpt.unlink(missing_ok=True)

    n_failed = int((df["status"] == "failed").sum())
    if n_failed:
        print(f"  WARNING: {n_failed}/{len(df)} runs FAILED and are excluded "
              f"from statistics. Breakdown:")
        for (algo, msg), n in (df[df.status == "failed"]
                               .groupby(["algorithm", "error_msg"]).size().items()):
            print(f"    {n:4d}  {algo}  {msg}")

    if traces:
        grid = np.unique(np.geomspace(1, problems[0].budget, TRACE_POINTS).astype(int))
        np.savez_compressed(
            out / "traces.npz",
            grid=grid,
            # Sorted by run index for the same determinism reason as runs.csv.
            **{k: np.vstack([t for _, t in sorted(v, key=lambda p: p[0])])
               for k, v in traces.items()},
        )

    manifest = {
        "master_seed": master_seed,
        "n_runs": n_runs,
        "n_problems": len(problems),
        "n_algorithms": len(algorithms),
        "total_runs": len(jobs),
        "failed_runs": n_failed,
        "failure_rate": round(n_failed / max(1, len(df)), 5),
        "problems": [asdict(p) for p in problems],
        "algorithms": [a if isinstance(a, str) else a for a in algorithms],
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": round(float(df["seconds"].sum()), 2),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return df
