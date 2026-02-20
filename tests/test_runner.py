"""Budget and seeding are the two things that make a comparison honest."""
import numpy as np
import pytest

from mhbench.runner import ProblemSpec, run_experiment
from mhbench.problems import ioh_problem
from mhbench.problems.base import BudgetExhausted
from mhbench.algorithms import build


def test_budget_is_enforced_not_requested():
    """The problem refuses evaluation 101, it does not trust the algorithm."""
    p = ioh_problem(1, 1, 5, budget=100)
    for _ in range(100):
        p(np.zeros(5))
    with pytest.raises(BudgetExhausted):
        p(np.zeros(5))


def test_every_algorithm_stops_exactly_at_budget():
    for key in ["random", "de", "pso", "ga", "cmaes"]:
        p = ioh_problem(1, 1, 5, budget=200)
        build(key)(p, np.random.default_rng(0))
        assert p.evaluations == 200, f"{key} used {p.evaluations}"


def test_reset_clears_state():
    p = ioh_problem(1, 1, 5, budget=50)
    build("de")(p, np.random.default_rng(0))
    assert p.evaluations == 50
    p.reset()
    assert p.evaluations == 0 and p.best_value == np.inf and not p.trace.evals


def test_same_master_seed_reproduces_exactly(tmp_path):
    specs = [ProblemSpec("ioh", dim=5, budget=300, fid=1)]
    kw = dict(problems=specs, algorithms=["de"], n_runs=3, workers=2, keep_traces=False)
    a = run_experiment(master_seed=42, out_dir=tmp_path / "a", **kw)
    b = run_experiment(master_seed=42, out_dir=tmp_path / "b", **kw)
    c = run_experiment(master_seed=43, out_dir=tmp_path / "c", **kw)
    assert list(a["error"]) == list(b["error"])
    assert list(a["seed"]) == list(b["seed"])
    assert list(a["error"]) != list(c["error"])


def test_seeds_are_distinct_across_runs(tmp_path):
    specs = [ProblemSpec("ioh", dim=5, budget=200, fid=1)]
    df = run_experiment(specs, ["de"], n_runs=10, master_seed=7,
                        out_dir=tmp_path, workers=2, keep_traces=False)
    assert df["seed"].nunique() == 10


def test_trace_is_monotone_non_increasing(tmp_path):
    specs = [ProblemSpec("ioh", dim=5, budget=500, fid=1)]
    run_experiment(specs, ["de"], n_runs=2, master_seed=1,
                   out_dir=tmp_path, workers=1, keep_traces=True)
    z = np.load(tmp_path / "traces.npz")
    key = [k for k in z.files if k != "grid"][0]
    for row in z[key]:
        finite = row[np.isfinite(row)]
        assert np.all(np.diff(finite) <= 1e-12), "best-so-far must not increase"


def test_one_failing_algorithm_does_not_abort_the_experiment(tmp_path):
    """A diverging optimizer must not destroy a multi-hour run.

    Regression test: an earlier CMA-ES configuration raised AssertionError deep
    inside `cma` and killed a 40-minute experiment at the final gather. Here an
    invalid population size makes DE raise inside its own search loop.
    """
    specs = [ProblemSpec("ioh", dim=5, budget=200, fid=1)]
    good = {"name": "de", "params": {"NP": 20}}
    bad = {"name": "de", "params": {"NP": -5}}      # raises inside the algorithm
    df = run_experiment(specs, [good, bad], n_runs=4, master_seed=5,
                        out_dir=tmp_path, workers=2, keep_traces=False)

    assert len(df) == 8, "every job must produce a record, failures included"
    assert (df["status"] == "ok").sum() == 4
    failed = df[df["status"] == "failed"]
    assert len(failed) == 4
    assert failed["error"].isna().all(), "a failed run has no target error"
    assert failed["error_msg"].str.len().gt(0).all()


def test_unregistered_algorithm_is_recorded_not_raised(tmp_path):
    """Construction failures are recorded too, not propagated."""
    specs = [ProblemSpec("ioh", dim=5, budget=100, fid=1)]
    df = run_experiment(specs, ["de", "no_such_algorithm"], n_runs=2,
                        master_seed=1, out_dir=tmp_path, workers=2,
                        keep_traces=False)
    assert len(df) == 4
    bad = df[df["status"] == "failed"]
    assert len(bad) == 2
    assert bad["error_msg"].str.contains("KeyError").all()


def test_failed_runs_are_excluded_from_statistics(tmp_path):
    """Failures stay in runs.csv but must never enter a median or a rank."""
    import pandas as pd
    from mhbench.report import ok_runs, failure_report

    df = pd.DataFrame({
        "problem": ["p1"] * 6, "algorithm": ["a"] * 3 + ["b"] * 3,
        "run": [0, 1, 2] * 2, "error": [1.0, 2.0, float("nan"), 5.0, 6.0, 7.0],
        "status": ["ok", "ok", "failed", "ok", "ok", "ok"],
        "error_msg": ["", "", "AssertionError: boom", "", "", ""],
    })
    assert len(ok_runs(df)) == 5
    assert ok_runs(df)["error"].notna().all()
    fr = failure_report(df)
    assert len(fr) == 1 and fr.iloc[0]["failed_runs"] == 1
