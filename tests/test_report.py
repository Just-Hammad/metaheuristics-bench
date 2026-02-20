"""Report generation must work end-to-end on a tiny experiment, so `make all`
cannot break silently between real runs."""
import numpy as np
import pandas as pd
import pytest

from mhbench.runner import ProblemSpec, run_experiment
from mhbench.report import (
    final_error_table, markdown_error_table, ranks_table,
    per_problem_winners, convergence, ecdf, rank_plot,
)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    out = tmp_path_factory.mktemp("tiny")
    specs = [ProblemSpec("ioh", dim=5, budget=600, fid=f) for f in (1, 2, 3, 8)]
    df = run_experiment(specs, ["random", "de", "pso"], n_runs=6, master_seed=3,
                        out_dir=out, workers=3, keep_traces=True)
    return df, out


def test_final_error_table_shape(tiny):
    df, _ = tiny
    t = final_error_table(df)
    assert len(t) == df["problem"].nunique() * df["algorithm"].nunique()
    assert (t["n_runs"] == 6).all()
    assert t["median"].notna().all()


def test_markdown_table_bolds_a_winner_per_row(tiny):
    df, _ = tiny
    md = markdown_error_table(df)
    body = [l for l in md.splitlines() if l.startswith("| BBOB")]
    assert body, "no data rows rendered"
    for line in body:
        assert line.count("**") == 2, f"exactly one winner per row: {line}"


def test_ranks_are_a_permutation(tiny):
    df, _ = tiny
    ranks, res = ranks_table(df)
    assert set(ranks["algorithm"]) == set(df["algorithm"].unique())
    assert ranks["avg_rank"].between(1, df["algorithm"].nunique()).all()
    assert "friedman_p" in res


def test_no_posthoc_when_omnibus_not_significant():
    """A non-significant Friedman must not be followed by a post-hoc table."""
    from mhbench.stats import full_comparison
    rng = np.random.default_rng(0)
    n = 20
    long = pd.DataFrame({
        "problem": np.repeat([f"p{i}" for i in range(n)], 3 * 5),
        "algorithm": np.tile(np.repeat(["a", "b", "c"], 5), n),
        "run": np.tile(np.arange(5), n * 3),
        "error": rng.normal(1.0, 1.0, n * 15),
    })
    res = full_comparison(long)
    if not res["significant"]:
        assert res["posthoc"] is None and res["effects"] is None


def test_winners_table_has_effect_size(tiny):
    df, _ = tiny
    w = per_problem_winners(df)
    assert len(w) == df["problem"].nunique()
    assert w["cliffs_delta"].between(-1, 1).all()
    assert w["magnitude"].isin(["negligible", "small", "medium", "large", "undefined"]).all()


def test_figures_render(tiny, tmp_path):
    df, out = tiny
    ranks, _ = ranks_table(df)
    tp = out / "traces.npz"
    for f in (convergence(tp, tmp_path / "c.png"),
              ecdf(tp, tmp_path / "e.png"),
              rank_plot(ranks, tmp_path / "r.png")):
        assert f.exists() and f.stat().st_size > 5000
