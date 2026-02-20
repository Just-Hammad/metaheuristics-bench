"""The three assertions that separate a statistics module from a
statistics-shaped function (12-build-guides.md section 2)."""
import numpy as np
import pandas as pd
import pytest

from mhbench.stats import cliffs_delta, delta_magnitude, friedman_ranks, posthoc_holm


def test_cliffs_delta_self_is_zero():
    x = np.array([1.0, 2, 3, 4, 5])
    assert cliffs_delta(x, x) == 0.0


def test_cliffs_delta_strict_dominance_is_one():
    a, b = np.array([10.0, 11, 12]), np.array([1.0, 2, 3])
    assert cliffs_delta(a, b) == 1.0
    assert cliffs_delta(b, a) == -1.0


def test_cliffs_delta_magnitude_labels():
    assert delta_magnitude(0.0) == "negligible"
    assert delta_magnitude(0.2) == "small"
    assert delta_magnitude(0.4) == "medium"
    assert delta_magnitude(0.9) == "large"


def test_cliffs_delta_sign_convention_lower_is_better():
    """Errors: lower is better, so a better algorithm has NEGATIVE delta."""
    better, worse = np.array([0.1, 0.2, 0.15]), np.array([5.0, 6.0, 5.5])
    assert cliffs_delta(better, worse) == -1.0


def test_friedman_recovers_known_ranking():
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame({
        "best": rng.normal(1.0, 0.1, n),
        "mid": rng.normal(5.0, 0.1, n),
        "worst": rng.normal(9.0, 0.1, n),
    })
    stat, p, ranks = friedman_ranks(df)
    assert p < 0.01
    assert list(ranks.index) == ["best", "mid", "worst"]
    assert ranks["best"] == pytest.approx(1.0)


def test_friedman_rejects_two_groups():
    df = pd.DataFrame({"a": [1.0, 2, 3], "b": [4.0, 5, 6]})
    with pytest.raises(ValueError):
        friedman_ranks(df)


def test_posthoc_holm_is_symmetric_and_labelled():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "a": rng.normal(1.0, 0.1, 25),
        "b": rng.normal(5.0, 0.1, 25),
        "c": rng.normal(9.0, 0.1, 25),
    })
    ph = posthoc_holm(df)
    assert list(ph.columns) == ["a", "b", "c"]
    assert ph.loc["a", "b"] == pytest.approx(ph.loc["b", "a"])
    assert ph.loc["a", "b"] < 0.05
