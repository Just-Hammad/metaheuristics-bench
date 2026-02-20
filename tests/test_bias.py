"""The structural-bias detector must be validated against known ground truth.

A bias test that has never been shown to fire on a biased sample, and stay
silent on an unbiased one, is not evidence of anything.
"""
import numpy as np
import pytest

from mhbench.bias import analyse, _holm


N, DIM, LB, UB = 300, 10, -5.0, 5.0


def _uniform(rng):
    return rng.uniform(LB, UB, size=(N, DIM))


def _boundary_biased(rng, frac=0.25):
    """Like uniform, but a fraction of coordinates are snapped to a bound --
    the signature that clipping-based repair produces."""
    X = rng.uniform(LB, UB, size=(N, DIM))
    mask = rng.random(X.shape) < frac
    X[mask] = np.where(rng.random(mask.sum()) < 0.5, LB, UB)
    return X


def _centre_biased(rng):
    return np.clip(rng.normal(0.0, 0.7, size=(N, DIM)), LB, UB)


def test_detector_is_silent_on_uniform_samples():
    """No false positive on the null. This is what makes a rejection mean
    something."""
    rng = np.random.default_rng(0)
    res = analyse({"uniform": _uniform(rng)})
    assert res.iloc[0]["verdict"] == "no evidence"


def test_detector_fires_on_boundary_bias():
    rng = np.random.default_rng(1)
    res = analyse({"edge": _boundary_biased(rng)})
    row = res.iloc[0]
    assert row["verdict"] == "biased"
    assert row["n_dims_biased_holm"] == DIM
    assert row["direction"] == "outward"


def test_detector_fires_on_centre_bias_and_names_the_direction():
    rng = np.random.default_rng(2)
    row = analyse({"centre": _centre_biased(rng)}).iloc[0]
    assert row["verdict"] == "biased"
    assert row["direction"] == "inward"
    assert row["centre_ratio"] < 1.0


def test_verdicts_are_only_the_declared_tiers():
    rng = np.random.default_rng(3)
    res = analyse({"u": _uniform(rng), "e": _boundary_biased(rng),
                   "c": _centre_biased(rng)})
    assert set(res["verdict"]) <= {"biased", "suggestive", "no evidence"}


def test_holm_correction_is_monotone_and_bounded():
    p = [0.001, 0.01, 0.03, 0.2, 0.9]
    adj = _holm(p)
    assert np.all(adj >= np.asarray(p)), "correction may only increase p-values"
    assert np.all(adj <= 1.0)
    assert np.all(np.diff(adj[np.argsort(p)]) >= -1e-12), "must be monotone in rank"


def test_holm_matches_hand_computed_values():
    """m=3: adjusted = max over the step-down path of (m - rank) * p."""
    adj = _holm([0.01, 0.04, 0.03])
    assert adj[0] == pytest.approx(0.03)   # 3 * 0.01
    assert adj[2] == pytest.approx(0.06)   # max(0.03, 2 * 0.03)
    assert adj[1] == pytest.approx(0.06)   # max(0.06, 1 * 0.04)
