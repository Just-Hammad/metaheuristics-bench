"""Built-in problems: classic functions, and the random objective used for bias testing.

Two roles:
  1. A dependency-free suite so the test suite never needs a compiled backend.
  2. `RandomObjective`, which carries no information about x and is the basis of
     the structural-bias experiment (see mhbench.experiments / RESULTS.md).
"""
from __future__ import annotations

import numpy as np
from .base import TrackedProblem


# -- classic functions, all with optimum 0 at the origin before shifting -------
def _sphere(x):
    return float(np.sum(x**2))


def _rosenbrock(x):
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2))


def _rastrigin(x):
    return float(10 * x.size + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def _ackley(x):
    n = x.size
    return float(
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
        - np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
        + 20
        + np.e
    )


def _griewank(x):
    i = np.arange(1, x.size + 1)
    return float(np.sum(x**2) / 4000.0 - np.prod(np.cos(x / np.sqrt(i))) + 1)


def _schwefel12(x):
    return float(np.sum(np.cumsum(x) ** 2))


CLASSIC = {
    "sphere": (_sphere, 5.0),
    "rosenbrock": (_rosenbrock, 5.0),
    "rastrigin": (_rastrigin, 5.0),
    "ackley": (_ackley, 32.0),
    "griewank": (_griewank, 600.0),
    "schwefel12": (_schwefel12, 5.0),
}


def make_classic(
    name: str, dim: int, budget: int, shift: np.ndarray | None = None
) -> TrackedProblem:
    """A classic function, optionally with its optimum translated by `shift`.

    The shift is the mechanism for the centre-bias test: an algorithm that
    performs worse purely because the optimum moved away from the domain centre
    was exploiting the benchmark's geometry, not searching.
    """
    fn, half = CLASSIC[name]
    lb = np.full(dim, -half)
    ub = np.full(dim, half)
    if shift is None:
        shift = np.zeros(dim)
    shift = np.asarray(shift, dtype=float)

    def shifted(x):
        return fn(x - shift)

    label = name if not shift.any() else f"{name}_shifted"
    return TrackedProblem(shifted, dim, lb, ub, label, optimum=0.0, budget=budget)


class RandomObjective:
    """f(x) ~ U(0,1), independent of x.

    Carries zero information about the search space, so an unbiased optimizer
    should leave its sampled points uniformly distributed over the domain.
    Any concentration is structural bias in the algorithm itself.

    Deterministic given the seed, so runs are reproducible.
    """

    def __init__(self, seed: int):
        self._rng = np.random.default_rng(seed)

    def __call__(self, x):
        return float(self._rng.random())


def make_random_objective(dim: int, budget: int, seed: int) -> TrackedProblem:
    return TrackedProblem(
        RandomObjective(seed),
        dim,
        np.full(dim, -5.0),
        np.full(dim, 5.0),
        "random_objective",
        optimum=0.0,
        budget=budget,
    )
