"""CMA-ES with IPOP restarts, via Hansen's `cma` package.

NOT reimplemented: `cma` is the reference implementation by the algorithm's
author. A hand-rolled CMA-ES would be strictly worse and would make any negative
finding about it uninterpretable.

WHY RESTARTS ARE REQUIRED, NOT OPTIONAL
---------------------------------------
CMA-ES has its own termination criteria and is designed to stop when it has
converged or stagnated. A fixed evaluation budget far larger than it needs
(here 10,000 x D) therefore poses a question the bare algorithm does not answer:
what should it do with the remaining budget?

Driving a single instance past its own stopping criteria is not a neutral
choice -- it drives the step size to overflow and the covariance update to NaN.
An earlier version of this file did exactly that and crashed after 40 minutes
of compute with `assert np.isfinite(vectors[0][0])` inside cma's sampler.

The standard answer is IPOP-CMA-ES (Auger & Hansen 2005): on termination,
restart from a fresh point with the population size doubled. That spends a
large budget the way the algorithm's own literature says to spend it, and it is
what BBOB comparisons of CMA-ES actually run.
"""
from __future__ import annotations

import warnings

import numpy as np

from .base import Algorithm, run_to_exhaustion
from ..problems.base import BudgetExhausted


class CMAES(Algorithm):
    name = "CMA-ES/IPOP"

    def __init__(self, sigma0_frac: float = 0.3, popsize_factor: int = 2,
                 max_restarts: int = 20):
        self.sigma0_frac = sigma0_frac
        self.popsize_factor = popsize_factor
        self.max_restarts = max_restarts
        self.params = {"sigma0": f"{sigma0_frac}*span",
                       "ipop": f"x{popsize_factor}"}

    @run_to_exhaustion
    def __call__(self, problem, rng):
        import cma

        sigma0 = float(np.mean(problem.ub - problem.lb) * self.sigma0_frac)
        popsize = None

        for restart in range(self.max_restarts):
            x0 = problem.random_point(rng)
            opts = {
                "bounds": [list(problem.lb), list(problem.ub)],
                "seed": int(rng.integers(1, 2**31 - 1)),
                "verbose": -9, "verb_log": 0, "verb_disp": 0,
            }
            if popsize is not None:
                opts["popsize"] = popsize

            with warnings.catch_warnings():
                # cma warns loudly on ill-conditioning; the restart is the response.
                warnings.simplefilter("ignore")
                es = cma.CMAEvolutionStrategy(list(x0), sigma0, opts)

                while not es.stop():
                    X = es.ask()
                    ys = [problem(np.asarray(x)) for x in X]   # may raise BudgetExhausted
                    try:
                        es.tell(X, ys)
                    except (AssertionError, FloatingPointError, ValueError):
                        # Numerical breakdown: abandon this instance and restart
                        # rather than propagating. Recorded via the restart count.
                        break

            popsize = (popsize or es.popsize) * self.popsize_factor

        # Budget left after max_restarts: spend it uniformly rather than idling,
        # so the comparison remains budget-matched.
        while True:
            problem(problem.random_point(rng))
