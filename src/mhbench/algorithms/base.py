"""Algorithm contract.

Every algorithm is a callable `(problem, rng) -> None`. It optimises by calling
`problem(x)` and stops when `BudgetExhausted` is raised. It never counts its own
evaluations and never reads the budget to decide what to do -- that keeps the
comparison honest (see 12-build-guides.md section 1).

Bound handling is CLIPPING everywhere, applied uniformly. This is a design
choice, not a neutral default: it biases search toward the boundary for
algorithms that overshoot. It is documented here and stated in the README
because an undocumented repair rule silently becomes part of the algorithm.
"""
from __future__ import annotations

import numpy as np
from ..problems.base import BudgetExhausted


class Algorithm:
    name = "base"
    params: dict = {}

    def __call__(self, problem, rng: np.random.Generator) -> None:
        raise NotImplementedError

    def label(self) -> str:
        """Precise identification for table captions. 'DE' is not an algorithm;
        'DE/rand/1/bin F=0.5 CR=0.9 NP=50' is."""
        if not self.params:
            return self.name
        inner = " ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name} {inner}"


def run_to_exhaustion(fn):
    """Decorator: swallow BudgetExhausted, which is the normal termination path."""

    def wrapper(self, problem, rng):
        try:
            fn(self, problem, rng)
        except BudgetExhausted:
            pass

    return wrapper
