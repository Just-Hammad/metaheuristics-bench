"""Problem interface and the evaluation-counting wrapper.

Every backend (built-in, ioh/BBOB, opfunu/CEC) is wrapped in `TrackedProblem`.
Budget accounting therefore lives in ONE place and cannot drift between backends.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


class BudgetExhausted(Exception):
    """Raised when an algorithm requests an evaluation past its budget."""


@dataclass
class Trace:
    """Compact best-so-far record.

    Only improvements are stored, which is lossless for a monotone
    best-so-far curve and avoids holding one float per evaluation.
    """

    evals: list[int] = field(default_factory=list)
    best: list[float] = field(default_factory=list)

    def record(self, n_evals: int, value: float) -> None:
        self.evals.append(n_evals)
        self.best.append(value)

    def resample(self, grid: np.ndarray) -> np.ndarray:
        """Best-so-far at each budget in `grid` (step function, right-continuous)."""
        if not self.evals:
            return np.full(len(grid), np.inf)
        e = np.asarray(self.evals)
        b = np.asarray(self.best)
        idx = np.searchsorted(e, grid, side="right") - 1
        out = np.where(idx < 0, np.inf, b[np.clip(idx, 0, None)])
        return out


class TrackedProblem:
    """Wraps a raw objective and owns the evaluation budget.

    The algorithm never counts evaluations; this does. That removes the single
    most common source of unfair comparison (an algorithm that quietly spends
    extra evaluations inside its own machinery).
    """

    def __init__(
        self,
        fn,
        dim: int,
        lb: np.ndarray,
        ub: np.ndarray,
        name: str,
        optimum: float = 0.0,
        budget: int | None = None,
    ):
        self._fn = fn
        self.dim = int(dim)
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.name = name
        self.optimum = float(optimum)
        self.budget = budget
        self.evaluations = 0
        self.best_value = np.inf
        self.best_x: np.ndarray | None = None
        self.trace = Trace()

    # -- core ---------------------------------------------------------------
    def __call__(self, x) -> float:
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or x.size != self.dim:
            raise ValueError(f"{self.name}: expected shape ({self.dim},), got {x.shape}")
        if self.budget is not None and self.evaluations >= self.budget:
            raise BudgetExhausted(
                f"{self.name}: budget {self.budget} exhausted at eval {self.evaluations}"
            )
        y = float(self._fn(x))
        self.evaluations += 1
        if y < self.best_value:
            self.best_value = y
            self.best_x = x.copy()
            self.trace.record(self.evaluations, y)
        return y

    def evaluate_batch(self, X) -> np.ndarray:
        """Convenience for population-based algorithms. Counts one eval per row."""
        return np.array([self(row) for row in np.atleast_2d(X)])

    # -- state --------------------------------------------------------------
    @property
    def remaining(self) -> int:
        return np.inf if self.budget is None else max(0, self.budget - self.evaluations)

    @property
    def target_error(self) -> float:
        """Distance to the known optimum. The reported metric."""
        return self.best_value - self.optimum

    def reset(self) -> None:
        self.evaluations = 0
        self.best_value = np.inf
        self.best_x = None
        self.trace = Trace()

    def random_point(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.lb, self.ub)

    def clip(self, x) -> np.ndarray:
        return np.clip(x, self.lb, self.ub)

    def __repr__(self) -> str:
        return f"TrackedProblem({self.name}, d={self.dim}, budget={self.budget})"
