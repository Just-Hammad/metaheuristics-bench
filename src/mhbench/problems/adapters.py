"""Backend adapters. Every suite arrives as a TrackedProblem so the runner,
the budget accounting and the reporting never branch on backend.

Backends
--------
ioh    : BBOB (24 fns) and CEC 2022. Has its own harness; preferred.
opfunu : CEC 2017 (and 2005-2022). Pure Python, functions only - NOT in ioh.
         See 12-build-guides.md section 0: this is why both exist.
"""
from __future__ import annotations

import numpy as np
from .base import TrackedProblem


def ioh_problem(fid: int, instance: int, dim: int, budget: int, suite: str = "BBOB") -> TrackedProblem:
    """One ioh problem, wrapped.

    `optimum` comes from ioh itself (`f.optimum.y`), so target error is exact
    rather than assumed to be zero.
    """
    import ioh

    cls = {"BBOB": ioh.ProblemClass.BBOB, "CEC2022": ioh.ProblemClass.CEC2022}[suite]
    f = ioh.get_problem(fid, instance, dim, cls)
    return TrackedProblem(
        fn=lambda x: f(list(x)),
        dim=dim,
        lb=np.asarray(f.bounds.lb, dtype=float),
        ub=np.asarray(f.bounds.ub, dtype=float),
        name=f"{suite}_f{fid}_i{instance}",
        optimum=float(f.optimum.y),
        budget=budget,
    )


def ioh_suite(dim: int, budget: int, instance: int = 1, suite: str = "BBOB", fids=None):
    import ioh

    cls = {"BBOB": ioh.ProblemClass.BBOB, "CEC2022": ioh.ProblemClass.CEC2022}[suite]
    if fids is None:
        fids = sorted(cls.problems.keys())
    return [ioh_problem(fid, instance, dim, budget, suite) for fid in fids]


def opfunu_problem(fid: int, dim: int, budget: int, year: str = "2017") -> TrackedProblem:
    """One CEC function via opfunu. CEC 2017 is not available through ioh."""
    import opfunu

    cls_name = f"F{fid}{year}"
    matches = opfunu.get_functions_by_classname(cls_name)
    if not matches:
        raise ValueError(f"opfunu has no {cls_name}")
    f = matches[0](ndim=dim)
    lb = np.asarray(f.lb, dtype=float)
    ub = np.asarray(f.ub, dtype=float)
    # CEC functions are conventionally shifted so f(x*) = 100 * fid
    optimum = float(getattr(f, "f_global", 0.0))
    return TrackedProblem(
        fn=f.evaluate,
        dim=dim,
        lb=lb,
        ub=ub,
        name=f"CEC{year}_f{fid}",
        optimum=optimum,
        budget=budget,
    )


def opfunu_suite(dim: int, budget: int, year: str = "2017", fids=None, exclude=(2,)):
    """CEC 2017 suite.

    F2 is excluded by default and this is DELIBERATE and DECLARED: it is widely
    dropped for numerical instability in high dimensions. 12-build-guides.md
    section 1 - silently dropping a function is the practice this repo criticises,
    so the exclusion is a named argument and it is reported in the results.
    """
    if fids is None:
        fids = [i for i in range(1, 31) if i not in set(exclude)]
    return [opfunu_problem(fid, dim, budget, year) for fid in fids]
