"""Classical metaheuristics, implemented from their published definitions.

Variants are named precisely. There is no single 'PSO' or 'DE'; the variant and
its parameters are part of the result and appear in every table caption.
"""
from __future__ import annotations

import numpy as np
from .base import Algorithm, run_to_exhaustion


class RandomSearch(Algorithm):
    """Uniform random sampling. The floor every other algorithm must clear.

    Not a strawman -- a control. An algorithm that fails to beat this on a
    problem has told you something about the problem or about itself.
    """

    name = "RandomSearch"

    @run_to_exhaustion
    def __call__(self, problem, rng):
        while True:
            problem(problem.random_point(rng))


class DE(Algorithm):
    """DE/rand/1/bin (Storn & Price 1997)."""

    name = "DE/rand/1/bin"

    def __init__(self, F: float = 0.5, CR: float = 0.9, NP: int = 50):
        self.F, self.CR, self.NP = F, CR, NP
        self.params = {"F": F, "CR": CR, "NP": NP}

    @run_to_exhaustion
    def __call__(self, problem, rng):
        d, NP = problem.dim, self.NP
        pop = rng.uniform(problem.lb, problem.ub, size=(NP, d))
        fit = problem.evaluate_batch(pop)
        while True:
            for i in range(NP):
                r = rng.choice(np.delete(np.arange(NP), i), size=3, replace=False)
                a, b, c = pop[r]
                mutant = problem.clip(a + self.F * (b - c))
                cross = rng.random(d) < self.CR
                if not cross.any():
                    cross[rng.integers(d)] = True
                trial = np.where(cross, mutant, pop[i])
                ft = problem(trial)
                if ft <= fit[i]:
                    pop[i], fit[i] = trial, ft


class PSO(Algorithm):
    """PSO with Clerc-Kennedy constriction (Clerc & Kennedy 2002).

    Deliberately NOT labelled 'Standard PSO 2011' -- SPSO-2011 uses a
    rotation-invariant hypersphere update this does not implement. Naming it
    accurately matters: the two are different algorithms and are routinely
    conflated in published comparisons.
    """

    name = "PSO/constriction"

    def __init__(self, swarm: int = 40, c1: float = 2.05, c2: float = 2.05):
        phi = c1 + c2
        self.chi = 2.0 / abs(2 - phi - np.sqrt(phi**2 - 4 * phi))
        self.swarm, self.c1, self.c2 = swarm, c1, c2
        self.params = {"N": swarm, "c1": c1, "c2": c2, "chi": round(self.chi, 4)}

    @run_to_exhaustion
    def __call__(self, problem, rng):
        d, N = problem.dim, self.swarm
        span = problem.ub - problem.lb
        X = rng.uniform(problem.lb, problem.ub, size=(N, d))
        V = rng.uniform(-span, span, size=(N, d)) * 0.1
        F = problem.evaluate_batch(X)
        P, PF = X.copy(), F.copy()
        g = int(np.argmin(PF))
        while True:
            r1, r2 = rng.random((N, d)), rng.random((N, d))
            V = self.chi * (V + self.c1 * r1 * (P - X) + self.c2 * r2 * (P[g] - X))
            V = np.clip(V, -span, span)
            X = problem.clip(X + V)
            for i in range(N):
                fi = problem(X[i])
                if fi < PF[i]:
                    P[i], PF[i] = X[i].copy(), fi
                    if fi < PF[g]:
                        g = i


class GA(Algorithm):
    """Real-coded GA: tournament selection, SBX crossover, polynomial mutation."""

    name = "GA/SBX"

    def __init__(self, NP: int = 50, pc: float = 0.9, eta_c: float = 20.0, eta_m: float = 20.0):
        self.NP, self.pc, self.eta_c, self.eta_m = NP, pc, eta_c, eta_m
        self.params = {"NP": NP, "pc": pc, "eta_c": eta_c, "eta_m": eta_m}

    def _sbx(self, p1, p2, rng):
        u = rng.random(p1.size)
        beta = np.where(u <= 0.5, (2 * u) ** (1 / (self.eta_c + 1)),
                        (1 / (2 * (1 - u))) ** (1 / (self.eta_c + 1)))
        return 0.5 * ((1 + beta) * p1 + (1 - beta) * p2), \
               0.5 * ((1 - beta) * p1 + (1 + beta) * p2)

    def _poly_mut(self, x, problem, rng):
        d = x.size
        pm = 1.0 / d
        mask = rng.random(d) < pm
        if not mask.any():
            return x
        u = rng.random(d)
        delta = np.where(u < 0.5, (2 * u) ** (1 / (self.eta_m + 1)) - 1,
                         1 - (2 * (1 - u)) ** (1 / (self.eta_m + 1)))
        return np.where(mask, x + delta * (problem.ub - problem.lb), x)

    @run_to_exhaustion
    def __call__(self, problem, rng):
        NP = self.NP
        pop = rng.uniform(problem.lb, problem.ub, size=(NP, problem.dim))
        fit = problem.evaluate_batch(pop)
        while True:
            # binary tournament
            idx = rng.integers(0, NP, size=(NP, 2))
            winners = np.where(fit[idx[:, 0]] <= fit[idx[:, 1]], idx[:, 0], idx[:, 1])
            parents = pop[winners]
            children = []
            for i in range(0, NP - 1, 2):
                p1, p2 = parents[i], parents[i + 1]
                c1, c2 = (self._sbx(p1, p2, rng) if rng.random() < self.pc else (p1.copy(), p2.copy()))
                children += [c1, c2]
            children = [problem.clip(self._poly_mut(c, problem, rng)) for c in children]
            cfit = np.array([problem(c) for c in children])
            # elitist (mu + lambda) replacement
            allp = np.vstack([pop, np.array(children)])
            allf = np.concatenate([fit, cfit])
            keep = np.argsort(allf)[:NP]
            pop, fit = allp[keep], allf[keep]
