# Methodology

Why each choice was made, and what it rules out. The short version lives in the README; this is the
version that answers a reviewer.

---

## 1. Why benchmark at all, given how much benchmarking already exists

A 2026 survey of reproducibility in evolutionary computation measured what published EC papers
actually ship alongside the manuscript:

| Practice | Share of surveyed papers |
|---|---|
| Any artifact beyond the manuscript | 36.9% |
| Both code **and** data | 10.7% |
| Automated parameter tuning | < 14% |
| Multiple-comparison correction | 13.7% |

The scarce contribution in this field is therefore not another optimizer. It is a comparison that
can be checked. This repository is built to clear that bar rather than to add an algorithm — see
§6 for what it deliberately does not do.

---

## 2. The experimental contract

| Element | Choice | What it rules out |
|---|---|---|
| **Budget** | Fixed function evaluations (2,000 × D), never wall-clock or generations | Wall-clock compares hardware and implementation language; generations compare population sizes. Neither compares algorithms |
| **Repetitions** | 30 (BBOB); 51 (CEC 2017, that suite's own protocol) | Below ~25 runs the non-parametric tests have no power to detect anything but enormous differences |
| **Seeds** | Spawned from one master seed via `numpy.random.SeedSequence`, recorded per run | `master_seed + i` produces correlated streams. Recording only the master seed makes a single run unreplayable |
| **Metric** | Target error `f(x_best) − f(x*)`, with `f(x*)` read from the suite | Assuming the optimum is 0 is wrong for BBOB (instance-dependent) and for CEC (shifted to 100·i) |
| **Aggregation** | Median and IQR | Mean ± std asserts a normality that bounded, heavily skewed error distributions do not have |
| **Bound handling** | Clipping, applied identically to all algorithms | An undocumented repair rule silently becomes part of the algorithm. Clipping is not neutral — it concentrates mass on the boundary for algorithms that overshoot — so it is declared rather than hidden |
| **Tuning** | None; published defaults throughout | Tuning one side and not the other is the most common way a published comparison misleads. Doing neither is the honest option at this scale; doing both is the better one, and is future work |

---

## 3. The statistical pipeline

```
per-run raw results
        ├── aggregate per (algorithm, problem) → median
        ▼
  [1] Friedman test, problems as blocks → average ranks
        │     not significant → STOP
        ▼
  [2] pairwise Wilcoxon signed-rank + Holm step-down
        ▼
  [3] Cliff's δ — per problem, on RAW RUNS
```

This follows the standard recommendation for comparing multiple classifiers or optimizers over
multiple problems (Demšar 2006; García & Herrera 2008): a non-parametric omnibus test first,
post-hoc pairwise comparisons only if the omnibus rejects, and p-values corrected for multiplicity.

### The distinction the API enforces

Steps 1 and 2 consume the **algorithms × problems matrix of aggregates**. Step 3 does not.

Cliff's δ measures stochastic dominance between two *samples*. Computed over one median per
problem, it merely restates the Friedman ranking. Computed over the raw runs *within* a problem, it
answers the question an effect size exists to answer: when A and B both run here, how often does A
win, and by how much does that exceed chance?

`mhbench.stats` gives these separate function names with separate input types
(`posthoc_holm(DataFrame)` vs `cliffs_delta(array, array)`), so the two cannot be silently
interchanged. `tests/test_stats.py` pins the sign convention — because these are **errors**, a
*negative* δ means the first algorithm is better, which is the opposite of the usual reading and an
easy place to publish a sign error.

### Two different Wilcoxon tests

| Question | Test | Data |
|---|---|---|
| Is A better than B **across the suite**? | Wilcoxon **signed-rank** (paired; problems are the pairs) | one aggregate per problem |
| Is A better than B **on this problem**? | Mann–Whitney / **rank-sum** (unpaired) | the raw runs |

Both appear in the literature and are routinely swapped. Every table states which was used.

---

## 4. The structural-bias experiment

### The question

Benchmark suites overwhelmingly place optima at or near the centre of the search domain. An
algorithm with an intrinsic pull toward the centre — through its initialisation, its recombination
operator, or convergence toward a population centroid — will score well on such suites for a reason
that has nothing to do with search quality.

Performance tables cannot detect this, because the artifact and genuine competence produce the same
numbers.

### The design

Optimise `f(x) ~ U(0,1)`, drawn independently of `x`. The objective carries **zero information**
about the search space, so no location can be justified by the problem. Over many independent runs,
an unbiased optimizer's final best position must be uniformly distributed over the domain.

Any systematic concentration is a property of the **algorithm**, not the problem.

This is the structural-bias methodology of Kononova, Caraffini, Bäck and colleagues.

### The test

For each algorithm, 200 independent runs on a fresh random objective, in 10 dimensions, at a
10,000-evaluation budget. The final best position is recorded for every run.

- **Per-dimension Kolmogorov–Smirnov** test of each coordinate against the uniform distribution,
  with **Holm correction across the 10 dimensions**.
- A **pooled KS** test over all coordinates. Reported as descriptive only: coordinates within one
  run are not independent, so the per-dimension tests carry the inference.
- **Median distance to the domain centre**, against the same statistic for uniformly sampled points,
  as an interpretable effect size (`centre_ratio` < 1 means the algorithm finishes closer to the
  centre than chance).

### What it cannot show

It detects bias **on a flat landscape**. An algorithm unbiased here may still behave badly on
structured landscapes, and an algorithm biased here is not thereby useless — the bias only matters
to the extent that benchmark optima sit where the bias points. What the test establishes is that a
performance ranking on a centre-optimum suite is **partly** measuring something other than search.

---

## 4b. Giving a fixed budget to an algorithm that stops on its own

CMA-ES terminates by its own criteria. A fixed budget far exceeding what it needs therefore raises a
question the bare algorithm does not answer, and the choice made is part of the experiment.

| Option | Consequence |
|---|---|
| Run one instance until the budget is spent | **Numerically invalid.** Past its stopping criteria the step size diverges and the covariance update produces NaN. An earlier version of this code did this and crashed a 40-minute experiment inside `cma`'s sampler |
| Stop when CMA-ES stops, leave the budget unspent | Not budget-matched; the comparison against algorithms that used the full budget is meaningless |
| **IPOP restarts** (Auger & Hansen 2005) — on termination, restart from a fresh point with the population doubled | **Chosen.** Spends the full budget the way the algorithm's own literature prescribes, and is what published BBOB comparisons of CMA-ES run |

`max_restarts` is capped, and any budget still unspent afterwards is consumed by uniform sampling so
that the evaluation count remains exactly matched across algorithms.

The general point generalises beyond CMA-ES: **a termination policy is part of an algorithm's
definition under a fixed budget**, exactly as a bound-handling rule is. A comparison that leaves
either unstated is not reconstructible.

---

## 5. Reproducibility apparatus

- `make all` regenerates every table and figure from raw results on a clean checkout
- `make experiment` re-runs the experiments from scratch
- `requirements.lock` pins exact versions; `pyproject.toml` declares ranges for installability
- `scripts/verify_claims.py` asserts the invariants directly against `runs.csv`: no run exceeded its
  budget, all seeds distinct and recorded, run counts match the manifest, no negative target errors
- The README's results block is **generated** by `make report`, never typed, so a stale claim is
  structurally impossible
- Each experiment writes a `manifest.json` recording config, platform, package versions, timestamp
  and the failed-run count
- Runs are checkpointed to `runs.partial.csv` as they complete, and a run that raises is **recorded
  as a failure and excluded from the statistics** rather than aborting the experiment. A failed run
  is a fact about the algorithm; it belongs in the results, not in a traceback

---

## 6. What this repository deliberately does not do

**It does not introduce a new metaheuristic.** The metaphor-driven literature is under sustained
and well-founded criticism, and a large fraction of recently proposed nature-inspired algorithms
have been shown to be rediscoveries of existing methods under new names. Adding to that pile would
undercut the argument the rest of this work is making.

**It does not claim a winner in general.** Results are reported per suite, per dimension and per
problem, with effect sizes. "Algorithm X is best" is not a claim this design can support, and no
comparison at this scale can support it.

**It does not tune.** See §2. This is a limitation, stated as one.

---

## References

- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *JMLR* 7.
- García, S. & Herrera, F. (2008). An extension on statistical comparisons of classifiers over
  multiple data sets for all pairwise comparisons. *JMLR* 9.
- Romano, J. et al. (2006). Exploring methods for evaluating group differences (Cliff's δ
  magnitude cut-points).
- Storn, R. & Price, K. (1997). Differential evolution. *Journal of Global Optimization* 11.
- Clerc, M. & Kennedy, J. (2002). The particle swarm — explosion, stability, and convergence.
  *IEEE TEC* 6(1).
- Hansen, N. & Ostermeier, A. (2001). Completely derandomized self-adaptation in evolution
  strategies. *Evolutionary Computation* 9(2).
- Hansen, N. et al. (2021). COCO: a platform for comparing continuous optimizers in a black-box
  setting. *Optimization Methods and Software* 36(1).
- Kononova, A. V., Caraffini, F., Bäck, T. et al. Structural bias in population-based algorithms.
- de Nobel, J., Ye, F., Vermetten, D., Wang, H., Doerr, C. & Bäck, T. IOHexperimenter: benchmarking
  platform for iterative optimization heuristics.
- Awad, N. H., Ali, M. Z., Liang, J. J., Qu, B. Y. & Suganthan, P. N. (2016). Problem definitions
  and evaluation criteria for the CEC 2017 competition on single-objective bound-constrained
  real-parameter numerical optimization.
