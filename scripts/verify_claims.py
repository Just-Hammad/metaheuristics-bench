#!/usr/bin/env python
"""Assert the repo's own reproducibility invariants against the raw results.

This is the check that the README's claims are not drifting from the data.
Run it in CI, and before citing any number.

Invariants
----------
1. No run exceeded its budget.
2. Every run used a distinct, recorded seed.
3. Run counts are exactly as configured (no silently dropped runs).
4. Target errors are non-negative -- a negative error means the optimum used
   for the suite is wrong, which invalidates everything downstream.
5. Derived tables exist for every experiment with raw results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FAIL = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAIL.append(msg)


def verify_experiment(d: Path):
    runs = d / "runs.csv"
    if not runs.exists():
        return
    if (d / "bias_summary.csv").exists():
        return          # the bias study has its own artifacts; checked below
    print(f"\n{d.parent.name}")
    df = pd.read_csv(runs)
    man = json.loads((d / "manifest.json").read_text())

    check((df["evaluations"] <= df["budget"]).all(),
          f"no run exceeded budget (max {df['evaluations'].max()} / {df['budget'].iloc[0]})")
    check(df["seed"].nunique() == len(df),
          f"all {len(df)} seeds distinct")
    check(len(df) == man["total_runs"],
          f"run count {len(df)} == configured {man['total_runs']}")
    check((df["error"] >= -1e-9).all(),
          f"all target errors non-negative (min {df['error'].min():.3e})")
    expected = man["n_problems"] * man["n_algorithms"] * man["n_runs"]
    check(len(df) == expected,
          f"problems x algorithms x runs = {expected} rows present")

    for t in ["table_final_error.csv", "table_ranks.csv", "summary.json"]:
        check((d / t).exists(), f"derived artifact present: {t}")


def main():
    print("mhbench :: reproducibility invariants")
    for d in sorted((ROOT / "experiments").glob("*/results")):
        verify_experiment(d)

    bias = ROOT / "experiments/exp02_structural_bias/results/bias_summary.csv"
    if bias.exists():
        print("\nexp02_structural_bias")
        b = pd.read_csv(bias)
        check(len(b) >= 3, f"{len(b)} algorithms tested for structural bias")
        check("verdict" in b.columns, "graded bias verdict present")
        check(b["verdict"].isin(["biased", "suggestive", "no evidence"]).all(),
              "every verdict is one of the three declared tiers")
        for tier in ["biased", "suggestive", "no evidence"]:
            names = [a.split()[0] for a in b[b["verdict"] == tier]["algorithm"]]
            print(f"  INFO  {tier:12s}: {', '.join(names) or '-'}")

    print("\n" + ("FAILED: " + str(len(FAIL)) if FAIL else "All invariants hold."))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
