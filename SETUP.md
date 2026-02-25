# Before publishing this repository

Two placeholders must be replaced with the real GitHub owner, and one artifact
must be minted. Nothing else is outstanding.

| Where | Placeholder | Replace with |
|---|---|---|
| `README.md` line 11 | `USERNAME` in the CI badge URL | your GitHub username |
| `CITATION.cff` `repository-code` | `USERNAME` | your GitHub username |

```bash
# after creating the GitHub repo:
sed -i '' "s/USERNAME/<your-github-username>/g" README.md CITATION.cff
```

## Archiving the raw results

`runs.csv`, `traces.npz` and `positions.npz` are gitignored by design — derived
tables and figures are committed so every claim in the README is checkable, but
the raw results belong in an archive with a persistent identifier rather than in
git history.

1. `make experiment` to regenerate everything from scratch
2. Zip `experiments/*/results/` and deposit on [Zenodo](https://zenodo.org)
3. Add the DOI to `CITATION.cff` and swap the DOI badge into `README.md`

The reproducibility survey this project is a response to found that persistent
archival is among the least-practised steps in the field. It costs an afternoon.

## Verifying a clean checkout

```bash
make setup && make test && make all && make verify
```

`make verify` asserts the invariants directly against the raw data: no run
exceeded its budget, every seed distinct and recorded, run counts matching the
manifest, no negative target errors.
