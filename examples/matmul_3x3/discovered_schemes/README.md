# Discovered schemes

Exact rank-23 schemes found by ShinkaEvolve runs. These are **not** verified against the
literature, so they are kept separate from `known_schemes/`, which the evaluator uses for
equivalence checks. Novelty is unproven.

| File | Rank | Source | Notes |
|---|---|---|---|
| `gen6_seed1001_rank23.npz` | 23 | Generation 6 of run `results_matmul_3x3_rounded`; program reproduced by calling `run_experiment(random_seed=1001, rank=23, time_budget=20.0)` | Entries in {-1,0,1}, 148 non-zeros. Matches `A @ B` on random integer pairs. Rank profile differs from Laderman and Oh-Hopcroft-Kim. Not matched by `symmetry.canonical_signature` against known schemes. |

Each file holds integer arrays `U`, `V`, `W` in the same convention as `known_schemes/`.
Re-check exactness with `evaluate.is_exact_discrete` before relying on a file.
