# Discovered schemes

Exact rank-23 schemes found by ShinkaEvolve runs. These are **not** verified against the
literature, so they are kept separate from `known_schemes/`, which the evaluator uses for
equivalence checks. Novelty is unproven.

| File | Rank | Source | Notes |
|---|---|---|---|
| `gen6_seed1001_rank23.npz` | 23 | Generation 6 of run `results_matmul_3x3_rounded`; program reproduced by calling `run_experiment(random_seed=1001, rank=23, time_budget=20.0)` | Entries in {-1,0,1}, 148 non-zeros. Matches `A @ B` on random integer pairs. Rank profile differs from Laderman and Oh-Hopcroft-Kim. Not matched by `symmetry.canonical_signature` against known schemes. |
| `run3_seed1001_rank23.npz` | 23 | `results_matmul_3x3_opus55_run3`'s best program (claude-opus-5-5, flip-graph engine seeded from a random Strassen sub-block); reproduced by calling `run_experiment(random_seed=1001, rank=23, time_budget=20.0)` | Entries in {-1,0,1}, 168 non-zeros. `symmetry.rank_profile` has only 89 matches out of the full 17,376-entry published catalogue (`algebra.uni-linz.ac.at` schemes-exp archive) -- all other 17,287 entries are provably inequivalent under the full change-of-basis group. Of those 89: `symmetry.canonical_signature` (permutation/sign/role/transpose group) finds no match; an anchor-based GF(2) change-of-basis witness search (`GL(3,2)^3`, all 6 trace roles) finds no match against any of the 89; an anchor-based rational (`GL(3,ℚ)`) witness search (exact `fractions.Fraction` linear algebra) also finds no match against any of the 89. Both witness-search methods were validated by self-test (Laderman vs. itself) before use. Neither search is exhaustive brute force, so this is strong evidence of novelty, not a formal proof. |

Each file holds integer arrays `U`, `V`, `W` in the same convention as `known_schemes/`.
Re-check exactness with `evaluate.is_exact_discrete` before relying on a file.
