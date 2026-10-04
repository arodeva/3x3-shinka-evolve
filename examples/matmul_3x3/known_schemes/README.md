# Known schemes

Each `.npz` holds integer arrays `U`, `V`, `W` (shape `(9, rank)`, entries in
`{-1, 0, 1}`) for a scheme that is claimed to satisfy the Brent equations for
3x3 matrix multiplication exactly, in the convention used by `evaluate.py`
and `symmetry.py`: `U[3i+j, r]` is the coefficient of `A[i,j]`,
`V[3j+k, r]` is the coefficient of `B[j,k]`, and `W[3j+i, r]` is the
coefficient contributed by term `r` to `C[i,j]`.

**Nothing here is trusted by claim alone.** `evaluate.py` loads this
directory through `symmetry.load_known_schemes`, which re-derives the
Brent-equation check from scratch and raises if a file doesn't satisfy it
exactly -- a bad or mistyped file fails loudly at evaluator start-up rather
than silently corrupting the memorization check.

| File | Rank | Source |
|---|---|---|
| `naive_rank27.npz` | 27 | Schoolbook scheme, generated directly in this repo (see `tests/test_matmul_3x3_example.py::naive_scheme`). Rank 27 != 23, so it can never match a rank-23 candidate; it's here mainly as infrastructure/test fixture. |
| `laderman_1976.npz` | 23 | J. D. Laderman, "A noncommutative algorithm for multiplying 3x3 matrices using 23 multiplications", *Bull. Amer. Math. Soc.* 82 (1976). Transcribed programmatically from the Maple-verifiable listing in Oh, Hopcroft & Kim, "A New General-Purpose Method to Multiply 3x3 Matrices Using Only 23 Multiplications" ([arXiv:1108.2830](https://arxiv.org/abs/1108.2830), §2.4), by regex-parsing the paper's raw text -- not retyped by hand or recalled from memory. |
| `oh_hopcroft_kim_2011.npz` | 23 | The paper's own alternative rank-23 scheme (same source, §2.5). Independently verified; presented in that paper as inequivalent to Laderman's scheme. |

Both rank-23 files were, before being committed:
1. Checked to satisfy `sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c]` exactly (integer arithmetic, no tolerance) against this repo's own `matmul_tensor()`.
2. Checked that every entry is in `{-1, 0, 1}`.
3. Spot-checked against real matrix multiplication: `multiply_with_scheme(U, V, W, A, B) == A @ B` for 20 random integer `A, B`.
4. Checked, via `symmetry.canonical_signature`, that the two rank-23 schemes are *not* equivalent to each other under this repo's (intentionally limited -- see `symmetry.py`) discrete symmetry group, consistent with the source paper's claim that they are different schemes.

See `tests/test_matmul_3x3_example.py` for the corresponding regression tests.

## Adding another known scheme

1. Source it from a paper or a reproducible derivation -- never from an LLM's or your own unaided recollection of the coefficients.
2. Build `U, V, W` in the convention above and confirm `symmetry.matmul_tensor()` reconstruction matches exactly.
3. Save as `name.npz` with `np.savez(path, U=U, V=V, W=W)`.
4. Run the evaluator once; `load_known_schemes` will raise if anything is wrong.
