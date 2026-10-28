"""Evaluator for 3x3 matrix multiplication schemes (Brent equations, fixed rank).

Rank-22 variant: no verified rank-22 scheme for general 3x3 matrix multiplication
is known to exist. Decades of dedicated search (flip-graph exploration, SAT) have
not found one; this is an exploratory probe, not an expected success. See
matmul_3x3/ for the rank-23 variant, where the best known scheme (Laderman,
1976) is still unbeaten.
"""

import argparse
import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from shinka.core import run_shinka_eval
import symmetry

N = 3
D = N * N
RANK = 22  # no known scheme exists at this rank; see module docstring
TIME_BUDGET = 40.0  # longer than the rank-23 variant: fewer degrees of freedom to fit
NUM_RUNS = 3
TOL = 1e-9
KNOWN_SCHEMES_DIR = os.path.join(os.path.dirname(__file__), "known_schemes")

# Empty by default: no verified rank-22 scheme exists to check against. Kept
# for structural parity with matmul_3x3/evaluate.py and in case a rank-22
# scheme is ever published (add it the same way: verified .npz, checked
# against the Brent equations by symmetry.load_known_schemes, never from memory).
_KNOWN_SIGNATURES = symmetry.load_known_schemes(
    sorted(glob.glob(os.path.join(KNOWN_SCHEMES_DIR, "*.npz")))
)


def matmul_tensor() -> np.ndarray:
    T = np.zeros((D, D, D), dtype=np.int64)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                T[N * i + j, N * j + k, N * k + i] = 1
    return T


def brent_residual(U, V, W) -> float:
    """Frobenius norm of T - sum_r u_r (x) v_r (x) w_r."""
    diff = matmul_tensor() - np.einsum("ar,br,cr->abc", U, V, W)
    return float(np.linalg.norm(diff))


def is_exact_discrete(U, V, W) -> bool:
    """Exact over Z with entries in {-1,0,1} (integer arithmetic, no tolerance)."""
    mats = [np.asarray(M) for M in (U, V, W)]
    if not all(np.all(np.abs(M - np.round(M)) == 0) for M in mats):
        return False
    Ui, Vi, Wi = (np.round(M).astype(np.int64) for M in mats)
    if max(np.abs(M).max() for M in (Ui, Vi, Wi)) > 1:
        return False
    return bool(np.array_equal(np.einsum("ar,br,cr->abc", Ui, Vi, Wi), matmul_tensor()))


def multiply_with_scheme(U, V, W, A, B) -> np.ndarray:
    """Compute A @ B using only the R products defined by the scheme."""
    m = (U.T @ A.reshape(-1)) * (V.T @ B.reshape(-1))
    c = W @ m  # index 3k+i holds C[i, k]
    return c.reshape(N, N).T


def validate(run_output) -> Tuple[bool, Optional[str]]:
    try:
        U, V, W = (np.asarray(M, dtype=float) for M in run_output)
    except Exception as e:  # noqa: BLE001
        return False, f"Expected (U, V, W) numeric arrays: {e}"
    for name, M in zip("UVW", (U, V, W)):
        if M.shape != (D, RANK):
            return False, f"{name} has shape {M.shape}, expected {(D, RANK)}"
        if not np.all(np.isfinite(M)):
            return False, f"{name} has non-finite entries"
    return True, None


def get_kwargs(run_idx: int) -> Dict[str, Any]:
    return {"random_seed": 1000 + run_idx, "rank": RANK, "time_budget": TIME_BUDGET}


def matches_known_scheme(U, V, W) -> Optional[str]:
    """Return the name of a known scheme this is equivalent to (up to term
    reordering, sign rescaling, and index/role symmetries), or None."""
    sig = symmetry.canonical_signature(np.round(U).astype(np.int64),
                                        np.round(V).astype(np.int64),
                                        np.round(W).astype(np.int64))
    for name, known_sig in _KNOWN_SIGNATURES.items():
        if sig == known_sig:
            return name
    return None


def score_run(U, V, W) -> Dict[str, Any]:
    """Score ladder: approximate (<=0.5) < exact {-1,0,1} after rounding (2+).

    See matmul_3x3/evaluate.py:score_run for the rationale. At rank 22 an exact
    hit would mean a scheme with no known precedent; if one is ever found here,
    treat it with extreme suspicion before celebrating -- re-verify independently
    (exact integer check, random A@B products) exactly as done for the rank-23
    discoveries, since a bug producing a false "exact" result is far more likely
    than a genuine rank-22 scheme.
    """
    Ur, Vr, Wr = (np.round(M) for M in (U, V, W))
    if is_exact_discrete(Ur, Vr, Wr):
        nnz = int(np.count_nonzero(Ur) + np.count_nonzero(Vr) + np.count_nonzero(Wr))
        known = matches_known_scheme(Ur, Vr, Wr)
        return {"score": 2.0 + (1.0 - nnz / (3 * D * RANK)), "residual": 0.0,
                "status": "exact_discrete", "nnz": nnz, "matches_known": known}
    rounded_res = brent_residual(Ur, Vr, Wr)
    return {"score": 0.5 / (1.0 + rounded_res),
            "residual": brent_residual(U, V, W), "rounded_residual": rounded_res,
            "status": "inexact", "nnz": None, "matches_known": None}


def aggregate(results: List[Any]) -> Dict[str, Any]:
    runs = [score_run(*(np.asarray(M, dtype=float) for M in r)) for r in results]
    n_exact = sum(r["status"] == "exact_discrete" for r in runs)
    n_novel = sum(r["status"] == "exact_discrete" and r["matches_known"] is None for r in runs)
    # Independent sanity check of exact schemes via random matrix products.
    rng = np.random.default_rng(0)
    for r, out in zip(runs, results):
        if r["status"] == "exact_discrete":
            U, V, W = (np.round(np.asarray(M)) for M in out)
            A, B = rng.integers(-5, 6, (N, N)), rng.integers(-5, 6, (N, N))
            assert np.array_equal(multiply_with_scheme(U, V, W, A, B), A @ B)
    text = "; ".join(
        f"run{i}: {r['status']} residual={r['residual']:.4f}"
        + (f" matches_known={r['matches_known']}" if r.get("matches_known") else "")
        for i, r in enumerate(runs)
    )
    return {
        "combined_score": float(np.mean([r["score"] for r in runs])),
        "public": {"rank": RANK, "num_exact_discrete": n_exact, "num_novel_exact": n_novel},
        "private": {"runs": runs},
        "extra_data": {},
        "text_feedback": f"Target rank {RANK} (no known scheme exists). {text}. "
                         f"Exact discrete solves: {n_exact}/{len(runs)} ({n_novel} unverified).",
    }


def main(program_path: str, results_dir: str):
    metrics, correct, err = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_experiment",
        num_runs=NUM_RUNS,
        get_experiment_kwargs=get_kwargs,
        validate_fn=validate,
        aggregate_metrics_fn=aggregate,
    )
    print("correct:", correct, "error:", err)
    print("combined_score:", metrics.get("combined_score"))
    print(metrics.get("text_feedback"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--program_path", default="initial.py")
    p.add_argument("--results_dir", default="results")
    a = p.parse_args()
    main(a.program_path, a.results_dir)
