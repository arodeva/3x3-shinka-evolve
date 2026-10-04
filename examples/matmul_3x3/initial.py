"""Seed program: search for a rank-R scheme for 3x3 matrix multiplication."""

import time

import numpy as np

N = 3
D = N * N  # flattened matrix dimension


def matmul_tensor() -> np.ndarray:
    """T[3i+j, 3j+k, 3k+i] = 1: <A,B,C^T> in trace form (cyclically symmetric)."""
    T = np.zeros((D, D, D))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                T[N * i + j, N * j + k, N * k + i] = 1.0
    return T


# EVOLVE-BLOCK-START
def find_scheme(rank: int, seed: int, time_budget: float):
    """
    Search for factor matrices U, V, W (each shape (9, rank)) with
        sum_r U[a, r] * V[b, r] * W[c, r] == T[a, b, c]   (Brent equations)
    where T = matmul_tensor(). Exact integer solutions with entries in
    {-1, 0, 1} are best. Return (U, V, W); the best attempt is scored even
    if it is not exact. Must respect time_budget (seconds).
    """
    rng = np.random.default_rng(seed)
    T = matmul_tensor()
    eye = np.eye(rank)
    deadline = time.time() + time_budget
    best, best_err = None, np.inf

    while time.time() < deadline:
        U, V, W = (rng.normal(scale=0.5, size=(D, rank)) for _ in range(3))
        lam = 1e-2
        for _ in range(400):
            if time.time() > deadline:
                break
            U = np.einsum("abc,br,cr->ar", T, V, W) @ np.linalg.inv(
                (V.T @ V) * (W.T @ W) + lam * eye
            )
            V = np.einsum("abc,ar,cr->br", T, U, W) @ np.linalg.inv(
                (U.T @ U) * (W.T @ W) + lam * eye
            )
            W = np.einsum("abc,ar,br->cr", T, U, V) @ np.linalg.inv(
                (U.T @ U) * (V.T @ V) + lam * eye
            )
            lam = max(lam * 0.98, 1e-9)

        err = np.linalg.norm(T - np.einsum("ar,br,cr->abc", U, V, W))
        # Balance column scales, then try rounding to integers.
        for r in range(rank):
            s = [np.abs(M[:, r]).max() + 1e-12 for M in (U, V, W)]
            g = np.prod(s) ** (1 / 3)
            U[:, r] *= g / s[0]
            V[:, r] *= g / s[1]
            W[:, r] *= g / s[2]
        Ur, Vr, Wr = np.round(U), np.round(V), np.round(W)
        err_r = np.linalg.norm(T - np.einsum("ar,br,cr->abc", Ur, Vr, Wr))
        if err_r < best_err:
            best, best_err = (Ur, Vr, Wr), err_r
        if err < best_err:
            best, best_err = (U, V, W), err
        if best_err < 1e-9:
            break
    return best


# EVOLVE-BLOCK-END


def run_experiment(random_seed: int = 0, rank: int = 23, time_budget: float = 15.0, **kwargs):
    """Entry point called by evaluate.py. Fixed; not evolved."""
    return find_scheme(rank, random_seed, time_budget)
