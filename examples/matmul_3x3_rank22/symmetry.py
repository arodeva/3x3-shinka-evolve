"""Equivalence checking for rank-R schemes of the 3x3 matmul tensor.

Two schemes (U,V,W) and (U',V',W') compute the "same" algorithm if one can be
obtained from the other by symmetries that trivially preserve the Brent
equations:

  1. Permuting the R rank-1 terms (order of summation doesn't matter).
  2. Rescaling a term's three factor-columns (a, b, c) with a*b*c = 1.
  3. Relabeling the underlying 3x3 matrix indices, and/or permuting which of
     U, V, W plays which role in the trilinear form trace(ABC) (valid because
     trace(ABC) = trace(BCA) = trace(CAB) = trace(C^T B^T A^T) = ...).

(3) is NOT hand-derived here: every candidate index-relabeling / role-swap /
transpose combination is verified computationally against the tensor itself
(`_discover_symmetries`), and only the ones that provably leave the tensor
invariant are kept. This avoids silently using a wrong symmetry group.

Limitation: this covers the "discrete" (signed monomial) symmetries above.
It does NOT check general GL(9) sandwiching equivalence, which is a much
larger continuous group. So this catches relabeled/reordered/rescaled copies
of a known scheme, but a scheme related by a non-monomial change of basis
would not be flagged. That is an intentional scope limit, not an oversight.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Tuple

import numpy as np

N = 3
D = N * N


def matmul_tensor() -> np.ndarray:
    T = np.zeros((D, D, D), dtype=np.int64)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                T[N * i + j, N * j + k, N * k + i] = 1
    return T


def _pair_perm_matrix(perm: Tuple[int, ...], transpose: bool) -> np.ndarray:
    """9x9 permutation matrix relabeling flattened-pair index 3x+y -> 3x'+y'
    under index relabeling `perm` (and swapping pair order if `transpose`)."""
    P = np.zeros((D, D), dtype=np.int64)
    for x in range(N):
        for y in range(N):
            a = N * x + y
            px, py = perm[x], perm[y]
            b = N * py + px if transpose else N * px + py
            P[b, a] = 1
    return P


# Candidate role assignments: which physical matrix (0=U's pair (i,j),
# 1=V's pair (j,k), 2=W's pair (k,i)) ends up in the U/V/W slot after the
# symmetry. All 6 permutations of the 3 roles are candidates; only those
# that preserve T (composed with an index relabeling/transpose) survive.
_ROLE_PERMS = list(itertools.permutations(range(3)))
_INDEX_PERMS = list(itertools.permutations(range(N)))


def _apply_symmetry(T: np.ndarray, role: Tuple[int, int, int], perm, transpose: bool):
    """Apply role permutation + index relabeling (+ optional transpose) to T
    directly (used only to discover/verify valid symmetries)."""
    P = _pair_perm_matrix(perm, transpose)
    # role tells us, for each new slot s in {0,1,2}, which old mode feeds it.
    Tp = np.transpose(T, axes=role)
    Tp = np.einsum("ap,bq,cr,pqr->abc", P, P, P, Tp)
    return Tp


def _discover_symmetries():
    """Return list of (role, perm, transpose) generators verified to leave
    the matmul tensor exactly invariant."""
    T = matmul_tensor()
    valid = []
    for role in _ROLE_PERMS:
        for perm in _INDEX_PERMS:
            for transpose in (False, True):
                if np.array_equal(_apply_symmetry(T, role, perm, transpose), T):
                    valid.append((role, perm, transpose))
    return valid


_SYMMETRIES = _discover_symmetries()

# Sign triples (a, b, c) in {-1, 1}^3 with a*b*c = 1 (rescaling a term keeps
# u_r (x) v_r (x) w_r unchanged only when the product of scale factors is 1;
# for the discrete {-1,0,1} case that means an even number of -1's).
_SIGN_TRIPLES = [s for s in itertools.product((1, -1), repeat=3) if np.prod(s) == 1]


def _apply_symmetry_to_scheme(U, V, W, role, perm, transpose):
    P = _pair_perm_matrix(perm, transpose)
    mats = [U, V, W]
    new = [P @ mats[role[0]], P @ mats[role[1]], P @ mats[role[2]]]
    return new[0], new[1], new[2]


def _canonical_column(u_col, v_col, w_col) -> Tuple[int, ...]:
    """Canonical representative of a single term under the 4 valid sign
    rescalings, as the lexicographically smallest signed version."""
    best = None
    for a, b, c in _SIGN_TRIPLES:
        cand = tuple((a * u_col).astype(int)) + tuple((b * v_col).astype(int)) + tuple(
            (c * w_col).astype(int)
        )
        if best is None or cand < best:
            best = cand
    return best


def canonical_signature(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> bytes:
    """Canonical form of a discrete {-1,0,1} scheme, invariant under term
    reordering, per-term sign rescaling, and the verified index/role
    symmetries of the matmul tensor. Equal schemes (up to those symmetries)
    get equal signatures.

    Caller must ensure U, V, W are integral with entries in {-1, 0, 1}.
    """
    U = np.asarray(U, dtype=np.int64)
    V = np.asarray(V, dtype=np.int64)
    W = np.asarray(W, dtype=np.int64)
    rank = U.shape[1]

    best_sig = None
    for role, perm, transpose in _SYMMETRIES:
        Up, Vp, Wp = _apply_symmetry_to_scheme(U, V, W, role, perm, transpose)
        cols = sorted(
            _canonical_column(Up[:, r], Vp[:, r], Wp[:, r]) for r in range(rank)
        )
        sig = tuple(cols)
        if best_sig is None or sig < best_sig:
            best_sig = sig
    return repr(best_sig).encode()


def load_known_schemes(paths: Iterable[str]):
    """Load known schemes from .npz files, each holding arrays U, V, W.
    Returns {name: canonical_signature}. Raises if a file's scheme does not
    exactly satisfy the Brent equations (never trust an unverified scheme)."""
    T = matmul_tensor()
    known = {}
    for path in paths:
        data = np.load(path)
        U, V, W = (np.round(data[k]).astype(np.int64) for k in ("U", "V", "W"))
        if np.abs(np.concatenate([U, V, W], axis=1)).max() > 1:
            raise ValueError(f"{path}: entries outside {{-1,0,1}}")
        if not np.array_equal(np.einsum("ar,br,cr->abc", U, V, W), T):
            raise ValueError(f"{path}: does not satisfy the Brent equations exactly")
        known[path] = canonical_signature(U, V, W)
    return known


def rank_profile(U: np.ndarray, V: np.ndarray, W: np.ndarray) -> tuple:
    """Sorted multiset of per-term rank triples.

    Each factor column reshapes to a 3x3 matrix, and the symmetries of the
    matmul tensor act on those matrices by left/right multiplication by
    invertible matrices (and transposition), which preserves rank. Sorting each
    term's triple absorbs the role permutation of U, V, W. So two schemes with
    different profiles are provably inequivalent. Equal profiles prove nothing.
    """
    profile = []
    for r in range(U.shape[1]):
        ranks = [
            int(np.linalg.matrix_rank(np.asarray(M[:, r], dtype=float).reshape(N, N)))
            for M in (U, V, W)
        ]
        profile.append(tuple(sorted(ranks)))
    return tuple(sorted(profile))


def provably_inequivalent(S1, S2) -> bool:
    """True if the rank profiles differ, which proves S1 and S2 are inequivalent
    under the full symmetry group. False means inconclusive, not equivalent."""
    return rank_profile(*S1) != rank_profile(*S2)
