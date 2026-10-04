import importlib.util
import sys
from pathlib import Path

import numpy as np

EX = Path(__file__).resolve().parents[1] / "examples" / "matmul_3x3"
if str(EX) not in sys.path:
    # evaluate.py does `import symmetry` as a same-directory sibling module,
    # the way it will actually be run (from the example directory).
    sys.path.insert(0, str(EX))


def _load(name):
    spec = importlib.util.spec_from_file_location(f"matmul3_{name}", EX / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def naive_scheme():
    """Rank-27 schoolbook scheme in the trace-form convention."""
    U = np.zeros((9, 27))
    V = np.zeros((9, 27))
    W = np.zeros((9, 27))
    r = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[3 * i + j, r] = V[3 * j + k, r] = W[3 * k + i, r] = 1
                r += 1
    return U, V, W


def test_naive_scheme_is_exact_and_multiplies():
    ev = _load("evaluate")
    U, V, W = naive_scheme()
    assert ev.brent_residual(U, V, W) == 0
    assert ev.is_exact_discrete(U, V, W)
    rng = np.random.default_rng(1)
    A, B = rng.integers(-9, 9, (3, 3)), rng.integers(-9, 9, (3, 3))
    assert np.array_equal(ev.multiply_with_scheme(U, V, W, A, B), A @ B)


def test_perturbed_scheme_not_exact():
    ev = _load("evaluate")
    U, V, W = naive_scheme()
    U[0, 0] = 0
    assert not ev.is_exact_discrete(U, V, W)
    assert ev.brent_residual(U, V, W) > 0


def test_validate_rejects_wrong_rank():
    ev = _load("evaluate")
    ok, msg = ev.validate(naive_scheme())
    assert not ok and "expected" in msg


def test_matches_known_scheme_after_reorder_and_sign_flip():
    """The known-scheme guard must survive trivial disguises: shuffling the
    term order and rescaling a term's three columns by (a, b, c), abc=1."""
    ev = _load("evaluate")
    U, V, W = (M.astype(int) for M in naive_scheme())
    rng = np.random.default_rng(2)
    perm = rng.permutation(27)
    Up, Vp, Wp = U[:, perm].copy(), V[:, perm].copy(), W[:, perm].copy()
    sym = _load("symmetry")
    for r in range(27):
        a, b, c = sym._SIGN_TRIPLES[rng.integers(len(sym._SIGN_TRIPLES))]
        Up[:, r] *= a
        Vp[:, r] *= b
        Wp[:, r] *= c
    assert ev.brent_residual(Up, Vp, Wp) == 0
    assert ev.matches_known_scheme(Up, Vp, Wp) is not None


def test_matches_known_scheme_none_for_different_rank():
    """A rank-23-shaped scheme should never match the rank-27 known scheme
    (different rank => different signature length), so this must return
    None rather than raise or false-positive."""
    ev = _load("evaluate")
    U, V, W = naive_scheme()
    assert ev.matches_known_scheme(U[:, :23], V[:, :23], W[:, :23]) is None


def test_laderman_scheme_file_is_rank23_exact_and_detected_as_known():
    """known_schemes/laderman_1976.npz must itself satisfy the Brent
    equations exactly (sourced from arXiv:1108.2830 sec. 2.4, the original
    Laderman 1976 scheme transcribed from the paper's Maple listing) and the
    evaluator must recognize it -- including under relabeling/rescaling."""
    ev = _load("evaluate")
    data = np.load(EX / "known_schemes" / "laderman_1976.npz")
    U, V, W = data["U"], data["V"], data["W"]
    assert U.shape == V.shape == W.shape == (9, 23)
    assert ev.is_exact_discrete(U, V, W)
    assert ev.matches_known_scheme(U, V, W) == str(EX / "known_schemes" / "laderman_1976.npz")

    rng = np.random.default_rng(3)
    A, B = rng.integers(-9, 9, (3, 3)), rng.integers(-9, 9, (3, 3))
    assert np.array_equal(ev.multiply_with_scheme(U, V, W, A, B), A @ B)

    # Disguise it (reorder + valid sign rescale) -- must still be caught.
    sym = _load("symmetry")
    perm = rng.permutation(23)
    Up, Vp, Wp = U[:, perm].copy(), V[:, perm].copy(), W[:, perm].copy()
    for r in range(23):
        a, b, c = sym._SIGN_TRIPLES[rng.integers(len(sym._SIGN_TRIPLES))]
        Up[:, r] *= a
        Vp[:, r] *= b
        Wp[:, r] *= c
    assert ev.brent_residual(Up, Vp, Wp) == 0
    assert ev.matches_known_scheme(Up, Vp, Wp) == str(EX / "known_schemes" / "laderman_1976.npz")


def test_oh_hopcroft_kim_scheme_is_rank23_exact_and_not_equal_to_laderman():
    """A second, independently-sourced rank-23 scheme (Oh, Hopcroft & Kim,
    2011, arXiv:1108.2830 sec. 2.5) must also verify exactly, and our
    (deliberately limited, see symmetry.py docstring) symmetry group must
    not conflate it with the Laderman scheme -- they are presented in the
    paper as genuinely different schemes."""
    ev = _load("evaluate")
    data = np.load(EX / "known_schemes" / "oh_hopcroft_kim_2011.npz")
    U, V, W = data["U"], data["V"], data["W"]
    assert ev.is_exact_discrete(U, V, W)
    sym = _load("symmetry")
    lad = np.load(EX / "known_schemes" / "laderman_1976.npz")
    sig_lad = sym.canonical_signature(lad["U"], lad["V"], lad["W"])
    sig_ohk = sym.canonical_signature(U, V, W)
    assert sig_lad != sig_ohk


def test_score_run_exact_discrete_beats_any_approximate():
    ev = _load("evaluate")
    U, V, W = naive_scheme()
    exact = ev.score_run(U[:, :27].astype(float), V.astype(float), W.astype(float))
    assert exact["status"] == "exact_discrete" and exact["score"] >= 2.0
    # Tiny float perturbation: still rounds to the exact scheme, so still exact.
    jitter = ev.score_run(U + 1e-6, V - 1e-6, W + 1e-7)
    assert jitter["status"] == "exact_discrete"


def test_score_run_approximate_never_exceeds_half():
    ev = _load("evaluate")
    U, V, W = naive_scheme()
    # A near-zero float residual that does NOT round to an exact scheme.
    U2 = U * 0.5
    r = ev.score_run(U2, V, W)
    assert r["status"] == "inexact"
    assert r["score"] <= 0.5
    assert r["residual"] > 0


def test_rank_profile_separates_known_rank23_schemes_and_is_invariant():
    sym = _load("symmetry")
    lad = np.load(EX / "known_schemes" / "laderman_1976.npz")
    ohk = np.load(EX / "known_schemes" / "oh_hopcroft_kim_2011.npz")
    S_lad = (lad["U"], lad["V"], lad["W"])
    S_ohk = (ohk["U"], ohk["V"], ohk["W"])
    assert sym.provably_inequivalent(S_lad, S_ohk)
    # Invariance: reordering terms and sign-rescaling must not change the profile.
    rng = np.random.default_rng(5)
    perm = rng.permutation(23)
    U, V, W = (M[:, perm].copy() for M in S_lad)
    for r in range(23):
        a, b, c = sym._SIGN_TRIPLES[rng.integers(len(sym._SIGN_TRIPLES))]
        U[:, r] *= a
        V[:, r] *= b
        W[:, r] *= c
    assert sym.rank_profile(U, V, W) == sym.rank_profile(*S_lad)
    assert not sym.provably_inequivalent((U, V, W), S_lad)


def test_discovered_scheme_is_exact_rank23_and_not_in_known_set():
    """Discovered schemes must verify exactly and must not be loaded as known
    schemes (known_schemes/ is the literature reference set)."""
    ev = _load("evaluate")
    d = np.load(EX / "discovered_schemes" / "gen6_seed1001_rank23.npz")
    U, V, W = d["U"], d["V"], d["W"]
    assert U.shape == V.shape == W.shape == (9, 23)
    assert ev.is_exact_discrete(U, V, W)
    assert ev.matches_known_scheme(U, V, W) is None
    assert not any("discovered" in name for name in ev._KNOWN_SIGNATURES)
