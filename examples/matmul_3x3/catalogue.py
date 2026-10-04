"""Parse and check against the published .exp catalogue of 3x3 matmul schemes.

The archive is third-party data, so it is downloaded on demand into a cache
outside the repository (default ~/.cache/shinka_matmul3x3) and never committed.

Each .exp file lists rank-one terms as `(a-form)*(b-form)*(c-form)` with entries
written like a11, b22, c31 (1-indexed rows and columns). The c-form matches the
convention used by `symmetry.matmul_tensor`, and every exact file verified
against it.
"""

from __future__ import annotations

import glob
import os
import re
import tarfile
import urllib.request
from typing import Iterator, Tuple

import numpy as np

import symmetry

URL = "http://www.algebra.uni-linz.ac.at/research/matrix-multiplication/schemes-exp.tgz"
DEFAULT_CACHE = os.path.expanduser("~/.cache/shinka_matmul3x3")
ARCHIVE_NAME = "schemes-exp.tgz"
EXTRACT_DIR = "schemes"

_TERM = re.compile(r"\((.*?)\)\*\((.*?)\)\*\((.*?)\)")
_ENTRY = re.compile(r"([+-]?)(%s)(\d)(\d)")


def fetch(cache_dir: str = DEFAULT_CACHE, url: str = URL) -> str:
    """Download and extract the archive if needed. Returns the extracted dir."""
    os.makedirs(cache_dir, exist_ok=True)
    archive = os.path.join(cache_dir, ARCHIVE_NAME)
    out = os.path.join(cache_dir, EXTRACT_DIR)
    if not os.path.exists(archive):
        urllib.request.urlretrieve(url, archive)
    if not os.path.isdir(out):
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(cache_dir, filter="data")
    return out


def _linear(expr: str, var: str) -> np.ndarray:
    v = np.zeros(9, dtype=np.int64)
    for sign, i, j in re.findall(r"([+-]?)" + var + r"(\d)(\d)", expr):
        v[3 * (int(i) - 1) + (int(j) - 1)] += -1 if sign == "-" else 1
    return v


def parse_exp(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse one .exp scheme into integer factor matrices U, V, W (9 x rank)."""
    U, V, W = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip().replace(" ", "")
            if not line:
                continue
            m = _TERM.fullmatch(line)
            if m is None:
                raise ValueError(f"{path}: unexpected line {line!r}")
            a, b, c = m.groups()
            U.append(_linear(a, "a"))
            V.append(_linear(b, "b"))
            W.append(_linear(c, "c"))
    return np.array(U).T, np.array(V).T, np.array(W).T


def iter_catalogue(root: str) -> Iterator[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (path, U, V, W) for every rank-23 scheme that verifies exactly."""
    T = symmetry.matmul_tensor()
    for path in sorted(glob.glob(os.path.join(root, "**", "*.exp"), recursive=True)):
        U, V, W = parse_exp(path)
        if U.shape[1] != 23:
            continue
        if not np.array_equal(np.einsum("ar,br,cr->abc", U, V, W), T):
            continue
        yield path, U, V, W


def check_scheme(U, V, W, root: str) -> dict:
    """Compare a scheme against the catalogue by canonical signature."""
    target = symmetry.canonical_signature(U, V, W)
    matches = []
    total = 0
    for path, cU, cV, cW in iter_catalogue(root):
        total += 1
        if symmetry.canonical_signature(cU, cV, cW) == target:
            matches.append(path)
    return {"catalogue_size": total, "matches": matches}
