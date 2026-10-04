"""Check a scheme (.npz with U, V, W) against the published catalogue.

Usage: python check_catalogue.py path/to/scheme.npz [--cache DIR]
The catalogue is downloaded on first use into the cache dir (outside the repo).
"""

import argparse
import sys

import numpy as np

import catalogue
import symmetry


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("scheme")
    p.add_argument("--cache", default=catalogue.DEFAULT_CACHE)
    args = p.parse_args()

    d = np.load(args.scheme)
    U, V, W = (np.round(d[k]).astype(np.int64) for k in "UVW")
    T = symmetry.matmul_tensor()
    if not np.array_equal(np.einsum("ar,br,cr->abc", U, V, W), T):
        print("scheme is not exact; refusing to compare")
        return 1
    root = catalogue.fetch(args.cache)
    result = catalogue.check_scheme(U, V, W, root)
    print(f"catalogue schemes checked: {result['catalogue_size']}")
    print(f"canonical matches: {len(result['matches'])}")
    for m in result["matches"]:
        print("  ", m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
