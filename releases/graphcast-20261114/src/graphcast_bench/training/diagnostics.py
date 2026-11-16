"""Host diagnostics, kept outside measured model computation."""

import jax
import numpy as np


def tree_numpy(tree):
    return jax.tree.map(lambda x: np.asarray(x), jax.device_get(tree))


def flatten(tree):
    return np.concatenate([np.asarray(x).ravel() for x in jax.tree.leaves(tree)])


def comparison(actual, reference):
    # Blockwise float64 to avoid several extra full-size double buffers.
    sq = 0.0
    ref_sq = 0.0
    l1 = 0.0
    ref_l1 = 0.0
    mx = 0.0
    for i in range(0, actual.size, 1_000_000):
        a = actual[i : i + 1_000_000].astype(np.float64)
        b = reference[i : i + 1_000_000].astype(np.float64)
        d = a - b
        sq += float(d @ d)
        ref_sq += float(b @ b)
        l1 += float(np.abs(d).sum())
        ref_l1 += float(np.abs(b).sum())
        mx = max(mx, float(np.max(np.abs(d))))
    return dict(
        relative_l2=float(np.sqrt(sq / ref_sq)) if ref_sq else None,
        relative_l1=l1 / ref_l1 if ref_l1 else None,
        rmse=float(np.sqrt(sq / actual.size)),
        max_abs=mx,
    )
