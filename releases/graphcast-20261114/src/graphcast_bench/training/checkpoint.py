"""Checkpoint tree identity and exact CPU/device reload checks."""

import hashlib
import pickle

import jax
import numpy as np


def tree_digest(tree):
    h = hashlib.sha256()
    for leaf in jax.tree.leaves(jax.device_get(tree)):
        a = np.asarray(leaf)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def snapshot_state(tree):
    """Make a finite host snapshot without changing the live training state."""
    host = jax.tree.map(np.asarray, jax.device_get(tree))
    if not all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(host)):
        raise ValueError("Nonfinite parameter or optimizer state in checkpoint")
    return host


def verify_reload(path, params, state):
    with path.open("rb") as f:
        saved = pickle.load(f)
    expected = (params, state)
    actual = (saved["params"], saved["opt_state"])
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert np.asarray(a).dtype == np.asarray(b).dtype
        np.testing.assert_array_equal(a, b)
    device = jax.device_put(actual)
    jax.block_until_ready(device)
    assert tree_digest(device) == tree_digest(expected)
    # Adam counter counts all scheduled updates, including zero learning rates.
    counts = [
        int(np.asarray(x))
        for x in jax.tree.leaves(saved["opt_state"])
        if np.asarray(x).shape == () and np.asarray(x).dtype.kind in "iu"
    ]
    assert counts and all(x == saved["updates"] for x in counts), counts
    return dict(
        status="passed",
        params_sha256=tree_digest(saved["params"]),
        optimizer_sha256=tree_digest(saved["opt_state"]),
        optimizer_counters=counts,
        device_reload=True,
    )
