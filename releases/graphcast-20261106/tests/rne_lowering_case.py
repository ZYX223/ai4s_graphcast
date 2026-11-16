"""Numerical contract: moving RNE must not differentiate through bitcasts.

Before the candidate module exists this deliberately runs the unchanged JAX
baseline, so the forward/backward comparisons expose the missing rounding.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from graphcast_bench.optimizations.rne import lowering

lowering.install(platform="cpu")


def reference_rne(x):
    x = np.asarray(x, np.float32)
    u = x.view(np.uint32)
    finite = (u & np.uint32(0x7F800000)) != np.uint32(0x7F800000)
    r = (u + np.uint32(0xFFF) + ((u >> 13) & np.uint32(1))) & np.uint32(0xFFFFE000)
    return np.where(finite, r, u).view(np.float32)


class LoweringRneTest(unittest.TestCase):
    def test_forward_operand_rounding(self):
        rng = np.random.default_rng(27)
        a = rng.normal(size=(17, 8)).astype(np.float32)
        b = rng.normal(size=(8, 7)).astype(np.float32)
        actual = jax.jit(lambda x, w: x @ w)(a, b)
        expected = reference_rne(a) @ reference_rne(b)
        np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=1e-6)

    def test_forward_and_both_gradients_after_nonlinear_producer(self):
        rng = np.random.default_rng(37)
        x = rng.normal(size=(17, 8)).astype(np.float32)
        w = rng.normal(size=(8, 7)).astype(np.float32)
        cot = rng.normal(size=(17, 7)).astype(np.float32)

        def loss(x, w, cot):
            return jnp.sum((jax.nn.silu(x) @ w) * cot)

        val, (gx, gw) = jax.jit(jax.value_and_grad(loss, (0, 1)))(x, w, cot)
        sig = 1 / (1 + np.exp(-x))
        activation = x * sig
        expected_value = np.sum((reference_rne(activation) @ reference_rne(w)) * cot)
        expected_gw = reference_rne(activation).T @ reference_rne(cot)
        expected_gx = (reference_rne(cot) @ reference_rne(w).T) * (sig + x * sig * (1 - sig))
        np.testing.assert_allclose(val, expected_value, rtol=2e-6, atol=1e-5)
        np.testing.assert_allclose(gw, expected_gw, rtol=3e-7, atol=1e-6)
        np.testing.assert_allclose(gx, expected_gx, rtol=2e-6, atol=2e-6)
        self.assertGreater(float(jnp.linalg.norm(gx)), 1)

    def test_checkpoint_preserves_forward_and_backward_contract(self):
        rng = np.random.default_rng(47)
        x = rng.normal(size=(19, 8)).astype(np.float32)
        w = rng.normal(size=(8, 7)).astype(np.float32)
        cot = rng.normal(size=(19, 7)).astype(np.float32)

        def loss(x, w):
            return jnp.sum(jax.checkpoint(lambda a, b: a @ b)(x, w) * cot)

        val, (gx, gw) = jax.jit(jax.value_and_grad(loss, (0, 1)))(x, w)
        np.testing.assert_allclose(
            val, np.sum((reference_rne(x) @ reference_rne(w)) * cot), rtol=2e-6, atol=1e-5
        )
        np.testing.assert_allclose(
            gx, reference_rne(cot) @ reference_rne(w).T, rtol=3e-7, atol=1e-6
        )
        np.testing.assert_allclose(
            gw, reference_rne(x).T @ reference_rne(cot), rtol=3e-7, atol=1e-6
        )

    def test_scalar_dot_for_norm_is_not_rounded(self):
        x = np.array([1.00021, -2.00031, 3.00019], np.float32)
        actual = jax.jit(lambda a: jnp.dot(a, a))(x)
        np.testing.assert_allclose(actual, np.dot(x, x), rtol=1e-7)


if __name__ == "__main__":
    unittest.main()
