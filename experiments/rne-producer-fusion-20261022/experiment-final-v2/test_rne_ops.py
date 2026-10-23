"""Spike correctness contract: preserve current HIP RNE semantics exactly."""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
from rne_ops import rne


def oracle(bits):
    # Independent quotient/remainder implementation, including ties to even.
    q, rem = bits >> 13, bits & 8191
    q += ((rem > 4096) | ((rem == 4096) & ((q & 1) != 0))).astype(np.uint32)
    return np.where((bits & 0x7f800000) == 0x7f800000, bits, q << 13).astype(np.uint32)


class RNETests(unittest.TestCase):
    def test_bit_patterns(self):
        special = np.array([0,0x80000000,1,0x80000001,0x7fffff,0x800000,
                            0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,
                            0x7fc01234,0xffc05678,0x7f801234],np.uint32)
        ties = np.array([0x3f801000,0x3f803000,0xbf801000,0xbf803000],np.uint32)
        random = np.random.RandomState(22).randint(0,2**32,100000,dtype=np.uint32)
        bits = np.r_[special,ties,random]
        actual = np.asarray(jax.jit(rne)(jnp.asarray(bits.view(np.float32)))).view(np.uint32)
        np.testing.assert_array_equal(actual,oracle(bits))

    def test_idempotent(self):
        x = jnp.asarray(np.random.RandomState(23).normal(size=10000).astype(np.float32))
        np.testing.assert_array_equal(np.asarray(jax.jit(rne)(x)).view(np.uint32),
                                      np.asarray(jax.jit(lambda v:rne(rne(v)))(x)).view(np.uint32))

    def test_concat_then_round(self):
        a = np.random.RandomState(24).normal(size=(29,7)).astype(np.float32)
        b = np.random.RandomState(25).normal(size=(29,11)).astype(np.float32)
        f = jax.jit(lambda x,y:rne(jnp.concatenate((x,y),axis=-1)))
        np.testing.assert_array_equal(np.asarray(f(a,b)).view(np.uint32),
                                      oracle(np.concatenate((a,b),axis=-1).view(np.uint32)))

    def test_reject_wrong_dtype(self):
        with self.assertRaises(TypeError):
            rne(jnp.ones((10,),dtype=jnp.float16))

if __name__ == '__main__':
    unittest.main(verbosity=2)
