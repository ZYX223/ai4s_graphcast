"""Known arithmetic catches missing backward GEMMs, scan lengths and dead work."""
import unittest
import jax,jax.numpy as jnp
from flops import summarize_jaxpr

class FlopsTests(unittest.TestCase):
 def test_weight_only_and_both_operand_gradients(self):
  x=jnp.ones((2,3));w=jnp.ones((3,4))
  f=lambda w,x:jnp.sum(x@w)
  self.assertEqual(summarize_jaxpr(jax.make_jaxpr(jax.value_and_grad(f))(w,x))['gemm_flops'],96)
  self.assertEqual(summarize_jaxpr(jax.make_jaxpr(jax.value_and_grad(f,argnums=(0,1)))(w,x))['gemm_flops'],144)
 def test_scan_multiplier_and_unused_dot(self):
  w=jnp.ones((3,4));xs=jnp.ones((5,2,3))
  def f(w,xs):
   def body(c,x):return c+jnp.sum(x@w),None
   unused=xs[0]@w
   return jax.lax.scan(body,jnp.float32(0),xs)[0]
  r=summarize_jaxpr(jax.make_jaxpr(jax.value_and_grad(f))(w,xs))
  self.assertEqual(r['gemm_flops'],480)
 def test_two_step_bptt_is_not_two_independent_gradients(self):
  def f(w,x):
   y=x@w;z=y@w
   return jnp.sum(y)+jnp.sum(z)
  r=summarize_jaxpr(jax.make_jaxpr(jax.value_and_grad(f))(jnp.ones((3,3)),jnp.ones((2,3))))
  self.assertEqual(r['gemm_flops'],180)

if __name__=='__main__':unittest.main()
