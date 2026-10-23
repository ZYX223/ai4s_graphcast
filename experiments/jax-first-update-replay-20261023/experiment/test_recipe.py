"""Catches horizon boundaries, truncated feedback, bad gradient averaging/resume."""
import unittest
import numpy as np
import jax, jax.numpy as jnp
import optax
import recipe

class RecipeTests(unittest.TestCase):
 def test_schedule_boundaries(self):
  rows=recipe.schedule()
  self.assertEqual(len(rows),38)
  self.assertEqual([r['rollout'] for r in rows[:16]],[1]*16)
  self.assertEqual([r['rollout'] for r in rows[16:]],[v for v in range(2,13) for _ in range(2)])
  self.assertEqual([rows[i]['lr'] for i in [0,5,6,15,16,37]],[0.,.001,.001,0.,3e-7,3e-7])
  self.assertEqual(sum(r['rollout'] for r in rows)*2,340)
 def test_sample_windows_fit_horizon_and_are_reproducible(self):
  a=recipe.sample_plan(14,27);b=recipe.sample_plan(14,27)
  self.assertEqual(a,b)
  self.assertEqual(a[-1]['starts'],[0,0])
  self.assertTrue(all(0<=s and s+2+r['rollout']<=14 for r in a for s in r['starts']))
  self.assertGreater(len(set(s for r in a[:16] for s in r['starts'])),1)
 def test_full_bptt_and_average_before_update(self):
  def loss(p,x):
   y1=p['w']*x;y2=p['w']*y1
   return (y1*y1+y2*y2)/2
  p={'w':jnp.array(2.)}
  loss_value,grads=recipe.accumulate(jax.value_and_grad(loss),p,[jnp.array(1.),jnp.array(2.)])
  self.assertAlmostEqual(float(loss_value),25.)
  self.assertAlmostEqual(float(grads['w']),45.)
 def test_optimizer_matches_manual_first_two_updates_and_resume(self):
  import pickle
  p={'w':jnp.array([[2.]]),'b':jnp.array([1.])};g={'w':jnp.array([[3.]]),'b':jnp.array([4.])}
  tx=recipe.optimizer(p);s=tx.init(p)
  def step(p,s,g):
   u,s=tx.update(g,s,p);p=jax.tree.map(lambda a,b:a+.01*b,p,u);return p,s
  p,s=step(p,s,g)
  np.testing.assert_allclose(np.asarray(p['w']),[[1.988]],rtol=1e-6)
  np.testing.assert_allclose(np.asarray(p['b']),[.99],rtol=1e-6)
  restored=pickle.loads(pickle.dumps(jax.device_get((p,s))))
  actual=step(p,s,g);resumed=step(*restored,g)
  for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(resumed)):np.testing.assert_array_equal(a,b)
  # Nonconstant gradient distinguishes beta2=.95 from the Optax default.
  g2=jax.tree.map(lambda x:-x*2,g);p2,_=step(p,s,g2)
  m=.1*3*.9+.1*(-6);v=.05*9*.95+.05*36
  expected=1.988-.01*((m/(1-.9**2))/(np.sqrt(v/(1-.95**2))+1e-8)+.1*1.988)
  self.assertAlmostEqual(float(p2['w'][0,0]),expected,places=6)

if __name__=='__main__':unittest.main()
