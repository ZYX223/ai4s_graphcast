"""Catch reinitialization/skipped steps and rejecting valid later checkpoints."""
import os, pickle, tempfile, unittest
from pathlib import Path
import jax
import jax.numpy as jnp
from runtime import verify_reload
from curriculum import select_updates
import recipe

class CurriculumTests(unittest.TestCase):
 def test_resume_horizon_boundary(self):
  plan=recipe.sample_plan(14,27)
  rows=select_updates(plan,2,16,38)
  self.assertEqual([r['update'] for r in rows],[17,18])
  with self.assertRaises(ValueError):select_updates(plan,2,15,38)
  self.assertEqual(len(select_updates(plan,1,0,38)),16)
 def test_restore_later_optimizer_count(self):
  p={'w':jnp.array([[1.]])};s=(jnp.array(18,dtype=jnp.int32),{'m':jnp.array([[0.]])})
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'ckpt.pkl'
   f.write_bytes(pickle.dumps(dict(params=p,opt_state=s,updates=18)))
   self.assertEqual(verify_reload(f,p,s)['optimizer_counters'],[18])
   f.write_bytes(pickle.dumps(dict(params=p,opt_state=s,updates=17)))
   with self.assertRaises(AssertionError):verify_reload(f,p,s)

if __name__=='__main__':unittest.main()
