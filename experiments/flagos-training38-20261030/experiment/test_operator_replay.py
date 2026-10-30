import unittest
import jax,numpy as np
from operator_replay import replay_group,pending_cases
class ReplayTests(unittest.TestCase):
 def test_backward_linear_returns_both_weight_and_input_gradients(self):
  # Multiple logical outputs must survive; the inference runner assumed one.
  equations=[dict(primitive='mul',params={},inputs=['x','w'],outputs=['dw']),dict(primitive='add',params={},inputs=['x','w'],outputs=['dx'])]
  f=replay_group(['x','w'],['dw','dx'],equations,{})
  y=f(np.array([2.,3.],np.float32),np.array([5.,7.],np.float32))
  np.testing.assert_array_equal(y[0],[10.,21.]);np.testing.assert_array_equal(y[1],[7.,10.])
 def test_autodiff_add_any_replays_as_add(self):
  f=replay_group(['x','y'],['z'],[dict(primitive='add_any',params={},inputs=['x','y'],outputs=['z'])],{})
  np.testing.assert_array_equal(f(np.array([2.],np.float32),np.array([3.],np.float32))[0],[5.])
 def test_resume_preserves_failed_case_and_never_duplicates_results(self):
  cases=[{'id':'a'},{'id':'b'},{'id':'c'}]
  rows=[{'id':'a','status':'passed'},{'id':'b','status':'process_failed'}]
  self.assertEqual([c['id'] for c in pending_cases(cases,rows)],['c'])
  with self.assertRaises(ValueError):pending_cases(cases,rows+rows[:1])
if __name__=='__main__':unittest.main()
