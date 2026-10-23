import unittest
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from model_hook import RneHook
from test_rne_ops import oracle

class HookTests(unittest.TestCase):
    def test_only_selected_linear_input_is_rounded(self):
        hook=RneHook({'selected':{'input_shape':[2,2]}})
        @hk.transform
        def model(x):
            with hk.intercept_methods(hook):
                return hk.Linear(2,name='selected')(x),hk.Linear(2,name='unselected')(x),x
        x=np.array([[1.0004,-1.0007],[2.001,3.003]],np.float32)
        params={name:{'w':np.eye(2,dtype=np.float32),'b':np.zeros(2,np.float32)} for name in ['selected','unselected']}
        a,b,c=jax.jit(model.apply)(params,None,x)
        np.testing.assert_array_equal(np.asarray(a).view(np.uint32),oracle(x.view(np.uint32)))
        np.testing.assert_array_equal(np.asarray(b),x);np.testing.assert_array_equal(np.asarray(c),x)
        hook.verify_traced()

    def test_wrong_shape_rejected(self):
        hook=RneHook({'selected':{'input_shape':[2,2]}})
        @hk.transform
        def model(x):
            with hk.intercept_methods(hook):return hk.Linear(2,name='selected')(x)
        with self.assertRaises(ValueError):model.init(jax.random.PRNGKey(0),jnp.ones((3,2)))

    def test_missing_target_rejected(self):
        hook=RneHook({'absent':{'input_shape':[2,2]}})
        with self.assertRaises(ValueError):hook.verify_traced()

if __name__=='__main__':unittest.main(verbosity=2)
