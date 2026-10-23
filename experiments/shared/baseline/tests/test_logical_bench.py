import unittest
import numpy as np

from graphcast_bench.logical_bench import decode_parameter, primitive_function, reference_samples, rne_numpy


class LogicalBenchTests(unittest.TestCase):
    def test_rne_ties_and_signs(self):
        bits = np.array([0x3f801000, 0x3f803000, 0xbf801000, 0xbf803000], np.uint32)
        np.testing.assert_array_equal(rne_numpy(bits.view(np.float32)).view(np.uint32),
                                      [0x3f800000, 0x3f804000, 0xbf800000, 0xbf804000])

    def test_gather_reference_and_execution(self):
        params = {'dimension_numbers': {'namedtuple': 'GatherDimensionNumbers', 'fields':
                  {'offset_dims': [1], 'collapsed_slice_dims': [0], 'start_index_map': [0],
                   'operand_batching_dims': [], 'start_indices_batching_dims': []}},
                  'slice_sizes': [1, 3], 'unique_indices': False, 'indices_are_sorted': False,
                  'mode': {'enum':'GatherScatterMode','name':'PROMISE_IN_BOUNDS'}, 'fill_value': None}
        x=np.arange(12, dtype=np.float32).reshape(4,3); idx=np.array([[3],[1],[3]], np.int32)
        actual=np.asarray(primitive_function('gather',params)(x,idx))
        expected=reference_samples('gather',params,[x,idx],actual.shape,np.arange(actual.size))
        np.testing.assert_array_equal(actual.ravel(),expected)

    def test_scatter_reference_keeps_duplicates_and_empty_nodes(self):
        params={'dimension_numbers':{'namedtuple':'ScatterDimensionNumbers','fields':
                 {'update_window_dims':[1],'inserted_window_dims':[0],'scatter_dims_to_operand_dims':[0],
                  'operand_batching_dims':[],'scatter_indices_batching_dims':[]}},
                'indices_are_sorted':False,'unique_indices':False,
                'mode':{'enum':'GatherScatterMode','name':'FILL_OR_DROP'},'update_jaxpr':{},'update_consts':[]}
        args=[np.ones((4,2),np.float32), np.array([[2],[0],[2]],np.int32),np.arange(6,dtype=np.float32).reshape(3,2)]
        actual=np.asarray(primitive_function('scatter-add',params)(*args))
        ref=reference_samples('scatter-add',params,args,actual.shape,np.arange(actual.size))
        np.testing.assert_array_equal(actual.ravel(),ref)

    def test_concat_and_slice_references(self):
        arrays=[np.arange(6,dtype=np.float32).reshape(2,3),np.full((2,2),-1,np.float32)]
        out=np.concatenate(arrays,axis=1)
        np.testing.assert_array_equal(reference_samples('concatenate',{'dimension':1},arrays,out.shape,np.arange(10)),out.ravel())
        sliced=out[:,1:4]
        np.testing.assert_array_equal(reference_samples('slice',{'start_indices':[0,1],'limit_indices':[2,4],'strides':None},[out],sliced.shape,np.arange(6)),sliced.ravel())

    def test_scalar_literals_remain_compile_time_constants(self):
        import tempfile
        import jax
        import jax.numpy as jnp
        from graphcast_bench.logical_inventory import export_jaxpr
        from graphcast_bench.logical_bench import Inventory
        with tempfile.TemporaryDirectory() as out:
            export_jaxpr(jax.make_jaxpr(lambda x:x/4)(jnp.ones(3)),out+'/inventory')
            inventory=Inventory(out+'/inventory');case={'group':inventory.data['groups'][0]}
            self.assertEqual(inventory.runtime_positions(case),[0])
            fn=inventory.function(case)
            np.testing.assert_array_equal(fn(jnp.array([4.,8.,12.])),[1,2,3])

    def test_large_tensor_sampling_does_not_truncate_flat_indices(self):
        from graphcast_bench.logical_bench import sample_values
        class LargeTensor:
            shape=(1618745,1,1536)
            def __getitem__(self,index):
                # JAX x64 is disabled: passing a flattened index >2**31
                # would overflow. Per-axis indices remain representable.
                self_outer.assertIsInstance(index,tuple)
                self_outer.assertEqual(len(index),3)
                return np.ravel_multi_index(index,self.shape).astype(np.float64)
        self_outer=self
        coords=np.array([[1618744,0,1535],[0,0,0]],np.int64)
        np.testing.assert_array_equal(sample_values(LargeTensor(),coords),[1618745*1536-1,0])


if __name__=='__main__': unittest.main()
