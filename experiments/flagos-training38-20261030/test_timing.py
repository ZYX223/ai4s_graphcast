import unittest
from timing import interval_union, jax_update_kernels
class TimingTests(unittest.TestCase):
 def test_union_excludes_overlap_and_gaps(self):
  self.assertEqual(interval_union([(1,4),(2,5),(8,10)]),6)
 def test_selects_only_gpu_kernels_inside_named_update(self):
  e=[dict(ph='M',pid=1,name='process_name',args={'name':'/device:GPU:0'}),
     dict(ph='X',pid=701,name='train-update-01',ts=10,dur=10),
     dict(ph='X',pid=1,name='Gemm',ts=11,dur=3),
     dict(ph='X',pid=1,name='RNE',ts=15,dur=2),
     dict(ph='X',pid=1,name='MemcpyHostToDevice',ts=11,dur=8),
     dict(ph='X',pid=701,name='CPU',ts=11,dur=8),
     dict(ph='X',pid=1,name='warmup',ts=1,dur=3)]
  r=jax_update_kernels(e)
  self.assertEqual(len(r[1]),2)
  self.assertAlmostEqual(interval_union([(a,b) for a,b,n in r[1]]),5000)
 def test_missing_gpu_cannot_report_zero_mfu(self):
  with self.assertRaises(ValueError):jax_update_kernels([dict(ph='X',pid=701,name='train-update-01',ts=10,dur=10)])
if __name__=='__main__':unittest.main()
