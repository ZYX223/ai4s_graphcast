"""Host overhead must never be counted as device compute time."""
import unittest
from trace_summary import account
class TraceTests(unittest.TestCase):
 def test_filters_host_and_unmeasured_and_accounts_overlap(self):
  def event(cat,b,e):return dict(cat=cat,ph='X',name='kernel',ts=b/1000,dur=(e-b)/1000,args=dict(BeginNs=str(b),EndNs=str(e)))
  events=[event('HIP',100,900),event('HIPOPS',50,90),event('HIPOPS',200,400),event('HIPOPS',300,500),event('HIPCOPY',510,580)]
  r=account(events,100,600)
  self.assertAlmostEqual(r['gpu_kernel_sum_ms'],.0004)
  self.assertAlmostEqual(r['gpu_kernel_busy_union_ms'],.0003)
  self.assertEqual(r['gpu_kernel_count'],2)
 def test_rejects_missing_device_events(self):
  with self.assertRaises(ValueError):account([],0,100)
if __name__=='__main__':unittest.main()
