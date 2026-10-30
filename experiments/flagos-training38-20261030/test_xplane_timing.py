import unittest
from tensorflow.tsl.profiler.protobuf import xplane_pb2
from xplane_timing import update_kernels

class XPlaneTimingTests(unittest.TestCase):
 def test_real_schema_time_origin_and_cpu_copy_exclusion(self):
  x=xplane_pb2.XSpace()
  h=x.planes.add(name='/host:CPU');h.event_metadata[1].name='train-update-01'
  l=h.lines.add(timestamp_ns=1000);l.events.add(metadata_id=1,offset_ps=10000,duration_ps=100000)
  g=x.planes.add(name='/device:GPU:0')
  for i,name in enumerate(['gemm','RNE','MemcpyHtoD']):g.event_metadata[i].name=name
  l=g.lines.add(timestamp_ns=1005,name='Stream #1(Kernel)')
  l.events.add(metadata_id=0,offset_ps=10000,duration_ps=20000)
  l.events.add(metadata_id=1,offset_ps=20000,duration_ps=20000)
  l.events.add(metadata_id=2,offset_ps=10000,duration_ps=50000)
  r=update_kernels(x)
  self.assertEqual(r,{1:[(1015,1035,'gemm'),(1025,1045,'RNE')]})
 def test_missing_device_and_duplicate_ranges_rejected(self):
  x=xplane_pb2.XSpace()
  with self.assertRaises(ValueError):update_kernels(x)
  h=x.planes.add(name='/host:CPU');h.event_metadata[1].name='train-update-01'
  l=h.lines.add(timestamp_ns=1)
  l.events.add(metadata_id=1,duration_ps=10000);l.events.add(metadata_id=1,duration_ps=10000)
  x.planes.add(name='/device:GPU:0')
  with self.assertRaises(ValueError):update_kernels(x)

if __name__=='__main__':unittest.main()
