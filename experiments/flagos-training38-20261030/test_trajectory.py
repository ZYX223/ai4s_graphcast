import copy,unittest
from trajectory import validate_trajectory
class TrajectoryTests(unittest.TestCase):
 def reports(self):
  common={'data_sha256':'d','model':{},'task':{},'optimizer':{},'seed':27,'source_revision':'s','tiny':False}
  out=[]
  for h,ids,a,b in [(1,range(1,17),'a','b'),(2,range(17,19),'b','c')]:
   out.append(dict(common,rollout=h,rows=[{'update':u} for u in ids],initial_params_sha256=a,initial_optimizer_sha256=a+'o',checkpoint_reload={'params_sha256':b,'optimizer_sha256':b+'o','optimizer_counters':[max(ids)]}))
  return out
 def test_finite_parameter_or_optimizer_chain_replacement_rejected(self):
  for key in ['initial_params_sha256','initial_optimizer_sha256']:
   rows=self.reports();validate_trajectory(rows);rows[1][key]='replaced'
   with self.assertRaises(ValueError):validate_trajectory(rows)
 def test_skipped_updates_rejected(self):
  rows=self.reports();rows[1]['rows'][0]['update']=16
  with self.assertRaises(ValueError):validate_trajectory(rows)
if __name__=='__main__':unittest.main()
