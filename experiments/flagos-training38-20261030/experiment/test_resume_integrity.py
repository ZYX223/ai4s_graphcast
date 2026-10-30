import hashlib,json,pickle,tempfile,unittest
from pathlib import Path
from runtime import validate_resume

class ResumeIntegrityTests(unittest.TestCase):
 def prepare(self,root):
  p=root/'checkpoint.pkl';p.write_bytes(pickle.dumps({'finite_weight':1.0}))
  r={'status':'completed','checkpoint_reload':{'status':'passed'},'checkpoint_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
  (root/'report.json').write_text(json.dumps(r));(root/'exit-code.txt').write_text('0\n')
  return p,r
 def test_finite_replacement_is_rejected_despite_matching_recipe_metadata(self):
  with tempfile.TemporaryDirectory() as d:
   p,r=self.prepare(Path(d));self.assertEqual(validate_resume(p)['checkpoint_sha256'],r['checkpoint_sha256'])
   p.write_bytes(pickle.dumps({'finite_weight':1.001}))
   with self.assertRaises(ValueError):validate_resume(p)
 def test_failed_or_not_finished_predecessor_cannot_resume(self):
  with tempfile.TemporaryDirectory() as d:
   p,r=self.prepare(Path(d));(p.parent/'exit-code.txt').write_text('139\n')
   with self.assertRaises(ValueError):validate_resume(p)
   (p.parent/'exit-code.txt').write_text('0\n');r['status']='running';(p.parent/'report.json').write_text(json.dumps(r))
   with self.assertRaises(ValueError):validate_resume(p)

if __name__=='__main__':unittest.main()
