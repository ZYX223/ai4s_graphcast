"""Exercise the actual frozen evaluator with CPU fixtures; never import Torch."""
import ast,copy,hashlib,json,math,os,signal,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
P=Path(__file__).resolve().parent
FROZEN=P.parents[2]/'scripts/training/run_official_epoch_training.py'
sys.path.insert(0,str(P.parent/'harness/evaluation'))
import validation_journal as j
BASELINE=False
def fixture():
 ns={'__file__':str(P/'evaluator_fixture.py')}
 exec(compile((P/'evaluator_fixture.py').read_text(),'evaluator_fixture','exec'),ns)
 return ns

def invoke(ns,fn,forward=None):
 return fn(args=NS(formal=True,seed=27,global_micro_batch_size=1,global_batch_size=2,
     validation_mode='full',validation_steps=2,rollout_steps=12),context=ns['context'],
     wrapped_unroll=forward or ns['forward'],base_model=ns['model'],config=None,stats=None,input_device=None)

IDENTITY={'checkpoint_sha256':'a'*64,'data_sha256':'b'*64,'windows_sha256':'c'*64,
          'flaggems_manifest_sha256':'d'*64,'runner_sha256':hashlib.sha256(FROZEN.read_bytes()).hexdigest()}

class Tests(unittest.TestCase):
 def test_completed_losses_survive_interruption(self):
  ns=fixture()
  with tempfile.TemporaryDirectory() as t:
   out=Path(t)/'rows'
   fn=ns['namespace']['evaluate_validation']
   if not BASELINE:fn=j.instrument(FROZEN.read_text(),ns['namespace'],out,IDENTITY,list(range(1,112)))
   def f(batch,**kw):
    if batch.index==4:raise RuntimeError('simulated interruption before fourth forward')
    return ns['forward'](batch,**kw)
   with self.assertRaisesRegex(RuntimeError,'simulated interruption'):invoke(ns,fn,f)
   self.assertEqual(len(list(out.glob('row-*.json'))),3,'completed losses must persist before interruption')
   if not BASELINE:self.assertEqual([x['row']['loss'] for x in j.read_prefix(out,IDENTITY)], [.1,.2,.3])

 @unittest.skipIf(BASELINE,'baseline regression only')
 def test_111_frozen_evaluator_results_unchanged(self):
  a=fixture();expected=invoke(a,a['namespace']['evaluate_validation'])
  with tempfile.TemporaryDirectory() as t:
   ns=fixture();out=Path(t)/'rows';fn=j.instrument(FROZEN.read_text(),ns['namespace'],out,IDENTITY,list(range(1,112)))
   actual=invoke(ns,fn)
   self.assertEqual(actual,expected);self.assertTrue(math.isclose(actual['mean_loss'],5.6,rel_tol=1e-15))
   self.assertEqual(len(j.read_prefix(out,IDENTITY)),111)
   self.assertTrue(j.verify_complete(out,IDENTITY,actual)['complete'])
   self.assertNotIn('torch',sys.modules)
   with self.assertRaises((AssertionError,FileExistsError)):j.instrument(FROZEN.read_text(),ns['namespace'],out,IDENTITY,list(range(1,112)))

 @unittest.skipIf(BASELINE,'baseline regression only')
 def test_invalid_or_corrupted_records_rejected(self):
  ns=fixture();original=invoke(ns,ns['namespace']['evaluate_validation']);row=original['steps'][0]
  for kind in ('nan','wrong_index','wrong_window','wrong_weight','wrong_count'):
   with self.subTest(kind=kind), tempfile.TemporaryDirectory() as t:
    out=Path(t)/'rows';rows=j.JournalRows(out,IDENTITY,list(range(1,112)));bad=copy.deepcopy(row)
    if kind=='nan':bad['per_variable_lead_losses']['a'][0]=float('nan')
    elif kind=='wrong_index':bad['batch_index']=1
    elif kind=='wrong_window':bad['origins_by_rank'][0]['local_window_indices']=[2]
    elif kind=='wrong_weight':bad['origins_by_rank'][0]['sample_weights']=[0]
    else:bad['valid_sample_count']=0
    with self.assertRaises((AssertionError,ValueError)):rows.append(bad)
    self.assertEqual(len(list(out.glob('row-*.json'))),0)
  with tempfile.TemporaryDirectory() as t:
   out=Path(t)/'rows';rows=j.JournalRows(out,IDENTITY,list(range(1,112)));rows.append(row)
   with self.assertRaises(AssertionError):j.read_prefix(out,{**IDENTITY,'checkpoint_sha256':'f'*64})
   f=out/'row-0000.json';v=json.loads(f.read_text());v['row']['loss']+=1;f.write_text(json.dumps(v))
   with self.assertRaises(AssertionError):j.read_prefix(out,IDENTITY)

 @unittest.skipIf(BASELINE,'baseline regression only')
 def test_actual_sigkill_leaves_committed_row(self):
  with tempfile.TemporaryDirectory() as t:
   out=Path(t)/'rows';ns=fixture();row=invoke(ns,ns['namespace']['evaluate_validation'])['steps'][0]
   code='import sys,json,os,signal;sys.path.insert(0,sys.argv[1]);import validation_journal as j;j.JournalRows(sys.argv[2],json.loads(sys.argv[3]),list(range(1,112))).append(json.loads(sys.argv[4]));os.kill(os.getpid(),signal.SIGKILL)'
   p=subprocess.run([sys.executable,'-c',code,str(P.parent/'harness/evaluation'),str(out),json.dumps(IDENTITY),json.dumps(row)])
   self.assertEqual(p.returncode,-9);self.assertEqual(j.read_prefix(out,IDENTITY)[0]['row'],row)

if __name__=='__main__':unittest.main()
