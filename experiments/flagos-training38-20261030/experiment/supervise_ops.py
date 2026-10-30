"""An abort in one compiler case must not erase evidence or all later cases."""
import json,os,subprocess,sys
from pathlib import Path
base=Path('/tmp/zyx-training38-operators')
cases=json.loads((base/'cases.json').read_text())['cases']
lookup={c['id']:c for c in cases}
for mode in ['native','fl_only']:
 out=base/('bench-'+mode);out.mkdir(exist_ok=True)
 for attempt in range(len(cases)+1):
  env=os.environ.copy();env['XLA_MODE']=mode;env['TRAINING_GEMM_LOG']=str(base/('gemm-'+mode+'.jsonl'));env['TMPDIR']=str(base/('tmp-'+mode));Path(env['TMPDIR']).mkdir(exist_ok=True)
  with (out/('attempt-%03d.log'%attempt)).open('w') as f:
   p=subprocess.run([sys.executable,'/tmp/zyx-training38-experiment/bench_training_ops.py'],env=env,stdout=f,stderr=subprocess.STDOUT)
  (out/('attempt-%03d-exit.txt'%attempt)).write_text(str(p.returncode)+'\n')
  r=json.loads((out/'report.json').read_text())
  if p.returncode==0:print('DONE',mode,r['status'],flush=True);break
  if 'active_case' not in r:raise RuntimeError('Failure outside a case: '+str(p.returncode))
  c=lookup[r['active_case']]
  assert c['id'] not in {v['id'] for v in r['rows']}
  r['rows'].append(dict(id=c['id'],operator=c['operator'],phase=c['phase'],signature=c['signature'],calls_per_microbatch=c['calls'],calls_per_update=2*c['calls'],status='process_failed',exit_code=p.returncode,error='Compiler/runtime process aborted; see attempt-%03d.log'%attempt))
  r.pop('active_case',None);r['status']='running';(out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
  print('RECORDED_FAILURE',mode,c['id'],p.returncode,flush=True)
