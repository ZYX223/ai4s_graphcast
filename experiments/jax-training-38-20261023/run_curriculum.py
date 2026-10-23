"""Sequential same-card curriculum; each mode resumes its own prior horizon."""
import datetime,json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'experiment'))
from trace_summary import summarize
from normalize_trace import normalize
device=12
while not (root/'data/source.json').exists():
 print('WAIT data verification',datetime.datetime.now().isoformat(),flush=True);time.sleep(30)
for rollout in range(1,13):
 for mode in ['fp32','rne_xf32']:
  out=root/('%s-r%02d'%(mode,rollout))
  if out.exists():
   report=json.loads((out/'report.json').read_text())
   if report['status']!='completed':raise RuntimeError('Existing failed/incomplete stage: '+str(out))
  else:
   while True:
    card=json.loads(subprocess.check_output(['/opt/hyhal/bin/hy-smi','-d',str(device),'--showuse','--showmeminfo','vram','--json'],universal_newlines=True))['card'+str(device)]
    if float(card['HCU use (%)'])==0 and float(card['vram Total Used Memory (MiB)'])<512:break
    print('WAIT GPU',device,card,flush=True);time.sleep(30)
   cmd=[sys.executable,str(root/'experiment/launch.py'),'train','--mode',mode,'--rollout',str(rollout),'--device',str(device),'--output',str(out)]
   if rollout>1:cmd+=['--resume','%s-r%02d/checkpoint.pkl'%(mode,rollout-1)]
   print('START',mode,rollout,datetime.datetime.now().isoformat(),flush=True)
   subprocess.run(cmd,check=True)
  if not (out/'device-timing.json').exists():
   normalize(out)
   summarize(out)
  report=json.loads((out/'report.json').read_text())
  assert report['status']=='completed' and not report.get('preflight',False) and not report['tiny']
  print('DONE',mode,rollout,report['rows'][-1]['update'],flush=True)
print('COMPLETE both modes38updates',flush=True)
