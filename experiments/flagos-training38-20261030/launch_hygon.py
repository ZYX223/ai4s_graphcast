"""Serial single-card stages, actual child exit, independent whole-card sampling."""
import datetime,json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent;remote='/tmp/zyx-flagos-training38-20261030'
probe=['/opt/hyhal/bin/hy-smi','-d','10','--showuse','--showmeminfo','vram','--json']
for horizon in range(1,13):
 name='flagos-r%02d'%horizon;out=root/name
 assert not out.exists(),out
 before=json.loads(subprocess.check_output(probe,universal_newlines=True));card=before['card10']
 assert float(card['HCU use (%)'])==0 and int(card['vram Total Used Memory (MiB)'])<100,before
 out.mkdir();(out/'device-before.json').write_text(json.dumps(before,indent=2))
 args=['--mode','rne_xf32','--rollout',str(horizon)]
 if horizon>1:args+=['--resume',remote+'/flagos-r%02d/checkpoint.pkl'%(horizon-1)]
 cmd=['docker','exec','flagos-jax-test','bash',remote+'/run-hygon.sh',name]+args
 (out/'launch.json').write_text(json.dumps(dict(command=cmd,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),baseline_modified=False),indent=2))
 with (out/'console.log').open('w') as log,(out/'host-device-memory.jsonl').open('w') as m:
  p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
  while p.poll() is None:
   try:m.write(json.dumps(dict(time_ns=int(time.time()*1e9),data=json.loads(subprocess.check_output(probe,universal_newlines=True))))+'\n');m.flush()
   except Exception as e:m.write(json.dumps(dict(error=str(e)))+'\n')
   time.sleep(1)
  code=p.wait()
 (out/'exit-code.txt').write_text(str(code)+'\n')
 subprocess.run(['docker','cp','flagos-jax-test:'+remote+'/'+name+'/.',str(out)],check=True)
 assert code==0 and int((out/'python-exit-code.txt').read_text())==0,(horizon,code)
 r=json.loads((out/'report.json').read_text());assert r['status']=='completed' and r['checkpoint_reload']['status']=='passed'
 print('COMPLETED',horizon,len(r['rows']),r['rows'][-1]['loss'],flush=True)
