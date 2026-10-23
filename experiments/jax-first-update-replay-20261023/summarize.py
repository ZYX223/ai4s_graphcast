"""Validate completed fixed-state replay and export compact result tables."""
import argparse,ast,collections,csv,hashlib,json,math,re,struct
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
folder=a.run.resolve();root=folder.parent
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
r=json.loads((folder/'report.json').read_text())
assert r['status']=='completed' and r['initial_state_unchanged']
assert (folder/'exit-code.txt').read_text().strip()=='0'
assert len(r['updates'])==10 and len(r['gradient_trials'])==4
assert r['fixed_starts']==[2,3] and r['accumulation']==2 and r['rollout']==1
assert r['shared_start_sha256']==sha(folder/'shared-start.pkl')
for name,h in r['experiment_sha256'].items():assert sha(folder/'source'/name)==h
for trial in r['gradient_trials']:
    name='%s-repeat%d-gradient.npy'%(trial['mode'],trial['repeat'])
    with (folder/name).open('rb') as f:
        assert f.read(6)==b'\x93NUMPY'
        version=f.read(2)
        fmt='<H' if version==b'\x01\x00' else '<I'
        length=struct.unpack(fmt,f.read(struct.calcsize(fmt)))[0]
        header=ast.literal_eval(f.read(length).decode('latin1'))
        assert header['descr']=='<f4' and not header['fortran_order'] and len(header['shape'])==1
        h=hashlib.sha256()
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        assert h.hexdigest()==trial['gradient_sha256']
for mode in ['fp32','rne_xf32']:
    assert len({x['optimizer_after_sha256'] for x in r['updates'] if x['training_precision']==mode})==1
for hlo in ['gradient-hlo.txt','forward-hlo.txt']:
    assert not re.search(r'\b(?:bf16|f16)\[',(folder/hlo).read_text())
logs=[json.loads(s) for s in (folder/'gemm.jsonl').read_text().splitlines()]
by_label=collections.defaultdict(list)
for x in logs:by_label[x['label']].append(x)
for audit in r['audits']:
    records=by_label[audit['label']]
    assert len(records)==audit['calls']
    assert all(x['mode']==int(audit['precision']=='rne_xf32') and x['rounded']==(audit['precision']=='rne_xf32') for x in records)
    if 'repeat' in audit['label']:assert audit['calls']==402
rows=[];before=r['pre_loss']['fp32']['mean']
for lr in r['learning_rates']:
    pair={x['training_precision']:x for x in r['updates'] if x['learning_rate']==lr}
    assert set(pair)=={'fp32','rne_xf32'}
    for mode,x in pair.items():
        assert x['starting_params_sha256']==r['initial_params_sha256']
        assert x['starting_optimizer_sha256']==r['initial_optimizer_sha256']
        assert x['batch_sha256']==r['batch_sha256']
        assert sha(folder/x['params_file'])==x['params_file_sha256']
        if lr==0:assert x['parameter_delta_norm']==0
    f,q=pair['fp32'],pair['rne_xf32']
    fl,ql=[x['post_loss']['fp32']['mean'] for x in [f,q]]
    rows.append(dict(learning_rate=lr,pre_loss_common_fp32=before,
        fp32_update_post_loss_fp32_eval=fl,rne_update_post_loss_fp32_eval=ql,
        fp32_loss_change_percent=(fl/before-1)*100,rne_loss_change_percent=(ql/before-1)*100,
        post_loss_between_precisions_relative_percent=abs(ql-fl)/abs(fl)*100,
        fp32_parameter_delta_norm=f['parameter_delta_norm'],rne_parameter_delta_norm=q['parameter_delta_norm'],
        delta_relative_difference=q['parameter_delta_vs_fp32_branch']['relative_l2'],
        full_parameter_relative_difference=q['parameters_vs_fp32_branch']['relative_l2']))
with (root/'learning-rate-results.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
modules=r['comparisons']['per_module_gradient']
with (root/'gradient-by-module.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(modules[0]));w.writeheader();w.writerows(modules)
summary=dict(status='completed',run=str(folder),report_sha256=sha(folder/'report.json'),
    fixed_starts=r['fixed_starts'],initial_params_sha256=r['initial_params_sha256'],
    initial_optimizer_sha256=r['initial_optimizer_sha256'],batch_sha256=r['batch_sha256'],
    optimizer_state_origin=r['optimizer_state_origin'],pre_loss=r['pre_loss'],rows=rows,
    gradient_trials=r['gradient_trials'],comparisons={k:v for k,v in r['comparisons'].items() if k!='per_module_gradient'},
    scope='Single update on one fixed two-microbatch sample. Primary loss always FP32-evaluated. Does not establish a safe long-run learning rate, convergence, or 40-step forecast acceptance.',
    evidence_validation_passed=True,initial_failed_attempt=dict(path=str(root/'run'),reason='Lazy BFC allocation failed on 72GiB backward temporary after forward evaluations; retained raw record. Completed rerun preallocates 70% VRAM; model math and starting conditions unchanged.'))
(root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(pre_loss=summary['pre_loss'],rows=rows,comparisons=summary['comparisons']),indent=2))
