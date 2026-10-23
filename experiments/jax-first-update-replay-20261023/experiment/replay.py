"""Fixed-state first nonzero update: precision x learning-rate control.

All branches start at the same original parameters and the same Adam state
after the historical FP32 zero-learning-rate update. No branch is continued
from another branch. Primary post-update evaluation is always FP32.
"""
import ctypes as C
import dataclasses
import gc
import hashlib
import importlib.metadata as md
import json
import os
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from graphcast import checkpoint,data_utils,graphcast
from graphcast_bench.assets import load_case,sha256,verify_source,write_json
from model import make_loss
from precision import Precision
from data_transfer import to_device
from original_train import tree_numpy,flatten,comparison
import recipe

OUT=Path('/output')
LRS=[0.,3e-7,2e-6,2e-5,2e-4]

def digest(tree):
    h=hashlib.sha256()
    leaves,structure=jax.tree_util.tree_flatten(tree)
    h.update(str(structure).encode())
    for leaf in leaves:
        a=np.asarray(leaf)
        h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes())
    return h.hexdigest()

def norm(a):
    return float(np.sqrt(sum(float(np.square(a[i:i+1_000_000].astype(np.float64)).sum())
                             for i in range(0,a.size,1_000_000))))

def vector_compare(actual,reference):
    r=comparison(actual,reference)
    dot=0.;different=0;active=0
    for i in range(0,actual.size,1_000_000):
        a=actual[i:i+1_000_000].astype(np.float64)
        b=reference[i:i+1_000_000].astype(np.float64)
        dot+=float(a@b)
        mask=(a!=0)|(b!=0)
        active+=int(mask.sum());different+=int(((np.sign(a)!=np.sign(b))&mask).sum())
    an,bn=norm(actual),norm(reference)
    r.update(actual_norm=an,reference_norm=bn,cosine=dot/(an*bn) if an*bn else None,
             sign_difference_fraction=different/active if active else None)
    return r

def restore_flat(flat,template):
    leaves,structure=jax.tree_util.tree_flatten(template)
    cursor=0;result=[]
    for leaf in leaves:
        result.append(np.asarray(flat[cursor:cursor+leaf.size]).reshape(leaf.shape).copy())
        cursor+=leaf.size
    assert cursor==flat.size
    return jax.tree_util.tree_unflatten(structure,result)

def main():
    case,prepared=load_case('/weekly');upstream=verify_source()
    for name,h in upstream['files'].items():
        if name.endswith('.py'):assert sha256(Path('/official')/name)==h
    experiment={p.name:sha256(p) for p in Path('/experiment').iterdir() if p.is_file()}
    dev=jax.devices()[0];assert len(jax.devices())==1 and 'BW1200' in dev.device_kind
    assert not jax.config.jax_enable_x64 and os.environ.get('ROCBLAS_MATH_MODE') is None
    hip=C.CDLL('libamdhip64.so');bus=C.create_string_buffer(64)
    assert hip.hipDeviceGetPCIBusId(bus,64,0)==0 and bus.value.decode().lower()==os.environ['EXPECTED_PCI']
    control=Precision();control.select('fp32','bootstrap')
    with open('/weekly/assets/jax_weights.npz','rb') as f:w=checkpoint.load(f,graphcast.CheckPoint)
    params_host=w.params;p0=jax.device_put(params_host);p0_flat=flatten(params_host)
    assert p0_flat.dtype==np.float32 and np.isfinite(p0_flat).all()
    source=json.loads(Path('/training/data/source.json').read_text())
    raw_path=Path('/training/data/hres-20220101-025-13-12.nc')
    assert sha256(raw_path)==source['sha256']
    raw=xr.load_dataset(raw_path,decode_timedelta=True)
    plan=recipe.sample_plan(raw.sizes['time'],27)
    assert plan[0]['lr']==0 and plan[1]['lr']==2e-4 and plan[1]['starts']==[2,3]
    stats={n:xr.load_dataset('/weekly/assets/stats/'+n+'.nc').astype(np.float32)
           for n in ['stddev_by_level','mean_by_level','diffs_stddev_by_level']}
    batches=[]
    for start in [2,3]:
        selected=raw.isel(time=slice(start,start+3))
        b=data_utils.extract_inputs_targets_forcings(selected,target_lead_times=slice('6h','6h'),
            **dataclasses.asdict(w.task_config))
        b=tuple(x.astype(np.float32) for x in b)
        assert b[0].sizes['time']==2 and b[1].sizes['time']==1
        assert all(np.isfinite(v.values).all() for ds in b for v in ds.data_vars.values())
        batches.append(to_device(b))
    jax.block_until_ready(batches)
    del raw;gc.collect()
    batch_sha=digest(batches)
    old_report=json.loads(Path('/training/fp32-r01/report.json').read_text())
    historic_g1=np.load('/training/fp32-r01/gradient-01.npy',mmap_mode='r')
    assert hashlib.sha256(historic_g1.tobytes()).hexdigest()==old_report['rows'][0]['gradient_sha256']
    historic_g2=np.load('/training/fp32-r01/gradient-02.npy',mmap_mode='r')
    assert hashlib.sha256(historic_g2.tobytes()).hexdigest()==old_report['rows'][1]['gradient_sha256']
    optimizer=recipe.optimizer(params_host)

    @jax.jit
    def update(params,state,grads,lr):
        updates,new_state=optimizer.update(grads,state,params)
        return jax.tree.map(lambda p,u:p+lr*u,params,updates),new_state

    @jax.jit
    def direction(params,state,grads):
        return optimizer.update(grads,state,params)

    zero_state=jax.device_put(optimizer.init(params_host))
    pg1=jax.device_put(restore_flat(historic_g1,params_host))
    untouched,shared_state=update(p0,zero_state,pg1,jnp.float32(0))
    jax.block_until_ready((untouched,shared_state))
    assert np.array_equal(flatten(tree_numpy(untouched)),p0_flat)
    integer_leaves=[int(x) for x in jax.tree.leaves(tree_numpy(shared_state)) if x.dtype.kind in 'iu']
    assert integer_leaves==[1],integer_leaves
    state_sha=digest(shared_state);param_sha=digest(p0)
    with (OUT/'shared-start.pkl').open('wb') as f:
        pickle.dump(dict(params=params_host,opt_state=tree_numpy(shared_state),historical_updates=1,
            source_gradient_sha256=old_report['rows'][0]['gradient_sha256'],starts=[2,3]),f,protocol=4)
    del zero_state,pg1,untouched;gc.collect()

    loss_fn=make_loss(w.model_config,w.task_config,stats)
    def loss_apply(params,batch):
        loss,state=loss_fn.apply(params,{},jax.random.PRNGKey(27),*batch)
        assert not state
        return loss
    report=dict(schema='graphcast.fixed_first_nonzero_update.v1',status='compiling',
        source=upstream,experiment_sha256=experiment,prepared_case=prepared,
        model=dataclasses.asdict(w.model_config),task=dataclasses.asdict(w.task_config),
        device=dict(index=int(os.environ['PHYSICAL_DEVICE']),pci=bus.value.decode(),driver_name=dev.device_kind,user_name='BW1100'),
        versions={n:md.version(n) for n in ['jax','jaxlib','dm-haiku','optax','numpy']},
        data_sha256=source['sha256'],fixed_starts=[2,3],accumulation=2,rollout=1,seed=27,
        memory_preallocate=os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'],memory_fraction=os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'],
        initial_params_sha256=param_sha,initial_optimizer_sha256=state_sha,batch_sha256=batch_sha,
        shared_start_sha256=sha256(OUT/'shared-start.pkl'),learning_rates=LRS,
        optimizer_state_origin='Shared historical FP32 step-1 gradient reconstructed into Adam state at count=1; original RNE step-1 state intentionally excluded.',
        precision='FP32 storage; native FP32 GEMM or fresh both-operand RNE+XF32 in forward, backward and recomputation.',
        loss_evaluation='Same fixed batch. All branches evaluated with a common FP32 forward; RNE forward also recorded.',
        gradient_trials=[],updates=[],audits=[],comparisons={})
    def save():write_json(OUT/'report.json',report)
    save()
    print('FIXED START VERIFIED',param_sha,state_sha,batch_sha,flush=True)
    t=time.perf_counter()
    grad=jax.jit(jax.value_and_grad(loss_apply)).lower(p0,batches[0]).compile()
    report['gradient_compile_seconds']=time.perf_counter()-t
    report['gradient_memory_analysis']=str(grad.memory_analysis())
    (OUT/'gradient-hlo.txt').write_text(grad.as_text())
    print('GRADIENT COMPILED',report['gradient_compile_seconds'],report['gradient_memory_analysis'],flush=True)
    t=time.perf_counter()
    forward=jax.jit(loss_apply).lower(p0,batches[0]).compile()
    report['forward_compile_seconds']=time.perf_counter()-t
    (OUT/'forward-hlo.txt').write_text(forward.as_text())
    print('FORWARD COMPILED',report['forward_compile_seconds'],flush=True)
    average=jax.jit(lambda a,b:jax.tree.map(lambda x,y:(x+y)*.5,a,b))

    def checked_call(fn,precision,label,*args):
        control.select(precision,label);before=control.counts();t=time.perf_counter()
        value=fn(*args);jax.block_until_ready(value)
        counts=control.delta(before)
        assert counts['calls']>0 and counts['rounded']==(counts['calls'] if precision=='rne_xf32' else 0)
        audit=dict(label=label,precision=precision,host_synchronized_seconds=time.perf_counter()-t,**counts)
        report['audits'].append(audit)
        return value

    def evaluate(params,precision,label):
        vals=[float(checked_call(forward,precision,label+'-micro%d'%i,params,b)) for i,b in enumerate(batches)]
        assert np.isfinite(vals).all()
        return dict(mean=float(np.mean(vals)),micro_losses=vals)

    report['pre_loss']={mode:evaluate(p0,mode,'before-'+mode) for mode in ['fp32','rne_xf32']}
    print('PRE LOSS',report['pre_loss'],flush=True)
    report['status']='gradient_trials';save()
    gradients={};directions={};grad_flats={}
    for mode,repeat in [('fp32',1),('fp32',2),('rne_xf32',1),('rne_xf32',2)]:
        label=mode+'-repeat%d'%repeat
        t=time.perf_counter()
        l0,g0=checked_call(grad,mode,label+'-micro0',p0,batches[0])
        l1,g1=checked_call(grad,mode,label+'-micro1',p0,batches[1])
        g=average(g0,g1);jax.block_until_ready(g)
        host=tree_numpy(g);flat=flatten(host)
        assert np.isfinite(flat).all()
        np.save(OUT/(label+'-gradient.npy'),flat,allow_pickle=False)
        grad_flats[label]=flat
        row=dict(mode=mode,repeat=repeat,loss=(float(l0)+float(l1))*.5,
            micro_losses=[float(l0),float(l1)],gradient_norm=norm(flat),gradient_sha256=hashlib.sha256(flat.tobytes()).hexdigest(),
            seconds=time.perf_counter()-t)
        if repeat==1:
            gradients[mode]=host
            control.select('fp32','optimizer-direction-'+mode)
            d,new_state=direction(p0,shared_state,g);jax.block_until_ready((d,new_state))
            directions[mode]=flatten(tree_numpy(d))
            np.save(OUT/(mode+'-optimizer-direction.npy'),directions[mode],allow_pickle=False)
            row['optimizer_direction_norm']=norm(directions[mode])
            del d,new_state
        report['gradient_trials'].append(row);save()
        print('GRADIENT',row,flush=True)
        del g0,g1,g,l0,l1,host;gc.collect()
    c=report['comparisons']
    c['gradient_rne_vs_fp32']=vector_compare(grad_flats['rne_xf32-repeat1'],grad_flats['fp32-repeat1'])
    c['gradient_fp32_repeat_floor']=vector_compare(grad_flats['fp32-repeat2'],grad_flats['fp32-repeat1'])
    c['gradient_rne_repeat_floor']=vector_compare(grad_flats['rne_xf32-repeat2'],grad_flats['rne_xf32-repeat1'])
    c['gradient_fp32_vs_historical_update2']=vector_compare(grad_flats['fp32-repeat1'],historic_g2)
    c['optimizer_direction_rne_vs_fp32']=vector_compare(directions['rne_xf32'],directions['fp32'])
    c['per_module_gradient']=[]
    for module in gradients['fp32']:
        a=flatten(gradients['rne_xf32'][module]);b=flatten(gradients['fp32'][module])
        c['per_module_gradient'].append(dict(module=module,**vector_compare(a,b)))
    del grad_flats;gc.collect()
    report['status']='learning_rate_branches';save()

    for lr in LRS:
        lr_name='%.0e'%lr
        reference_after=None
        for mode in ['fp32','rne_xf32']:
            assert digest(p0)==param_sha and digest(shared_state)==state_sha
            g=jax.device_put(gradients[mode])
            control.select('fp32','optimizer-'+mode+'-'+lr_name)
            updated,new_state=update(p0,shared_state,g,jnp.float32(lr))
            jax.block_until_ready((updated,new_state))
            updated_host=tree_numpy(updated);flat=flatten(updated_host)
            assert np.isfinite(flat).all()
            delta=flat.astype(np.float64)-p0_flat.astype(np.float64)
            if lr==0:assert np.array_equal(flat,p0_flat)
            else:assert np.any(flat!=p0_flat)
            path=OUT/('params-'+mode+'-'+lr_name+'.pkl')
            with path.open('wb') as f:pickle.dump(updated_host,f,protocol=4)
            row=dict(training_precision=mode,learning_rate=lr,params_file=path.name,params_file_sha256=sha256(path),
                starting_params_sha256=param_sha,starting_optimizer_sha256=state_sha,batch_sha256=batch_sha,
                parameter_change_vs_start=comparison(flat,p0_flat),parameter_delta_norm=norm(delta),
                optimizer_after_sha256=digest(new_state),
                post_loss={ev:evaluate(updated,ev,'post-'+mode+'-'+lr_name+'-eval-'+ev) for ev in ['fp32','rne_xf32']})
            row['fp32_loss_change']=row['post_loss']['fp32']['mean']-report['pre_loss']['fp32']['mean']
            if mode=='fp32':reference_after=flat.copy()
            else:
                row['parameters_vs_fp32_branch']=comparison(flat,reference_after)
                row['parameter_delta_vs_fp32_branch']=vector_compare(delta,reference_after.astype(np.float64)-p0_flat.astype(np.float64))
            report['updates'].append(row);save()
            print('BRANCH',mode,lr,'post_loss',row['post_loss'],'delta_norm',row['parameter_delta_norm'],flush=True)
            del updated,new_state,g,updated_host,flat,delta;gc.collect()
    assert digest(p0)==param_sha and digest(shared_state)==state_sha and digest(batches)==batch_sha
    assert {p.name:sha256(p) for p in Path('/experiment').iterdir() if p.is_file()}==experiment
    assert len(report['gradient_trials'])==4 and len(report['updates'])==10
    report.update(status='completed',initial_state_unchanged=True,jax_memory=dev.memory_stats())
    save();print('COMPLETED controlled first nonzero update replay',flush=True)

if __name__=='__main__':main()
