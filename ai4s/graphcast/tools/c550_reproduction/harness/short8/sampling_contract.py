"""Bounded single-process sampler. All overrides are process-local and restored."""
import json, math
from dataclasses import replace
from pathlib import Path
from statistics import mean

ROLLOUTS=(1,1,1,6,6,6,12,12,12)
PAIR_GATE_PCT=5.0

def validate_row(row,step,rollout):
    required=dict(step=step,optimizer_step=step,rollout_steps=rollout,grad_ok=True,global_batch_size=2,
        global_micro_batch_size=1,gradient_accumulation_steps=2,valid_sample_count=2,inference_only=False)
    if any(row.get(k)!=v for k,v in required.items()):raise ValueError('unexpected diagnostic update or workload')
    for key in ('loss','end_to_end_step_seconds','training_flops'):
        x=row.get(key)
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):raise ValueError('nonfinite metric')
    if row['end_to_end_step_seconds']<=0 or not math.isclose(row['training_flops'],173142566854656*rollout,rel_tol=1e-9):
        raise ValueError('invalid time or workload FLOPs')

def read_declared_segments(run_dir,steps,rollouts):
    if not steps or len(steps)!=len(rollouts) or steps!=list(range(steps[0],steps[-1]+1)):raise ValueError('invalid declared range')
    rows=[]
    for path in sorted((Path(run_dir)/'metrics').glob('segment_*.jsonl')):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if len(rows)!=len(steps) or not all(isinstance(r,dict) and type(r.get('optimizer_step')) is int for r in rows):
        raise ValueError('missing, duplicate or invalid rows')
    rows.sort(key=lambda r:r['optimizer_step'])
    for row,step,r in zip(rows,steps,rollouts):validate_row(row,step,r)
    return rows

def pair_spread(times):
    if len(times)!=2 or any(not math.isfinite(t) or t<=0 for t in times):raise ValueError('invalid sample pair')
    return (max(times)-min(times))/mean(times)*100

class Schedule:
    def __init__(self,original):self.original=original
    def __getattr__(self,name):return getattr(self.original,name)
    def value_for_update(self,index):
        if type(index) is not int or not 0<=index<9:raise ValueError('outside bounded sampling sequence')
        value=self.original.value_for_update(index)
        if index<3:return value
        return replace(value,update_index=index,phase_index=2,phase_name='np43_single_process_r'+str(ROLLOUTS[index]),
            phase_update_index=index-3,learning_rate=3e-7,rollout_steps=ROLLOUTS[index])

class Controller:
    def __init__(self,runner,completion=None):
        self.runner=runner;self.completion=completion;self.args=None;self.rows=[];self.initial_load_count=0;self.decision=None
        bindings={'resolve_training_schedule':self.resolve,'append_jsonl':self.append,'read_metric_segments':self.read,
            'load_model_only_checkpoint':self.load,'load_resume_checkpoint':self.no_resume,'save_validated_training_checkpoint':self.finish}
        self.originals={k:getattr(runner,k) for k in bindings}
        for name,fn in bindings.items():setattr(runner,name,fn)
    def resolve(self,args):
        required=dict(schedule_recipe='official_2_to_12',warmup_updates=6,training_updates=10,finetune_updates=22,
            stop_after_global_step=9,checkpoint_every_steps=10000,resume_training_checkpoint='',global_batch_size=2,
            global_micro_batch_size=1,seed=27,profile_hotspots=False,gradient_sync='manual',data_parallel_replicas=1)
        for key,want in required.items():
            if getattr(args,key)!=want:raise ValueError('sampling contract mismatch: '+key)
        self.args=args;original,source=self.originals['resolve_training_schedule'](args)
        return Schedule(original),source
    def load(self,*a,**kw):
        if self.initial_load_count:raise ValueError('repeated model initialization')
        result=self.originals['load_model_only_checkpoint'](*a,**kw);self.initial_load_count+=1
        return result
    def no_resume(self,*a,**kw):raise ValueError('single process has no intermediate checkpoint reload')
    def append(self,path,value):
        is_metric=Path(path).parent.name=='metrics' and Path(path).name.startswith('segment_')
        if is_metric:
            step=len(self.rows)+1
            if step>9:raise ValueError('sample budget exceeded')
            validate_row(value,step,ROLLOUTS[step-1])
        result=self.originals['append_jsonl'](path,value)
        if is_metric:
            self.rows.append(dict(value))
            if len(self.rows) in (8,9):
                spread=pair_spread([r['end_to_end_step_seconds'] for r in self.rows[-2:]])
                self.decision=dict(pair_spread_pct=spread,threshold_pct=PAIR_GATE_PCT,stable=spread<=PAIR_GATE_PCT,
                    timed_r12_steps=[r['step'] for r in self.rows[-2:]],actual_updates=len(self.rows),
                    extra_third_long_update=len(self.rows)==9,all_samples_preserved=True)
                if len(self.rows)==8 and self.decision['stable']:self.args.stop_after_global_step=8
        return result
    def read(self,run_dir):
        if self.decision is None or len(self.rows)!=self.args.stop_after_global_step:raise ValueError('incomplete sampling boundary')
        rows=read_declared_segments(run_dir,list(range(1,len(self.rows)+1)),list(ROLLOUTS[:len(self.rows)]))
        keys=('step','optimizer_step','rollout_steps','end_to_end_step_seconds','step_seconds','data_load_seconds',
              'loss','grad_ok','global_batch_size','global_micro_batch_size','gradient_accumulation_steps','valid_sample_count',
              'inference_only','training_flops','learning_rate','grad_norm_before_clip','sample_indices','sample_provenance')
        if any(any(a.get(k)!=b.get(k) for k in keys) for a,b in zip(rows,self.rows)):
            raise ValueError('on-disk metrics differ from observed metrics')
        return rows
    def finish(self,**kw):
        step=kw['cursor'].global_optimizer_step
        if self.initial_load_count!=1 or self.decision is None or step!=len(self.rows) or step!=self.args.stop_after_global_step or kw['final']:
            raise ValueError('unexpected checkpoint boundary')
        if self.completion:self.completion(kw,self)
        return dict(path=None,sha256=None,checkpoint_written=False,reason='NP43 single-process diagnostic',
            actual_total_optimizer_step=step,actual_new_optimizer_updates=step,formal_mfu_eligible=False)
    def close(self):
        for name,fn in self.originals.items():setattr(self.runner,name,fn)

def install(runner,completion=None):return Controller(runner,completion)
