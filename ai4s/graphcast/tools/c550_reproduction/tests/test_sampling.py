import ast, importlib.util, json, sys, tempfile, types, unittest
from pathlib import Path
P=Path(__file__).resolve().parent
F=P.parents[2]
H=P.parent/'harness/short8'

def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def frozen():
    f=F/'scripts/training/run_official_epoch_training.py';tree=ast.parse(f.read_text())
    names={'resolve_training_schedule','_next_official_update_data_request'}
    ns=types.ModuleType('cpu_frozen')
    ns.__dict__.update(TrainingSchedule=load(F/'src/graphcast_compat/training_schedule.py','cpu_schedule').TrainingSchedule,
        OFFICIAL_2_TO_12_RECIPE='official_2_to_12',CURRICULUM_STAGE_COUNT=11,CURRICULUM_ROLLOUT_START=2,CURRICULUM_ROLLOUT_END=12)
    body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body,type_ignores=[])),str(f),'exec'),ns.__dict__)
    return ns

def row(step,r,seconds):
    return dict(step=step,optimizer_step=step,rollout_steps=r,end_to_end_step_seconds=seconds,loss=1.,grad_ok=True,
        global_batch_size=2,global_micro_batch_size=1,gradient_accumulation_steps=2,valid_sample_count=2,inference_only=False,
        training_flops=173142566854656*r)

class Sampling(unittest.TestCase):
    def setUp(self):
        self.assertTrue((H/'sampling_contract.py').is_file(),'single-process contract not implemented')
        self.api=load(H/'sampling_contract.py','sampling_contract')
        self.m=frozen();self.saved=[];self.loads=[]
        self.m.append_jsonl=lambda p,v:self.saved.append((p,v))
        self.m.read_metric_segments=lambda p:None
        self.m.load_resume_checkpoint=lambda *a,**k:None
        self.m.load_model_only_checkpoint=lambda *a,**k:self.loads.append(a) or ('state',{},'sha')
        self.m.save_validated_training_checkpoint=lambda **kw:self.fail('must not save checkpoint')
        self.m._official_update_data_request=lambda **kw:kw
        self.c=self.api.install(self.m)
        self.args=types.SimpleNamespace(schedule_recipe='official_2_to_12',warmup_updates=6,training_updates=10,finetune_updates=22,
            stop_after_global_step=9,checkpoint_every_steps=10000,resume_training_checkpoint='',global_batch_size=2,
            global_micro_batch_size=1,seed=27,profile_hotspots=False,gradient_sync='manual',data_parallel_replicas=1)
        self.schedule,_=self.m.resolve_training_schedule(self.args)
    def feed(self,last=(240,241)):
        for s,r,t in [(1,1,35),(2,1,20),(3,1,20.2),(4,6,150),(5,6,120),(6,6,121),(7,12,last[0]),(8,12,last[1])]:
            self.m.append_jsonl(Path('/run/metrics/segment_00000001.jsonl'),row(s,r,t))
    def test_schedule_keeps_actual_indices_and_original_r1_lr(self):
        v=[self.schedule.value_for_update(i) for i in range(9)]
        self.assertEqual([x.rollout_steps for x in v],[1,1,1,6,6,6,12,12,12])
        self.assertEqual([x.update_index for x in v],list(range(9)))
        # Literal values verified against the actual original R1 metric records.
        self.assertEqual([x.learning_rate for x in v[:3]],[0.,0.0002,0.0004])
        with self.assertRaises(ValueError):self.schedule.value_for_update(9)
    def test_two_stable_long_samples_stop_without_extra_update(self):
        self.feed();self.assertEqual(self.args.stop_after_global_step,8)
        self.assertEqual(self.c.decision['timed_r12_steps'],[7,8])
        self.assertEqual(len(self.saved),8)
    def test_unstable_long_samples_use_third_and_preserve_first(self):
        self.feed((270,240));self.assertEqual(self.args.stop_after_global_step,9)
        self.m.append_jsonl(Path('/run/metrics/segment_00000001.jsonl'),row(9,12,241))
        self.assertEqual(self.c.decision['timed_r12_steps'],[8,9]);self.assertTrue(self.c.decision['stable'])
        self.assertEqual(self.c.rows[6]['end_to_end_step_seconds'],270)
    def test_unstable_three_samples_do_not_claim_prediction(self):
        self.feed((270,240));self.m.append_jsonl(Path('/run/metrics/segment_00000001.jsonl'),row(9,12,280))
        self.assertFalse(self.c.decision['stable']);self.assertEqual(self.args.stop_after_global_step,9)
    def test_prefetch_is_fixed_horizon_before_adaptive_terminal_decision(self):
        kw=dict(current_epoch=0,current_batch_index=6,current_plan=types.SimpleNamespace(optimizer_steps=16),training_epoch_limit=3,
            planned_optimizer_updates=38,target_optimizer_updates=38,stop_after_global_step=9,
            dataset=types.SimpleNamespace(replacement_indices=lambda **k:(k['update_index'],k['target_rollout_steps'])),
            schedule=self.schedule,seed=27,global_batch_size=2,microbatch_plan=object(),rank=0,replicas=1)
        x=self.m._next_official_update_data_request(current_global_step=8,**kw)
        self.assertEqual(x['global_positions'],(8,12));self.assertTrue(x['allow_replacement'])
        self.feed();kw['stop_after_global_step']=self.args.stop_after_global_step
        self.assertIsNone(self.m._next_official_update_data_request(current_global_step=8,**kw))
    def test_only_one_initial_load_no_resume_no_checkpoint(self):
        self.m.load_model_only_checkpoint('/weights.pt');self.assertEqual(len(self.loads),1)
        with self.assertRaises(ValueError):self.m.load_model_only_checkpoint('/weights.pt')
        with self.assertRaises(ValueError):self.m.load_resume_checkpoint('/cp.pt')
        self.feed();record=self.m.save_validated_training_checkpoint(cursor=types.SimpleNamespace(global_optimizer_step=8),final=False)
        self.assertFalse(record['checkpoint_written'])
        with self.assertRaises(ValueError):self.m.save_validated_training_checkpoint(cursor=types.SimpleNamespace(global_optimizer_step=7),final=False)
    def test_bad_workload_or_metrics_rejected(self):
        for key,bad in [('global_batch_size',1),('profile_hotspots',True),('resume_training_checkpoint','/cp.pt')]:
            old=getattr(self.args,key);setattr(self.args,key,bad)
            with self.assertRaises(ValueError):self.m.resolve_training_schedule(self.args)
            setattr(self.args,key,old)
        for change in [dict(step=2),dict(rollout_steps=6),dict(grad_ok=False),dict(end_to_end_step_seconds=float('nan'))]:
            with self.assertRaises(ValueError):self.m.append_jsonl(Path('/run/metrics/segment_00000001.jsonl'),dict(row(1,1,30),**change))
    def test_restore_all_hooks(self):
        originals=dict(self.c.originals);self.c.close()
        for name,fn in originals.items():self.assertIs(getattr(self.m,name),fn)
    def test_postloop_enrichment_allowed_but_timing_drift_rejected(self):
        self.feed()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'metrics').mkdir();f=p/'metrics/segment_00000001.jsonl'
            rows=[dict(r,mfu_self_check=True,busy_profiled=False) for r in self.c.rows]
            f.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            self.assertEqual(len(self.m.read_metric_segments(p)),8)
            rows[-1]['end_to_end_step_seconds']+=1
            f.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError):self.m.read_metric_segments(p)

if __name__=='__main__':unittest.main(verbosity=2)
