"""Bind frozen training and summarize a bounded diagnostic without changing its math."""
import hashlib, importlib.util, json, runpy, sys
from pathlib import Path
from sampling_contract import install, ROLLOUTS
STATE={}

def bind(plan):
    runner_path=Path(plan['model'])/'scripts/training/run_official_epoch_training.py'
    original=runpy.run_path
    def dispatch(path,*a,**kw):
        if Path(path).resolve()!=runner_path.resolve():return original(path,*a,**kw)
        if STATE:raise RuntimeError('runner may execute only once')
        digest=hashlib.sha256(runner_path.read_bytes()).hexdigest()
        if digest!=plan['runner_sha256']:raise RuntimeError('frozen runner drift')
        spec=importlib.util.spec_from_file_location('np43_frozen_runner',runner_path)
        runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner)
        out=Path(plan['output_dir'])
        def completion(kw,ctl):
            import torch
            model,optimizer=kw['model'],kw['optimizer'];params=list(model.parameters());states=optimizer.state_dict()['state']
            if not all(bool(torch.isfinite(p.detach()).all()) for p in params):raise FloatingPointError('nonfinite parameter')
            if not all(bool(torch.isfinite(v).all()) for r in states.values() for v in r.values() if isinstance(v,torch.Tensor)):
                raise FloatingPointError('nonfinite optimizer')
            steps=sorted({float(r['step']) for r in states.values()});end=len(ctl.rows)
            if len(params)!=264 or len(states)!=256 or steps!=[float(end)]:raise RuntimeError('wrong update counts')
            final,hashes=runner.validated_replicated_model_hashes(kw['context'],model)
            (out/'final-state.json').write_text(json.dumps(dict(parameter_count=264,optimizer_state_count=256,optimizer_steps=steps,
                all_parameters_finite=True,all_optimizer_finite=True,final_model_hash=final,rank_hashes=hashes,
                initial_lineage=kw['initial_lineage'],checkpoint_written=False,initial_load_count=ctl.initial_load_count,
                resume_load_count=0,decision=ctl.decision,measured_outside_update_e2e=True),indent=2))
        ctl=install(runner,completion);STATE.update(controller=ctl,plan=plan,finalized=False)
        (out/'sampling-contract.json').write_text(json.dumps(dict(scope=plan['comparison_scope'],
            runner_sha256=digest,actual_step_budget=list(range(1,10)),rollout_budget=list(ROLLOUTS),
            min_updates=8,max_updates=9,pair_gate_pct=5.,r1_learning_rates=[0.,0.0002,0.0004],long_r_learning_rate=3e-7,
            init='original model-only checkpoint, one load',checkpoint_sha256=plan['checkpoint_sha256'],
            input_selection='unchanged replacement_indices with actual zero-based update indices; never pretend late-course state',
            r12_rule='after update8, stop if pair7,8 spread<=5%; otherwise execute9; use pair8,9 and reject prediction if unstable',
            r1_timed=[2,3],r6_timed=[5,6],discarded_priming=[1,4],all_costs_counted=True,
            prefetch='unchanged depth1; optional update9 data may be prefetched before stop, its cost stays in wall time',
            no_new_checkpoint=True,no_resume=True,formal_mfu_eligible=False,train_body_modified=False),indent=2))
        try:raise SystemExit(runner._main())
        finally:ctl.close();runpy.run_path=original
    runpy.run_path=dispatch

def finalize():
    if not STATE or STATE['finalized']:return
    plan=STATE['plan'];out=Path(plan['output_dir']);path=out/'run/summary.json'
    if not path.exists():return
    ctl=STATE['controller'];summary=json.loads(path.read_text());rows=summary['metrics'];end=len(ctl.rows)
    if summary['status']!='STOPPED' or summary['completed_optimizer_steps']!=end or len(rows)!=end:
        raise RuntimeError('incorrect diagnostic stop')
    if list(out.rglob('*.pt')) or not (out/'final-state.json').is_file():raise RuntimeError('checkpoint or final-state violation')
    summary['parent_schedule_digest']=summary.pop('schedule_digest')
    summary.update(np43_diagnostic=True,actual_new_optimizer_updates=end,formal_mfu_eligible=False,
        sampling_contract='sampling-contract.json',checkpoint_written=False)
    path.write_text(json.dumps(summary,indent=2))
    (out/'sparse-window.json').write_text(json.dumps(dict(all_metrics=rows,decision=ctl.decision,
        actual_new_optimizer_updates=end,actual_cursor=summary['cursor'],initial_load_count=ctl.initial_load_count,
        validated_mfu_prediction=None,full_course_mfu=None,state_transfer_validated=False),indent=2))
    from prediction import evaluate
    (out/'prediction.json').write_text(json.dumps(evaluate(rows),indent=2))
    STATE['finalized']=True
