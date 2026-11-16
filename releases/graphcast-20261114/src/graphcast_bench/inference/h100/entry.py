"""Read explicit H100 context, install hooks, and retain audits on every exit."""
import argparse
from pathlib import Path

from graphcast_bench.assets import read_json, write_json
from graphcast_bench.platforms.context import context_arguments
from graphcast_bench.progress import emit, phase


def record_failure(output, error):
    try:
        write_json(output/'worker-error.json',dict(error=repr(error),type=type(error).__name__))
        path=output/'run/report.json'
        if path.is_file():
            report=read_json(path)
            report.update(status='failed',error=repr(error))
            write_json(path,report)
    except Exception as diagnostic:
        emit('Failure','Could not save failure diagnostic: '+repr(diagnostic),'WARN')


def run_with_audits(action, save_audits, output):
    try:
        result=action()
    except BaseException as primary:
        record_failure(output,primary)
        try:
            save_audits()
        except Exception as error:
            try:
                write_json(output/'optimization-audit-error.json',dict(
                    status='incomplete_after_inference_error',primary_error=repr(primary),audit_error=repr(error)))
            except Exception as diagnostic:
                emit('Failure','Could not save audit diagnostic: '+repr(diagnostic),'WARN')
        raise
    try:
        save_audits()
    except BaseException as error:
        record_failure(output,error)
        raise
    return result


def parse_arguments(argv, single):
    context, remaining=context_arguments(argv)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform',choices=('h100',),required=True)
    parser.add_argument('--mode',choices=('tf32',),required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--repeats',type=int)
    args=parser.parse_args(remaining)
    if not args.profile or args.data.resolve()!=context.path('assets').resolve() or args.output.resolve()!=(context.output/'run').resolve():
        raise ValueError('H100 command paths/profile differ from runtime context')
    if single != (context.config['recipe']=='single'):
        raise ValueError('H100 entry differs from selected recipe')
    if args.repeats is not None and args.repeats!=context.config['inference']['repeats']:
        raise ValueError('H100 command repeats differ from runtime context')
    return context


def validate_single(output):
    from graphcast_bench.data.inference import load_input_dataset
    from graphcast_bench.engines.operational_h100.frozen.src.graphcast_bench.metrics import compare
    from .provenance import FROZEN_SHA256

    with phase('Accuracy','Compare baseline and candidate at +6h'):
        baseline,candidate=[load_input_dataset(output/'run'/('round%d-tf32'%i)/'prediction.nc') for i in (1,2)]
        result=compare(candidate,baseline,lead_hours=6)
        result.update(reference='Same-process prior-best mesh2grid3 TF32 first +6h forecast',
            screening_threshold_percent=0.1,
            status='passed' if result['overall_mean_percent']<0.1 and result['z500_percent']<0.1 else 'failed',
            scope='One model step after independent 5-step warmup and input reset for each original/candidate; not full-rollout acceptance or MFU.',
            frozen_benchmark_sha256=FROZEN_SHA256)
        write_json(output/'single-step-validation.json',result)
        if result['status']!='passed':
            raise RuntimeError('Single-step numerical screening failed')
        emit('Accuracy','Single-step screening passed | overall relative MAE={:.6g}% | Z500={:.6g}%'.format(
            result['overall_mean_percent'],result['z500_percent']))


def main(argv=None, single=False):
    # Delivery selects the stage and serializes its context. Keep hook setup here;
    # runner owns independent forecast rounds and their measured time windows.
    context=parse_arguments(argv,single)
    try:
        with phase('Initialize','Import H100 dependencies and verify worker sources'):
            from . import runner
            from .hooks import HookPolicy
            from .provenance import source_files, write_manifest
            from graphcast_bench.validation.provenance import verify_maintained

            runner.validate_context(context)
            verify_maintained()
            manifest=write_manifest(context)
            policy=HookPolicy(context)

        def execute():
            report=runner.main(context,policy)
            if source_files()!=manifest['sources']:
                raise RuntimeError('H100 worker code changed during execution')
            return report

        run_with_audits(execute,lambda:policy.save_audits(context.output),context.output)
        if single:
            validate_single(context.output)
    except BaseException as error:
        record_failure(context.output,error)
        emit('Failure',type(error).__name__+': '+str(error),'ERROR')
        raise
    return 0
