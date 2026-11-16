"""Host delivery entry: parse requests, prepare sessions, and monitor execution.

The host and container share delivery.py. Public arguments prepare a host
session; internal --inside loads saved context and dispatches the engine.
Top-level dependencies use only the stdlib so --list/--plan work before Docker.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from .catalog import checked_path, read, resolve_plan, sha256, verify_artifact
from .containers import ensure_container, exec_command, work_directory
from .devices import select, admit
from graphcast_bench.platforms.environment import delivery_environment
from graphcast_bench.progress import emit, phase
from .options import ALIASES, WORKLOADS, INFERENCE_DEFAULTS, TRAINING_DEFAULTS, prepare_resume


def choose(title, values):
    if not sys.stdin.isatty():
        raise ValueError("Non-interactive runs require --platform, --case and --workload")
    for index, value in enumerate(values, 1):
        print("{}: {}".format(index, value))
    index = int(input(title + ": ")) - 1
    if not 0 <= index < len(values):
        raise ValueError("Selection is out of range")
    return values[index]


def verify_source(release):
    manifest = read(release / "release.json")
    for name, expected in manifest["files"].items():
        if sha256(checked_path(release, name)) != expected:
            raise ValueError("Source SHA256 mismatch: " + name)
    return manifest


def inside_check(release):
    """Check imports in the pinned container Python without initializing GPUs."""
    import importlib.metadata as md
    import jax, haiku, optax, xarray, scipy  # noqa: F401

    # Never enumerate devices or initialize an accelerator in an environment check.
    result = dict(
        python=sys.version,
        versions={
            n: md.version(n) for n in ("jax", "jaxlib", "dm-haiku", "optax", "numpy", "xarray")
        },
        gpu_execution=False,
    )
    if os.environ.get("GRAPHCAST_DELIVERY_PLATFORM") == "h100":
        helper = release / "src/graphcast_bench/engines/operational_h100/clone3_compat_exec.py"
        subprocess.check_call(
            [
                sys.executable,
                str(helper),
                "--",
                sys.executable,
                "-c",
                'print("H100 process launcher passed")',
            ]
        )
        result["process_launcher"] = "passed"
    print(json.dumps(result, indent=2))
    return 0


def parse_arguments():
    """Parse public options and internal --inside callbacks from the container."""
    parser = argparse.ArgumentParser(description="GraphCast delivery launcher")
    parser.add_argument("--platform", choices=("bw1100", "h100"))
    parser.add_argument("--case", dest="case_id")
    parser.add_argument("--workload", choices=WORKLOADS,
                        help="infer/train: custom runs; infer-single/infer-full40/train-full38: fixed reproduction presets")
    inference = parser.add_argument_group("Inference controls (--workload infer only)")
    inference.add_argument("--steps", type=int, help="Measured forecast steps, 1..40 (default: 40)")
    inference.add_argument("--warmup-steps", type=int, help="Forecast steps per warmup, 1..40 (default: 5)")
    inference.add_argument("--repeats", type=int, help="Independent measured rounds (default: 3)")
    inference.add_argument("--warmup-scope", choices=("once", "per-repeat"),
                           help="once: warm up before the first round; per-repeat: warm up before every round (default)")
    training = parser.add_argument_group("Training controls (--workload train only)")
    training.add_argument("--end-stage", type=int, help="Last rollout stage, 1..12 (default: 12)")
    training.add_argument("--max-updates", type=int, help="Cumulative update limit including resumed updates, 1..38 (default: 38)")
    # Preserve an empty shell argument so validation can diagnose an unset variable.
    training.add_argument("--resume", help="Checkpoint file or one run directory under --work; a directory selects the latest valid checkpoint and resumes its updates/stages")
    training.add_argument("--save-every", type=int, help="Save every N cumulative optimizer updates; default: save at stage boundaries")
    training.add_argument("--warmup-iters", type=int, help="Warmup iterations per stage, at least 1 (default: 1); measured state is unchanged")
    parser.add_argument("--devices", nargs="+", type=int)
    parser.add_argument("--work", type=Path, default=Path.home() / "graphcast-work")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--verify-full", action="store_true",
                        help="Rehash all material files; default: reuse verification records for unchanged files")
    parser.add_argument("--inside", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--inside-check", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main(release=None):
    """Dispatch container callbacks directly; resolve host plans before setup."""
    release = Path(release or Path(__file__).resolve().parents[3]).resolve()
    args = parse_arguments()
    # docker exec re-enters delivery.py; these branches bypass host setup.
    if args.inside:
        from .runtime import run

        return run(read(args.inside))
    if args.inside_check:
        return inside_check(release)
    cases = [read(f) for f in sorted((release / "configs/cases").glob("*.json"))]
    if args.list:
        for case in cases:
            print(
                "{} | inference: 1 GPU / training: {} GPUs | {}".format(
                    case["id"], case["training_cards"], ", ".join(
                        ["infer", "train"] + [ALIASES.get(w, w) for w in case["workloads"]])
                )
            )
            if case["pending"]:
                print("  Pending: " + ", ".join(case["pending"]))
        return 0
    platform = args.platform or (
        "bw1100"
        if Path("/dev/kfd").exists()
        else "h100"
        if shutil.which("nvidia-smi")
        else choose("Select platform", ["bw1100", "h100"])
    )
    case_id = args.case_id or (
        cases[0]["id"]
        if len(cases) == 1
        else "operational-025deg-13l-b1"
        if args.check
        else choose("Select case", [c["id"] for c in cases])
    )
    case = next((c for c in cases if c["id"] == case_id), None)
    if case is None:
        raise ValueError("Unknown case: " + case_id)
    workload = args.workload or ("infer-single" if args.check else choose(
        "Select workload", ["infer", "train"] + [ALIASES.get(w, w) for w in case["workloads"]]))
    overrides = {key: getattr(args, key) for key in list(INFERENCE_DEFAULTS) + list(TRAINING_DEFAULTS)}
    plan = resolve_plan(release, platform, case_id, workload, overrides)
    plan["run_options"] = prepare_resume(plan["run_options"], args.work)
    if plan["kind"] == "training":
        plan["horizons"] = [stage["horizon"] for stage in plan["run_options"]["training"]["stages"]]
    if args.plan:
        # Plan mode returns before creating work directories, containers, or admitting devices.
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    return run_host(release, plan, args, workload)


def run_host(release, plan, args, workload):
    """Hold the environment lock across preparation, computation and log finalization."""
    bundle = release.parent.parent
    if release.parent.name != 'releases':
        raise ValueError('Launch from the delivery entry releases/<version>/scripts/run.sh')
    work = work_directory(bundle, args.work)
    lock_name = plan['artifacts']['environment'].replace('/', '-') + '.lock'
    with (work / lock_name).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This environment already has an active task in the selected work directory')
        if plan['kind'] == 'inference' and not args.check:
            from .host_log import host_log

            runs = work / 'runs' / release.name
            runs.mkdir(parents=True, exist_ok=True)
            output = Path(tempfile.mkdtemp(prefix=plan['case'] + '-' + workload + '-', dir=str(runs)))
            with host_log(output / 'host.log') as recording:
                print('Results: ' + str(output), flush=True)
                print('Session log: ' + str(output / 'host.log'), flush=True)
                print('Worker log: ' + str(output / 'console.log'), flush=True)
                recording.exit_code = run_session(release, plan, args, workload, work, output)
                return recording.exit_code
        return run_session(release, plan, args, workload, work)


def run_session(release, plan, args, workload, work, output=None):
    """Prepare materials/devices and launch under run_host's lock and optional log."""
    platform, case_id = plan['platform'], plan['case']
    bundle = release.parent.parent
    env_path = plan['artifacts']['environment']
    verification = dict(cache_dir=work / '.verification-cache', force=args.verify_full)
    with phase('Validation', 'Verify source and environment materials', heartbeat_seconds=30):
        source = verify_source(release)
        locks = read(release / 'configs/artifacts.lock.json')['artifacts']
        environment = verify_artifact(bundle, env_path, locks[env_path], **verification)
        if not args.check:
            for key, relative in plan["artifacts"].items():
                if key != "environment":
                    verify_artifact(bundle, relative, locks[relative], **verification)
    with phase('Container', 'Prepare the runtime environment', heartbeat_seconds=30):
        name = ensure_container(bundle, work, env_path, environment)
    # Mount bundle as /bundle (read-only) and work as /work (writable).
    # Context uses container paths; output keeps the host path for logs.
    runtime_release = "/bundle/" + str(release.relative_to(bundle))
    common_env = delivery_environment(runtime_release, platform)
    if args.check:
        # Check CPU imports through the same entry, then return before launching a worker.
        command = exec_command(
            name,
            environment["python"],
            runtime_release,
            ["--inside-check"],
            dict(common_env, JAX_PLATFORMS="cpu", XLA_FLAGS=""),
        )
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
        )
        print(result.stdout)
        record = dict(
            exit_code=result.returncode,
            container=name,
            image=environment["image"],
            environment=env_path,
            source_sha256=sha256(release / "release.json"),
            output=result.stdout,
            gpu_execution=False,
        )
        target = work / ("environment-check-" + env_path.split("/")[-1] + ".json")
        target.write_text(json.dumps(record, indent=2) + "\n")
        return result.returncode
    devices = select(platform, plan["cards"], args.devices)
    admission = admit(platform, devices)
    if plan['kind'] == 'inference' and platform == 'bw1100':
        values = plan['run_options']['inference']
        emit('Run', 'Inference | BW1100 GPU {} | steps={} | rounds={} | warmup_steps={} | warmup_scope={}'.format(
            devices[0]['index'], values['steps'], values['repeats'], values['warmup_steps'],
            'once' if values['warmup_scope'] == 'once' else 'per-repeat'))
        emit('Monitoring', 'Using host telemetry (utilization, VRAM); clocks are not sampled')
    if output is None:
        runs = work / "runs" / release.name
        runs.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix=case_id + "-" + workload + "-", dir=str(runs)))
    context = dict(
        plan,
        release=runtime_release,
        output="/work/" + str(output.relative_to(work)),
        python=environment["python"],
        devices=devices,
        owner_uid=os.getuid(),
        owner_gid=os.getgid(),
        build="/work/build/"
        + release.name
        + "-"
        + sha256(release / "release.json")[:12]
        + "-"
        + locks[env_path][:12],
    )
    for key, value in plan["artifacts"].items():
        if key != "environment":
            context[key] = "/bundle/" + value
    (output / "context.json").write_text(json.dumps(context, indent=2) + "\n")
    (output / "admission.json").write_text(json.dumps(admission, indent=2) + "\n")
    # Pass argv directly; Docker re-enters delivery.py with the pinned container Python.
    command = exec_command(
        name,
        environment["python"],
        runtime_release,
        ["--inside", context["output"] + "/context.json"],
        common_env,
    )
    (output / "launch.json").write_text(
        json.dumps(
            dict(command=command, release=source, environment_manifest_sha256=locks[env_path]),
            indent=2,
        )
        + "\n"
    )
    if plan['kind'] != 'inference':
        print("Results: " + str(output), flush=True)
        print("Worker log: " + str(output / "console.log"), flush=True)
    return execute_and_monitor(context, command, work, output, name, bundle=bundle)



def execute_and_monitor(context, command, work, output, name, bundle=None):
    """Run and monitor docker exec under the caller's lock; record exit status and clean up."""
    platform, devices = context["platform"], context["devices"]
    from .monitor import sample

    from .console import ConsoleRelay

    child, relay, log = None, None, None
    transient_pids = set()
    readable = context.get('kind') in ('inference', 'training')
    try:
        # Setup may take time; recheck device availability immediately before launch.
        admit(platform, devices)
        log = (output / "console.log").open("w", encoding='utf-8')
        child = subprocess.Popen(command, stdout=subprocess.PIPE if readable else log,
                                 stderr=subprocess.STDOUT)
        if readable:
            path_map = {'/work': str(work)}
            if bundle is not None:
                path_map['/bundle'] = str(bundle)
            relay = ConsoleRelay(child.stdout, log, path_map=path_map)
            relay.start()
        while child.poll() is None:
            if relay:
                relay.check()
            try:
                record = sample(context, output, name)
                if relay and isinstance(record, dict) and record.get('warning'):
                    relay.warn_once('monitor-unavailable', record['warning'])
                if relay and isinstance(record, dict) and record.get('transient_pids'):
                    if not transient_pids:
                        relay.note('Monitoring', 'Some sampled processes exited; disappearance was verified. See stage process-monitor.jsonl files under ' + str(output))
                    transient_pids.update(record['transient_pids'])
            except (OSError, ValueError) as exc:
                with (output / "monitor-errors.log").open("a") as errors:
                    errors.write(repr(exc) + "\n")
                if relay:
                    relay.warn_once('monitor-error', 'Some host telemetry is unavailable; see ' + str(output / 'monitor-errors.log'))
            time.sleep(1)
        code = child.wait()
        # Persist the process outcome before log draining or cleanup can fail.
        (output / "exit-code.txt").write_text(str(code) + "\n")
    except BaseException as exc:
        if child is not None and child.poll() is None:
            # This container belongs to the locked session; stop it before reaping the child.
            subprocess.call(["docker", "stop", "--time", "10", name])
        actual = child.wait() if child is not None else None
        if context["engine"] == "full38" or readable:
            (output / "host-abort.json").write_text(json.dumps(dict(
                error=repr(exc), status="aborted", docker_exec_exit_code=actual,
                stage_child_exit="Read stage exit-code.txt if present; otherwise unknown"), indent=2)+"\n")
            if actual is not None:
                (output / "exit-code.txt").write_text(str(actual)+"\n")
        if readable:
            emit('Failure', 'Run {}; actual child exit code: {}; full log: {}'.format(
                'interrupted' if isinstance(exc, KeyboardInterrupt) else 'aborted',
                actual, output / 'console.log'), 'WARN' if isinstance(exc, KeyboardInterrupt) else 'ERROR')
        raise
    finally:
        already_aborting = sys.exc_info()[0] is not None
        if readable:
            emit('Cleanup', 'Stopping owned container processes and finalizing outputs...')
        # Drivers/compilers require root; return ownership only for this session's output/build
        # trees without following symlinks. Run cleanup on success, failure, or interruption.
        owned = [context["output"]]
        if (work / Path(context["build"]).relative_to("/work")).exists():
            owned.append(context["build"])
        try:
            cleanup_code = cleanup_owned_container(name, owned)
            if readable:
                emit('Cleanup', 'Container stopped' if cleanup_code == 0 else 'Container stop failed; inspect: ' + name,
                     'INFO' if cleanup_code == 0 else 'WARN')
        finally:
            already_aborting = already_aborting or sys.exc_info()[0] is not None
            try:
                if relay:
                    try:
                        relay.finish()
                    except Exception as exc:
                        (output / 'log-relay-error.json').write_text(json.dumps(dict(
                            error=repr(exc), docker_exec_exit_code=child.returncode), indent=2) + '\n')
                        emit('Failure', 'Log finalization failed; see ' + str(output / 'log-relay-error.json'), 'ERROR')
                        if not already_aborting:
                            raise
            finally:
                if child is not None and child.stdout is not None:
                    child.stdout.close()
                if log is not None:
                    log.close()
    if readable:
        from .outcome import summarize_run

        summary = summarize_run(context, output, code)
        if code != 0:
            reason = relay.failure_reason if relay else None
            if reason:
                reason = relay.host_text(reason)
            emit('Failure', '{} | exit_code={} | diagnostics={}'.format(
                reason or 'Worker failed without a structured cause', code, output / 'console.log'), 'ERROR')
            if relay and not relay.has_direct_failure:
                try:
                    print(relay.host_text(''.join(list(relay.tail)[-8:])), end='', flush=True)
                except (OSError, ValueError):
                    pass
        monitoring = summary['monitoring']
        if summary['performance']['status'] == 'not_accepted':
            emit('Monitoring', 'Performance not accepted: {} | foreign_mapping_samples={} | warning_samples={} | summary={}'.format(
                summary['performance']['reason'], monitoring['foreign_mapping_samples'],
                monitoring['warning_samples'], output / 'run-summary.json'), 'WARN')
        emit('Complete', 'execution={} | accuracy={} | performance={} | exit_code={}'.format(
            summary['execution']['status'], summary['accuracy']['status'],
            summary['performance']['status'], code), 'INFO' if code == 0 else 'ERROR')
        if transient_pids:
            emit('Monitoring', 'Recorded {} exited processes during sampling; diagnostics are retained'.format(len(transient_pids)))
        emit('Results', 'Run summary: ' + str(output / 'run-summary.json'))
        reports = list(output.glob('*/run/report.json')) + list(output.glob('r[0-9][0-9]/report.json'))
        if (output / 'run/report.json').is_file():
            reports.append(output / 'run/report.json')
        for report in sorted(reports):
            emit('Results', 'Run report: ' + str(report))
    if code == 0:
        message = "Reproduction completed: " if context["run_options"]["preset"] else "Execution completed: "
    else:
        message = "Run failed; inspect: "
    try:
        print(message + str(output), flush=True)
    except (OSError, ValueError):
        pass
    return code


def cleanup_owned_container(name, owned):
    """Attempt stop even when diagnostic storage or ownership restoration fails."""
    cleanup_log = None
    try:
        cleanup_log = tempfile.TemporaryFile(mode='w+', encoding='utf-8')
    except OSError as exc:
        emit('Cleanup', 'Cleanup diagnostics unavailable: ' + str(exc), 'WARN')

    def execute(command):
        try:
            return subprocess.call(command, stdout=cleanup_log if cleanup_log is not None else subprocess.DEVNULL,
                                   stderr=subprocess.STDOUT)
        except OSError as exc:
            emit('Cleanup', 'Unable to run {}: {}'.format(' '.join(command), exc), 'WARN')
            return None

    ownership_code = cleanup_code = None
    try:
        try:
            ownership_code = execute(['docker', 'exec', name, 'chown', '-R', '-h',
                                      str(os.getuid()) + ':' + str(os.getgid())] + owned)
        finally:
            # Stop persistent compiler processes too; keep the container for reuse.
            cleanup_code = execute(['docker', 'stop', '--time', '10', name])
    finally:
        if cleanup_log is not None:
            try:
                if ownership_code != 0 or cleanup_code != 0:
                    cleanup_log.seek(0)
                    emit('Cleanup', 'Cleanup diagnostics: ' + cleanup_log.read().strip(), 'WARN')
            except OSError as exc:
                emit('Cleanup', 'Unable to read cleanup diagnostics: ' + str(exc), 'WARN')
            finally:
                try:
                    cleanup_log.close()
                except OSError as exc:
                    emit('Cleanup', 'Unable to finalize cleanup diagnostics: ' + str(exc), 'WARN')
    return cleanup_code
