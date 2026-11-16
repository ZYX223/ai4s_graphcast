"""Host sampling belongs to each active stage, alongside its actual exit code."""

import json
import subprocess
import time
from pathlib import Path
from .devices import bw_guard, inventory, admit


def append(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")


def sample(context, host_output, container):
    state = host_output / "active-stage.json"
    if not state.exists():
        return
    stage = json.loads(state.read_text())
    # Older adapters emitted path-only records. Normalize once so every branch
    # uses the same compatibility default, including fail-closed monitoring.
    status = stage.get("status", "running")
    if status not in ("running", "analyzing", "waiting"):
        return
    # A child cannot redirect host monitoring outside this fresh run directory.
    relative = Path(stage["path"]).relative_to(context["output"])
    target = (host_output / relative).resolve()
    if (
        target != host_output.resolve() and host_output.resolve() not in target.parents
    ) or not target.is_dir():
        return
    if status == 'waiting':
        admission = target / 'stage-admission.json'
        if not admission.exists():
            try:
                value = dict(status='passed', resource=admit(context['platform'], context['devices']))
            except Exception as exc:
                value = dict(status='rejected', error=repr(exc))
            temporary = admission.with_suffix('.tmp')
            temporary.write_text(json.dumps(value, indent=2)+'\n')
            temporary.replace(admission)
        return
    guard_issue = None
    stamp = int(time.time() * 1e9)
    record = dict(time_ns=stamp)
    try:
        raw_pids = subprocess.check_output(
            ["docker", "top", container, "-eo", "pid"], universal_newlines=True
        )
        if context["platform"] == "bw1100":
            raw_usage, diagnostic = bw_guard().query_smi(['--showuse', '--showmeminfo', 'vram', '--json'])
            if diagnostic.strip():
                record['stderr'] = diagnostic
                record['warning'] = 'Host utilization query reported diagnostics; see ' + str(target / 'host-memory.jsonl')
            raw = json.loads(raw_usage)
            selected = {str(d["index"]): raw["card" + str(d["index"])] for d in context["devices"]}
            record.update(devices=selected, value={"card" + k: v for k, v in selected.items()})
            own = [int(x) for x in raw_pids.splitlines()[1:] if x.strip().isdigit()]
            processes = bw_guard().probe_processes()
            record['transient_pids'] = processes.get('transient_pids', [])
            if context['engine'] == 'full38':
                # GPU queries may observe children started after the first top.
                refreshed = subprocess.check_output(['docker','top',container,'-eo','pid'], universal_newlines=True)
                own = sorted(set(own) | {int(x) for x in refreshed.splitlines()[1:] if x.strip().isdigit()})
            foreign = {
                str(d["index"]): [
                    int(pid)
                    for pid, cards in processes["by_pid"].items()
                    if d["index"] in cards and int(pid) not in own
                ]
                for d in context["devices"]
            }
            if not processes['known'] or processes['unknown_pids'] or any(foreign.values()):
                guard_issue = 'Unknown or foreign process ownership on selected devices'
                record['warning'] = 'Host monitoring could not fully establish device process ownership; see ' + str(target / 'process-monitor.jsonl')
            append(
                target / "process-monitor.jsonl",
                dict(
                    time_ns=stamp,
                    processes=processes,
                    container_pids=own,
                    foreign_card_pids=foreign,
                ),
            )
        else:
            devices = {r["uuid"]: r for r in inventory("h100")}
            selected = {
                d["uuid"]: {
                    "memory_mib": devices[d["uuid"]]["memory_mib"],
                    "utilization": devices[d["uuid"]]["utilization"],
                }
                for d in context["devices"]
            }
            apps = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid",
                    "--format=csv,noheader,nounits",
                ],
                universal_newlines=True,
            )
            own = {int(x) for x in raw_pids.splitlines()[1:] if x.strip().isdigit()}
            if context['engine'] == 'full38':
                refreshed = subprocess.check_output(['docker','top',container,'-eo','pid'], universal_newlines=True)
                own |= {int(x) for x in refreshed.splitlines()[1:] if x.strip().isdigit()}
                raw_pids = 'PID\n' + '\n'.join(str(x) for x in sorted(own))
            indices = {d['uuid']: d.get('index', d['uuid']) for d in context['devices']}
            by_pid = {}
            foreign = {str(index): [] for index in indices.values()}
            for line in apps.splitlines():
                fields = [x.strip() for x in line.split(',')]
                if fields and fields[0] in selected:
                    pid, index = int(fields[1]), indices[fields[0]]
                    by_pid.setdefault(str(pid), []).append(index)
                    if pid not in own:
                        foreign[str(index)].append(pid)
                        guard_issue = 'Foreign GPU process on selected devices'
            if guard_issue:
                record['warning'] = guard_issue + '; see ' + str(target / 'process-monitor.jsonl')
            append(target / 'process-monitor.jsonl', dict(
                time_ns=stamp, processes=dict(known=True, unknown_pids=[], by_pid=by_pid),
                container_pids=sorted(own), foreign_card_pids=foreign,
            ))
            record.update(
                devices=selected,
                value=max(v["memory_mib"] for v in selected.values()),
                compute_apps=apps,
                container_pids=raw_pids,
            )
    except Exception as exc:
        record["error"] = repr(exc)
        record['warning'] = 'Some host telemetry is unavailable; see ' + str(target / 'host-memory.jsonl')
        diagnostic = getattr(exc, 'stderr', None)
        if diagnostic:
            diagnostic = diagnostic.decode(errors='replace') if isinstance(diagnostic, bytes) else diagnostic
            record['stderr'] = record.get('stderr', '') + diagnostic
    append(target / "host-memory.jsonl", record)
    if context['engine'] == 'full38' and status == 'running':
        # Abort only this owned delivery container; never signal foreign jobs.
        if 'error' in record:
            guard_issue = 'Resource monitor failed: ' + record['error']
        if guard_issue:
            (target/'resource-abort.json').write_text(json.dumps(dict(
                time_ns=stamp, reason=guard_issue, action='stop owned delivery container'),indent=2)+'\n')
            raise RuntimeError(guard_issue)
    return record
