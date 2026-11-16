"""Fail-closed admission; a quiet utilization sample alone is insufficient."""
import ast
import json
import re
import subprocess
import tempfile
from pathlib import Path
from graphcast_bench.progress import plain

KFD_PROC_ROOT = Path('/sys/class/kfd/kfd/proc')
KFD_TOPOLOGY_ROOT = Path('/sys/class/kfd/kfd/topology/nodes')
PROC_ROOT = Path('/proc')
DEVICES_CGROUP_ROOT = Path('/sys/fs/cgroup/devices')
DRM_ROOT = Path('/sys/class/drm')


def cgroup_gpu_scope(pid):
    """Read the process's enforced v1 device permissions and visible GPU IDs.

    A container may expose fewer than the host's 16 GPUs. Do not infer its
    scope from environment variables, Docker labels, or missing counters.
    Unsupported or unreadable policies leave the process unresolved.
    """
    try:
        membership = (PROC_ROOT / str(pid) / 'cgroup').read_text()
        groups = [line.split(':', 2)[2] for line in membership.splitlines()
                  if len(line.split(':', 2)) == 3 and
                  'devices' in line.split(':', 2)[1].split(',')]
        if len(groups) != 1 or not groups[0].startswith('/') or '..' in Path(groups[0]).parts:
            return None
        policy = (DEVICES_CGROUP_ROOT / groups[0].lstrip('/') / 'devices.list').resolve()
        policy.relative_to(DEVICES_CGROUP_ROOT.resolve())
        raw_rules = policy.read_text()
        rules = []
        for line in raw_rules.splitlines():
            match = re.fullmatch(r'([acb]) (\*|[0-9]+):(\*|[0-9]+) ([rwm]+)', line.strip())
            if not match:
                return None
            kind, major, minor, access = match.groups()
            if kind in ('a', 'c') and set(access) & set('rw'):
                rules.append((major, minor))
        allowed, host_ids = set(), set()
        for node in KFD_TOPOLOGY_ROOT.iterdir():
            gpu = int((node / 'gpu_id').read_text().strip())
            if gpu == 0:
                continue
            host_ids.add(gpu)
            properties = dict(line.split(None, 1) for line in (node / 'properties').read_text().splitlines())
            minor = int(properties['drm_render_minor'])
            major_text, minor_text = (DRM_ROOT / ('renderD' + str(minor)) / 'dev').read_text().strip().split(':')
            if int(minor_text) != minor or not major_text.isdigit():
                return None
            if any(a in ('*', major_text) and b in ('*', minor_text) for a, b in rules):
                allowed.add(gpu)
        if len(host_ids) != 16 or not allowed:
            return None
        return dict(source=str(policy), membership=membership, rules=raw_rules,
                    gpu_ids=sorted(allowed), host_gpu_ids=sorted(host_ids))
    except (OSError, ValueError, KeyError):
        return None

def parse_processes(raw):
    clean=re.sub(r'\x1b\[[0-9;]*m','',raw)
    result={'raw':raw,'known':False,'by_pid':{},'unknown_pids':[]}
    if 'End of SMI Log' not in clean:return result
    if re.search(r'error|failed|permission denied',clean,re.I):return result
    if re.search(r'(?m)^No KFD PIDs currently running!\s*$',clean):
        # This is a valid alternate SMI response, not a failed PID query.
        # Keep it untrusted until probe_processes also checks kernel state.
        if re.search(r'(?m)^PID:',clean):return result
        result['empty_smi_report']=True
        return result
    if 'PIDs for KFD processes:' not in clean:return result
    records=re.split(r'(?m)^PID:\s*',clean)[1:]
    try:
        for record in records:
            pid,body=record.split('\n',1)
            pid=int(pid.strip())
            match=re.search(r'(?m)^\s*HCU Index:[ \t]*([^\n]*)',body)
            value=match.group(1).strip() if match else ''
            indices=[]
            if value:
                chunks=re.findall(r'\[[^\[\]]*\]',value)
                if not chunks or re.sub(r'\[[^\[\]]*\]','',value).strip():
                    raise ValueError('Unexpected HCU mapping')
                for chunk in chunks:
                    group=ast.literal_eval(chunk)
                    if not isinstance(group,list):raise ValueError('Unexpected HCU mapping')
                    indices.extend(group)
            cards=[int(v) for v in indices]
            if any(c<0 or c>=16 for c in cards):raise ValueError('Unknown device index')
            result['by_pid'][str(pid)]=cards
            if not cards:result['unknown_pids'].append(pid)
        result['known']=True
    except (ValueError,TypeError,SyntaxError):pass
    return result

def process_disappeared(pid):
    """Require readable kernel roots and repeated ENOENT evidence for both PID views."""
    try:
        for _ in range(2):
            for root in (PROC_ROOT, KFD_PROC_ROOT):
                if not root.is_dir():
                    return False
                try:
                    (root / str(pid)).stat()
                except FileNotFoundError:
                    continue
                return False
        return True
    except OSError:
        return False


def process_diagnostics(stderr):
    vanished, unresolved = [], []
    for line in plain(stderr).splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(
            r'Unable to open queues directory for process ([0-9]+): No such file or directory',
            line.strip())
        if match and process_disappeared(int(match.group(1))):
            vanished.append(int(match.group(1)))
        else:
            unresolved.append(line)
    return sorted(set(vanished)), unresolved


def query_smi(arguments):
    """Keep driver stderr on success and include it in visible command failures."""
    command = ['/opt/hyhal/bin/hy-smi'] + arguments
    with tempfile.TemporaryFile() as errors:
        try:
            raw = subprocess.check_output(command, stderr=errors)
        except subprocess.CalledProcessError as exc:
            errors.seek(0)
            diagnostic = errors.read().decode('utf-8', errors='replace')
            failure = RuntimeError('hy-smi {} failed with exit code {}: {}'.format(
                ' '.join(arguments), exc.returncode, plain(diagnostic).strip()))
            failure.stderr = diagnostic
            raise failure from exc
        errors.seek(0)
        stderr = errors.read().decode('utf-8', errors='replace')
    return raw, stderr


def probe_processes():
    # Capture vendor stderr so sampling cannot flood the user's terminal.
    raw, stderr = query_smi(['--showpids'])
    result = parse_processes(raw.decode())
    vanished, unresolved = process_diagnostics(stderr)
    result.update(stderr=stderr, transient_pids=vanished, diagnostic_errors=unresolved)
    if unresolved:
        result['known'] = False
    result['kernel_no_device_evidence']={}
    if result.get('empty_smi_report'):
        try:
            snapshots=[sorted(p.name for p in KFD_PROC_ROOT.iterdir()) for _ in range(2)]
            result['kernel_empty_process_evidence']={'source':str(KFD_PROC_ROOT),'snapshots':snapshots}
            result['known']=not unresolved and all(not entries for entries in snapshots)
        except OSError as exc:
            result['kernel_empty_process_evidence']={'error':str(exc)}
        return result
    if result['known']:
        unresolved=[]
        for pid in result['unknown_pids']:
            evidence=verify_no_gpu_state(pid)
            if evidence is None:unresolved.append(pid)
            else:result['kernel_no_device_evidence'][str(pid)]=evidence
        result['unknown_pids']=unresolved
    return result

def verify_no_gpu_state(pid):
    """Resolve an empty SMI mapping only with complete, repeated kernel evidence.

    Opening KFD alone is not GPU ownership. Any queue, nonzero VRAM, missing
    counter, incomplete topology or read failure leaves the process unresolved.
    For a restricted container, require counters for its entire enforced device
    scope and verify that scope again around both kernel-state snapshots.
    This does not exempt a process that SMI mapped to any device.
    """
    try:
        gpu_ids={int((node/'gpu_id').read_text().strip()) for node in KFD_TOPOLOGY_ROOT.iterdir()}
        gpu_ids.discard(0)
        if len(gpu_ids)!=16:return None
        base=KFD_PROC_ROOT/str(pid)
        expected={'vram_'+str(gid) for gid in gpu_ids}
        scope = None
        if {p.name for p in base.glob('vram_[0-9]*')} != expected:
            scope = cgroup_gpu_scope(pid)
            if scope is None or set(scope['host_gpu_ids']) != gpu_ids:
                return None
            expected = {'vram_' + str(gid) for gid in scope['gpu_ids']}
            if {p.name for p in base.glob('vram_[0-9]*')} != expected:
                return None
        pasid=(base/'pasid').read_text().strip()
        if not pasid.isdigit():return None
        snapshots=[]
        for _ in range(2):
            if scope is not None and cgroup_gpu_scope(pid) != scope:return None
            if {p.name for p in base.glob('vram_[0-9]*')} != expected:return None
            if list((base/'queues').iterdir()):return None
            values={name:int((base/name).read_text().strip()) for name in sorted(expected)}
            if any(value!=0 for value in values.values()):return None
            if list((base/'queues').iterdir()):return None
            if (base/'pasid').read_text().strip()!=pasid:return None
            if scope is not None and cgroup_gpu_scope(pid) != scope:return None
            snapshots.append(values)
        evidence = {'source':str(base),'pasid':pasid,'queues':[],
                    'vram_bytes_snapshots':snapshots,'gpu_count':len(expected)}
        if scope is not None:evidence['device_scope'] = scope
        return evidence
    except (OSError,ValueError):
        return None

def require_idle(sample, card, processes=None):
    import math
    diagnostics = plain(processes.get('stderr', '')).strip() if processes else ''
    detail = '; driver diagnostics: ' + diagnostics if diagnostics else ''
    try:
        row=sample['card%d'%card]
        utilization=float(row['HCU use (%)'])
        memory=float(row['vram Total Used Memory (MiB)'])
    except (KeyError,TypeError,ValueError):
        raise RuntimeError('Selected card usage is unavailable; do not start timing')
    if not (math.isfinite(utilization) and math.isfinite(memory)) or utilization!=0 or memory<0 or memory>512:
        raise RuntimeError('Card%d is occupied: utilization=%s%%, memory=%sMiB'%(card,utilization,memory))
    if processes and processes.get('unknown_pids'):
        raise RuntimeError('Cannot establish GPU ownership for KFD PIDs {}; no job was started. '
                           'Retry after those processes settle, or inspect their KFD queues/device permissions.'
                           .format(processes['unknown_pids']) + detail)
    if not processes or not processes.get('known'):
        raise RuntimeError('KFD process/device mapping is unavailable or incomplete; do not start timing' + detail)
    if any(card in cards for cards in processes['by_pid'].values()):
        raise RuntimeError('Card%d has an existing KFD process; do not start timing'%card)
    return True

if __name__=='__main__':
    import json,subprocess,sys
    card=int(sys.argv[1])
    raw=subprocess.check_output(['/opt/hyhal/bin/hy-smi','-d',str(card),'--showuse','--showmeminfo','vram','--json'])
    require_idle(json.loads(raw),card,probe_processes())
    print('Card%d idle at admission; this check is not a reservation'%card,flush=True)
