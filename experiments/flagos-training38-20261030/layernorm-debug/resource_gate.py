"""Admission check before starting a device timing process."""
def require_idle(sample, card):
    import math
    try:
        row=sample['card%d'%card]
        utilization=float(row['HCU use (%)'])
        memory=float(row['vram Total Used Memory (MiB)'])
    except (KeyError,TypeError,ValueError):
        raise RuntimeError('Selected card usage is unavailable; do not start timing')
    if not (math.isfinite(utilization) and math.isfinite(memory)) or utilization!=0 or memory<0 or memory>512:
        raise RuntimeError('Card%d is occupied: utilization=%s%%, memory=%sMiB'%(card,utilization,memory))
    return True

if __name__=='__main__':
    import json,subprocess,sys
    card=int(sys.argv[1])
    raw=subprocess.check_output(['/opt/hyhal/bin/hy-smi','-d',str(card),'--showuse','--showmeminfo','vram','--json'])
    require_idle(json.loads(raw),card)
    print('Card%d idle at admission; this check is not a reservation'%card,flush=True)
