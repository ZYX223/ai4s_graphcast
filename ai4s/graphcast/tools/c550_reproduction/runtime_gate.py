"""Recheck a fresh host attestation and actual single-device PCI before tensors."""
import ctypes
import json
import os
from pathlib import Path

from reproduce import read, sha, validate_gate, verify_job


def check_runtime(plan):
    job = Path(__file__).resolve().parent
    meta = read(job / 'job.json')
    gate = read(job / 'resource-gate.json')
    validate_gate(gate, meta['device'], verify_job(job))
    if sha(job / 'resource-evidence.txt') != gate['raw_evidence_sha256']:
        raise ValueError('host evidence drift')
    device = meta['device']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != device['visible_device']:
        raise ValueError('visibility drift')
    lib = ctypes.CDLL(device['runtime_library'])
    lib.mcGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.mcGetDeviceCount.restype = ctypes.c_int
    lib.mcDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.mcDeviceGetPCIBusId.restype = ctypes.c_int
    count, bus = ctypes.c_int(), ctypes.create_string_buffer(64)
    if lib.mcGetDeviceCount(ctypes.byref(count)) != 0 or count.value != 1:
        raise RuntimeError('expected exactly one visible C550 device')
    if lib.mcDeviceGetPCIBusId(bus, len(bus), 0) != 0:
        raise RuntimeError('PCI query failed')
    if bus.value.decode().lower() != device['pci_bus'].lower():
        raise RuntimeError('runtime PCI differs from the host resource gate')
    (Path(plan['output_dir']) / 'runtime-placement.json').write_text(json.dumps(
        dict(count=count.value, pci=bus.value.decode(), physical_gpu=device['physical_gpu'])))
