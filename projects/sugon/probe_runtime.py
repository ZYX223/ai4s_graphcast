"""Record device identity and the actual mounted source before inference."""
import ctypes
import hashlib
import json
import os
import platform
from pathlib import Path

import torch
import triton
import flag_gems
import graphcast_compat.google_small as model

assert torch.version.hip and torch.cuda.is_available()
assert torch.cuda.device_count() == 1
bus = ctypes.create_string_buffer(64)
assert ctypes.CDLL("libamdhip64.so").hipDeviceGetPCIBusId(bus, len(bus), 0) == 0
assert bus.value.decode().lower() == os.environ["GRAPHCAST_EXPECTED_PCI"]
assert Path(model.__file__).resolve().is_relative_to("/workspace/ai4s/graphcast/src")
assert Path(flag_gems.__file__).resolve().is_relative_to("/opt/graphcast/source/flaggems")
assert (torch.arange(1024, device="cuda") + 1).sum().item() == 524800
torch.cuda.synchronize()
root = Path("/workspace")
sources = {}
for directory in (root / "ai4s/graphcast", root / "projects/sugon"):
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".sh", ".yaml", ".json", ".sha256"}:
            sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
result = {
    "python": platform.python_version(), "torch": torch.__version__,
    "hip": torch.version.hip, "triton": triton.__version__,
    "device_name": torch.cuda.get_device_name(0), "device_count": 1,
    "pci_bus": bus.value.decode(), "small_compute_passed": True,
    "graphcast_module": model.__file__, "flaggems_module": flag_gems.__file__,
    "source_sha256": sources,
    "compile": False, "profiler": False, "graph_capture": False,
}
Path("/outputs/runtime.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: value for key, value in result.items() if key != "source_sha256"}), flush=True)
