"""Independent conversion checks and native JAX GEMM routing checks."""

import ctypes as C
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

base = Path(os.environ["GRAPHCAST_ROUTER_LIBRARY"]).parent
out = Path(sys.argv[1])
out.mkdir(exist_ok=False)
lib = Path(os.environ["GRAPHCAST_ROUTER_LIBRARY"])
router = C.CDLL(str(lib))
hip = C.CDLL("libamdhip64.so")
hip.hipMalloc.argtypes = [C.POINTER(C.c_void_p), C.c_size_t]
hip.hipMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
router.graphcast_round_into.argtypes = [
    C.c_void_p,
    C.c_void_p,
    C.c_int,
    C.c_int,
    C.c_int,
    C.c_void_p,
]
router.graphcast_stage_select.argtypes = [C.c_uint64, C.c_char_p]
router.graphcast_round_select.argtypes = [C.c_uint64]
router.graphcast_stage_calls.restype = C.c_uint64
router.graphcast_round_calls.restype = C.c_uint64
router.graphcast_stage_unknown.restype = C.c_uint64


def oracle(x):
    """Integer quotient/remainder rule, independent from GPU add-mask formula."""
    u = x.view(np.uint32).copy()
    finite = (u & 0x7F800000) != 0x7F800000
    q = u >> 13
    remainder = u & 8191
    up = (remainder > 4096) | ((remainder == 4096) & ((q & 1) == 1))
    rounded = ((q + up.astype(np.uint32)) << 13).astype(np.uint32)
    return np.where(finite, rounded, u).astype(np.uint32).view(np.float32)


rng = np.random.default_rng(20261016)
edge = np.array(
    [
        0,
        0x80000000,
        1,
        0x80000001,
        0x7F800000,
        0xFF800000,
        0x7FC01234,
        0xFF801234,
        0x7F7FFFFF,
        0xFF7FFFFF,
        0x007FFFFF,
    ],
    dtype=np.uint32,
)
# Midpoints, retained-even/odd values, both signs, and multiple exponents.
ties = np.array(
    [
        (sign | (exp << 23) | (mantissa << 13) | rem)
        for sign in (0, 0x80000000)
        for exp in (0, 1, 100, 127, 200, 254)
        for mantissa in (0, 1, 2, 511, 512, 1023)
        for rem in (4095, 4096, 4097)
    ],
    dtype=np.uint32,
)
bits = np.concatenate([edge, ties, rng.integers(0, 2**32, size=100000, dtype=np.uint32)])
x = bits.view(np.float32)
conversion = []
for rows, cols, ld in [(len(x), 1, len(x)), (37, 53, 41)]:
    span = (cols - 1) * ld + rows
    src = np.resize(x, span).copy()
    dst = np.full(span, -12345, dtype=np.float32)
    a = C.c_void_p()
    b = C.c_void_p()
    assert hip.hipMalloc(C.byref(a), src.nbytes) == 0
    assert hip.hipMalloc(C.byref(b), dst.nbytes) == 0
    assert hip.hipMemcpy(a, src.ctypes.data, src.nbytes, 1) == 0
    assert hip.hipMemcpy(b, dst.ctypes.data, dst.nbytes, 1) == 0
    assert router.graphcast_round_into(a, b, rows, cols, ld, None) == 0
    assert hip.hipDeviceSynchronize() == 0
    assert hip.hipMemcpy(dst.ctypes.data, b, dst.nbytes, 2) == 0
    indices = np.concatenate([np.arange(rows) + col * ld for col in range(cols)])
    expected = np.full(span, -12345, dtype=np.float32)
    expected[indices] = oracle(src[indices])
    assert np.array_equal(dst.view(np.uint32), expected.view(np.uint32))
    untouched = src.copy()
    assert hip.hipMemcpy(untouched.ctypes.data, a, src.nbytes, 2) == 0
    assert np.array_equal(untouched.view(np.uint32), src.view(np.uint32))
    assert hip.hipFree(a) == 0 and hip.hipFree(b) == 0
    conversion.append({"rows": rows, "cols": cols, "ld": ld, "passed": True})

activations = [rng.normal(size=(m, 512)).astype(np.float32) for m in (512, 1024, 1536)]
weights = [rng.normal(size=(512, 512)).astype(np.float32) for _ in range(3)]
rounded = [oracle(x) for x in (*activations, *weights)]
mapping = []
for bit, w in enumerate(weights):
    for form in (w, w.T, oracle(w), oracle(w).T):
        mapping.append(
            f"{w.nbytes} {hashlib.sha256(np.ascontiguousarray(form).tobytes()).hexdigest()} {bit} probe_{bit}"
        )
(out / "weights.tsv").write_text("\n".join(mapping) + "\n")
os.environ["GRAPHCAST_STAGE_WEIGHTS"] = str(out / "weights.tsv")
os.environ["GRAPHCAST_STAGE_LOG"] = str(out / "router.jsonl")
# The native router must receive the probe's weight map before JAX initializes.
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

assert len(jax.devices("gpu")) == 1
bus = C.create_string_buffer(64)
assert hip.hipDeviceGetPCIBusId(bus, len(bus), 0) == 0
assert bus.value.decode().lower() == os.environ["GRAPHCAST_EXPECTED_PCI"]
assert os.environ.get("ROCBLAS_MATH_MODE") is None
fn = jax.jit(lambda *args: tuple(jnp.matmul(args[i], args[i + 3]) for i in range(3)))
placed = jax.device_put((*activations, *weights))
pre = jax.device_put(tuple(rounded))
outputs = {}
metrics = []
policies = [
    ("fp32", 0, 0, placed),
    ("xf32", 7, 0, placed),
    ("rne", 7, 7, placed),
    ("explicit_rne_xf32", 7, 0, pre),
    ("mixed", 7, 2, placed),
    ("fp32_repeat", 0, 0, placed),
    ("rne_repeat", 7, 7, placed),
]
for label, mask, rnd, values in policies:
    router.graphcast_stage_select(mask, label.encode())
    router.graphcast_round_select(rnd)
    before = router.graphcast_stage_calls()
    before_round = router.graphcast_round_calls()
    result = fn(*values)
    jax.block_until_ready(result)
    assert router.graphcast_stage_calls() - before == 3
    assert router.graphcast_round_calls() - before_round == bin(rnd).count("1")
    outputs[label] = [np.asarray(v) for v in result]
    for i, y in enumerate(outputs[label]):
        truth = activations[i].astype(np.float64) @ weights[i].astype(np.float64)
        metrics.append(
            {"policy": label, "matrix": i, "mae_vs_fp64": float(np.mean(np.abs(y - truth)))}
        )
for i in range(3):
    assert np.array_equal(outputs["rne"][i], outputs["explicit_rne_xf32"][i])
    assert np.array_equal(outputs["rne"][i], outputs["rne_repeat"][i])
    assert np.array_equal(outputs["mixed"][i], outputs["rne" if i == 1 else "xf32"][i])
    assert np.array_equal(outputs["fp32"][i], outputs["fp32_repeat"][i])
    assert not np.array_equal(outputs["rne"][i], outputs["xf32"][i])
for old, new in zip((*activations, *weights), placed):
    assert np.array_equal(old, np.asarray(new))
trace = [json.loads(line) for line in (out / "router.jsonl").read_text().splitlines()]
assert len(trace) == 21 and router.graphcast_stage_unknown() == 0
for label, mask, rnd, _ in policies:
    rows = [r for r in trace if r["label"] == label]
    assert len(rows) == 3
    assert all(
        r["mode"] == int(bool(mask & (1 << r["bit"])))
        and r["rne_inputs"] == bool(rnd & (1 << r["bit"]))
        and r["previous_mode"] == 0
        and r["status"] == 0
        for r in rows
    )
report = {
    "status": "passed",
    "conversion": conversion,
    "checks": metrics,
    "calls": 21,
    "pre_rounded_native_xf32_match_bitwise": True,
    "inputs_unchanged": True,
    "fp32_restoration_bitwise": True,
    "pci": bus.value.decode(),
    "router_sha256": hashlib.sha256(lib.read_bytes()).hexdigest(),
}
(out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
