"""Replay logical JAX operators from the frozen forward graph, not GPU kernels."""

import gc
import hashlib
import json
import statistics
import time
from importlib import metadata
from pathlib import Path

import numpy as np

from .assets import ROOT, code_snapshot, load_case, read_json, sha256, verify_source, write_json


def rne_numpy(x):
    x = np.asarray(x, np.float32)
    bits = x.view(np.uint32)
    q, rem = bits >> 13, bits & 8191
    q = q + ((rem > 4096) | ((rem == 4096) & ((q & 1) != 0))).astype(np.uint32)
    rounded = (q << 13).astype(np.uint32)
    return np.where((bits & 0x7F800000) == 0x7F800000, bits, rounded).view(np.float32)


def decode_parameter(v):
    import jax

    if isinstance(v, list):
        return tuple(decode_parameter(x) for x in v)
    if isinstance(v, dict):
        if "dtype" in v:
            return np.dtype(v["dtype"])
        if "enum" in v:
            return getattr(getattr(jax.lax, v["enum"]), v["name"])
        if "namedtuple" in v:
            return getattr(jax.lax, v["namedtuple"])(
                **{k: decode_parameter(x) for k, x in v["fields"].items()}
            )
        if "repr" in v:
            raise ValueError("Unsupported opaque parameter: " + str(v))
        return {k: decode_parameter(x) for k, x in v.items()}
    return v


def primitive_function(name, params):
    import jax

    if name == "scatter-add":
        p = {
            k: decode_parameter(v)
            for k, v in params.items()
            if k not in ("update_jaxpr", "update_consts")
        }
        return lambda *args: jax.lax.scatter_add(*args, **p)
    p = decode_parameter(params)
    primitive = getattr(jax.lax, name.replace("-", "_") + "_p")
    return lambda *args: primitive.bind(*args, **p)


def sample_values(x, coordinates):
    """Gather only requested values from device; no full activation CPU copy."""
    coordinates = np.asarray(coordinates, np.int64)
    if not x.shape:
        return np.full(len(coordinates), float(np.asarray(x)), np.float64)
    # Per-axis indices stay int32-safe even for tensors with >2**31 elements.
    # JAX x64 is disabled in the pinned model environment.
    return np.asarray(x[tuple(coordinates[:, i] for i in range(len(x.shape)))], dtype=np.float64)


def broadcast_coordinates(coordinates, shape):
    if not shape:
        return np.zeros((len(coordinates), 0), np.int64)
    c = coordinates[:, -len(shape) :].copy()
    for i, s in enumerate(shape):
        if s == 1:
            c[:, i] = 0
    return c


def reference_samples(kind, params, args, out_shape, sample):
    """Independent NumPy FP64/scalar-index references for selected output elements."""
    sample = np.asarray(sample, np.int64)
    coords = (
        np.array(np.unravel_index(sample, out_shape)).T
        if out_shape
        else np.zeros((len(sample), 0), np.int64)
    )
    if kind in ("add", "sub", "mul", "div"):
        a, b = [sample_values(x, broadcast_coordinates(coords, x.shape)) for x in args]
        return {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide}[kind](a, b)
    if kind == "silu":
        x = sample_values(args[0], coords)
        return x / (1 + np.exp(-x))
    if kind == "linear":
        x, w, b = args
        k = w.shape[0]
        xc = np.repeat(coords, len(range(k)), axis=0)
        xc[:, -1] = np.tile(np.arange(k), len(coords))
        wc = np.stack([np.tile(np.arange(k), len(coords)), np.repeat(coords[:, -1], k)], axis=1)
        a = rne_numpy(sample_values(x, xc)).astype(np.float64).reshape(-1, k)
        ww = rne_numpy(sample_values(w, wc)).astype(np.float64).reshape(-1, k)
        return (a * ww).sum(axis=1) + sample_values(b, coords[:, -1:])
    if kind == "layer_norm":
        x = args[0]
        channels = x.shape[-1]
        c = np.repeat(coords, channels, axis=0)
        c[:, -1] = np.tile(np.arange(channels), len(coords))
        xx = sample_values(x, c).reshape(-1, channels)
        scale, offset = args[-2:]
        # Official hk.LayerNorm uses eps=1e-5 and population variance.
        return (xx[np.arange(len(coords)), coords[:, -1]] - xx.mean(axis=1)) / np.sqrt(
            xx.var(axis=1) + 1e-5
        ) * sample_values(scale, coords[:, -1:]) + sample_values(offset, coords[:, -1:])
    if kind == "broadcast_in_dim":
        c = coords[:, params["broadcast_dimensions"]].copy()
        for i, s in enumerate(args[0].shape):
            if s == 1:
                c[:, i] = 0
        return sample_values(args[0], c)
    if kind in ("reshape", "squeeze"):
        dims = params.get("dimensions") if kind == "reshape" else None
        if dims is None:
            return sample_values(args[0], np.array(np.unravel_index(sample, args[0].shape)).T)
        perm_shape = tuple(args[0].shape[d] for d in dims)
        pc = np.array(np.unravel_index(sample, perm_shape)).T
        return sample_values(args[0], pc[:, np.argsort(dims)])
    if kind == "transpose":
        return sample_values(args[0], coords[:, np.argsort(params["permutation"])])
    if kind == "slice":
        stride = np.asarray(params["strides"] or [1] * len(out_shape))
        return sample_values(args[0], coords * stride + np.asarray(params["start_indices"]))
    if kind == "dynamic_slice":
        starts = [
            min(max(int(np.asarray(v)), 0), d - s)
            for v, d, s in zip(args[1:], args[0].shape, params["slice_sizes"])
        ]
        return sample_values(args[0], coords + np.array(starts))
    if kind == "concatenate":
        dim = params["dimension"]
        result = np.empty(len(coords))
        begin = 0
        for x in args:
            mask = (coords[:, dim] >= begin) & (coords[:, dim] < begin + x.shape[dim])
            c = coords[mask].copy()
            c[:, dim] -= begin
            if mask.any():
                result[mask] = sample_values(x, c)
            begin += x.shape[dim]
        return result
    if kind == "gather":
        d = params["dimension_numbers"]["fields"]
        if d.get("operand_batching_dims") or d.get("start_indices_batching_dims"):
            raise ValueError("Batched gather reference not implemented")
        offsets = d["offset_dims"]
        batch = [i for i in range(len(out_shape)) if i not in offsets]
        idx = np.asarray(args[1])
        starts = (
            idx[tuple(coords[:, batch].T)]
            if batch
            else np.broadcast_to(idx, (len(coords), idx.shape[-1]))
        )
        operand = np.zeros((len(coords), args[0].ndim), np.int64)
        for j, axis in enumerate(d["start_index_map"]):
            operand[:, axis] = starts[:, j]
        window = [i for i in range(args[0].ndim) if i not in d["collapsed_slice_dims"]]
        for od, axis in zip(offsets, window):
            operand[:, axis] += coords[:, od]
        return sample_values(args[0], operand)
    if kind == "scatter-add":
        d = params["dimension_numbers"]["fields"]
        if d["scatter_dims_to_operand_dims"] != [0] or d["inserted_window_dims"] != [0]:
            raise ValueError("Unexpected scatter topology")
        indices = np.asarray(args[1]).reshape(-1)
        chunks = []
        lengths = []
        for c in coords:
            edges = np.flatnonzero(indices == c[0])
            lengths.append(len(edges))
            ec = np.tile(c, (len(edges), 1))
            ec[:, 0] = edges
            chunks.append(ec)
        values = sample_values(args[2], np.concatenate(chunks)) if sum(lengths) else np.empty(0)
        result = sample_values(args[0], coords)
        offset = 0
        for i, n in enumerate(lengths):
            result[i] += values[offset : offset + n].sum(dtype=np.float64)
            offset += n
        return result
    raise ValueError("Missing independent reference for " + kind)


class Inventory:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.data = read_json(self.folder / "inventory.json")
        self.variables = self.data["variables"]
        self.equations = {r["id"]: r for r in self.data["equations"]}
        self.producers = {v: r for r in self.data["equations"] for v in r["outputs"]}
        self.archive = np.load(self.folder / "constants.npz", allow_pickle=False)
        self.constants = {}

    def constant(self, key):
        if key in self.constants:
            return self.constants[key]
        if key in self.archive.files:
            self.constants[key] = self.archive[key]
            return self.constants[key]
        e = self.producers.get(key)
        if e is None or np.prod(self.variables[key]["shape"], dtype=np.int64) > 10000000:
            return None
        kind = e["primitive"]
        if kind not in (
            "device_put",
            "broadcast_in_dim",
            "reshape",
            "squeeze",
            "transpose",
            "slice",
            "add",
            "sub",
            "mul",
            "div",
            "lt",
            "select_n",
            "convert_element_type",
        ):
            return None
        xs = [self.constant(v) for v in e["inputs"]]
        if any(x is None for x in xs):
            return None
        p = e["params"]
        if kind == "device_put":
            out = xs[0]
        elif kind == "broadcast_in_dim":
            shape = [1] * len(p["shape"])
            for i, d in enumerate(p["broadcast_dimensions"]):
                shape[d] = xs[0].shape[i]
            out = np.broadcast_to(xs[0].reshape(shape), p["shape"])
        elif kind == "reshape":
            out = (
                xs[0].transpose(p["dimensions"]).reshape(p["new_sizes"])
                if p["dimensions"]
                else xs[0].reshape(p["new_sizes"])
            )
        elif kind == "squeeze":
            out = np.squeeze(xs[0], axis=tuple(p["dimensions"]))
        elif kind == "transpose":
            out = xs[0].transpose(p["permutation"])
        elif kind == "slice":
            out = xs[0][
                tuple(
                    slice(a, b, c)
                    for a, b, c in zip(
                        p["start_indices"], p["limit_indices"], p["strides"] or [1] * xs[0].ndim
                    )
                )
            ]
        elif kind == "select_n":
            out = np.where(xs[0], xs[2], xs[1])
        elif kind == "convert_element_type":
            out = xs[0].astype(p["new_dtype"].get("dtype", p["new_dtype"]))
        else:
            out = {
                "add": np.add,
                "sub": np.subtract,
                "mul": np.multiply,
                "div": np.divide,
                "lt": np.less,
            }[kind](*xs)
        self.constants[key] = np.asarray(out)
        return self.constants[key]

    def cases(self, hlo):
        known = {
            h
            for layer in read_json(ROOT / "kernels/xf32_rne/weights.json")["layers"]
            for h in layer["sha256"].values()
        }
        selected, excluded = {}, []
        for g in self.data["groups"]:
            kind = g["operator"]
            if kind == "device_put" or not any(
                self.variables[v]["dtype"].startswith("float") for v in g["outputs"]
            ):
                excluded.append(
                    dict(
                        group=g,
                        reason="Device placement or integer index/control helper; not a floating model operator",
                    )
                )
                continue
            if kind == "linear":
                dot = next(
                    self.equations[i]
                    for i in g["equation_ids"]
                    if self.equations[i]["primitive"] == "dot_general"
                )
                w = self.constant(dot["inputs"][1])
                fingerprint = hashlib.sha256(np.ascontiguousarray(w).tobytes()).hexdigest()
                if fingerprint not in known:
                    tail = g["scope"].split("/scan_body/")[-1] + "/dot_general"
                    if tail in hlo:
                        raise ValueError("Unrecognized Linear still appears in optimized HLO")
                    excluded.append(
                        dict(
                            group=g,
                            reason="Matrix product absent from optimized HLO and 84 runtime weight fingerprints; compile-time precomputed",
                        )
                    )
                    continue
            inputs = [self.variables[v] for v in g["inputs"]]
            outputs = [self.variables[v] for v in g["outputs"]]
            scalar_and_index = []
            for v in g["inputs"]:
                d = self.variables[v]
                if d["shape"] == [] or not d["dtype"].startswith("float"):
                    c = self.constant(v)
                    scalar_and_index.append(
                        hashlib.sha256(np.ascontiguousarray(c).tobytes()).hexdigest()
                        if c is not None
                        else "generated-zero"
                    )
            eqs = [self.equations[i] for i in g["equation_ids"]]
            signature = dict(
                operator=kind,
                inputs=[(x["shape"], x["dtype"]) for x in inputs],
                outputs=[(x["shape"], x["dtype"]) for x in outputs],
                equations=[(e["primitive"], e["params"]) for e in eqs],
                index_or_scalar=scalar_and_index,
            )
            key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
            if key not in selected:
                selected[key] = dict(
                    id=kind.replace("-", "_") + "-" + key[:12],
                    operator=kind,
                    group=g,
                    signature=signature,
                    calls=0,
                    scopes=[],
                )
            selected[key]["calls"] += 1
            selected[key]["scopes"].append(g["scope"])
        return list(selected.values()), excluded

    def runtime_positions(self, case):
        return [
            i
            for i, key in enumerate(case["group"]["inputs"])
            if not (self.variables[key]["shape"] == [] and self.constant(key) is not None)
        ]

    def function(self, case):
        import jax

        g = case["group"]
        eqs = [self.equations[i] for i in g["equation_ids"]]
        functions = [primitive_function(e["primitive"], e["params"]) for e in eqs]
        positions = self.runtime_positions(case)
        literals = {
            key: np.asarray(self.constant(key), dtype=np.dtype(self.variables[key]["dtype"]))
            for i, key in enumerate(g["inputs"])
            if i not in positions
        }

        def replay(*args):
            env = dict(literals)
            env.update(zip((g["inputs"][i] for i in positions), args))
            for e, fn in zip(eqs, functions):
                value = fn(*(env[v] for v in e["inputs"]))
                vals = value if isinstance(value, (tuple, list)) else (value,)
                env.update(zip(e["outputs"], vals))
            if len(g["outputs"]) != 1:
                raise ValueError("Expected one logical result")
            return env[g["outputs"][0]]

        return jax.jit(replay)


def run(folder, data, output, repeats=3, iterations=20, smoke=False):
    import jax
    import jax.numpy as jnp

    from .graph_operator_bench import device_identity
    from .router import Router

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    identity = device_identity()
    verify_source()
    _, prepared = load_case(data)
    source = code_snapshot()
    inventory = Inventory(folder)
    capture = Path(folder).parent
    capture_report = read_json(capture / "report.json")
    if (
        capture_report.get("status") != "completed"
        or not capture_report.get("logical_catalog_only")
        or capture_report.get("mode") != "rne_xf32"
    ):
        raise RuntimeError("Inventory must come from a completed RNE+XF32 catalog capture")
    if (
        capture_report["data"] != prepared
        or sha256(capture / "optimized-hlo.txt") != capture_report["optimized_hlo_sha256"]
    ):
        raise RuntimeError("Inventory data or optimized HLO differs from completed capture")
    hlo = (capture / "optimized-hlo.txt").read_text()
    cases, excluded = inventory.cases(hlo)
    if sum(c["calls"] for c in cases if c["operator"] == "linear") != 84:
        raise RuntimeError("Logical catalog does not match the 84 steady-state model GEMMs")
    write_json(output / "catalog.json", dict(cases=cases, excluded=excluded))
    router = Router()
    router.select("rne_xf32", "logical_replay")
    inputs_hash = {
        str(p): sha256(p)
        for p in (
            Path(folder) / "inventory.json",
            Path(folder) / "constants.npz",
            capture / "optimized-hlo.txt",
            capture / "report.json",
        )
    }
    report = dict(
        status="running",
        mode="rne_xf32",
        **identity,
        code_sha256=source,
        data=prepared,
        sources=inputs_hash,
        versions={k: metadata.version(k) for k in ("jax", "jaxlib", "numpy", "dm-haiku")},
        timing="Synchronized JAX wall time, 5 warmups per round, median of round medians; rounds and calls are recorded below. Compile, inputs and correctness excluded. Linear includes RNE, GEMM and bias; RNE subitems are not additive.",
        input_policy="Frozen real weight/bias/normalization/index constants where available; seeded synthetic float activations. Representative case per exact graph signature; differing float parameter values may share one case.",
        rounds=repeats,
        iterations_per_round=iterations,
        rows=[],
    )
    write_json(output / "report.json", report)
    selected = list({c["operator"]: c for c in reversed(cases)}.values()) if smoke else cases
    for ci, case in enumerate(selected):
        print("Logical", ci + 1, "/", len(selected), case["id"], "calls", case["calls"], flush=True)
        args = []
        g = case["group"]
        for ai, key in enumerate(g["inputs"]):
            desc = inventory.variables[key]
            shape = tuple(desc["shape"])
            dtype = np.dtype(desc["dtype"])
            constant = inventory.constant(key)
            if constant is not None:
                x = jax.device_put(np.asarray(constant))
            elif dtype.kind not in "fc":
                x = jnp.zeros(shape, dtype)
            elif case["operator"] == "scatter-add" and ai == 0:
                x = jnp.zeros(shape, dtype)
            else:
                x = jax.random.uniform(
                    jax.random.PRNGKey(20261022 + ai), shape, dtype=jnp.float32, minval=-1, maxval=1
                )
            args.append(x)
        jax.block_until_ready(args)
        runtime_args = [args[i] for i in inventory.runtime_positions(case)]
        fn = inventory.function(case)
        before = router.snapshot()
        y = fn(*runtime_args)
        jax.block_until_ready(y)
        after = router.snapshot()
        if case["operator"] == "linear" and (
            after[0] - before[0] != 1 or after[1] - before[1] != 1
        ):
            raise RuntimeError("Linear replay did not execute one RNE+XF32 GEMM")
        if router.library.graphcast_stage_unknown() != 0:
            raise RuntimeError("Unknown replay weight")
        size = y.size
        sample = np.unique(
            np.r_[0, size - 1, np.random.RandomState(20261022).randint(0, size, min(256, size))]
        )
        actual = sample_values(y, np.array(np.unravel_index(sample, y.shape)).T)
        eq = inventory.equations[g["equation_ids"][0]]
        ref = reference_samples(case["operator"], eq["params"], args, y.shape, sample)
        tolerance = (
            (3e-3, 5e-4)
            if case["operator"] == "scatter-add"
            else ((1e-3, 2e-4) if case["operator"] == "linear" else (2e-5, 2e-5))
        )
        np.testing.assert_allclose(actual, ref, atol=tolerance[0], rtol=tolerance[1])
        if not np.isfinite(actual).all():
            raise RuntimeError("Nonfinite logical output")
        np.savez(
            output / (case["id"] + "-samples.npz"),
            flat_indices=sample,
            actual=actual,
            reference=ref,
        )
        (output / (case["id"] + "-hlo.txt")).write_text(fn.lower(*runtime_args).compile().as_text())
        cache = fn._cache_size()
        rounds = []
        start_calls = router.snapshot()
        for _ in range(repeats):
            for _ in range(5):
                jax.block_until_ready(fn(*runtime_args))
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                y = fn(*runtime_args)
                jax.block_until_ready(y)
                times.append((time.perf_counter() - start) * 1000)
            rounds.append(times)
        if cache != fn._cache_size():
            raise RuntimeError("Compilation inside timing")
        end_calls = router.snapshot()
        if case["operator"] == "linear":
            expected = repeats * (5 + iterations)
            if (
                end_calls[0] - start_calls[0] != expected
                or end_calls[1] - start_calls[1] != expected
            ):
                raise RuntimeError("Timed Linear call count mismatch")
        medians = [statistics.median(r) for r in rounds]
        row = dict(
            id=case["id"],
            operator=case["operator"],
            calls=case["calls"],
            input_shapes=[list(x.shape) for x in args],
            output_shape=list(y.shape),
            round_samples_ms=rounds,
            round_median_ms=medians,
            median_ms=statistics.median(medians),
            status="passed",
            oracle_samples=len(sample),
            max_abs_error=float(np.max(np.abs(actual - ref))),
            atol=tolerance[0],
            rtol=tolerance[1],
            native_calls=end_calls[0] - start_calls[0],
            rne_calls=end_calls[1] - start_calls[1],
            jit_cache_unchanged=True,
        )
        report["rows"].append(row)
        write_json(output / "report.json", report)
        print(case["id"], round(row["median_ms"], 6), "ms", flush=True)
        del args, runtime_args, fn, y, x
        gc.collect()
        jax.clear_caches()
    load_case(data)
    if source != code_snapshot():
        raise RuntimeError("Code changed during logical replay")
    if any(sha256(Path(p)) != h for p, h in inputs_hash.items()):
        raise RuntimeError("Inventory changed during replay")
    totals = {}
    for row in report["rows"]:
        total = totals.setdefault(row["operator"], dict(cases=0, calls=0, weighted_ms=0))
        total["cases"] += 1
        total["calls"] += row["calls"]
        total["weighted_ms"] += row["calls"] * row["median_ms"]
    report.update(
        status="passed", complete_inventory=not smoke, summary=totals, code_and_data_unchanged=True
    )
    write_json(output / "report.json", report)
    print(json.dumps(totals, indent=2), flush=True)
