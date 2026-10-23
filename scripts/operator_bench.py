#!/usr/bin/env python3
"""Build/run standalone native GEMM/RNE and real-topology JAX graph operators."""

import argparse
import csv
import ctypes as C
import json
import statistics
import subprocess
from importlib import metadata
from pathlib import Path

from graphcast_bench.assets import (
    ROOT,
    code_snapshot,
    load_case,
    read_json,
    sha256,
    verify_source,
    write_json,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--iterations", type=int, default=20)
    a = p.parse_args()
    if a.repeats < 1 or a.iterations < 1:
        p.error("repeats and iterations must be positive")
    # Same process-only THP setting as the official model runner.
    libc = C.CDLL(None, use_errno=True)
    if libc.prctl(41, 1, 0, 0, 0) != 0:
        raise OSError(C.get_errno(), "Could not disable THP for this process")
    official = verify_source()
    case, prepared = load_case(a.data)
    config = read_json(ROOT / "configs/hygon.json")
    if metadata.version("jax") != config["jax_version"]:
        raise RuntimeError("Wrong JAX runtime")
    catalog = read_json(ROOT / "configs/operators.json")
    if sha256(ROOT / "kernels/xf32_rne/weights.json") not in catalog["sources"].values():
        raise RuntimeError("Operator catalog does not match frozen checkpoint map")
    a.output.mkdir(parents=True, exist_ok=False)
    exe = a.output / "gemm_bench"
    command = [
        "/opt/dtk/bin/hipcc",
        "--offload-arch=" + config["arch"],
        "-O2",
        str(ROOT / "kernels/operator_bench/gemm_bench.hip"),
        str(ROOT / "kernels/xf32_rne/rne_kernel.hip"),
        "-o",
        str(exe),
        "-lrocblas",
    ]
    built = subprocess.run(command, capture_output=True, text=True)
    (a.output / "build.log").write_text(built.stdout + built.stderr)
    built.check_returncode()
    write_json(
        a.output / "build.json",
        {
            "command": command,
            "binary_sha256": sha256(exe),
            "compiler": subprocess.check_output(["/opt/dtk/bin/hipcc", "--version"], text=True),
        },
    )
    rows = []
    report = {
        "schema": "graphcast.operator_benchmark.v1",
        "status": "running",
        "mode": "rne_xf32",
        "source": official,
        "code_sha256": code_snapshot(),
        "data": prepared,
        "catalog_sha256": sha256(ROOT / "configs/operators.json"),
        "jax": metadata.version("jax"),
        "rounds": a.repeats,
        "iterations_per_round": a.iterations,
        "gemm_shapes": rows,
        "input_policy": catalog["input_policy"],
        "timing_scope": "HIP event averages over repeated operations, 5 warmups, allocations and correctness excluded. No interposer/routing overhead. Synthetic inputs at real model shapes; not end-to-end performance.",
    }
    for shape in catalog["gemm_shapes"]:
        print("Native operators:", shape["id"], flush=True)
        args = (
            [str(exe)]
            + [str(shape[k]) for k in ("m", "n", "k")]
            + [str(a.repeats), str(a.iterations)]
        )
        run = subprocess.run(args, capture_output=True, text=True)
        (a.output / (shape["id"] + ".stderr.log")).write_text(run.stderr)
        if run.returncode:
            raise RuntimeError("Native benchmark failed: " + shape["id"] + "\n" + run.stderr)
        result = json.loads(run.stdout)
        result["calls_per_step"] = shape["calls_per_step"]
        result["id"] = shape["id"]
        for timing in result["timings"].values():
            timing["median_ms"] = statistics.median(timing["round_ms"])
        rows.append(result)
        write_json(a.output / (shape["id"] + ".json"), result)
        write_json(a.output / "report.json", report)
    weighted = {}
    for name in rows[0]["timings"]:
        per_round = [
            sum(r["calls_per_step"] * r["timings"][name]["round_ms"][i] for r in rows)
            for i in range(a.repeats)
        ]
        weighted[name] = {"round_ms": per_round, "median_ms": statistics.median(per_round)}
    report["frequency_weighted_standalone_ms_per_model_step"] = weighted
    with (a.output / "gemm-timings.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["shape", "calls_per_step", "mode", "round", "milliseconds_per_call"])
        for row in rows:
            for mode, values in row["timings"].items():
                for i, value in enumerate(values["round_ms"], 1):
                    writer.writerow([row["id"], row["calls_per_step"], mode, i, value])
    write_json(a.output / "report.json", report)
    # Native subprocesses have released their buffers before importing JAX.
    from graphcast_bench.graph_operator_bench import run_graph_operators

    report["graph_operators"] = run_graph_operators(
        a.data, a.output / "graph", a.repeats, a.iterations
    )
    if code_snapshot() != report["code_sha256"]:
        raise RuntimeError("Code changed during execution")
    _, final_prepared = load_case(a.data)
    if final_prepared != prepared:
        raise RuntimeError("Prepared data manifest changed during execution")
    report["code_and_data_unchanged"] = True
    report["status"] = "passed"
    write_json(a.output / "report.json", report)
    print(
        json.dumps(
            {"status": "passed", "gemm_shapes": len(rows), "weighted_native_ms": weighted}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
