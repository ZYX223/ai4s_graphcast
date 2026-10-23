#!/usr/bin/env python3
"""Validate a single RNE+XF32 model-step trace and export operator timings."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphcast_bench.assets import ROOT, read_json, sha256, write_json
from graphcast_bench.profile_summary import category, summarize_events


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("Output exists")
    report = read_json(a.run / "run/report.json")
    audit = read_json(a.run / "run/step-audit.json")
    if (
        report["status"] != "completed"
        or report["mode"] != "rne_xf32"
        or report["measured_steps"] != 1
    ):
        raise ValueError("Expected a completed single-step RNE+XF32 profile")
    row = report["rows"][0]
    if row["jit_cache_before"] != row["jit_cache_after"]:
        raise ValueError("Compilation inside profile range")
    recorded = [r for r in audit if r["label"] == row["run"]]
    if (
        len(recorded) != 1
        or recorded[0]["native_calls"] != 84
        or recorded[0]["rounded_calls"] != 84
    ):
        raise ValueError("Expected 84 model GEMMs and double-input rounding")
    trace = read_json(a.run / "model-trace.json")["traceEvents"]
    hlo = (a.run / "run/optimized-hlo.txt").read_text()
    result = summarize_events(trace, hlo)
    if (
        result["categories"].get("gemm", {}).get("calls") != 84
        or result["categories"].get("rne", {}).get("calls") != 168
    ):
        raise ValueError("Device trace does not cover exactly one model step")
    result.update(
        mode="rne_xf32",
        platform="hygon",
        product_name="BW1100",
        driver_device_name=report["device"],
        pci=report["pci"],
        profile_rollout_ms=row["rollout_wall_seconds"] * 1000,
        profile_model_call_ms=row["apply_sum_seconds"] * 1000,
        status="passed",
        jit_cache_unchanged=True,
        sources={
            str(path.resolve()): sha256(path)
            for path in (
                a.run / "model-trace.json",
                a.run / "run/optimized-hlo.txt",
                a.run / "run/report.json",
                a.run / "run/step-audit.json",
            )
        },
    )
    pairs = []
    log = a.run / "model-gemm.jsonl"
    if log.exists():
        native = [json.loads(line) for line in log.open()]
        native = [r for r in native if r["label"] == row["run"]]
        relevant = sorted(
            (
                e
                for e in trace
                if e.get("ph") == "X"
                and e.get("cat") == "HIPOPS"
                and category(e["name"], []) in ("rne", "gemm")
            ),
            key=lambda e: e["ts"],
        )
        if len(native) != 84 or len(relevant) != 252:
            raise ValueError("Cannot match device kernels to native calls")
        known = {
            x["module"]: x for x in read_json(ROOT / "kernels/xf32_rne/weights.json")["layers"]
        }
        for i, call in enumerate(native):
            events = relevant[i * 3 : i * 3 + 3]
            if [category(e["name"], []) for e in events] != ["rne", "rne", "gemm"]:
                raise ValueError("Kernel ordering is not round-A, round-B, GEMM")
            if len({e["tid"] for e in events}) != 1:
                raise ValueError("Operand/GEMM kernels use different streams")
            weight = known[call["layer"]]
            if (
                weight["shape"] != [call["k"], call["m"]]
                or call["weight_sha256"] not in weight["sha256"].values()
            ):
                raise ValueError("Runtime weight shape/fingerprint mismatch")
            if call["mode"] != 1 or call["rne_inputs"] is not True or call["status"] != 0:
                raise ValueError("Wrong GEMM precision/rounding")
            pairs.append(
                {
                    "index": i + 1,
                    "layer": call["layer"],
                    "m": call["m"],
                    "n": call["n"],
                    "k": call["k"],
                    "weight_rne_ms": events[0]["dur"] / 1000,
                    "activation_rne_ms": events[1]["dur"] / 1000,
                    "gemm_ms": events[2]["dur"] / 1000,
                }
            )
        result["layer_timings"] = pairs
        result["rne_split"] = {
            "weight_ms": sum(r["weight_rne_ms"] for r in pairs),
            "activation_ms": sum(r["activation_rne_ms"] for r in pairs),
            "method": "Match 84 native log calls to 84 same-stream device triples (round A, round B, GEMM). A has frozen weight shape [m,k], B is activation [k,n]. Checkpoint fingerprints and shapes checked for each call.",
        }
        result["sources"][str(log.resolve())] = sha256(log)
    a.output.mkdir(parents=True)
    write_json(a.output / "summary.json", result)
    for filename, rows in (("kernels.csv", result["kernels"]), ("layers.csv", pairs)):
        if not rows:
            continue
        with (a.output / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "gpu_kernel_count", "gpu_kernel_sum_ms", "categories")
            },
            indent=2,
        )
    )
    if "rne_split" in result:
        print(json.dumps(result["rne_split"], indent=2))


if __name__ == "__main__":
    main()
