#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any

PLATFORMS = ("h100", "910c", "bw3000", "s5000", "c550", "810e")
MODES = ("strict_fp32", "fast_fp32", "bf16")
SIZES = (8192, 12288, 16384, 24576)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def median_ci_by_process(
    processes: list[list[float]], *, seed: int, draws: int = 20_000
) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        pooled = []
        for _ in processes:
            process = processes[rng.randrange(len(processes))]
            pooled.extend(process[rng.randrange(len(process))] for _ in process)
        estimates.append(statistics.median(pooled))
    estimates.sort()
    return estimates[int(draws * 0.025)], estimates[int(draws * 0.975)]


def summarize_mode(records: list[dict[str, Any]], mode_index: int) -> dict[str, Any]:
    by_size: dict[int, list[list[float]]] = {size: [] for size in SIZES}
    seconds_by_size: dict[int, list[list[float]]] = {size: [] for size in SIZES}
    correctness = []
    controls = set()
    identities = set()
    for record in records:
        correctness.append(record["correctness"])
        controls.add(json.dumps(record["precision"]["controls"], sort_keys=True))
        identity = record["runtime_identity"]
        identities.add(
            (
                identity["device_name"],
                identity["properties"].get("total_memory"),
                identity["versions"]["torch"],
            )
        )
        for size_record in record["sizes"]:
            size = size_record["n"]
            by_size[size].append(size_record["tflops"])
            seconds_by_size[size].append(size_record["seconds"])

    if len(records) != 6 or any(len(processes) != 6 for processes in by_size.values()):
        raise RuntimeError("expected six independent processes per mode and size")

    size_summaries = {}
    for size_index, size in enumerate(SIZES):
        processes = by_size[size]
        samples = [value for process in processes for value in process]
        seconds = [value for process in seconds_by_size[size] for value in process]
        ci_low, ci_high = median_ci_by_process(
            processes, seed=20260901 + mode_index * 100 + size_index
        )
        process_medians = [statistics.median(process) for process in processes]
        median_of_process_medians = statistics.median(process_medians)
        block_span = (max(process_medians) - min(process_medians)) / median_of_process_medians
        seconds_cv = statistics.stdev(seconds) / statistics.fmean(seconds)
        size_summaries[str(size)] = {
            "process_count": len(processes),
            "sample_count": len(samples),
            "median_tflops": statistics.median(samples),
            "median_ci95_low": ci_low,
            "median_ci95_high": ci_high,
            "mean_tflops": statistics.fmean(samples),
            "sample_seconds_cv": seconds_cv,
            "process_median_tflops": process_medians,
            "process_median_span_fraction": block_span,
        }

    primary = size_summaries["24576"]
    diagnostic = size_summaries["16384"]
    change = primary["median_tflops"] / diagnostic["median_tflops"] - 1.0
    best_size = max(SIZES, key=lambda size: size_summaries[str(size)]["median_tflops"])
    unstable = primary["sample_seconds_cv"] > 0.03 or primary["process_median_span_fraction"] > 0.05
    return {
        "controls": [json.loads(item) for item in sorted(controls)],
        "runtime_identities": [
            {"device_name": item[0], "total_memory": item[1], "torch": item[2]}
            for item in sorted(identities)
        ],
        "correctness": {
            "all_passed": all(item["passed"] and item["finite"] for item in correctness),
            "max_normalized_rmse": max(item["normalized_rmse"] for item in correctness),
            "limit": correctness[0]["limit"],
        },
        "sizes": size_summaries,
        "primary_size": 24576,
        "primary_median_tflops": primary["median_tflops"],
        "primary_median_ci95": [
            primary["median_ci95_low"],
            primary["median_ci95_high"],
        ],
        "primary_unstable": unstable,
        "change_16384_to_24576_fraction": change,
        "roof_not_established_by_predeclared_rule": change > 0.05,
        "best_diagnostic_size": best_size,
        "best_diagnostic_median_tflops": size_summaries[str(best_size)]["median_tflops"],
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"output already exists: {args.output}")
    payload: dict[str, Any] = {
        "schema": "dense_gemm_runtime_roof_aggregate.v1",
        "input_root": str(args.input.resolve()),
        "platforms": {},
    }
    for platform_index, platform in enumerate(PLATFORMS):
        files = sorted((args.input / platform).glob("block-*/*/result.json"))
        if len(files) != 18:
            raise RuntimeError(f"{platform}: expected 18 results, found {len(files)}")
        records = [json.loads(path.read_text()) for path in files]
        source_hashes = {record["source"]["sha256"] for record in records}
        if len(source_hashes) != 1:
            raise RuntimeError(f"{platform}: source hashes differ")
        platform_payload = {
            "result_count": len(records),
            "source_sha256": next(iter(source_hashes)),
            "modes": {},
        }
        for mode_index, mode in enumerate(MODES):
            selected = [record for record in records if record["mode"] == mode]
            platform_payload["modes"][mode] = summarize_mode(
                selected, mode_index=platform_index * len(MODES) + mode_index
            )
        payload["platforms"][platform] = platform_payload

    serialized = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
