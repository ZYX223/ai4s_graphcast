#!/usr/bin/env python3
"""Reconcile model/data/runtime evidence and export actual GEMM shapes (CPU only)."""

import argparse
import csv
import datetime
import json
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphcast_bench.assets import ROOT, load_case, read_json, sha256, verify_source, write_json
from graphcast_bench.operator_catalog import build_catalog


def command_record(command):
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--hlo", type=Path, required=True)
    parser.add_argument(
        "--historical-run",
        type=Path,
        required=True,
        help="Directory containing the original report.json",
    )
    parser.add_argument("--historical-launch", type=Path, required=True)
    parser.add_argument("--label", default="fp32:1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    # Verify first; do not emit a passed artifact if an input is inconsistent.
    official = verify_source()
    case, prepared = load_case(args.data)
    trace = [json.loads(line) for line in args.trace.open() if line.strip()]
    catalog = build_catalog(
        trace, args.hlo.read_text(), read_json(ROOT / "kernels/xf32_rne/weights.json"), args.label
    )
    report_path = args.historical_run / "report.json"
    old = read_json(report_path)
    launch = read_json(args.historical_launch)
    if old["source"]["commit"] != official["commit"]:
        raise ValueError("Historical official version differs")
    if old["case_sha256"] != case["source_case_sha256"]:
        raise ValueError("Historical case differs")
    if old["weight_map_sha256"] != sha256(ROOT / "kernels/xf32_rne/weights.tsv"):
        raise ValueError("Historical weight map differs")
    if old["status"] != "completed" or old["measured_steps_per_run"] != 40:
        raise ValueError("Historical run is incomplete")
    per_round = []
    for row in old["rows"]:
        times = row["apply_seconds"]
        if len(times) != 40:
            raise ValueError("Historical run lacks 40 step timings")
        wall = row["rollout_wall_seconds"]
        model = sum(times)
        if wall < model or min(times) <= 0:
            raise ValueError("Inconsistent historical timing boundaries")
        per_round.append(
            {
                "run": row["run"],
                "mode": row.get("mode", row.get("policy")),
                "model_calls_seconds": model,
                "rollout_seconds": wall,
                "outside_model_calls_seconds": wall - model,
            }
        )
    summary = []
    for mode in sorted({r["mode"] for r in per_round}):
        selected = [r for r in per_round if r["mode"] == mode]
        summary.append(
            dict(
                mode=mode,
                rounds=len(selected),
                **{
                    k: statistics.median(r[k] for r in selected)
                    for k in (
                        "model_calls_seconds",
                        "rollout_seconds",
                        "outside_model_calls_seconds",
                    )
                },
            )
        )
    args.output.mkdir(parents=True)
    catalog["sources"] = {
        str(p.resolve()): sha256(p)
        for p in (args.trace, args.hlo, ROOT / "kernels/xf32_rne/weights.json")
    }
    write_json(args.output / "operator-catalog.json", catalog)
    with (args.output / "gemm-shapes.csv").open("w", newline="") as f:
        fields = [
            "id",
            "m",
            "n",
            "k",
            "calls_per_step",
            "flops_per_call",
            "weight_bytes",
            "activation_bytes",
            "output_bytes",
            "rne_read_write_bytes_per_call",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(catalog["gemm_shapes"])
    current = {
        name: command_record(cmd)
        for name, cmd in (
            ("identity", ["id"]),
            (
                "devices",
                [
                    "/opt/hyhal/bin/hy-smi",
                    "--showproductname",
                    "--showbus",
                    "--showmeminfo",
                    "vram",
                    "--json",
                ],
            ),
            (
                "docker_access",
                [
                    "docker",
                    "image",
                    "inspect",
                    launch["platform_config"]["image"],
                    "--format",
                    "{{.Id}}",
                ],
            ),
            (
                "docker_service",
                [
                    "systemctl",
                    "show",
                    "docker",
                    "-p",
                    "ActiveState",
                    "-p",
                    "ExecMainStartTimestamp",
                ],
            ),
            ("socket_acl", ["getfacl", "-p", "/var/run/docker.sock"]),
        )
    }
    pci = old["pci"].lower()
    pci_present = pci in current["devices"]["stdout"].lower()
    checks = {
        "official_source_unchanged": True,
        "data_weights_reference_hashes_match": True,
        "historical_case_and_weight_map_match": True,
        "historical_pci_present_on_current_host": pci_present,
        "hlo_runtime_catalog_match": True,
        "current_container_image_accessible": current["docker_access"]["returncode"] == 0,
    }
    audit = {
        "schema": "graphcast.inference_audit.v1",
        "recorded_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "repo": str(ROOT),
        "commit": command_record(["git", "-C", str(ROOT), "rev-parse", "HEAD"])["stdout"].strip(),
        "reported_hardware_model": "BW1100",
        "reported_hardware_source": "User correction on 2026-10-22",
        "driver_name_policy": "Keep BW1200 as raw driver identity; expected_device validates that string, not the product label.",
        "status": "records_verified_runtime_available"
        if checks["current_container_image_accessible"]
        else "records_verified_runtime_access_blocked",
        "checks": checks,
        "current_host": current,
        "historical": {
            "pci": pci,
            "jax": old["jax_distribution"],
            "image": launch["platform_config"]["image"],
            "launch": str(args.historical_launch.resolve()),
            "launch_sha256": sha256(args.historical_launch),
            "report": str(report_path.resolve()),
            "report_sha256": sha256(report_path),
        },
        "prepared_data": prepared,
        "historical_timing_boundary": "Model calls include synchronized inference, pending transfers and routing/RNE overhead. Difference from rollout is host/autoregressive work outside these calls, not all CPU overhead.",
        "historical_timing_rounds": per_round,
        "historical_timing_medians": summary,
        "new_gpu_measurements": False,
    }
    write_json(args.output / "audit.json", audit)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "checks": checks,
                "calls": catalog["calls_per_step"],
                "shapes": catalog["unique_gemm_shapes"],
                "historical_timing_medians": summary,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
