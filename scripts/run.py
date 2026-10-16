#!/usr/bin/env python3
"""Host launcher (Python 3.6+); all numerical work runs in a pinned container."""

import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action", choices=("prepare", "check", "compare", "probe", "benchmark", "profile")
    )
    p.add_argument("--platform", choices=("h100", "hygon"), required=True)
    p.add_argument("--output", type=Path, required=True, help="A new directory; never overwritten")
    p.add_argument("--data", type=Path)
    p.add_argument("--case", type=Path)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--candidate", type=Path)
    p.add_argument("--reference", type=Path)
    p.add_argument("--device", type=int)
    p.add_argument("--mode", choices=("tf32", "fp32", "xf32", "rne_xf32", "all"))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--gpu-access", choices=("volume", "standard"), default="volume")
    args = p.parse_args()
    config = json.loads((ROOT / "configs" / (args.platform + ".json")).read_text())
    mode = args.mode or ("tf32" if args.platform == "h100" else "rne_xf32")
    if (
        args.repeats < 1
        or (args.platform == "h100" and mode != "tf32")
        or (args.platform == "hygon" and mode == "tf32")
    ):
        p.error("Unsupported mode/repeats for this platform")
    if args.action == "profile" and args.platform != "h100":
        p.error("The included profiler entry is for H100")
    if args.action in ("benchmark", "profile") and (
        args.data is None or not (args.data / "manifest.json").is_file()
    ):
        p.error("--data must point to a prepared JAX case")
    if args.action == "prepare" and (args.case is None or args.baseline is None):
        p.error("prepare requires --case and --baseline")
    if args.action == "compare" and (args.candidate is None or args.reference is None):
        p.error("compare requires --candidate and --reference")
    output = args.output.resolve()
    if output.exists():
        p.error("Output already exists: " + str(output))
    device = args.device if args.device is not None else config["device_default"]
    if device < 0:
        p.error("Device index must be nonnegative")
    gpu = args.action in ("probe", "benchmark", "profile")
    access = []
    gpu_env = {}
    telemetry = None
    if gpu and args.platform == "h100":
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(device),
                "--query-gpu=uuid,pci.bus_id,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            universal_newlines=True,
        )
        row = [x.strip() for x in next(csv.reader([raw]))]
        if int(row[2]) >= 256 or int(row[3]) != 0:
            raise RuntimeError("Requested H100 is busy")
        pci = ":".join([row[1].split(":")[0][-4:]] + row[1].split(":")[1:])
        access = ["--runtime", "nvidia"]
        if args.gpu_access == "volume":
            access += ["-v", "/dev/null:/var/run/nvidia-container-devices/" + str(device)]
        else:
            access += ["--gpus", "device=" + row[0]]
        gpu_env = {
            "NVIDIA_VISIBLE_DEVICES": row[0],
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
            "CUDA_VISIBLE_DEVICES": "0",
        }
        telemetry = raw
    elif gpu:
        raw = subprocess.check_output(
            [
                "/opt/hyhal/bin/hy-smi",
                "-d",
                str(device),
                "--showuse",
                "--showmeminfo",
                "vram",
                "--showbus",
                "--json",
            ],
            universal_newlines=True,
        )
        card = json.loads(raw)["card" + str(device)]
        if float(card["HCU use (%)"]) != 0 or float(card["vram Total Used Memory (MiB)"]) >= 256:
            raise RuntimeError("Requested Hygon accelerator is busy")
        pci = card["PCI Bus"].split()[0]
        access = [
            "--device",
            "/dev/kfd",
            "--device",
            "/dev/mkfd",
            "--device",
            "/dev/dri",
            "--security-opt",
            "seccomp=unconfined",
            "--security-opt",
            "label=disable",
        ]
        gpu_env = {
            "ROCR_VISIBLE_DEVICES": str(device),
            "HIP_VISIBLE_DEVICES": "0",
            "CUDA_VISIBLE_DEVICES": "0",
        }
        telemetry = raw
    output.mkdir(parents=True, exist_ok=False)
    env = {
        "GRAPHCAST_PLATFORM": args.platform,
        "GRAPHCAST_PYTHON": config["python"],
        "OUTPUT_UID": str(os.getuid()),
        "OUTPUT_GID": str(os.getgid()),
        **gpu_env,
    }
    if gpu:
        env.update(GRAPHCAST_EXPECTED_PCI=pci.lower(), GRAPHCAST_PHYSICAL_DEVICE=str(device))
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--ipc=private",
        "--shm-size=8g",
        "--cpus=16",
        "--memory=64g",
    ] + access
    command += ["-v", str(ROOT) + ":/repo:ro", "-v", str(output) + ":/output", "-w", "/repo"]
    if args.platform == "hygon":
        command += ["-v", "/opt/hyhal:/opt/hyhal:ro"]
    if args.data:
        command += ["-v", str(args.data.resolve()) + ":/data:ro"]
    if args.action == "prepare":
        command += [
            "-v",
            str(args.case.resolve()) + ":/source-case:ro",
            "-v",
            str(args.baseline.resolve()) + ":/source-reference.npz:ro",
        ]
    if args.action == "compare":
        command += [
            "-v",
            str(args.candidate.resolve()) + ":/candidate.nc:ro",
            "-v",
            str(args.reference.resolve()) + ":/reference" + args.reference.suffix + ":ro",
        ]
        env["GRAPHCAST_REFERENCE_SUFFIX"] = args.reference.suffix
    for key, value in env.items():
        command += ["-e", key + "=" + value]
    command += [
        "--entrypoint",
        "bash",
        config["image"],
        "/repo/scripts/in_container.sh",
        args.action,
    ]
    if args.action in ("benchmark", "profile"):
        command += ["--mode", mode, "--repeats", str(args.repeats)]
    launch = {
        "command": command,
        "platform_config": config,
        "device_before": telemetry,
        "started_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    (output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    with (output / "console.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    (output / "exit-code.txt").write_text(str(result.returncode) + "\n")
    print("\n".join((output / "console.log").read_text().splitlines()[-20:]))
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
