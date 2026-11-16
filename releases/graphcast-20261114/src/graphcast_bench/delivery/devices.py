"""Explicit device selection and fail-closed admission before compute."""

import json
import subprocess

from . import resource_gate


def bw_guard():
    # Shared admission/monitoring must also work in an Operational-only export.
    return resource_gate


def inventory(platform):
    if platform == "bw1100":
        sample = json.loads(
            subprocess.check_output(
                [
                    "/opt/hyhal/bin/hy-smi",
                    "--showuse",
                    "--showmeminfo",
                    "vram",
                    "--showbus",
                    "--json",
                ]
            )
        )
        return [
            dict(
                index=int(k[4:]),
                pci=v["PCI Bus"].split()[0].lower(),
                uuid=None,
                memory_mib=float(v["vram Total Used Memory (MiB)"]),
                utilization=float(v["HCU use (%)"]),
            )
            for k, v in sorted(sample.items())
            if k.startswith("card")
        ]
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        universal_newlines=True,
    )
    rows = []
    for line in raw.splitlines():
        index, uuid, pci, name, memory, usage = [p.strip() for p in line.split(",")]
        if "H100" not in name:
            continue
        rows.append(
            dict(
                index=int(index),
                uuid=uuid,
                pci=pci[-12:].lower(),
                memory_mib=float(memory),
                utilization=float(usage),
            )
        )
    return rows


def select(platform, count, indices=None):
    rows = inventory(platform)
    if indices is None:
        for row in rows:
            print(
                "GPU {index}: PCI {pci}, utilization {utilization}%, VRAM used {memory_mib} MiB".format(
                    **row
                )
            )
        indices = [
            int(x)
            for x in input("Enter the {} physical GPU indices assigned to this run (space-separated): ".format(count)).split()
        ]
    if len(indices) != count or len(set(indices)) != count:
        raise ValueError("Expected {} distinct physical GPU indices".format(count))
    table = {row["index"]: row for row in rows}
    if any(i not in table for i in indices):
        raise ValueError("Selected GPU does not belong to the requested platform")
    return [table[i] for i in indices]


def admit(platform, devices):
    current = {r["index"]: r for r in inventory(platform)}
    for selected in devices:
        row = current[selected["index"]]
        if row["pci"] != selected["pci"] or row["uuid"] != selected["uuid"]:
            raise RuntimeError("Device identity changed")
        if row["utilization"] != 0 or not 0 <= row["memory_mib"] <= 512:
            raise RuntimeError("Selected GPU is busy; wait until it is idle: " + str(row["index"]))
    if platform == "bw1100":
        guard = bw_guard()
        sample = json.loads(
            subprocess.check_output(
                ["/opt/hyhal/bin/hy-smi", "--showuse", "--showmeminfo", "vram", "--json"]
            )
        )
        processes = guard.probe_processes()
        for device in devices:
            guard.require_idle(sample, device["index"], processes)
        return dict(devices=list(current.values()), processes=processes)
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        universal_newlines=True,
    )
    selected = {d["uuid"] for d in devices}
    if any(line.split(",")[0].strip() in selected for line in apps.splitlines()):
        raise RuntimeError("Selected H100 already has a compute process")
    return dict(devices=list(current.values()), compute_apps=apps)
