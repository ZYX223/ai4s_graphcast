"""Resolve an explicitly chosen physical card using the host's device inventory."""

import argparse
import json
import re
import subprocess
import sys

PCI = r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]"


def parse_devices(inventory):
    """Read physical card indices and PCI addresses from hy-smi JSON."""
    if not isinstance(inventory, dict):
        raise ValueError("Invalid device information format")
    devices = {}
    for name, info in inventory.items():
        match = re.fullmatch(r"card([0-9]+)", name)
        if not match:
            continue
        bus = info.get("PCI Bus", "") if isinstance(info, dict) else ""
        pci = re.match(r"^\s*(" + PCI + r")(?=\s|$)", str(bus))
        if not pci:
            raise ValueError("Unable to read the device PCI address: " + name)
        devices[int(match[1])] = {
            "pci": pci[1].lower(),
            "use": info.get("HCU use (%)", "unknown"),
            "used": info.get("vram Total Used Memory (MiB)", "unknown"),
            "total": info.get("vram Total Memory (MiB)", "unknown"),
        }
    if not devices:
        raise ValueError("No readable Hygon devices found")
    return devices


def read_devices():
    result = subprocess.run(
        ["/opt/hyhal/bin/hy-smi", "--showuse", "--showmeminfo", "vram", "--showbus", "--json"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=30,
    )
    return parse_devices(json.loads(result.stdout))


def select_device(devices, device=None, expected_pci=None, input_stream=None, output_stream=None):
    """Require a card choice; idle utilization does not imply an allocation."""
    if device is None:
        input_stream = sys.stdin if input_stream is None else input_stream
        output_stream = sys.stderr if output_stream is None else output_stream
        print("GPU  Utilization(%)  Used/Total VRAM(MiB)", file=output_stream)
        for index, info in sorted(devices.items()):
            print(
                f"{index:>4}  {info['use']:>9}  {info['used']}/{info['total']}",
                file=output_stream,
            )
        print("Enter the physical GPU index assigned to this run: ", end="", file=output_stream, flush=True)
        value = input_stream.readline().strip()
        if not re.fullmatch(r"[0-9]+", value):
            raise ValueError("Enter a valid physical GPU index; specify it explicitly for non-interactive runs")
        device = int(value)
    if device not in devices:
        raise ValueError("Physical GPU not found: " + str(device))
    pci = devices[device]["pci"]
    if expected_pci is not None and expected_pci.lower() != pci:
        raise ValueError(f"Physical GPU {device} has PCI address {pci}, which differs from the requested value")
    return device, pci


def main():
    parser = argparse.ArgumentParser(description="Select an assigned Hygon GPU and read its PCI address")
    parser.add_argument("device", nargs="?", type=int, help="Assigned physical GPU index; prompt if omitted")
    parser.add_argument("pci", nargs="?", help="Optional expected PCI address for an additional identity check")
    args = parser.parse_args()
    try:
        device, pci = select_device(read_devices(), args.device, args.pci)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, "GraphCast: " + str(error) + "\n")
    print(f"Using physical GPU {device} (PCI {pci})", file=sys.stderr)
    print(device, pci)


if __name__ == "__main__":
    main()
