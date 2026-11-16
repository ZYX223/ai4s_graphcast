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
        raise ValueError("设备信息格式不正确")
    devices = {}
    for name, info in inventory.items():
        match = re.fullmatch(r"card([0-9]+)", name)
        if not match:
            continue
        bus = info.get("PCI Bus", "") if isinstance(info, dict) else ""
        pci = re.match(r"^\s*(" + PCI + r")(?=\s|$)", str(bus))
        if not pci:
            raise ValueError("无法读取设备的 PCI 地址：" + name)
        devices[int(match[1])] = {
            "pci": pci[1].lower(),
            "use": info.get("HCU use (%)", "未知"),
            "used": info.get("vram Total Used Memory (MiB)", "未知"),
            "total": info.get("vram Total Memory (MiB)", "未知"),
        }
    if not devices:
        raise ValueError("未发现可读取的海光设备")
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
        print("卡号  使用率(%)  已用/总显存(MiB)", file=output_stream)
        for index, info in sorted(devices.items()):
            print(
                f"{index:>4}  {info['use']:>9}  {info['used']}/{info['total']}",
                file=output_stream,
            )
        print("请输入已分配给本任务的卡号: ", end="", file=output_stream, flush=True)
        value = input_stream.readline().strip()
        if not re.fullmatch(r"[0-9]+", value):
            raise ValueError("请输入有效卡号；非交互运行可使用 bash scripts/reproduce.sh 卡号")
        device = int(value)
    if device not in devices:
        raise ValueError("未找到物理卡：" + str(device))
    pci = devices[device]["pci"]
    if expected_pci is not None and expected_pci.lower() != pci:
        raise ValueError(f"物理卡 {device} 的 PCI 地址为 {pci}，与指定值不符")
    return device, pci


def main():
    parser = argparse.ArgumentParser(description="选择已分配的海光卡，自动读取 PCI 地址")
    parser.add_argument("device", nargs="?", type=int, help="已分配的物理卡号；省略时交互选择")
    parser.add_argument("pci", nargs="?", help="可选：额外核对预期 PCI 地址")
    args = parser.parse_args()
    try:
        device, pci = select_device(read_devices(), args.device, args.pci)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, "GraphCast: " + str(error) + "\n")
    print(f"使用物理卡 {device}（PCI {pci}）", file=sys.stderr)
    print(device, pci)


if __name__ == "__main__":
    main()
