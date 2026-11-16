"""Device selection must resolve the chosen card without guessing an allocation."""

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from graphcast_bench.platforms.device import parse_devices, select_device


class DeviceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.devices = parse_devices(
            {
                "card0": {
                    "PCI Bus": "0000:05:00.0 --> SN: N/A --> OAM ID: 8",
                    "HCU use (%)": "0.0",
                    "vram Total Memory (MiB)": "147440",
                    "vram Total Used Memory (MiB)": "22351",
                },
                "card3": {
                    "PCI Bus": "0000:4C:00.0 --> SN: N/A --> OAM ID: 11",
                    "HCU use (%)": "0.0",
                    "vram Total Memory (MiB)": "147440",
                    "vram Total Used Memory (MiB)": "62829",
                },
            }
        )

    def test_card_number_resolves_pci_without_manual_address(self):
        self.assertEqual(select_device(self.devices, 3), (3, "0000:4c:00.0"))

    def test_interactive_choice_shows_usage_and_resolves_selected_card(self):
        output = io.StringIO()
        selected = select_device(
            self.devices, input_stream=io.StringIO("3\n"), output_stream=output
        )
        self.assertEqual(selected, (3, "0000:4c:00.0"))
        self.assertIn("62829", output.getvalue())
        self.assertIn("147440", output.getvalue())

    def test_empty_input_never_selects_a_card_automatically(self):
        for value in ("", "\n"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                select_device(
                    self.devices, input_stream=io.StringIO(value), output_stream=io.StringIO()
                )

    def test_unknown_and_invalid_card_numbers_are_rejected(self):
        for value in ("1", "-1", "3; true", "x"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                select_device(
                    self.devices, input_stream=io.StringIO(value), output_stream=io.StringIO()
                )

    def test_legacy_explicit_pci_must_match_the_selected_card(self):
        self.assertEqual(
            select_device(self.devices, 3, "0000:4C:00.0"), (3, "0000:4c:00.0")
        )
        with self.assertRaises(ValueError):
            select_device(self.devices, 3, "0000:05:00.0")

    def test_missing_or_malformed_device_information_is_rejected(self):
        for value in ({}, [], {"card3": {}}, {"card3": {"PCI Bus": "unavailable"}}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_devices(value)


class ReproductionSelectionTests(unittest.TestCase):
    def run_entry(self, arguments, answer):
        """Stub hardware discovery and stop at the first expensive build operation."""
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            wrapper = folder / "python"
            wrapper.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "from unittest.mock import patch\n"
                "if sys.argv[1:3] == ['-m', 'graphcast_bench.platforms.device']:\n"
                "    from graphcast_bench.platforms import device\n"
                "    sys.argv = ['device'] + sys.argv[3:]\n"
                "    inventory = {3: {'pci': '0000:4c:00.0', 'use': '0.0', "
                "'used': '4', 'total': '147440'}}\n"
                "    with patch.object(device, 'read_devices', return_value=inventory):\n"
                "        device.main()\n"
                "elif sys.argv[1:4] == ['scripts/build_kernels.py', '--kind', 'inference']:\n"
                "    Path(os.environ['BUILD_BOUNDARY']).write_text(json.dumps(\n"
                "        {key: os.environ[key] for key in ('DEVICE', 'DEVICE_PCI')}))\n"
                "    sys.exit(19)\n"
                "else:\n"
                "    sys.exit(20)\n"
            )
            wrapper.chmod(0o755)
            boundary = folder / "boundary.json"
            result = subprocess.run(
                ["bash", "scripts/reproduce.sh", *arguments],
                cwd=root,
                input=answer,
                capture_output=True,
                text=True,
                timeout=10,
                env=dict(
                    os.environ,
                    PATH=str(folder) + os.pathsep + os.environ["PATH"],
                    PYTHONPATH=str(root / "src"),
                    PYTHONDONTWRITEBYTECODE="1",
                    BUILD_ROOT=str(folder / "build"),
                    BUILD_BOUNDARY=str(boundary),
                    RUN_DIR=str(folder),
                    CONFIG_PATH=str(folder / "paths.json"),
                    HIPCC="unused-compiler",
                    CONDA_PREFIX=str(folder),
                ),
            )
            return result, json.loads(boundary.read_text()) if boundary.exists() else None

    def test_interactive_and_card_only_entries_forward_resolved_device_to_workload(self):
        for arguments, answer in (([], "3\n"), (["3"], ""), (["3", "0000:4c:00.0"], "")):
            with self.subTest(arguments=arguments):
                result, selection = self.run_entry(arguments, answer)
                self.assertEqual(result.returncode, 19, result.stdout + result.stderr)
                self.assertEqual(selection, {"DEVICE": "3", "DEVICE_PCI": "0000:4c:00.0"})

    def test_failed_selection_stops_before_build_or_gpu_work(self):
        for arguments, answer in (([], ""), (["9"], ""), (["3", "0000:05:00.0"], "")):
            with self.subTest(arguments=arguments):
                result, selection = self.run_entry(arguments, answer)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIsNone(selection)


if __name__ == "__main__":
    unittest.main()
