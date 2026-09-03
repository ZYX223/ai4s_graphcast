from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALCULATOR = PROJECT_ROOT / "tools" / "dense_gemm_peak" / "calculate_path_matched_mfu.py"


class DenseGemmPeakMfuTest(unittest.TestCase):
    @staticmethod
    def valid_payload() -> dict[str, object]:
        payload = {
            "schema": "graphcast.path_matched_mfu_input.v1",
            "useful_flops_40": 120_000_000_000_000,
            "platforms": [
                {
                    "platform": "h100",
                    "execution_precision": "tf32",
                    "denominator_mode": "fast_fp32",
                    "e2e_seconds": [10.0, 10.0, 10.0],
                    "roof_median_tflops": 20.0,
                    "roof_ci95": [19.0, 21.0],
                    "operator_takeover": {
                        "requested": False,
                        "observed_ops": [],
                        "missing_ops": [],
                        "status": "not_applicable_native",
                        "evidence": [],
                    },
                },
                *[
                    {
                        "platform": name,
                        "execution_precision": "strict_fp32",
                        "denominator_mode": "strict_fp32",
                        "e2e_seconds": [10.0, 10.0, 10.0],
                        "roof_median_tflops": 20.0,
                        "roof_ci95": [19.0, 21.0],
                        "operator_takeover": {
                            "requested": True,
                            "observed_ops": [
                                "addmm",
                                "cat",
                                "native_layer_norm",
                                "index_add_",
                                "silu",
                                "index",
                                "add",
                            ],
                            "missing_ops": [],
                            "status": "passed",
                            "evidence": [{"path": "report.json", "sha256": "a" * 64}],
                        },
                    }
                    for name in ("910c", "bw3000", "s5000", "c550", "810e")
                ],
            ],
        }
        payload["platforms"][-1]["operator_takeover"]["observed_ops"] = [
            "cat",
            "native_layer_norm",
            "index_add_",
            "silu",
            "index",
            "add",
        ]
        payload["platforms"][-1]["operator_takeover"]["missing_ops"] = ["addmm"]
        return payload

    def run_calculator(
        self, payload: dict[str, object]
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, object] | None]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.json"
            output_path = root / "output.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(CALCULATOR),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            result = (
                json.loads(output_path.read_text(encoding="utf-8"))
                if output_path.exists()
                else None
            )
        return completed, result

    def test_calculates_empirical_path_matched_mfu(self) -> None:
        completed, result = self.run_calculator(self.valid_payload())

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIsNotNone(result)
        self.assertEqual(result["schema"], "graphcast.path_matched_mfu.v1")
        self.assertEqual(result["metric"], "empirical_path_matched_mfu")
        self.assertFalse(result["is_vendor_theoretical_mfu"])
        self.assertEqual(len(result["platforms"]), 6)
        self.assertAlmostEqual(result["platforms"][0]["mean_e2e_seconds"], 10.0)
        self.assertAlmostEqual(result["platforms"][0]["effective_tflops"], 12.0)
        self.assertAlmostEqual(result["platforms"][0]["mfu"], 0.6)
        self.assertEqual(
            result["platforms"][0]["operator_takeover"]["display"],
            "不适用（Native baseline）",
        )
        self.assertEqual(result["platforms"][1]["operator_takeover"]["display"], "7/7")
        self.assertEqual(result["platforms"][-1]["operator_takeover"]["display"], "6/7")

    def test_rejects_tf32_execution_with_strict_fp32_denominator(self) -> None:
        payload = self.valid_payload()
        payload["platforms"][0]["denominator_mode"] = "strict_fp32"

        completed, result = self.run_calculator(payload)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIsNone(result)
        self.assertIn("does not match denominator mode", completed.stderr)

    def test_rejects_inconsistent_core7_takeover_partition(self) -> None:
        payload = self.valid_payload()
        payload["platforms"][1]["operator_takeover"]["missing_ops"] = ["addmm"]

        completed, result = self.run_calculator(payload)

        self.assertNotEqual(completed.returncode, 0)
        self.assertIsNone(result)
        self.assertIn("must partition the seven core operators", completed.stderr)


if __name__ == "__main__":
    unittest.main()
