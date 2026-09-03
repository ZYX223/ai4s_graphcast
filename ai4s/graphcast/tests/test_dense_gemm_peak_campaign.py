from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = PROJECT_ROOT / "tools" / "dense_gemm_peak" / "run_platform_campaign.sh"


class DenseGemmPeakCampaignContractTests(unittest.TestCase):
    def test_requires_explicit_remote_access_tool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            completed = subprocess.run(
                [
                    "bash",
                    str(CAMPAIGN),
                    "h100",
                    "h100",
                    "0",
                    "test-campaign",
                    temporary_directory,
                ],
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("remote access tool is required", completed.stderr)


if __name__ == "__main__":
    unittest.main()
