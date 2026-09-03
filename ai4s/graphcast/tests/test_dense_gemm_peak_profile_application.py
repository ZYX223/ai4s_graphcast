from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = PROJECT_ROOT / "tools" / "dense_gemm_peak" / "profile_application.sh"


class ProfileApplicationContractTests(unittest.TestCase):
    def test_forwards_the_sealed_profile_contract_without_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            capture = temporary_root / "captured.json"
            source = temporary_root / "torch_dense_gemm.py"
            result = temporary_root / "result.json"
            source.write_text("# sealed source fixture\n", encoding="utf-8")
            fake_python = temporary_root / "fake-python"
            fake_python.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "json.dump(sys.argv[1:], open(os.environ['CAPTURE_PATH'], 'w'))\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            environment = {
                **os.environ,
                "CAPTURE_PATH": str(capture),
                "PROFILE_PYTHON": str(fake_python),
                "PROFILE_SOURCE": str(source),
                "PROFILE_MODE": "fast_fp32",
                "PROFILE_DEVICE": "cuda:0",
                "PROFILE_SIZE": "8192",
                "PROFILE_WARMUP": "1",
                "PROFILE_ITERATIONS": "3",
                "PROFILE_OUTPUT": str(result),
            }
            subprocess.run(["bash", str(WRAPPER)], env=environment, check=True)

            self.assertEqual(
                json.loads(capture.read_text(encoding="utf-8")),
                [
                    str(source),
                    "--mode",
                    "fast_fp32",
                    "--device",
                    "cuda:0",
                    "--sizes",
                    "8192",
                    "--warmup",
                    "1",
                    "--iterations",
                    "3",
                    "--output",
                    str(result),
                ],
            )


if __name__ == "__main__":
    unittest.main()
