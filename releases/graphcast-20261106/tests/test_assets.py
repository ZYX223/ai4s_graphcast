import hashlib
import tempfile
import unittest
from pathlib import Path

from graphcast_bench.assets import verify_files


class AssetTests(unittest.TestCase):
    def test_tamper_and_missing_file_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / "input.bin"
            path.write_bytes(b"fixed input")
            files = {"input.bin": hashlib.sha256(path.read_bytes()).hexdigest()}
            verify_files(root, files)
            path.write_bytes(b"changed input")
            with self.assertRaises(ValueError):
                verify_files(root, files)
            path.unlink()
            with self.assertRaises((ValueError, FileNotFoundError)):
                verify_files(root, files)

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                verify_files(Path(d), {"../other": "0" * 64})


if __name__ == "__main__":
    unittest.main()
