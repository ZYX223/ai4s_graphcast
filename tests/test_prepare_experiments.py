"""Workspace preparation must not overwrite runs or leave container-broken links."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_experiments", ROOT / "scripts/prepare_experiments.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareExperimentsTests(unittest.TestCase):
    def test_rejects_existing_output_and_missing_assets_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileExistsError):
                MODULE.prepare(root / "missing", root)
            with self.assertRaises(FileNotFoundError):
                MODULE.prepare(root / "missing", root / "new")
            self.assertFalse((root / "new").exists())

    def test_training_inputs_resolve_inside_the_training_mount(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets, output = root / "assets", root / "work"
            for name in (
                "jax-reorganization-20261016/prepared/data",
                "jax-single-card-training-20261022/dependencies",
                "jax-single-card-training-20261022/data",
                "jax-training-38-20261023/data",
            ):
                (assets / name).mkdir(parents=True)
            data = assets / "jax-training-38-20261023/data"
            (data / "source.json").write_text("{}")
            (data / "hres-20220101-025-13-12.nc").write_bytes(b"test input")
            for mode in ("fp32", "rne_xf32"):
                for horizon in (1, 12):
                    folder = assets / "jax-training-38-20261023" / ("%s-r%02d" % (mode, horizon))
                    folder.mkdir()
                    for name in (
                        "report.json",
                        "checkpoint.pkl",
                        "gradient-01.npy",
                        "gradient-02.npy",
                    ):
                        (folder / name).write_bytes(b"test asset")
            MODULE.prepare(assets, output, "training", True)
            training = output / "jax-training-38-20261023"
            self.assertFalse((training / "data").is_symlink())
            self.assertFalse((training / "data/hres-20220101-025-13-12.nc").is_symlink())
            self.assertEqual(
                (training / "data/hres-20220101-025-13-12.nc").read_bytes(), b"test input"
            )
            self.assertFalse((training / "fp32-r01/checkpoint.pkl").is_symlink())
            self.assertIn("GRAPHCAST_DEVICE", (training / "run_curriculum.py").read_text())
            self.assertTrue((output / "workspace.json").is_file())
            # Reports are independent copies: later summaries must not edit originals.
            (training / "fp32-r01/report.json").write_text("changed")
            self.assertEqual(
                (assets / "jax-training-38-20261023/fp32-r01/report.json").read_bytes(),
                b"test asset",
            )

    def test_archive_hashes_match_measured_sources(self):
        self.assertGreater(MODULE.verify_archive(), 0)


if __name__ == "__main__":
    unittest.main()
