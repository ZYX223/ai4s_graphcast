"""Training semantics and process-local RNE lowering regression."""

import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from graphcast_bench.training.curriculum import select_updates
from graphcast_bench.training.recipe import sample_plan, schedule


class TrainingContractTests(unittest.TestCase):
    def test_fixed_schedule_and_contiguous_resume(self):
        plan = schedule()
        self.assertEqual(len(plan), 38)
        self.assertEqual([r["update"] for r in plan if r["lr"] == 0], [1, 16])
        self.assertEqual([r["rollout"] for r in plan], [1] * 16 + list(np.repeat(range(2, 13), 2)))
        self.assertEqual([r["update"] for r in select_updates(plan, 2, 16, 38)], [17, 18])
        with self.assertRaises(ValueError):
            select_updates(plan, 3, 16, 38)

    def test_sample_plan_is_repeatable_with_real_future_targets(self):
        a = sample_plan(14, 27)
        self.assertEqual(a, sample_plan(14, 27))
        for row in a:
            self.assertEqual(len(row["starts"]), 2)
            self.assertTrue(all(0 <= s <= 12 - row["rollout"] for s in row["starts"]))
        with self.assertRaises(ValueError):
            sample_plan(13, 27)

    def test_lowering_forward_gradients_and_remat_in_fresh_process(self):
        case = Path(__file__).with_name("rne_lowering_case.py")
        result = subprocess.run(
            [sys.executable, str(case), "-q"],
            env=dict(os.environ, JAX_PLATFORMS="cpu"),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_training_hook_dependencies_install_in_explicit_order(self):
        script = """
import tempfile
from pathlib import Path
from graphcast_bench.training.entry import install_remat
from graphcast_bench.optimizations.training import regular3, layernorm_stats, mesh_vjp, final_mesh, g2m_mesh
from graphcast_bench.optimizations.training.remat import CANDIDATE
with tempfile.TemporaryDirectory() as folder:
    install_remat(Path(folder), CANDIDATE)
    for module in (regular3, layernorm_stats, mesh_vjp, final_mesh, g2m_mesh):
        module.install()
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
