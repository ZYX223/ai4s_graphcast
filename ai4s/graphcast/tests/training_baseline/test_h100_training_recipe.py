from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]


def load_training_recipe():
    path = (
        ROOT
        / "src/graphcast_compat/training_recipe.py"
    )
    spec = importlib.util.spec_from_file_location("training_recipe_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ToyParameters(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matrix = torch.nn.Parameter(torch.ones(3, 2))
        self.vector = torch.nn.Parameter(torch.ones(3))
        self.scalar = torch.nn.Parameter(torch.ones(()))


class H100TrainingRecipeTests(unittest.TestCase):
    def test_multistep_mesh_edge_chunking_preserves_outputs_and_gradients(self):
        program = textwrap.dedent(
            """
                from pathlib import Path
                import sys
                import torch

                repository = Path(sys.argv[1])
                sys.path.insert(0, str(repository / "src"))
                from graphcast_compat.training_model import (
                    GoogleSmallConfig,
                    GoogleSmallGraphNet,
                )

                torch.manual_seed(19)
                config = GoogleSmallConfig(
                    latent_size=4,
                    hidden_layers=1,
                    gnn_msg_steps=2,
                )

                def build(chunked):
                    return GoogleSmallGraphNet(
                        edge_types=("edge",),
                        edge_node_sets={"edge": ("sender", "receiver")},
                        node_types=("sender", "receiver"),
                        node_input_dims={"sender": 3, "receiver": 3},
                        edge_input_dims={"edge": 2},
                        node_processor_input_dims={
                            "sender": 4,
                            "receiver": 8,
                        },
                        edge_processor_input_dims={"edge": 12},
                        node_output_dims=None,
                        extra_final_node_output_dims=None,
                        edge_update_chunk_sizes=(
                            {"edge": 2} if chunked else None
                        ),
                        message_passing_steps=2,
                        config=config,
                    )

                base_nodes = {
                    "sender": torch.randn(1, 3, 3),
                    "receiver": torch.randn(1, 4, 3),
                }
                base_edges = {"edge": torch.randn(1, 5, 2)}
                indices = {
                    "edge": (
                        torch.tensor([0, 1, 2, 0, 1]),
                        torch.tensor([0, 1, 2, 3, 0]),
                    )
                }

                def execute(network, checkpoint_steps):
                    nodes = {
                        name: value.detach().clone().requires_grad_(True)
                        for name, value in base_nodes.items()
                    }
                    edges = {
                        name: value.detach().clone().requires_grad_(True)
                        for name, value in base_edges.items()
                    }
                    out_nodes, out_edges = network(
                        nodes,
                        edges,
                        indices,
                        checkpoint_steps=checkpoint_steps,
                        checkpoint_chunks=True,
                    )
                    loss = sum(value.square().sum() for value in out_nodes.values())
                    loss = loss + sum(
                        value.square().sum() for value in out_edges.values()
                    )
                    loss.backward()
                    outputs = tuple(out_nodes.values()) + tuple(out_edges.values())
                    input_grads = tuple(value.grad for value in nodes.values()) + tuple(
                        value.grad for value in edges.values()
                    )
                    parameter_grads = tuple(
                        parameter.grad for parameter in network.parameters()
                    )
                    return outputs, input_grads, parameter_grads

                for checkpoint_steps in (False, True):
                    full = build(False)
                    chunked = build(True)
                    chunked.load_state_dict(full.state_dict())
                    full_result = execute(full, checkpoint_steps)
                    chunked_result = execute(chunked, checkpoint_steps)
                    for full_group, chunked_group in zip(
                        full_result,
                        chunked_result,
                    ):
                        for full_value, chunked_value in zip(
                            full_group,
                            chunked_group,
                        ):
                            torch.testing.assert_close(
                                full_value,
                                chunked_value,
                                # Chunk boundaries change the FP32 reduction
                                # order.  Vendor BLAS implementations can
                                # therefore differ by a few ULPs even though
                                # the complete outputs and gradients agree.
                                rtol=5.0e-5,
                                atol=5.0e-6,
                            )
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", program, str(ROOT)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_graphcast_adamw_accepts_official_zero_lr_warmup_boundary(self):
        recipe = load_training_recipe()
        model = ToyParameters()

        optimizer = recipe.build_graphcast_adamw(model, lr=0.0)

        self.assertEqual(optimizer.defaults["lr"], 0.0)
        self.assertTrue(
            all(group["lr"] == 0.0 for group in optimizer.param_groups)
        )

    def test_graphcast_adamw_rejects_invalid_learning_rates(self):
        recipe = load_training_recipe()
        model = ToyParameters()

        for learning_rate in (-1.0, float("nan"), float("inf")):
            with self.subTest(learning_rate=learning_rate):
                with self.assertRaisesRegex(
                    ValueError,
                    "finite and non-negative",
                ):
                    recipe.build_graphcast_adamw(
                        model,
                        lr=learning_rate,
                    )

    def test_graphcast_adamw_uses_paper_hyperparameters_and_matrix_only_decay(self):
        recipe = load_training_recipe()
        model = ToyParameters()

        optimizer = recipe.build_graphcast_adamw(model, lr=2.5e-4)

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.defaults["lr"], 2.5e-4)
        self.assertEqual(optimizer.defaults["betas"], (0.9, 0.95))
        decay_by_parameter = {
            id(parameter): float(group["weight_decay"])
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertEqual(decay_by_parameter[id(model.matrix)], 0.1)
        self.assertEqual(decay_by_parameter[id(model.vector)], 0.0)
        self.assertEqual(decay_by_parameter[id(model.scalar)], 0.0)

    def test_official_gradient_clip_reports_preclip_norm_and_caps_global_norm(self):
        recipe = load_training_recipe()
        model = ToyParameters()
        model.matrix.grad = torch.full_like(model.matrix, 20.0)
        model.vector.grad = torch.full_like(model.vector, 20.0)
        model.scalar.grad = torch.full_like(model.scalar, 20.0)

        preclip_norm = recipe.clip_graphcast_gradients(model)

        self.assertGreater(preclip_norm, recipe.GRAD_CLIP_MAX_NORM)
        postclip_norm = torch.linalg.vector_norm(
            torch.cat(
                [
                    parameter.grad.detach().reshape(-1)
                    for parameter in model.parameters()
                ]
            )
        )
        self.assertLessEqual(
            float(postclip_norm),
            recipe.GRAD_CLIP_MAX_NORM + 1.0e-4,
        )

    def test_gradient_clip_rejects_nonfinite_and_missing_gradients(self):
        recipe = load_training_recipe()
        model = ToyParameters()
        with self.assertRaises(FloatingPointError):
            recipe.clip_graphcast_gradients(model)
        for value in (float("nan"), float("inf")):
            model.matrix.grad = torch.full_like(model.matrix, value)
            with self.assertRaises(RuntimeError):
                recipe.clip_graphcast_gradients(model)

    def test_h100_checkpointed_graph_records_actual_recomputed_linear_flops(self):
        program = textwrap.dedent(
            """
                import json
                from pathlib import Path
                import sys
                import torch

                repository = Path(sys.argv[1])
                sys.path.insert(0, str(repository / "src"))
                from graphcast_compat.training_model import (
                    GoogleSmallConfig,
                    GoogleSmallGraphNet,
                )
                from graphcast_compat.recompute_flops import RecomputeFlopCounter

                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                config = GoogleSmallConfig(
                    latent_size=4,
                    hidden_layers=1,
                    gnn_msg_steps=1,
                )
                network = GoogleSmallGraphNet(
                    edge_types=("edge",),
                    edge_node_sets={"edge": ("sender", "receiver")},
                    node_types=("sender", "receiver"),
                    node_input_dims={"sender": 3, "receiver": 3},
                    edge_input_dims={"edge": 2},
                    node_processor_input_dims={"sender": 4, "receiver": 8},
                    edge_processor_input_dims={"edge": 12},
                    node_output_dims=None,
                    extra_final_node_output_dims=None,
                    edge_update_chunk_sizes=None,
                    message_passing_steps=1,
                    config=config,
                ).to(device)
                nodes = {
                    "sender": torch.randn(
                        1, 3, 3, device=device, requires_grad=True
                    ),
                    "receiver": torch.randn(
                        1, 4, 3, device=device, requires_grad=True
                    ),
                }
                edges = {
                    "edge": torch.randn(
                        1, 5, 2, device=device, requires_grad=True
                    )
                }
                indices = {
                    "edge": (
                        torch.tensor([0, 1, 2, 0, 1], device=device),
                        torch.tensor([0, 1, 2, 3, 0], device=device),
                    )
                }
                counter = RecomputeFlopCounter()
                with counter.track():
                    out_nodes, out_edges = network(
                        nodes,
                        edges,
                        indices,
                        checkpoint_steps=True,
                    )
                    loss = sum(value.sum() for value in out_nodes.values())
                    loss = loss + sum(value.sum() for value in out_edges.values())
                    loss.backward()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                print(json.dumps({
                    "device": str(device),
                    "recompute_flops": counter.recompute_flops,
                    "recompute_linear_calls": counter.recompute_linear_calls,
                }))
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", program, str(ROOT)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        measured = json.loads(result.stdout)
        self.assertGreater(measured["recompute_flops"], 0)
        self.assertGreater(measured["recompute_linear_calls"], 0)


if __name__ == "__main__":
    unittest.main()
