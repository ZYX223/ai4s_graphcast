from __future__ import annotations

import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphcast_compat import training_state


@pytest.mark.parametrize("saved,expected,compatible", [
    ({}, {}, True),
    ({}, {"mesh_edge_input_projection": "concatenated"}, True),
    ({"mesh_edge_input_projection": "concatenated"}, {}, True),
    ({"mesh_edge_input_projection": "factorized"}, {}, False),
    ({}, {"mesh_edge_input_projection": "factorized"}, False),
    ({"mesh_edge_input_projection": "factorized"}, {"mesh_edge_input_projection": "factorized"}, True),
])
def test_resume_projection_normalizes_legacy_default_without_allowing_mode_switch(saved, expected, compatible):
    payload = {"metadata": {"resume_contract": saved}}
    if compatible:
        training_state.validate_resume_contract(payload, expected)
    else:
        with pytest.raises(ValueError, match="mesh_edge_input_projection"):
            training_state.validate_resume_contract(payload, expected)


def _run_update(model, optimizer, features, targets):
    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(model(features), targets).backward()
    optimizer.step()


def test_cpu_rng_round_trip_restores_all_generators():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    state = training_state.capture_rng_state(cuda_device_indices=())
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))
    for _ in range(5):
        random.random()
        np.random.random()
        torch.rand(())

    training_state.restore_rng_state(state, cuda_device_indices=())

    assert (random.random(), float(np.random.random()), float(torch.rand(()))) == expected


def test_checkpoint_restore_reproduces_the_next_optimizer_update(tmp_path):
    torch.manual_seed(31)
    model = torch.nn.Linear(2, 1, dtype=torch.float64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    first_features = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
    first_targets = torch.tensor([[0.5]], dtype=torch.float64)
    next_features = torch.tensor([[3.0, 0.25], [-1.0, 2.0]], dtype=torch.float64)
    next_targets = torch.tensor([[1.5], [-0.75]], dtype=torch.float64)
    _run_update(model, optimizer, first_features, first_targets)
    cursor = training_state.TrainingCursor(
        epoch=2,
        next_batch_index=4,
        global_optimizer_step=7,
        epoch_plan_digest="a" * 64,
    )
    payload = training_state.build_training_checkpoint(
        model_state={name: value.detach().cpu() for name, value in model.state_dict().items()},
        optimizer_state=optimizer.state_dict(),
        schedule_state={"name": "training", "digest": "b" * 64},
        cursor=cursor,
        rng_states_by_rank=[training_state.capture_rng_state(cuda_device_indices=())],
        metadata={
            "resume_contract": {
                "corpus_digest": "c" * 64,
                "schedule_digest": "b" * 64,
                "topology": {"dp": 1, "mp": 4},
            }
        },
    )
    checkpoint = tmp_path / "checkpoint.pt"
    latest = tmp_path / "latest.json"
    record = training_state.atomic_save_checkpoint(checkpoint, payload, latest_path=latest)

    _run_update(model, optimizer, next_features, next_targets)
    uninterrupted_model = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    uninterrupted_optimizer_digest = training_state.state_object_sha256(optimizer.state_dict())

    restored_model = torch.nn.Linear(2, 1, dtype=torch.float64)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1.0e-3)
    loaded = training_state.load_training_checkpoint(checkpoint)
    restored_model.load_state_dict(loaded["model_state"])
    restored_optimizer.load_state_dict(loaded["optimizer_state"])
    training_state.validate_resume_contract(
        loaded,
        {
            "corpus_digest": "c" * 64,
            "schedule_digest": "b" * 64,
            "topology": {"dp": 1, "mp": 4},
        },
    )
    _run_update(restored_model, restored_optimizer, next_features, next_targets)

    for name, expected in uninterrupted_model.items():
        torch.testing.assert_close(restored_model.state_dict()[name], expected, rtol=0, atol=0)
    assert training_state.state_object_sha256(restored_optimizer.state_dict()) == (
        uninterrupted_optimizer_digest
    )
    assert loaded["cursor"] == cursor.to_dict()
    assert record.sha256 == training_state.sha256_file(checkpoint)
    assert json.loads(latest.read_text(encoding="utf-8"))["global_optimizer_step"] == 7
    assert not list(tmp_path.glob("*.tmp-*"))


def test_corrupted_checkpoint_is_rejected_by_checksum(tmp_path):
    payload = training_state.build_training_checkpoint(
        model_state={},
        optimizer_state={},
        schedule_state={"digest": "b" * 64},
        cursor=training_state.TrainingCursor(
            epoch=0,
            next_batch_index=0,
            global_optimizer_step=0,
            epoch_plan_digest="a" * 64,
        ),
        rng_states_by_rank=[training_state.capture_rng_state(cuda_device_indices=())],
        metadata={"resume_contract": {}},
    )
    checkpoint = tmp_path / "checkpoint.pt"
    training_state.atomic_save_checkpoint(checkpoint, payload)
    with checkpoint.open("ab") as stream:
        stream.write(b"corruption")

    with pytest.raises(ValueError, match="SHA256"):
        training_state.load_training_checkpoint(checkpoint)
