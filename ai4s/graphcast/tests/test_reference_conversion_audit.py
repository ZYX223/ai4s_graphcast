import importlib

import numpy as np
import pytest
import torch


def test_conversion_audit_checks_names_transpose_dtype_and_all_parameters(tmp_path):
    jax, target = tmp_path / "jax.npz", tmp_path / "torch.pt"
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.savez(jax, **{"params:gnn/~_networks_builder/mlp/~/linear_0:w": values,
                    "params:gnn/~_networks_builder/norm:scale": np.ones(3, dtype=np.float32)})
    state = {"gnn.mlp.linear_0.weight": torch.from_numpy(values.T.copy()), "gnn.norm.weight": torch.ones(3)}
    torch.save(state, target)
    audit = importlib.import_module("scripts.inference.audit_reference_conversion")
    result = audit.audit_conversion(jax, target)
    assert result["status"] == "passed"
    assert result["parameter_count"] == 2
    state["gnn.mlp.linear_0.weight"][0, 0] += 1
    torch.save(state, target)
    with pytest.raises(ValueError, match="parameter"):
        audit.audit_conversion(jax, target)


@pytest.mark.parametrize("change", ["extra", "missing", "dtype"])
def test_conversion_audit_rejects_partial_or_cast_checkpoints(tmp_path, change):
    jax, target = tmp_path / "jax.npz", tmp_path / "torch.pt"
    np.savez(jax, **{"params:gnn/norm:scale": np.ones(3, dtype=np.float32)})
    state = {"gnn.norm.weight": torch.ones(3)}
    if change == "extra":
        state["unknown.weight"] = torch.ones(1)
    elif change == "missing":
        state.clear()
    else:
        state["gnn.norm.weight"] = state["gnn.norm.weight"].to(torch.bfloat16)
    torch.save(state, target)
    with pytest.raises(ValueError):
        importlib.import_module("scripts.inference.audit_reference_conversion").audit_conversion(jax, target)
