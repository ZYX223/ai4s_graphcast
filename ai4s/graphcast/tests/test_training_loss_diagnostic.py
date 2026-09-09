"""Behavior tests for the team loss algorithm and strict comparison inputs."""
import csv
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/training/diagnose_training_loss.py"


@lru_cache(maxsize=1)
def diagnostic():
    assert SCRIPT.is_file(), "training loss diagnostic has not been integrated"
    spec = importlib.util.spec_from_file_location("training_loss_diagnostic", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def start_record():
    return {"event": "start", "requested_backend": "native", "resolved_precision": "tf32",
        "resolved_backend": {"flagos_mode": "off", "flagos_status": "disabled"},
        "loss_comparison": {
        "start_optimizer_step": 0, "starting_parameters_sha256": "a" * 64,
        "starting_optimizer_sha256": "b" * 64,
        "resume_contract": {
            "data_sha256": ["c" * 64], "stats_sha256": ["d" * 64],
            "config_sha256": "e" * 64, "schedule_digest": "f" * 64,
            "window_indices_sha256": "1" * 64, "epoch_policy": "fixed_maximum_rollout_universe_v1",
            "seed": 7, "shuffle": True, "global_batch_size": 4,
            "micro_batch_size_per_rank": 1, "dp": 1, "mp": 4,
            "policy": {"storage_dtype": "float32", "tensor_core": "enabled",
                "per_variable_weights": {"2m_temperature": 1.},
                "optimizer": {"name": "AdamW", "betas": [.9, .95]},
                "normalization": "inputs_and_residuals_v1", "backend": "native"},
        },
    }}


def updates(losses=(20., 180.)):
    return [{"event": "update", "optimizer_step": i, "loss": loss,
             "epoch": 0, "batch_index": i - 1, "rollout_steps": 2,
             "phase": "curriculum", "learning_rate": .001,
             "effective_samples": 4, "accumulation_steps": 4,
             "finite": True, "gradient_norm": 1.}
            for i, loss in enumerate(losses, 1)]


def write_log(path, rows, start=None):
    entries = ([start] if start else []) + rows
    path.write_text("".join(json.dumps(row) + "\n" for row in entries))
    return path


def pair(tmp_path, candidate_losses=(24., 180.), reference_losses=(20., 180.)):
    reference = write_log(tmp_path / "native.jsonl", updates(reference_losses), start_record())
    fg_start = start_record()
    fg_start["requested_backend"] = "flaggems"
    fg_start["resolved_backend"] = {"flagos_mode": "flaggems", "flagos_status": "enabled_only_enable"}
    fg_start["loss_comparison"]["resume_contract"]["policy"]["backend"] = "flaggems"
    candidate = write_log(tmp_path / "flaggems.jsonl", updates(candidate_losses), fg_start)
    return candidate, reference


def compare(candidate, reference, **kwargs):
    return diagnostic().compare_losses(candidate, reference, start_step=1, end_step=2, **kwargs)


def test_uses_ratio_of_sums_not_mean_step_percent_and_allows_backend_difference(tmp_path):
    candidate, reference = pair(tmp_path)
    result = compare(candidate, reference)
    # |24-20|/(20+180)=2%; mean of step percentages would incorrectly be 10%.
    assert result["mean_relative_loss_percent"] == pytest.approx(2.)
    assert result["mean_step_relative_percent"] == pytest.approx(10.)
    assert result["status"] == "passed"
    assert result["max_step_relative_percent"] == pytest.approx(20.)
    assert result["weather_skill"] == "not_evaluated"


@pytest.mark.parametrize("losses,want", [((21., 21.), "passed"), ((21.01, 21.01), "failed_loss")])
def test_preserves_inclusive_five_percent_loss_threshold(tmp_path, losses, want):
    candidate, reference = pair(tmp_path, losses, (20., 20.))
    assert compare(candidate, reference)["status"] == want


@pytest.mark.parametrize("side", ["candidate", "reference", "both"])
def test_missing_steps_are_rejected_even_if_remaining_intersection_matches(tmp_path, side):
    candidate, reference = pair(tmp_path)
    for label, path in (("candidate", candidate), ("reference", reference)):
        if side in (label, "both"):
            write_log(path, updates()[:1], start_record())
    with pytest.raises(diagnostic().DiagnosticError, match="missing|incomplete"):
        compare(candidate, reference)


@pytest.mark.parametrize("bad_step", [1, 1.5, True, -1])
def test_duplicate_and_nonintegral_steps_cannot_be_overwritten_or_truncated(tmp_path, bad_step):
    module = diagnostic()
    candidate, reference = pair(tmp_path)
    rows = updates()
    rows[1]["optimizer_step"] = bad_step
    write_log(candidate, rows, start_record())
    with pytest.raises(module.DiagnosticError, match="step"):
        module.compare_losses(candidate, reference, start_step=1, end_step=2)


@pytest.mark.parametrize("field,value", [("loss", float("nan")), ("gradient_norm", float("inf")), ("finite", False)])
def test_nonfinite_or_failed_updates_cannot_pass(tmp_path, field, value):
    module = diagnostic()
    candidate, reference = pair(tmp_path)
    rows = updates(); rows[0][field] = value
    write_log(candidate, rows, start_record())
    with pytest.raises(module.DiagnosticError, match="finite"):
        module.compare_losses(candidate, reference, start_step=1, end_step=2)


@pytest.mark.parametrize("field,value", [("rollout_steps", 3), ("phase", "one_step"),
    ("learning_rate", .002), ("effective_samples", 3), ("batch_index", 3)])
def test_update_training_semantics_must_match(tmp_path, field, value):
    candidate, reference = pair(tmp_path)
    rows = updates(); rows[0][field] = value
    write_log(candidate, rows, start_record())
    result = compare(candidate, reference)
    assert result["status"] == "incomparable_training"
    assert any(field in reason for reason in result["comparison_issues"])


@pytest.mark.parametrize("field,value", [("global_batch_size", 8), ("seed", 8),
    ("data_sha256", ["9" * 64]), ("schedule_digest", "9" * 64)])
def test_run_training_semantics_must_match(tmp_path, field, value):
    candidate, reference = pair(tmp_path)
    start = start_record(); start["loss_comparison"]["resume_contract"][field] = value
    write_log(candidate, updates(), start)
    result = compare(candidate, reference)
    assert result["status"] == "incomparable_training"
    assert any(field in reason for reason in result["comparison_issues"])


def test_different_starting_weights_and_optimizer_are_not_a_backend_parity_run(tmp_path):
    candidate, reference = pair(tmp_path)
    for field in ("starting_parameters_sha256", "starting_optimizer_sha256"):
        start = start_record(); start["loss_comparison"][field] = "9" * 64
        write_log(candidate, updates(), start)
        assert compare(candidate, reference)["status"] == "incomparable_training"


def test_legacy_csv_column_mapping_measures_but_cannot_claim_verified_training(tmp_path):
    candidate = tmp_path / "candidate.csv"
    reference = tmp_path / "reference.csv"
    candidate.write_text("iteration,train_loss\n1,24\n2,180\n")
    reference.write_text("step,jax_loss\n1,20\n2,180\n")
    result = compare(candidate, reference, step_column="iteration", loss_column="train_loss",
                     baseline_step_column="step", baseline_loss_column="jax_loss")
    assert result["mean_relative_loss_percent"] == pytest.approx(2.)
    assert result["status"] == "unverified_comparison"


def test_csv_can_use_explicit_start_metadata_without_renaming_native_as_jax(tmp_path):
    candidate, reference = tmp_path / "candidate.csv", tmp_path / "reference.csv"
    for path in (candidate, reference):
        rows = updates()
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    metadata = tmp_path / "start.json"
    metadata.write_text(json.dumps(start_record()))
    result = compare(candidate, reference, candidate_metadata=metadata, baseline_metadata=metadata)
    assert result["status"] == "passed"


def test_inactive_learning_rate_and_zero_reference_are_not_acceptance(tmp_path):
    candidate, reference = pair(tmp_path)
    for path in (candidate, reference):
        rows = updates()
        for row in rows:
            row["learning_rate"] = 0.
        write_log(path, rows, start_record())
    assert compare(candidate, reference)["status"] == "unverified_comparison"
    candidate, reference = pair(tmp_path, (0., 0.), (0., 0.))
    assert compare(candidate, reference)["status"] == "invalid_reference"


def test_explicit_interval_ignores_other_complete_updates_but_does_not_hide_duplicate_steps(tmp_path):
    candidate, reference = pair(tmp_path)
    for path in (candidate, reference):
        write_log(path, updates((20., 180., 400.)), start_record())
    result = compare(candidate, reference)
    assert result["evaluated_steps"] == [1, 2]
    assert result["status"] == "passed"


def test_context_is_associated_with_each_resume_segment(tmp_path):
    candidate, reference = pair(tmp_path)
    rows = updates()
    second_start = start_record()
    second_start["loss_comparison"]["start_optimizer_step"] = 1
    second_start["loss_comparison"]["starting_parameters_sha256"] = "9" * 64
    write_log(candidate, [rows[0], second_start, rows[1]], start_record())
    assert compare(candidate, reference)["status"] == "incomparable_training"


def test_state_hash_record_feeds_legacy_diagnostic_without_training_loop(tmp_path):
    sys.path.insert(0, str(ROOT / "src"))
    from graphcast_compat.training_diagnostics import training_start_record
    from graphcast_compat.training_recipe import build_graphcast_adamw
    import torch
    from types import SimpleNamespace
    contract = start_record()["loss_comparison"]["resume_contract"]
    def session():
        torch.manual_seed(7)
        model = torch.nn.Linear(1, 1)
        return SimpleNamespace(model=model, optimizer=build_graphcast_adamw(model, lr=.001),
            cursor=SimpleNamespace(global_optimizer_step=0), resume_contract=contract)
    paths = [tmp_path / "first.jsonl", tmp_path / "second.jsonl"]
    for path in paths:
        trainer = session()
        start = training_start_record(trainer, {
            "requested_backend": "native", "resolved_precision": "tf32",
            "resolved_backend": {"flagos_mode": "off", "flagos_status": "disabled"}})
        rows = updates()
        write_log(path, rows, start)
    result = compare(*paths)
    assert result["mean_relative_loss_percent"] == 0.
    assert result["status"] == "passed"


@pytest.mark.parametrize("name,empty", [("data_sha256", []), ("policy", {})])
def test_empty_comparison_evidence_is_not_verified(tmp_path, name, empty):
    candidate, reference = pair(tmp_path)
    start = start_record(); start["loss_comparison"]["resume_contract"][name] = empty
    for path in (candidate, reference):
        write_log(path, updates(), start)
    assert compare(candidate, reference)["status"] == "unverified_comparison"


@pytest.mark.parametrize("a,b", [("-0.001", "-0.001"), ("-0.001", "0.001")])
def test_invalid_learning_rate_is_rejected_even_when_another_update_is_active(tmp_path, a, b):
    module = diagnostic()
    candidate, reference = pair(tmp_path)
    for path, value in ((candidate, a), (reference, b)):
        rows = updates(); rows[0]["learning_rate"] = value
        write_log(path, rows, start_record())
    with pytest.raises(module.DiagnosticError, match="learning_rate"):
        module.compare_losses(candidate, reference, start_step=1, end_step=2)


def test_cli_is_stdlib_only_explicit_baseline_and_outputs_nonpassing_exit_status(tmp_path):
    assert SCRIPT.is_file(), "training loss diagnostic has not been integrated"
    candidate, reference = pair(tmp_path)
    output = tmp_path / "diagnostic.json"
    # A real isolated interpreter cannot load site-packages (torch, numpy, xarray).
    command = [sys.executable, "-I", "-S", str(SCRIPT), str(candidate),
               "--baseline", str(reference), "--start-step", "1", "--end-step", "2",
               "--output-json", str(output)]
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    assert result["status"] == "passed"
    assert result["baseline"] == str(reference.resolve())
    assert result["metric_passed"] is True
    assert result["weather_skill"] == "not_evaluated"
    # Do not overwrite a prior result, especially not an input log.
    repeated = subprocess.run(command, text=True, capture_output=True)
    assert repeated.returncode != 0
    assert json.loads(output.read_text()) == result
    command[-1] = str(candidate)
    collided = subprocess.run(command, text=True, capture_output=True)
    assert collided.returncode != 0
    assert candidate.read_text().startswith('{"event": "start"')


@pytest.mark.parametrize("threshold", [-1., float("nan"), float("inf")])
def test_invalid_threshold_cannot_manufacture_pass(tmp_path, threshold):
    module = diagnostic()
    candidate, reference = pair(tmp_path)
    with pytest.raises(module.DiagnosticError, match="threshold"):
        module.compare_losses(candidate, reference, start_step=1, end_step=2, threshold=threshold)


@pytest.mark.parametrize("backend,want", [("flaggems", "incomparable_training"), (None, "unverified_comparison")])
def test_reference_must_be_identified_as_native_not_just_named_baseline(tmp_path, backend, want):
    candidate, reference = pair(tmp_path)
    start = start_record(); start["requested_backend"] = backend
    start["loss_comparison"]["resume_contract"]["policy"]["backend"] = backend
    if backend == "flaggems":
        start["resolved_backend"] = {"flagos_mode": "flaggems", "flagos_status": "enabled_only_enable"}
    write_log(reference, updates(), start)
    assert compare(candidate, reference)["status"] == want


@pytest.mark.parametrize("runtime,want", [
    ({"flagos_mode": "flaggems", "flagos_status": "enabled_only_enable"}, "incomparable_training"),
    ({"flagos_mode": "off", "flagos_status": "unsupported"}, "incomparable_training"),
    ({}, "unverified_comparison"),
])
def test_reference_request_alone_does_not_prove_native_resolution(tmp_path, runtime, want):
    candidate, reference = pair(tmp_path)
    start = start_record(); start["resolved_backend"] = runtime
    write_log(reference, updates(), start)
    assert compare(candidate, reference)["status"] == want


def test_resolved_precision_must_match_requested_math_family(tmp_path):
    candidate, reference = pair(tmp_path)
    start = start_record(); start["resolved_precision"] = "ieee_fp32"
    write_log(reference, updates(), start)
    assert compare(candidate, reference)["status"] == "incomparable_training"


def test_hf32_and_tf32_are_compatible_requests_not_equal_vendor_strings(tmp_path):
    candidate, reference = pair(tmp_path)
    start = start_record(); start["resolved_precision"] = "hf32"
    write_log(candidate, updates(), start)
    assert compare(candidate, reference)["status"] == "passed"


def source_start(start=0):
    return {"schema": "graphcast.training_segment_comparison.v1",
        "start_optimizer_step": start,
        "states_by_rank": [{"rank": 0, "parameters_sha256": "a" * 64,
                            "optimizer_sha256": "b" * 64}],
        "data_sha256": ["c" * 64], "stats_sha256": {"mean.nc": "d" * 64},
        "config_sha256": "e" * 64, "resolved_precision": "tf32",
        "optimizer": {"name": "AdamW", "betas": [.9, .95]},
        "per_variable_weights": {"2m_temperature": 1.},
        "execution_policy": {"data_prefetch_depth": 0, "split_cache_max_gib": 0.,
                             "backward_execution": "runtime_default"},
        "normalization": "inputs_and_residuals_v1",
        "runtime_by_rank": [{"rank": 0, "mode": "native", "enabled": False,
            "operators": [], "skipped": [], "runtime": {
                "backend": "nvidia", "device_type": "cuda",
                "flagos_mode": "off", "flagos_status": "disabled"}}],
        "resume_contract": {"precision": "fp32", "tensor_core_enabled": True,
            "schedule_digest": "f" * 64, "seed": 7,
            "training_sampling": "trajectory_with_replacement_per_optimizer_update",
            "global_batch_size": 2, "global_micro_batch_size": 1,
            "local_micro_batch_size": 1, "gradient_accumulation_steps": 2,
            "data_parallel_replicas": 1, "mp_devices_per_replica": 4,
            "gradient_sync": "manual", "activation_checkpoint": True,
            "rollout_checkpoint_segments": 1, "rollout_checkpoint_implementation": "reentrant",
            "single_step_rollout_checkpoint": "enabled", "mesh_checkpoint_granularity": "step",
            "loss_grid_chunk_size": 32, "graph_topology_version": "canonical_v1"}}


def source_rows(losses):
    return [{"step": i, "optimizer_step": i, "loss": loss, "epoch": 0,
        "epoch_batch_index": i - 1, "rollout_steps": 2, "schedule_phase": "warmup",
        "learning_rate": .001, "valid_sample_count": 2,
        "gradient_accumulation_steps": 2, "grad_ok": True, "grad_norm_before_clip": 1.,
        "training_sampling": "trajectory_with_replacement_per_optimizer_update",
        "origins_by_rank": [{"rank": 0, "global_window_indices": [4, 4],
            "local_window_indices": [4, 4], "sample_weights": [1., 1.],
            "global_sample_weights": [1., 1.], "reference_datetimes": ["t4", "t4"],
            "target_datetimes_by_sample": [["t5", "t6"], ["t5", "t6"]]}]}
        for i, loss in enumerate(losses, 1)]


def source_segment(root, rows, metadata):
    metrics = root / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    path = metrics / f"segment_{metadata['start_optimizer_step'] + 1:08d}.jsonl"
    write_log(path, rows)
    path.with_suffix(".start.json").write_text(json.dumps(metadata))
    return path


@pytest.mark.parametrize("losses,want", [([2., 4.], "passed"),
    ([2.1, 4.2], "passed"), ([2.2, 4.4], "failed_loss"),
    ([2.1000000001, 4.2000000002], "failed_loss")])
def test_source_final_metrics_and_sidecar_support_hand_derived_gate(tmp_path, losses, want):
    candidate, reference = tmp_path / "candidate", tmp_path / "reference"
    path = source_segment(candidate, source_rows(losses), source_start())
    source_segment(reference, source_rows([2., 4.]), source_start())
    # Source finalization rewrites only metric rows; start evidence survives.
    write_log(path, source_rows(losses))
    assert compare(candidate, reference)["status"] == want
    assert compare(path, reference)["status"] == want


@pytest.mark.parametrize("change", ["order", "precision", "parameters", "optimizer", "native_replaced", "memory", "backend", "weights", "cache"])
def test_source_evidence_mismatches_cannot_qualify(tmp_path, change):
    candidate, reference = tmp_path / "candidate", tmp_path / "reference"
    start, rows = source_start(), source_rows([2., 4.])
    source_segment(reference, rows, start)
    if change == "order": rows[0]["origins_by_rank"][0]["global_window_indices"] = [4, 5]
    if change == "precision": start["resolved_precision"] = "ieee_fp32"
    if change in ("parameters", "optimizer"): start["states_by_rank"][0][change + "_sha256"] = "9" * 64
    if change == "native_replaced": start["runtime_by_rank"][0]["enabled"] = True
    if change == "memory": start["resume_contract"]["loss_grid_chunk_size"] = 64
    if change == "backend": start["runtime_by_rank"][0]["runtime"]["backend"] = "hygon"
    if change == "weights": start["per_variable_weights"] = {"2m_temperature": 2.}
    if change == "cache": start["execution_policy"]["split_cache_max_gib"] = 8.
    source_segment(candidate, rows, start)
    assert compare(candidate, reference)["status"] == "incomparable_training"


def test_source_native_and_actual_enabled_flaggems_are_comparable(tmp_path):
    start = source_start()
    reference = source_segment(tmp_path / "reference", source_rows([2., 4.]), start)
    runtime = start["runtime_by_rank"][0]
    runtime.update(mode="flaggems", enabled=True, operators=["silu"])
    runtime["runtime"].update(flagos_mode="flaggems", flagos_status="enabled_only_enable")
    candidate = source_segment(tmp_path / "candidate", source_rows([2.1, 4.2]), start)
    assert compare(candidate, reference)["status"] == "passed"


@pytest.mark.parametrize("missing", ["step", "sidecar", "origins", "state", "runtime"])
def test_source_missing_evidence_never_becomes_verified(tmp_path, missing):
    candidate, reference = tmp_path / "candidate", tmp_path / "reference"
    rows, start = source_rows([2., 4.]), source_start()
    source_segment(reference, rows, start)
    if missing == "step": rows.pop()
    if missing == "origins": rows[0].pop("origins_by_rank")
    if missing == "state": start["states_by_rank"][0].pop("optimizer_sha256")
    if missing == "runtime": start["runtime_by_rank"] = []
    path = source_segment(candidate, rows, start)
    if missing == "sidecar": path.with_suffix(".start.json").unlink()
    if missing == "step":
        with pytest.raises(diagnostic().DiagnosticError, match="missing"):
            compare(candidate, reference)
    else:
        assert compare(candidate, reference)["status"] == "unverified_comparison"


def test_source_resume_binds_each_segment_state_and_cli_is_stdlib_only(tmp_path):
    candidate, reference = tmp_path / "candidate", tmp_path / "reference"
    for root in (candidate, reference):
        source_segment(root, source_rows([2., 4.])[:1], source_start())
        source_segment(root, source_rows([2., 4.])[1:], source_start(1))
    command = [sys.executable, "-I", "-S", str(SCRIPT), str(candidate),
        "--baseline", str(reference), "--start-step", "1", "--end-step", "2",
        "--output-json", str(tmp_path / "result.json")]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    start = source_start(1); start["states_by_rank"][0]["parameters_sha256"] = "9" * 64
    source_segment(candidate, source_rows([2., 4.])[1:], start)
    assert compare(candidate, reference)["status"] == "incomparable_training"


@pytest.mark.parametrize("losses,metadata,want", [((32., 180.), True, 1), ((24., 180.), False, 2)])
def test_cli_loss_failure_and_missing_context_return_distinct_nonzero_codes(tmp_path, losses, metadata, want):
    candidate, reference = pair(tmp_path, losses)
    if not metadata:
        write_log(candidate, updates(losses))
    result = subprocess.run([sys.executable, "-I", "-S", str(SCRIPT), str(candidate),
        "--baseline", str(reference), "--start-step", "1", "--end-step", "2",
        "--output-json", str(tmp_path / "result.json")], capture_output=True, text=True)
    assert result.returncode == want, result.stdout + result.stderr
