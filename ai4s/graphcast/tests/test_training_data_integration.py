from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
CONFIG_PATH = ROOT / "configs/inference/google_graphcast_operational_25km.yaml"
UPSTREAM_COMMIT = "08cf73625c9d12bd9aaa038868bcb2fe488f2a22"
MODEL_LEVELS = np.asarray(
    [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
    dtype=np.int32,
)
SOURCE_LEVELS = MODEL_LEVELS[::-1]


def _write_source(path: Path, *, start: str, time_count: int) -> Path:
    times = np.datetime64(start) + np.arange(time_count) * np.timedelta64(6, "h")
    lat = np.asarray([-0.5, 0.5], dtype=np.float32)
    lon = np.asarray([0.0, 0.25], dtype=np.float32)
    surface = np.arange(time_count * 2 * 2, dtype=np.float32).reshape(1, time_count, 2, 2)
    atmosphere = np.stack(
        [surface + level_index * 10_000 for level_index in range(len(SOURCE_LEVELS))],
        axis=2,
    )
    common = {"batch": [0], "time": times, "lat": lat, "lon": lon}
    with_level = {**common, "level": SOURCE_LEVELS}
    variables: dict[str, xr.DataArray] = {}
    for offset, name in enumerate(
        (
            "2m_temperature",
            "mean_sea_level_pressure",
            "10m_v_component_of_wind",
            "10m_u_component_of_wind",
            "total_precipitation_6hr",
            "toa_incident_solar_radiation",
        )
    ):
        variables[name] = xr.DataArray(
            surface + offset * 100_000,
            dims=("batch", "time", "lat", "lon"),
            coords=common,
        )
    for offset, name in enumerate(
        (
            "temperature",
            "geopotential",
            "u_component_of_wind",
            "v_component_of_wind",
            "vertical_velocity",
            "specific_humidity",
        )
    ):
        variables[name] = xr.DataArray(
            atmosphere + offset * 1_000_000,
            dims=("batch", "time", "level", "lat", "lon"),
            coords=with_level,
        )
    variables["geopotential_at_surface"] = xr.DataArray(
        np.arange(4, dtype=np.float32).reshape(2, 2),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
    )
    variables["land_sea_mask"] = xr.DataArray(
        np.ones((2, 2), dtype=np.float32),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
    )
    xr.Dataset(variables).to_netcdf(path)
    return path


def test_real_window_uses_two_inputs_configured_targets_and_model_levels(tmp_path):
    sys.path.insert(0, str(SOURCE_ROOT))
    try:
        from graphcast_compat.official_training_data import OfficialTrainingDataset
    finally:
        sys.path.remove(str(SOURCE_ROOT))

    source = _write_source(tmp_path / "source.nc", start="2017-01-01T00:00:00", time_count=7)
    dataset = OfficialTrainingDataset(
        source,
        config_path=CONFIG_PATH,
        data_role="train",
        dataset_kind="era5",
        training_stage="era5_pretrain",
        rollout_steps=3,
    )
    try:
        sample = dataset.sample_indices(
            window_indices=(1,),
            sample_weights=(1.0,),
            target_rollout_steps=3,
        )
    finally:
        dataset.close()

    assert sample.rollout_steps == 3
    assert sample.dataset_rollout_steps == 3
    assert sample.inputs.sizes["time"] == 2
    assert sample.targets.sizes["time"] == 3
    assert sample.forcings.sizes["time"] == 3
    np.testing.assert_array_equal(sample.inputs.level.values, MODEL_LEVELS)
    np.testing.assert_array_equal(sample.targets.level.values, MODEL_LEVELS)
    np.testing.assert_array_equal(
        sample.inputs.time.values,
        np.asarray([np.timedelta64(-6, "h"), np.timedelta64(0, "h")]),
    )
    np.testing.assert_array_equal(
        sample.targets.time.values,
        np.asarray(
            [np.timedelta64(6, "h"), np.timedelta64(12, "h"), np.timedelta64(18, "h")]
        ),
    )
    # The source levels are reversed. Level 50 is source position 12, and lead
    # one is source time position 2; output order follows the model config.
    assert float(sample.inputs["temperature"].isel(batch=0, time=0, level=0, lat=1, lon=0)) == 120_000.0
    assert float(sample.targets["temperature"].isel(batch=0, time=0, level=0, lat=1, lon=0)) == 120_008.0


def test_corpus_and_epoch_plan_keep_global_and_rank_local_padding_metadata(tmp_path):
    sys.path.insert(0, str(SOURCE_ROOT))
    try:
        from graphcast_compat.epoch_plan import EpochPlan
        from graphcast_compat.official_training_corpus import OfficialTrainingCorpus
    finally:
        sys.path.remove(str(SOURCE_ROOT))

    source = _write_source(tmp_path / "source.nc", start="2017-01-01T00:00:00", time_count=5)
    manifest = tmp_path / "corpus.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "integration",
                "dataset_kind": "era5",
                "training_stage": "era5_pretrain",
                "data_role": "train",
                "shards": [{"path": source.name}],
            }
        ),
        encoding="utf-8",
    )
    plan = EpochPlan.create(range(3), epoch=0, seed=27, global_batch_size=2, shuffle=False)
    assert plan.batches[0].window_indices == (0, 1)
    assert plan.batches[1].window_indices == (2, 2)
    assert plan.batches[1].sample_weights == (1.0, 0.0)

    corpus = OfficialTrainingCorpus(manifest, config_path=CONFIG_PATH, rollout_steps=1)
    try:
        rank_zero = corpus.sample_indices(
            window_indices=plan.batches[1].window_indices,
            sample_weights=plan.batches[1].sample_weights,
            rank=0,
            replicas=2,
        )
        rank_one = corpus.sample_indices(
            window_indices=plan.batches[1].window_indices,
            sample_weights=plan.batches[1].sample_weights,
            rank=1,
            replicas=2,
        )
    finally:
        corpus.close()

    assert rank_zero.window_indices == (2,)
    assert rank_one.window_indices == (2,)
    assert rank_zero.sample_weights == (1.0,)
    assert rank_one.sample_weights == (0.0,)
    assert rank_zero.global_window_indices == (2, 2)
    assert rank_one.global_window_indices == (2, 2)
    assert rank_zero.global_valid_sample_count == 1
    assert rank_one.global_valid_sample_count == 1


def test_upstream_bridge_imports_data_and_losses_without_jax_or_flaggems():
    code = """
import json
import sys
from graphcast import UPSTREAM_COMMIT, data_utils, losses
from graphcast_compat.official_training_corpus import OfficialTrainingCorpus
print(json.dumps({
    'commit': UPSTREAM_COMMIT,
    'data_utils': data_utils.__file__,
    'losses': losses.__file__,
    'forbidden': sorted(name for name in sys.modules if name == 'jax' or name.startswith('jax.') or name == 'flaggems' or name.startswith('flaggems.')),
}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["commit"] == UPSTREAM_COMMIT
    assert report["forbidden"] == []
    assert "/src/third_party/weathernext/graphcast/data_utils.py" in report["data_utils"]
    assert "/src/third_party/weathernext/graphcast/losses.py" in report["losses"]
