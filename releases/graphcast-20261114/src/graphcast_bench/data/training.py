"""Load checked real targets and form deterministic two-input windows."""

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr
from graphcast import data_utils

from graphcast_bench.training import recipe


def prepare_training_data(weights, rollout, data_root, assets, tiny=False, preflight=False):
    data_root, assets = Path(data_root), Path(assets)
    source = json.loads((data_root / "source.json").read_text())
    data_path = data_root / "hres-20220101-025-13-12.nc"
    h = hashlib.sha256()
    with data_path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    assert h.hexdigest() == source["sha256"]
    raw = xr.load_dataset(
        data_root / ("hres-20220101-025-13-1.nc" if preflight else "hres-20220101-025-13-12.nc"),
        decode_timedelta=True,
    )
    if tiny:
        raw = raw.isel(lat=slice(None, None, 120), lon=slice(None, None, 120))
        cfg = dataclasses.replace(
            weights.model_config, resolution=30, mesh_size=1, latent_size=8, gnn_msg_steps=2
        )
    else:
        cfg = weights.model_config
    assert tiny or (
        raw.sizes["lat"] == 721 and raw.sizes["lon"] == 1440 and raw.sizes["level"] == 13
    )
    assert (raw.sizes["time"] >= 14 or preflight) and raw.sizes["batch"] == 1
    if preflight:
        assert rollout == 1
        plan = [dict(r, starts=[0, 0]) for r in recipe.schedule()]
    else:
        plan = recipe.sample_plan(raw.sizes["time"], 27)
    stats = {
        n: xr.load_dataset(assets / "assets/stats" / (n + ".nc")).astype(
            np.float32
        )
        for n in ["stddev_by_level", "mean_by_level", "diffs_stddev_by_level"]
    }

    def get_batch(start):
        selected = raw.isel(time=slice(start, start + rollout + 2))
        b = data_utils.extract_inputs_targets_forcings(
            selected,
            target_lead_times=slice("6h", str(6 * rollout) + "h"),
            **dataclasses.asdict(weights.task_config),
        )
        b = tuple(x.astype(np.float32) for x in b)
        assert b[0].sizes["time"] == 2 and b[1].sizes["time"] == rollout
        assert all(np.isfinite(v.values).all() for ds in b for v in ds.data_vars.values())
        assert all(np.any(v.values != 0) for v in b[1].data_vars.values())
        return b

    return cfg, source, plan, stats, get_batch
