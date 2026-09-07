# Training source provenance

## Pinned project source

The real-window training data implementation is reused from the project team's
`graphcast-multiplatform` repository at commit
`6c5654c10c32a9a9b018eac88e4653a2b58be13d`.

The following files are byte-identical to that commit:

- `src/graphcast_compat/epoch_plan.py`
- `src/graphcast_compat/official_solar.py`
- `src/graphcast/__init__.py`
- `src/graphcast/solar_radiation.py`

The implementation retains lazy NetCDF access, cross-shard window assembly,
deterministic sampling, rank-local/global sample metadata, and zero-weight
epoch padding from that source. The NumPy solar module and the local
`graphcast.solar_radiation` adapter preserve the colleague Torch-only import
bridge; they replace only the upstream module's JAX dependency.

## Pinned Google DeepMind source

The minimal upstream dependency is from Google DeepMind WeatherNext commit
`08cf73625c9d12bd9aaa038868bcb2fe488f2a22` under
`src/third_party/weathernext/`. Its Apache-2.0 `LICENSE` and
`UPSTREAM_COMMIT` record are included.

The following upstream files are byte-identical to the pinned revision:

- `graphcast/data_utils.py`
- `graphcast/losses.py`
- `graphcast/xarray_tree.py`

The upstream JAX `graphcast/solar_radiation.py` is intentionally not imported
or copied into the runtime path. The project bridge resolves that name to the
Torch-compatible NumPy adapter above. Consequently, importing `data_utils`,
`losses`, or the training corpus does not import JAX or FlagGems.

## Local integration adaptations

- Existing target `graphcast_compat.data`, `normalization`, `google_small`, and
  backend modules remain unchanged; the imported data code reads the target
  operational YAML contract through its existing `config_path` interface.
- `src/graphcast_compat/official_training_data.py` and
  `official_training_corpus.py` start from the colleague implementations but
  have three local source/lifecycle adaptations: numeric-time sources preserve
  a singleton `time` dimension while still rejecting non-singleton extra or
  misaligned datetime coordinates; corpus first-shard bootstrap uses an
  internal metadata-only dataset mode while each shard's time deltas and the
  requested rollout are validated against the assembled global corpus
  timeline, and corpus close delegates to `OfficialTrainingDataset.close()`
  so extraction-cache storage is released. Standalone dataset validation is
  unchanged.
- `pandas` and `typing_extensions` are declared as direct runtime dependencies
  of the imported bridge/upstream modules. JAX is not a runtime dependency.
- Behavior tests were adapted to pytest and the target operational 13-level
  configuration. Their in-memory NetCDF fixtures verify two-frame inputs,
  configurable future targets, pressure-level reordering, seeded ordering,
  cross-shard continuity, and padded/rank-local weights.

No HRES conversion, deployment, dataset, checkpoint, or runtime-launch code is
included in this integration.

## Differentiable training primitives

The following Task 2 modules are derived from the same pinned project-team
commit `6c5654c10c32a9a9b018eac88e4653a2b58be13d`:

- `src/graphcast_compat/autoregressive_training.py`
- `src/graphcast_compat/gradient_accumulation.py`
- `src/graphcast_compat/training_recipe.py`
- `src/graphcast_compat/training_schedule.py`
- `src/graphcast_compat/training_state.py`
- `src/graphcast_compat/distributed_training.py`

The imported behavior retains differentiable normalized-residual feedback,
official WeatherNext latitude and pressure-level weighting, sample-masked
losses, unequal-microbatch scaling, the published AdamW parameter grouping and
gradient clip, update-indexed learning-rate/rollout schedules, rank-local RNG
state, semantic optimizer-boundary checkpoints, and checksum-protected atomic
checkpoint I/O.

Local integration adaptations are deliberately narrow:

- Autoregressive batch construction imports the target's existing
  `graphcast_compat.data` and `normalization` functions. Reentrant and
  non-reentrant rollout checkpointing use PyTorch's built-in checkpoint
  implementation when the colleague profiling package is absent; no broad
  profiling module was imported.
- `training_accelerator.py` contains only device discovery/context, collective
  backend selection, synchronization, and accelerator RNG calls needed by
  training state and distributed context. Vendor imports are delegated to the
  existing target backend. It does not contain precision, FlagGems, memory,
  environment-repair, or runtime-routing policy.
- Distributed execution is DP-over-MP: each process owns a disjoint contiguous
  model-parallel device group, and gradients are synchronized once per update
  in deterministic parameter-name order. The colleague's unused DDP wrapper
  and profiler-result merge functions are intentionally omitted so the target
  engine cannot accidentally apply a second gradient average or inherit broad
  profiling code.
- Resume identity is expressed by caller-supplied semantic fields in
  `metadata.resume_contract`; deployment paths and unrelated source hashes are
  not required. The historical `cuda_device_indices` checkpoint field name is
  retained for compatibility while its `accelerator_type` distinguishes CUDA,
  MUSA, and NPU state.

The focused pytest ports use hand-derived tensor values and real CPU Gloo
collectives. They cover two-lead full BPTT, checkpointed/uncheckpointed
loss-gradient equality, upstream weighting, zero sample padding, unequal
microbatch updates, schedule boundaries, restored next-update equivalence, and
valid-sample DP scaling. No source-text assertion or accelerator acceptance
claim is included.
