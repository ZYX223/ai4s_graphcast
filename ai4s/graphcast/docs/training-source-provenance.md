# Training source provenance

## Pinned project source

The approved colleague training baseline is reused from
`flagos-adaptation-lab/flagos-graphcast-adaptation` at commit
`b7667b42da708db86010fe2d962e5fd0363ffa0a`.

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

## Colleague training model and primitives

The following Task 1 modules are byte-identical in Python AST to the pinned
colleague source:

- `src/graphcast_compat/autoregressive_training.py`
- `src/graphcast_compat/data_prefetch.py`
- `src/graphcast_compat/distributed_training.py`
- `src/graphcast_compat/epoch_plan.py`
- `src/graphcast_compat/gradient_accumulation.py`
- `src/graphcast_compat/recompute_flops.py`
- `src/graphcast_compat/training_recipe.py`
- `src/graphcast_compat/training_schedule.py`
- `src/graphcast_compat/training_state.py`

`src/graphcast_compat/training_model.py` is the pinned colleague implementation
with one target-required call-site adaptation: its graph initialization passes
`canonical=True` explicitly because the shared target graph builder defaults
to the unchanged inference behavior.

The imported behavior retains differentiable normalized-residual feedback,
optional per-variable diagnostics, update-indexed with-replacement sampling,
official WeatherNext latitude and pressure-level weighting, sample-masked
losses, unequal-microbatch scaling, AdamW grouping and gradient clipping,
rank-local RNG state, checksum-protected checkpoints, depth-one CPU prefetch,
real DDP wrapping with unused-parameter support, source manual-synchronization
APIs, and exact activation-recompute FLOP counters.

Local integration adaptations are deliberately narrow:

- `google_small.py` adds only source factory hooks. Its defaults remain
  `linear_type = torch.nn.Linear` and
  `graphnet_type = GoogleSmallGraphNet`; the training subclasses select
  `PrecisionLinear` without changing inference parameter names, shapes,
  initialization order, or values.
- `graph.py` adds the colleague canonical mesh/topology helpers and keeps the
  target `src/graphcast_compat/build_mesh.py` loader. The optional
  `canonical` argument defaults to `False`, inference passes `False`, and
  training passes `True`. The imported
  `assets/google_mesh6_vertices.npy` has shape `(40962, 3)`, dtype
  `float32`, and SHA256
  `aa2462b34f8811cf11006eea1462ed558652cc7c4ce9b3f62ca4786c8814e40a`.
- `accelerator.py` retains the source runner's public device, memory,
  synchronization, RNG, and environment APIs. Its
  `configure_float32_math` resolves the accelerator and delegates directly to
  the target `backend.configure_float32_math`, preserving the target's MUSA
  matmul and platform policy rather than copying the source policy.
- `precision.py` retains the runner-used `PrecisionLinear` and
  `configure_model_precision(model, precision, fp8_backend=...)` call shape,
  but exposes only approved FP32 training. The source's BF16/FP16/FP8 training
  implementation is intentionally omitted and non-FP32 requests fail closed.
- `official_training_data.py` and `official_training_corpus.py` retain the
  target numeric-time, corpus metadata-bootstrap, and delegated-close fixes
  described above. All other named primitive modules remain source-equivalent.
- DDP callers perform accumulation inside `no_sync` and rely on the DDP
  reducer exactly once; the separate manual synchronization method remains for
  its source-compatible non-DDP path and returns zero without collectives for
  DP=1.

Focused tests use seeded hand-derived tensors and real CPU Gloo subprocesses.
They cover inference/training parameter identity, training-only
`PrecisionLinear`, optional diagnostics, two-lead checkpointed gradient
equality, deterministic replacement sampling, prefetch ordering/failure,
canonical mesh shape/dtype/digests, DP=1 zero synchronization, two-rank DDP
accumulation with unused parameters, and bounded peer failure. These are
primitive/integration checks, not full-model accelerator acceptance.

## Formal training entry integration (Task 3)

`training_engine.py`, `training_runtime.py`, `scripts/training/run_training.py`
and `train_official.sh` are slim target-side orchestration composing the pinned
data/primitives above; the colleague's broad epoch runner is not copied.
`configs/training/operational_tf32_schedule.json` preserves the source
`configs/training/published_era5_311k.json` values: 1000 warmup, 299000
one-step cosine, and 11000 curriculum updates at 3e-7, rollout 2 through 12
in increments every 1000 updates. Budget overrides are explicit.

The variable multipliers are from pinned
`scripts/training/run_official_parallel_training.py:75–81`: 2m temperature
1.0, 10m u/v wind, mean sea-level pressure and total precipitation 0.1;
atmospheric variables retain the imported loss builder's default 1.0.
The ordinary source epoch policy is preserved: eligibility is fixed by the
maximum configured rollout, with shorter trajectories extracted by phase.
The source's special update-indexed replacement sampler is not used.

The target's public model placement API already supports the required four
stages, so no shared model or inference behavior was changed. Precision and
strict FlagGems registration reuse the existing target backend. Unknown
forward/backward coverage and training MFU are explicitly unverified/unavailable.
Content hashes identify ordered corpus shards and stats; model/task/channel
configuration, training policy, optimizer, schedule, batch topology, order seed,
and the next epoch cursor are validated when restoring. Deployment paths and
unrelated source hashes are not resume identity.

Task 3 evidence includes hand-derived FP32 BPTT and accumulated updates, real
two-rank CPU Gloo including zero-valid rank participation, collective nonfinite
aborts, exact model/optimizer/RNG/cursor continuation, and subprocess CLI tests
that forbid importing accelerator/data payload modules during dry-run. These
are unit/integration checks, not full-model accelerator or performance evidence.

## Lightweight training loss diagnostic

`scripts/training/diagnose_training_loss.py` adapts the project team's
`flagos-graphcast-operator-handoff/tests/checkpoint_prediction_validation/diagnostic_package/diagnose_training_loss.py`.
The source file read on 2026-09-07 has SHA256
`117d635c1e9732aeb6852110fba8def994acc6647a485f3dd487a98545ea52be`.
It is a project-team diagnostic, not a Google GraphCast weather-skill standard.
The original file is unchanged; its historical small-1deg JAX baseline CSV is
not imported or silently used for operational 25km training.

The reused loss metric is `sum(abs(candidate-reference))/sum(abs(reference))*100`,
with the original inclusive `<= 5%` default. Per-step relative mean/max remain
explanatory metrics, not replacements for that ratio of sums. Adaptations are:

- Require an explicit reference and inclusive optimizer-step interval, reject
  incomplete coverage, duplicate/nonintegral steps and nonfinite loss/gradient
  evidence. Support both logs' column names, including `optimizer_step/loss`.
- Check the shared training recipe and per-update epoch/batch, rollout, phase,
  learning rate and effective samples. A numerically small loss error without
  comparison metadata is measured but unverified, never a formal pass. An
  all-zero-reference or all-zero-learning-rate interval is not acceptance.
- Serialize a small start event in `updates.jsonl` per training invocation,
  reusing the existing resume contract and hashing initial parameters/optimizer
  once outside update timing. Static graph buffers are not hashed. Metadata is
  associated with each invocation, including appended resume segments; it does
  not add a new checkpoint schema or change training numerical behavior.
- Compare semantic recipe fields rather than the entire resume contract:
  Native/FlagGems backend names and vendor/version details can differ. Current
  controlled comparisons require the same DP/MP and microbatch layout as well
  as global batch; alternative layouts need a separate comparison protocol.
  The reference must separately identify as Native with FlagGems disabled;
  both runs' resolved backend/precision must agree with their requests. TF32
  and HF32 are compatible request families, not proof of hardware instructions.
- Replace JAX-specific report labels with generic reference labels and emit
  strict JSON plus exit codes. Do not overwrite inputs or existing results.

The standalone diagnostic imports only the Python standard library. It does
not run training, load checkpoints or allocate accelerator memory, and it does
not compute weather ACC or declare trained-checkpoint weather acceptance.
