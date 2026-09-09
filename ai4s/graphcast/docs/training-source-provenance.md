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

## Canonical training entry integration (2026-09-09, Task 2)

The sole formal engine is the pinned source
`scripts/training/run_official_epoch_training.py`, with
`official_epoch_training_file_validation.py`. Public `run_training.py`,
`train_official.sh` and `train_pipeline.sh` delegate to it. The superseded
target engine and its execution tests are retired. Task 3 moves the remaining
legacy start-record helper to `training_diagnostics.py` and removes
`training_runtime.py`; no alternate loop is retained.

Source deviations are limited to:

- Import consumed parallel-runner helpers from `training_support.py`, not its
  second trainer. Extracted definitions: PER_VARIABLE_WEIGHTS, json_dump,
  append_jsonl, state_dict_cpu, state_dict_sha256, tensor_components_sha256,
  differing_component_names, configure_float32_math, synchronize,
  reset_peak_memory, memory_by_device, profile_config, validate_formal_dataset,
  validate_metric. Float32 configuration delegates to the existing target
  backend instead of the source accelerator's separate implementation.
- Route replacement setup through `training_backend.py`. It uses the target
  configure_flagos_environment/resolve_runtime_backend and the target training
  YAML, maps layer_norm to native_layer_norm before registration, checks every
  selected op, and retains profile/version/registration/exported-function-source
  metadata inside the existing operator_replacement object. Native imports no
  FlagGems and sets FLAGOS_REPLACE=0. No source operator allowlist is imported.
- The source-native runner defaults tensor-core to enabled and peak to zero
  (unavailable); nonfinite/negative peaks and positive peaks without a source
  are rejected. No engineering peak is claimed as verified.
- Model-only initialization also accepts historical nonempty raw tensor
  state_dict files. Source envelope checks and exact-resume schema/recipe
  validation remain intact. Raw checkpoints do not qualify as exact resume.
- Resume equality uses stable backend/operator/version policy, excluding rank
  device, recording path and registration/export observations. Complete runtime
  evidence and measured cache status are retained per invocation and rank at
  `processes/from_XXXXXXXX/rank_R/runtime.json`, outside update timing. Immutable
  run_config records the cache budget rather than changing preload/validation
  durations. These corrections make fresh-process and cache-enabled resumes usable.
- Source-native validation explicitly requests the existing per-variable loss
  diagnostics from EpochAutoregressiveModule. Training leaves that option off.
  This corrects the source's zero-column validation indexing bug without changing
  the loss/update algorithm. CPU tests verify unchanged training loss/gradients
  and a real source-native validation report covering all variables and 12 leads.

The source loop, update-indexed trajectory sampling with replacement, active
rollout eligibility, weighted BPTT, AdamW, gradient clipping, CPU cache/prefetch,
MUSA serial backward, checkpoint schema3 and metric segment layout are
preserved. Source checkpoint hashes validate checkpoint/state integrity;
the source resume contract records corpus/config/stats paths. It does not
rehash all input content or permit path relocation like the retired target
contract, and is documented accordingly. Prefix stops save numbered checkpoints.

`training_platform.py` adapts execution/allocator/visibility policies from
source `train_pipeline.sh` without its source locks, deploy paths, environment
activation, default physical-card reservations, source maintenance, operator
allowlists or engineering peaks. The public default global batch is 16; local
micro batch 1 is multiplied by DP before forwarding. MP4 defaults to source
manual sync, MP2 to DDP. DP1/DP2 both launch via torchrun unless already inside
a rank; the source initializes a process group even at DP1.
DDP's no_sync encloses forward/backward and no second
manual averaging occurs. Ascend MP2 uses split8/step/non_reentrant from the
source README's validated example instead of inheriting MP4 split16/reentrant.
Explicit public CLI overrides win. Source allocator, Ascend launch blocking,
MUSA descriptor limit and T-Head process teardown settings are retained.
Runtime installation remains the caller's responsibility.

`configs/training/official_2_to_12.json` serializes the built-in source recipe:
1000 warmup, 299000 half-cosine, 11000 curriculum, 2–12 leads with equal budgets.
Public budget overrides retain these semantics; stopping a prefix does not
change schedule identity. The superseded operational_tf32_schedule.json is removed.
The source published_era5_311k, epoch_rehearsal_v1 and hres_adaptation_v1
manifests are copied unchanged for source-native use. Manifest schedules retain
the source's epoch semantics; only official_2_to_12 selects replacement sampling.

`hotspot_profiler.py`, `profiled_mfu.py` and `ascend_profile.py` are copied
verbatim from b7667b4. Profiling stays opt-in; the optional ijson parser has a
stdlib fallback. Source tests for pretrained initialization, optimizer/recompute
and MFU/profiling are imported under tests/training_baseline with path-depth
adaptation; the source-text check for the unported old loss runner is omitted.

New behavioral tests execute public CLI subprocesses, resolve their argv through
the actual source parser/schedule, verify raw initialization versus exact resume,
exercise real target YAML/backend routing with a runtime boundary fixture, and
run the actual source loop with real tiny model/corpus/AdamW/Gloo DDP. Tiny CPU
fixtures substitute only hardware setup/placement and full-size acceptance
gates, preserving canonical graph building and the real training loop.
Six-update tests cross into 2/3-lead BPTT and prove exact model, optimizer,
cursor and loss continuation with prefetch off/on; changed seed is rejected.
These are CPU correctness checks, not full-model device, timing or weather-skill
acceptance. Source-segment paired diagnostic metadata and actual call
observation are described below.

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
- Retain reading old start events; the canonical source now writes a sibling
  `metrics/segment_XXXXXXXX.start.json` per invocation. This survives every
  in-memory metric-list rewrite. Gather real parameters/optimizer fingerprints
  on every rank after initialization/resume and before update timing; rank zero
  hashes ordered corpus shard bytes, three consumed stats files and model YAML.
  Static graph buffers are not hashed. This adds startup I/O, not an optimizer
  update or a checkpoint schema/immutable-resume identity change.
- Compare semantic recipe fields rather than the entire resume contract:
  Native/FlagGems replacement choices can differ. Source-format comparisons
  require equal resolved platform backend/device type, actual initial state,
  content hashes, full non-replacement source recipe, optimizer/variable weights,
  prefetch/cache/backward execution policy and per-rank sample provenance. Paths
  for corpus/config/stats/schedule source are replaced by content/recipe evidence;
  volatile rank device/PID/cache durations remain observations. Current
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

The inclusive 5% boundary allows only 8 binary64 ULPs around 5 (about 7.1e-15
percentage points) to avoid rejecting hand-derived `[2,4]` versus `[2.1,4.2]`.
The ratio-of-sums definition is unchanged; genuine excess still fails.

### Opt-in call observer (Task 3)

`scripts/validation/trace_training_flaggems.py` is adapted from the identically
named pinned colleague file (SHA256
`778f456afa029da52a78859bed87f3509a8ef3b1d0bed210185855726f0f40af`).
It retains logging handlers for FlagGems and the selected vendor hierarchy,
stack-based tensor device/source discovery, source recompute markers and a
torchrun/runpy invocation of the canonical runner. Deliberate deviations:

- Require the target `GRAPHCAST_FLAGOS_MODE=flaggems`, backend and accelerator
  environment, rather than trusting stale `FLAGOS_REPLACE=1`.
- Count startup as initialization (no optimizer update), scope forward to the
  canonical EpochAutoregressiveModule boundary, preserve backward/recompute,
  and label AdamW.step calls optimizer before flushing actual state steps.
- Include rank, canonical trainer path, logged-function count kind and explicit
  observed-only coverage. Refuse to append another invocation to an existing
  rank file and restore Torch functions/logger settings on normal Python exit
  or exception. The source's deliberate MUSA process-exit bypass remains intact;
  each completed update is already flushed before that path.
- Normal training never imports the observer. No operator implementation,
  installed FlagGems file, inference default, YAML, sampler or training algorithm
  changes are introduced by Task 3.

Tests exercise real tiny canonical CPU training/finalization/resume plus real
Torch autograd/AdamW call-phase behavior. They do not prove full-model device
correctness, logged coverage on an installed vendor runtime, performance or
trained-checkpoint weather skill. The controller owns the separately frozen
public-entry Native/FlagGems device pair and source/package evidence.
