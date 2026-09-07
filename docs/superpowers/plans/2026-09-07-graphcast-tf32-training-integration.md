# GraphCast TF32 Training Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the approved colleague training capabilities while retaining the unified project's tested inference path and TF32/FlagGems controls.

**Architecture:** Reuse data and training primitives from the pinned colleague source, retaining a minimal pinned DeepMind dependency. Introduce a separate formal training entry with a small reusable update engine and checkpoint lifecycle; adapt the existing model instead of replacing it.

**Tech Stack:** Python, PyTorch, NumPy, xarray, NetCDF, pytest; platform Torch supplied externally.

**Spec:** `docs/superpowers/specs/2026-09-07-graphcast-tf32-training-integration.md`

## Global Constraints

- Target worktree: `/mdata/data/zhangyuxin/wxt/ai4s-graphcast-profile-routing-worktree`; base `3e435216adb7af56a32e02a4a8490dde3d405eb0`.
- Colleague source: `/mdata/data/zhangym/zym/graphcast-multiplatform` at `6c5654c10c32a9a9b018eac88e4653a2b58be13d`, read-only via `sudo -n -u zhangym` where needed. Read its applicable AGENTS instructions before access.
- Preserve inference defaults/tests and the existing short `train_mp` entry. Do not overwrite shared model/data/normalization/backend modules with colleague versions.
- FP32 storage with existing TF32/HF32 controls only; no BF16, FP8, autocast, implicit Native fallback or FlagGems source changes.
- Full-model device acceptance is MP=4, DP=1 or DP=2, complete 0.25-degree/mesh6/512/16-step/13-level model. CPU tensor fixtures are unit tests only.
- Keep gradient-bearing feedback, valid-sample weighting, optimizer-boundary restore, and configurable rollout. Native and FlagGems use the same training loop.
- Tests first; local file edits through apply_patch. Preserve unrelated work. Commit only assigned files, do not push or run accelerator jobs.
- Source modules and relevant behavior tests are reused from the pinned colleague tree, with provenance. No deployment tools, broad environment copies, datasets or outputs.

### Task 1: Real-window data and upstream dependencies

**Files:**
- Create under `ai4s/graphcast/src/graphcast_compat/`: `official_training_data.py`, `official_training_corpus.py`, `epoch_plan.py`, and `official_solar.py` only if required by the existing colleague bridge.
- Create the colleague `src/graphcast/` import bridge and minimal required `src/third_party/weathernext/` files under the target model root, with LICENSE and UPSTREAM_COMMIT.
- Create `ai4s/graphcast/tests/test_training_data_integration.py`, plus relevant data/corpus/epoch behavior tests ported from the colleague tests. Do not include HRES-converter or deployment tests unless their code is in scope.
- Create `ai4s/graphcast/docs/training-source-provenance.md` listing original modules, commit and local adaptations.
- Modify `ai4s/graphcast/requirements.txt` only for direct runtime dependencies actually required by the added modules, without adding JAX to the Torch runtime.

**Interfaces:**
- Consumes existing `GoogleSmallConfig`, `graphcast_compat.data` and `normalization` APIs; adapt imported callers when needed.
- Produces colleague `OfficialTrainingDataset`, `OfficialTrainingCorpus`, `TrainingSample`, `EpochPlan` interfaces, including `sample_weights`, `global_valid_sample_count`, and rank-local sample metadata.
- Upstream import `from graphcast import data_utils, losses, UPSTREAM_COMMIT` must work without importing JAX or FlagGems.

- [x] Add tests first. Port small in-memory NetCDF fixtures from colleague tests. Check actual two-input-frame / future-target extraction, configurable rollout, model pressure-level selection, fixed-seed ordering, and final-batch zero weighting. A hand-checked epoch case is:

```python
plan = EpochPlan.create(range(3), epoch=0, seed=27, global_batch_size=2)
assert plan.batches[0].window_indices == (0, 1)
assert plan.batches[1].window_indices == (2, 2)
assert plan.batches[1].sample_weights == (1.0, 0.0)
```

- [x] Run `python -m pytest ai4s/graphcast/tests/test_training_data_integration.py -q`; record expected failure due to absent modules/behavior, not a fixture typo.
- [x] Port the selected pinned implementations using apply_patch. Retain original algorithms, lazy NetCDF access and upstream copyright. Reuse the colleague Torch-only solar bridge only to satisfy the data dependency. Resolve config/stat differences at callers; avoid unrelated features.
- [x] Run focused tests, then `python -m pytest ai4s/graphcast/tests -q`; inspect imports with `PYTHONPATH=ai4s/graphcast/src python -c 'from graphcast import data_utils, losses; from graphcast_compat.official_training_corpus import OfficialTrainingCorpus'`.
- [x] Self-review source diff/provenance and commit assigned files as `feat(graphcast): integrate real-window training data`.

### Task 2: Differentiable training, batch and restore primitives

**Files:**
- Create under `ai4s/graphcast/src/graphcast_compat/`: `autoregressive_training.py`, `gradient_accumulation.py`, `training_recipe.py`, `training_schedule.py`, `training_state.py`, `distributed_training.py`.
- Add `training_accelerator.py` only as minimal communication/RNG/device adapter needed by the imported context; use the existing backend instead of duplicating precision or FlagGems management.
- Add focused tests corresponding to autoregressive training, accumulation, schedules, state and distributed valid-sample scaling. Extend source provenance.

**Interfaces:**
- Consumes Task 1 data/upstream functions and target `NormalizationStats`/config.
- Produces `prepare_autoregressive_batch(inputs, targets, forcings, *, config, stats, input_device, storage_device, per_variable_weights, sample_weights)` and `autoregressive_loss(model, batch, *, checkpoint_segments=0, checkpoint_implementation='non_reentrant', loss_grid_chunk_size=0)` with differentiable scalar loss and per-variable lead losses.
- Produces colleague `MicrobatchPlan`, `TrainingSchedule`, `TrainingCursor`, RNG capture/restore, `build_training_checkpoint`, atomic save/load, `build_graphcast_adamw`, `clip_graphcast_gradients`, and `DistributedContext`/valid-sample loss scaling.
- Standard update uses `optimizer.zero_grad(set_to_none=True)`, weighted microbatch loss `.backward()`, DP gradient synchronization once, global clip, then `optimizer.step()`.

- [x] Port relevant behavior tests first. Required hand-derived cases include a two-lead scalar residual model with full BPTT; checkpointed and uncheckpointed loss/gradient equality; nonuniform latitude/pressure weights against pinned upstream; zero-padding effects; full-batch versus unequal microbatch update equality. Use colleague fixtures where they assert real values, not source text.
- [x] Run selected new tests and record missing-feature failures.
- [x] Port the pinned primitives with minimal API adaptations. Keep direct official weighting functions. Keep FP32 loss arithmetic and state feedback attached. Use existing normalization/data modules. Preserve parameter grouping, zero-LR optimizer bookkeeping, and exact valid-sample scaling.
- [x] Keep semantic checkpoint fields and atomic/checksum integrity; avoid absolute deployment paths and unrelated code hashes as resume identity. Test restored next update, not merely object serialization.
- [x] For DP, retain disjoint MP device groups with DP ranks and the colleague manual synchronization mechanism; no inference of one-rank-one-GPU semantics. Adapt accelerator methods without copying stale precision flags. CPU Gloo fixtures can validate communication behavior.
- [x] Run the new focused tests and full target test suite. Commit assigned files as `feat(graphcast): integrate autoregressive training primitives`.

### Task 3: Formal TF32 training entry and integration

**Files:**
- Create `ai4s/graphcast/src/graphcast_compat/training_engine.py` (one optimizer update over weighted microbatches).
- Create `ai4s/graphcast/src/graphcast_compat/training_runtime.py` (model/precision/backend placement, data iteration and restore lifecycle).
- Create `ai4s/graphcast/scripts/training/run_training.py` and `train_official.sh` (CLI and launch only).
- Create `ai4s/graphcast/configs/training/operational_tf32_schedule.json` for parameterized one-step warmup/training and 2-to-12 curriculum; stage budgets overridable by CLI.
- Modify `ai4s/graphcast/scripts/training/README.md` and the GraphCast README with invocation, evidence limits, exact batch/DP/MP semantics and restore examples.
- Create `ai4s/graphcast/tests/test_training_engine.py`, `test_training_runtime.py`, `test_training_entry.py`.
- Modify shared `google_small.py` only if a necessary differentiable placement gap is demonstrated by tests. Existing inference must remain unchanged.

**Interfaces:**
- Consumes Tasks 1 and 2 plus existing `resolve_runtime_backend`, `configure_float32_math`, `configure_flagos_environment`, `GoogleSmallCompatibleModel`, checkpoint loading and config APIs.
- Produces `run_training_update(model, optimizer, microbatches, *, context, checkpoint_segments=0, loss_grid_chunk_size=0)`; each microbatch supplies an AutoregressiveBatch and its effective global-mean scale. Return loss/lead/variable diagnostics and pre-clip gradient norm, without serializing graph-bearing tensors.
- CLI exposes config, corpus, stats, pretrained checkpoint versus resume checkpoint, output directory, devices/DP, batch and microbatch, stage budgets, tensor-core enabled/disabled, Native/FlagGems selection, checkpoint segments/loss chunk, seed, and dry-run. Default formal precision is TF32/HF32 with FP32 storage.
- Reuse existing model stage placement: input stage, two halves of 16 mesh steps, output stage over four devices. Retain current activation checkpoint behavior; autoregressive segment checkpoint and weighted loss chunking wrap it.

- [x] Write integration tests first. Compare a complete two-update sequence with stop/save/reconstruct/resume; compare unequal/padded accumulated updates with a single valid global batch. Test schedule transitions and no repeated data on restore. Test engine with FP32 tensor fixtures; these are not full-model acceptance.
- [x] Add CLI tests using subprocess `--help` and `--dry-run` that validate MP/DP/batch arithmetic before allocating accelerator tensors or importing FlagGems. Verify unsupported precision and ambiguous pretrained/resume inputs fail clearly. No source-text tests.
- [x] Run tests and record missing-feature failures.
- [x] Implement the small update engine by composing the imported loss/optimizer/context functions. Normalize valid samples globally, synchronize/clip/update once. Handle zero-valid rank participation without skipping collectives. Fail on nonfinite loss/gradients.
- [x] Implement runtime and CLI using one code path for Native/FlagGems. Call existing float32 math configuration before model execution. Preserve FP32 storage, current inference defaults, and old short training scripts. Do not copy the 3844-line colleague runner.
- [x] Log requested and resolved precision/backend, topology, effective samples, update/epoch/rollout, data/compute times, loss, finite/grad norm, memory and checkpoint status. Label unmeasured coverage/MFU unavailable rather than inventing numbers. Checkpoint only after successful optimizer updates and preserve semantic restore consistency.
- [x] Run focused tests, full `python -m pytest ai4s/graphcast/tests -q`, shell syntax checks and CLI dry-run for DP=1 and DP=2. Verify no accelerator training or deployment side effect occurred. Update README and provenance with commands and next full-model validation boundary.
- [x] Commit assigned files as `feat(graphcast): add formal TF32 autoregressive training entry`.

## Review and handoff

Each task gets a task-scoped independent review after implementation. Final review checks cross-module data/weighting/precision/restore contracts and inference preservation. This implementation stage ends with local commits and CPU/CLI evidence, not a push or six-platform training-performance claim.

### Local code integration closeout — 2026-09-07

Tasks 1–3 are implemented and locally committed through `d185f726d9349a6e3ed0c0a988093755dbca604a`.
The previously interrupted whole-range review of `3e43521..d185f72` is now complete:
no new Critical/Important blocker; data, BPTT, valid-sample scaling, once-only DP
synchronization, TF32 control and exact-resume joins were checked. A fresh
pre-diagnostic suite passed 377 tests in 35.34 s. After the separately approved
lightweight loss diagnostic integration, the complete suite passed 424 tests
in 34.01 s; isolated runtime checks still expose the pre-existing NumPy/NetCDF
ABI warning, not a test failure. No environment change was made.

One known Minor limitation remains outside the prescribed binary-padding path:
the generic AR loss primitive's clamp_min(1) misnormalizes fractional sample
weights whose total is below one. The formal corpus and engine reject nonbinary
weights, so that case is not supported by this entry. It is not a claim of
general fractional-weight correctness.

The lightweight diagnostic's independent review found a Native-reference
identity false-pass; failing regressions reproduced it, the fix verifies
requested/resolved backend and precision, and re-review closed the issue.
Checkpoint weather-skill evaluation remains deferred by the user's later choice.

Read-only remote verification on this date found the requested
`zyx/flaggems-profile-routing` branch at `3e435216adb7af56a32e02a4a8490dde3d405eb0`.
Training integration is therefore local only. No remote push, accelerator
training, four-card full-model memory/communication validation, TF32 numerical
acceptance, backward coverage or training MFU has been completed by this stage.
