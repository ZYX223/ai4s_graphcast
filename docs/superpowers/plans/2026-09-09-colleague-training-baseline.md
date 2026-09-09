# Colleague Training Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Integrate the colleague training baseline into the existing GraphCast branch without regressing inference.
**Architecture:** Reuse the pinned colleague training model/primitives and one epoch engine. Adapt only target backend, public entry, diagnostics and repository paths; retain source execution semantics.
**Tech Stack:** PyTorch, vendor Torch, NumPy/xarray, pytest, CPU Gloo, FlagGems selected through existing backend.
**Spec:** `docs/superpowers/specs/2026-09-09-colleague-training-baseline-design.md`

## Global Constraints

- Source b7667b42da708db86010fe2d962e5fd0363ffa0a at /mdata/data/zhangyuxin/wxt/graphcast-colleague-review-20260909-e3d0uf is read-only.
- Work only in /mdata/data/zhangyuxin/wxt/ai4s-graphcast-training-baseline-20260909; do not touch the original dirty worktree.
- One formal colleague-based training engine; four devices default, optional two-device DDP; full operational model and with-replacement update sampling.
- Preserve inference behavior and target backend/FlagGems YAML; FP32 storage, TF32/HF32 default, no operator library or environment changes.
- Explicit-reference training loss diagnostic: complete requested interval, comparable metadata, ratio of sums <=5%; no historical JAX default.
- Use apply_patch for source adaptations. Preserve source copyright/provenance. No data, weights, outputs, source-sync automation or secrets in commits.
- Tests before adaptations; report RED/GREEN evidence. CPU fixture tests are not full-model accelerator acceptance.
- No subagents from workers; do not push. Root owns review, device acceptance and final push.

## Task 1: Shared definitions and colleague training primitives

**Files:**
- Add/adapt `ai4s/graphcast/src/graphcast_compat/training_model.py`, `accelerator.py`, `data_prefetch.py`, `precision.py`, `recompute_flops.py` from the identically named pinned source.
- Update `autoregressive_training.py`, `distributed_training.py`, `training_recipe.py`, `training_state.py`, `training_schedule.py`, `official_training_data.py`, `official_training_corpus.py`, `gradient_accumulation.py`, `epoch_plan.py` only as required to adopt pinned behavior. Retain the target numeric-time/corpus-bootstrap/close fixes.
- Add minimal factory hooks to `google_small.py` and optional canonical graph support to `graph.py`; import pinned small mesh asset and its provenance.
- Add `tests/test_colleague_training_model.py`, adapt/import focused primitive tests and CPU DDP accumulation/failure workers from source. Update training-source-provenance.md for this surface.

**Interfaces:** training_model.GoogleSmallConfig/GoogleSmallCompatibleModel and configure_pipeline_parallel/configure_four_stage_pipeline_parallel/configure_model_parallel methods remain source-compatible. DistributedContext retains wrap_ddp/manual synchronization APIs. accelerator.configure_float32_math delegates to backend.configure_float32_math after device resolution. Expose source data-prefetch APIs and tracked checkpoint counters unchanged.

- [ ] Step 1: Add behavior tests, run RED against target before source import. Example: build inference and training models from the same seeded config and check all parameter keys/shapes and values match; assert training instantiates PrecisionLinear without changing inference Linear class. Fixture tests may use tiny CPU graphs, never label these model acceptance.
```python
torch.manual_seed(27)
inference = InferenceModel(config)
torch.manual_seed(27)
training = TrainingModel(training_config)
assert list(inference.state_dict()) == list(training.state_dict())
for name, value in inference.state_dict().items():
    torch.testing.assert_close(value, training.state_dict()[name], rtol=0, atol=0)
```
- [ ] Step 2: Reuse source implementations; add class-level defaults `linear_type = nn.Linear` and `graphnet_type = GoogleSmallGraphNet` where the training subclass requires them. Preserve default inference graph building with `canonical=False`; training init explicitly passes `canonical=True`. Verify source mesh asset provenance and ndarray shape/dtype.
- [ ] Step 3: Port source primitives and tests. Cover duplicate-permitted deterministic replacement sampling, optional per-variable diagnostics, finite failure at update guard rather than every replay, two-lead gradient equality with checkpointing, real CPU Gloo DDP with accumulation and unused parameters, and source-safe DP1 zero synchronization count. Preserve the target backend math helper with a direct delegation rather than a copied divergent precision policy.
```python
def configure_float32_math(tensor_core_enabled, accelerator_type=None):
    from .backend import configure_float32_math as configure
    return configure(tensor_core_enabled, resolve_accelerator_type(accelerator_type))
```
- [ ] Step 4: Run focused training-model/primitives and existing inference tests. Document unavoidable pre-existing runtime warnings separately; do not suppress errors or change installed dependencies. Full target engine/entry integration belongs to Task 2, diagnostic adaptation Task 3.
- [ ] Step 5: Self-review and commit task-scoped changes, report exact source deviations and tests.

## Task 2: One canonical training engine and public entry

**Files:**
- Add pinned `scripts/training/run_official_epoch_training.py`, `official_epoch_training_file_validation.py`, `training_support.py` (only helper definitions consumed from source run_official_parallel_training.py; not its second trainer main).
- Add source `hotspot_profiler.py`, `profiled_mfu.py` and required profiling helpers, keeping diagnostics opt-in.
- Replace `scripts/training/run_training.py` and `train_official.sh` with thin adapters; import/adapt `train_pipeline.sh` platform policies in the training directory, without deployment/source-sync scripts.
- Replace old `training_engine.py`/`training_runtime.py` formal execution references with the canonical runner; remove superseded engine-only code/tests once equivalent new coverage exists. Keep diagnostic-used helpers until Task 3 migrates them.
- Add/import source schedules and source runner tests under tests/training_baseline; update training README, main README, provenance and minimal declared dependencies.

**Interfaces:** public Python `run_training.py --dry-run --device-type npu --mp 4 --dp 1 --global-batch-size 2 --micro-batch-size 1 --tensor-core enabled --backend native` resolves source engine argv/environment without importing Torch/data. Expose source native CLI through the canonical runner and document it. Existing target backend/config/ops/variant flags remain the sole replacement policy. Preserve source metrics/checkpoint formats; Task 3 adds paired diagnostic metadata on them.

- [ ] Step 1: Add RED entry tests: complete MP4/MP2 run contract, incompatible topology rejection, TF32 default, strict override, source with-replacement schedule, invalid peak unavailable, native import isolation, FlagGems config/ops forwarding. Test actual CLI subprocess output and failures, not source text.
```python
result = subprocess.run([sys.executable, str(ENTRY), '--dry-run', '--device-type', 'npu', '--mp', '2', '--dp', '1'], capture_output=True, text=True)
assert result.returncode == 0, result.stderr
contract = json.loads(result.stdout)
assert contract['mp'] == 2
assert contract['requested_precision'] == 'tf32_hf32'
```
- [ ] Step 2: Port source epoch engine and only consumed helper functions. Runtime dataset, window order, optimizer/update/rollout semantics, FP32 storage, prefetch, checkpoint/resume and source MUSA/Ascend safety stay source-derived. Reuse source defaults but target public training default requests fast FP32 and unknown peaks stay unavailable. No implicit long training in dry-run. Public model-only initialization accepts old raw tensor state_dict as well as source envelope while exact resume remains schema/recipe checked.
- [ ] Step 3: Adapt FlagGems at a small training bridge to existing configure_flagos_environment/resolve_runtime_backend; validate every selected op registration (including layer_norm/native_layer_norm alias) before returning success. Native mode does not import it. Explicit missing op is an error, not a downgrade. Do not override inference profiles or installed files.
- [ ] Step 4: Source runner tests, tiny real CPU integration with supplied fixture boundaries, resume/sampler checks, public command tests, inference regression. Preserve runtime profile versions and actual op-source evidence in metadata. Retire old engine paths only after tests point to the new engine.
- [ ] Step 5: Update copyable Native/FlagGems DP1/DP2 MP4/MP2 commands, schedules and semantic migration caveats. Commit and report remaining device validation needs.

## Task 3: Qualified paired diagnostics and call observation

**Files:**
- Adapt `scripts/training/diagnose_training_loss.py` to source metrics and start metadata using `training_diagnostics.py` if a reusable adapter is needed.
- Add source `scripts/validation/trace_training_flaggems.py` as an opt-in wrapper for the one canonical runner; preserve recompute markers and device/source/rank/update distinctions.
- Update canonical runner at initialization/update emission only to record existing metadata for comparability, outside performance timing. Do not add a second training loop or per-operator debug into normal execution.
- Adapt `tests/test_training_loss_diagnostic.py`, add source-metric and observer behavior tests; update training diagnostics README/provenance.

**Interfaces:** `diagnose_training_loss.py CANDIDATE --baseline REFERENCE --start-step N --end-step M` retains exit 0 passed, 1 metric fail, 2 incomparable/invalid. Match same initial params/optimizer, data/stats, sampler/global indices, precision/backend policy, batch/topology, schedule and memory policy. Actual source metrics cannot be made comparable by inventing missing evidence. `trace_training_flaggems.py --trace-dir DIR -- <canonical trainer args>` observes logs only on explicit request.

- [ ] Step 1: Add RED tests using minimal actual source-format metric rows and start metadata. Include identical pass, >5% fail, different global sample order, missing step, wrong precision, wrong initialization, native reference with replacements enabled, and resume segments. Hand-derived E examples: reference [2,4], candidate [2.1,4.2] ->5% inclusive pass; [2.2,4.4] ->10% fail.
- [ ] Step 2: Retain ratio-of-sums algorithm and explicit interval, add metadata conversion from source runner's real metadata/checkpoint contract. Emit initial params/optimizer fingerprint once outside update timer and identify with-replacement samples from source per-update metadata; never accept history intersection alone.
- [ ] Step 3: Integrate optional call observer, report phase/function calls (not kernel count) without claiming all-op coverage from registration. Preserve required observed source path and optimizer update. Normal performance execution does not load observer. Keep historical JAX baseline out of defaults and weather skill marked not evaluated by the loss script.
- [ ] Step 4: Run qualified diagnostics, runtime metadata, observer and source/inference regressions. Update docs on trained-checkpoint vs fixed-pretrained weather reference and MFU denominators; no new long weather evaluation.
- [ ] Step 5: Commit; prepare source freeze and reproducible full-model short paired commands for root device acceptance and final review.

## Root verification and delivery checklist

- [ ] Review each task with a separate reviewer and then whole branch; record ledger and resolve load-bearing issues.
- [ ] Run full appropriate target tests, import/source provenance and inference default comparisons.
- [ ] Check authorized device availability; preserve source/package identity, run paired Native/FlagGems full25km TF32/HF32 6/10/22 prefix with same init/data/GB/micro/MP and qualified loss diagnostic. If unavailable, report explicitly; do not repurpose occupied devices or claim device acceptance.
- [ ] Verify GitHub identity ZYX223 and exact target branch tip; commit and non-force push HEAD:zyx/flaggems-profile-routing. Do not create a PR or touch colleague remote branch. Preserve original worktree/campaigns.
