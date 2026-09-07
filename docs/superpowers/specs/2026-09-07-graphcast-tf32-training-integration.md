# GraphCast TF32 training integration

The user approved this scope in chat on 2026-09-07 and requested implementation.

## Scope

Keep `ZYX223/ai4s_graphcast:zyx/flaggems-profile-routing` as the unified project. Integrate training capabilities from `/mdata/data/zhangym/zym/graphcast-multiplatform` at `6c5654c10c32a9a9b018eac88e4653a2b58be13d`; the A100 handoff is deployment context, not a second algorithm to merge. Start from target commit `3e435216adb7af56a32e02a4a8490dde3d405eb0`.

Reuse colleague modules rather than rewriting their mathematical algorithms: real ERA5 windows and epoch plans; latitude/level/variable weighted differentiable autoregressive loss; AdamW, learning-rate and rollout schedules; sample-weighted accumulation; optimizer-boundary checkpoints and exact restoration; rollout checkpointing and necessary model-parallel placement adaptation. Preserve provenance and the pinned DeepMind dependency/license.

## Binding constraints

- Preserve existing inference code, defaults, optimization flags, and tests. Keep the existing short `train_mp` operator-regression entry available.
- Training tensors/parameters/loss remain FP32; TF32/HF32 is selected through the existing `backend.configure_float32_math`, not autocast or BF16. Native remains an independent baseline; do not introduce automatic Native fallbacks.
- Keep existing FlagGems runtime management and installed v5.3.5 plus two WIP overlays. Do not copy the colleague's old platform allowlists, precision implementation, environment repairs, or peak values.
- Do not claim forward/backward operator coverage based on inference coverage. Record requested backend and actual available routing evidence, with unverified coverage explicitly unverified.
- Full-model accelerator validation uses 0.25 degrees, mesh 6, latent 512, 16 GNN steps and 13 pressure levels. Small tensor fixtures are unit tests only, never model acceptance.
- Preserve differentiable autoregressive feedback, exact variable/channel ordering and normalization; weight padded samples zero. A full update averages valid samples across accumulation and DP.
- Preserve model parallelism for a single sample. The initial full-model configuration is MP=4, DP=1; DP=2 owns two disjoint four-device groups. This is not one-rank-per-GPU DDP and not a promise that TF32 reduces storage.
- Gradient accumulation steps, rollout length and optimizer updates are different dimensions. All are logged. Rollout is configurable and never hardcoded to 40.
- Checkpoint only on optimizer-update boundaries. Include model, optimizer, schedule/cursor, and rank-local RNG states; restore without repeating/skipping data or updates. Validate semantic compatibility, not unrelated source hashes or deployment paths.
- No datasets, weights, traces or run outputs in Git. Do not edit colleague source, other platform environments or FlagGems files. Do not push or launch full accelerator training in this implementation stage.

## Integration design

1. Add the data modules and minimal pinned upstream imports under `ai4s/graphcast/src`; adapt to the target's existing configuration/data/normalization APIs without overwriting those modules.
2. Add the independent training primitives and their behavior tests. Retain original APIs when feasible. Adapt accelerator access in the DP context to the existing backend with minimal glue, not a parallel backend subsystem.
3. Add a slim formal training entry separate from the short operator probe. Keep lifecycle/restore and CLI orchestration separated from the reusable update engine. Build the existing `GoogleSmallCompatibleModel`, set MP stage devices with its public placement API, and use its existing activation-checkpoint implementation. Add missing placement capability only when demonstrated necessary by the training adapter.
4. Use the existing model's ordinary differentiable path initially. Do not enable inference-only state/cache/buffer optimizations for training without gradient evidence. Port autoregressive segment checkpointing and chunked weighted loss. Keep further CPU prefetch/cache tuning for a later change.
5. Native and FlagGems share the same data/update loop and precision request. Preserve the current optional training allowlist interface without equating it to full coverage. Report optimizer step, epoch, rollout, valid samples, loss/variable losses, gradient finite/norm, data/compute timing and memory. Training MFU stays unavailable unless a separately verified useful-FLOPs numerator and matching device peak are supplied; do not recycle inference FLOPs.

## Validation and non-goals

Baseline: 304 target tests passed before edits. Bring over relevant behavior tests, not source-text assertions or unrelated HRES/deployment tests. Add integration tests for real sample extraction, weighted loss/BPTT, unequal/padded accumulation, schedule boundaries, checkpoint/resume equivalence, a training update using FP32 storage, and CLI dry-run contracts. Re-run existing inference tests.

The handoff has seven days of ERA5, not a long-term training/validation corpus. CPU tests and dry-run success do not establish full-model memory fit, six-platform training correctness, weather skill or performance. No device result is claimed without a corresponding run. Full-model short Native TF32 validation and FlagGems forward/backward tracing are the next stage after code review.
