# Colleague training baseline integration

Approved in conversation on 2026-09-09. The user selected the colleague training
implementation as the baseline and approved the six integration boundaries.

## Source and outcome

Target: `ZYX223/ai4s_graphcast:zyx/flaggems-profile-routing`.
Training source: `flagos-adaptation-lab/flagos-graphcast-adaptation` at
`b7667b42da708db86010fe2d962e5fd0363ffa0a`, checked out read-only at
`/mdata/data/zhangyuxin/wxt/graphcast-colleague-review-20260909-e3d0uf`.
Integration starts from local committed `f74b909565bc7f70a0394bc37d7a0a5efc317970`,
which descends from target remote `3e435216adb7af56a32e02a4a8490dde3d405eb0`.
The original dirty worktree and all campaign evidence stay untouched.

## Binding integration boundaries

- One formal training engine, based on the colleague epoch runner, not a second
  reimplementation. Reuse its update-indexed with-replacement sampling, weighted
  autoregressive BPTT, AdamW, accumulation, CPU prefetch/cache, exact resume and
  training-loop reductions. Preserve provenance and licenses.
- Full 721x1440 grid, mesh6, latent512, 16 Mesh steps and 13 levels. Four-device
  placement remains the default; expose optional two-device multi-device DDP.
  DDP accumulation encloses forward/backward in no_sync and never performs a
  second manual gradient average. Keep source platform-specific memory defaults,
  Ascend launch safety and multi-device MUSA serial backward.
- Independent training_model.py shares parameter definitions with inference.
  Do not change inference outputs, defaults, YAML, graph construction, precision,
  runtime routing or optimizations. Minimal factory hooks in shared definitions
  must retain the inference defaults. Canonical colleague graph generation is
  training-only; the existing inference graph builder remains the default.
- FP32 storage with TF32/HF32 enabled by the public training entry by default;
  explicit strict FP32 is supported. No BF16 or FP8 expansion. Delegate math
  configuration to the target backend, including MUSA matmul.allow_tf32.
- Keep the target backend and training YAML as the single FlagGems management
  policy. Native must not import FlagGems. Explicit invalid/unregistered requests
  fail closed. Add opt-in actual function-call observation, separate from timing;
  registration is not proof of complete forward/backward coverage.
- Preserve the target's explicit-reference, full-interval, metadata-qualified
  ratio-of-sums loss diagnostic with inclusive 5% threshold. Do not default to
  historical small-1deg JAX CSV. Add the colleague trained-checkpoint diagnostic
  guidance without claiming weather skill or numerical equivalence from imports.
- Do not publish fabricated training MFU: source FLOPs/recompute accounting may
  be retained with its limitations, but no default engineering peak is accepted
  as verified; peak absent means MFU unavailable. Profiling is off for timing.
- Keep old raw checkpoint weights loadable for model-only initialization; do not
  claim exact optimizer resume across changed sampler/topology/schema. Preserve
  the old engine in existing Git history, not a second formal execution path.
- No datasets, checkpoints, large traces, environment repair, credential files,
  deployment trees, cron/source-sync automation or FlagGems operator edits enter
  this source integration.

## Verification and delivery

Use focused RED/GREEN integration tests, colleague source tests where applicable,
inference regressions, real CPU Gloo DP/DDP tests, CLI dry-run and resume tests.
Then run a matched full-model short Native/FlagGems test on an authorized idle
platform if available, with immutable source and independent outputs. Keep full
6/10/22 schedule and stop at a prefix rather than compressing the learning-rate
schedule merely to shorten the gate. Record device evidence separately from CPU
checks, and do not claim untested 12-lead or multi-platform acceptance. Review,
commit, and non-force push to the requested remote branch after verification.

## Implementation structure

Keep the canonical colleague runner under scripts/training, extracting only the
helper functions it consumes from the older parallel runner, so there is no
second official training main. Existing run_training.py/train_official.sh become
thin entry/argument adapters to that engine. Source public train_pipeline.sh
platform policies are reused without its source-maintenance/deployment machinery.
New training primitives live in graphcast_compat, and diagnostics remain opt-in.
