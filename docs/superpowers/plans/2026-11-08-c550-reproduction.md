# C550 GraphCast reproduction delivery

User scope: publish the accepted GraphCast training, short MFU sampling,
checkpoint recovery and validation code with reproduction instructions. No new
GPU work is authorized or required by this packaging task.

## Contract

- Preserve the 144 accepted model/integration file hashes and frozen mathematical
  routines. Bind FlagGems to flagos-common commit f2a276f3741026f1234574f3beccde55c5bb5f8c.
- Four isolated modes: 38 continuous updates; 8 updates with a conditional ninth;
  CP36 -> updates 37/38; CP38 -> 111 R12 validation windows with zero updates.
- Explicit local paths, device mapping, fresh resource attestation, bounded
  execution and new output directories. No shared environment changes.
- Publish source, small manifests and tests; exclude data, checkpoints, caches,
  credentials, user identities and private deployment records.
- Describe historical hardware validation separately from CPU-only packaging
  checks. Predictions and checkpoint structural checks are not formal acceptance.

## Work

1. Restore seven missing integration files; organize frozen harnesses and
   path-parameterized templates, retaining provenance hashes.
2. Add prepare/check/run and result checks, preserving failure evidence and
   rejecting drift, stale gates and reuse of output directories.
3. Test command generation, prediction rejection, recovery lineage and durable
   validation records on CPU; review the entire patch and document limitations.
4. Commit and fast-forward the existing fork branch; verify the remote commit.

## Review focus

Portability substitutions must not alter model mathematics, schedules, input
identity or tolerances. Check failed/incomplete jobs cannot be called accepted;
new wrappers have no fresh hardware validation. Existing untracked profiling
files and uv.lock must remain untouched.

## Execution record

- Restored seven exact integration files; all 144 model file hashes match.
- Organized four modes and fixed external source/data/checkpoint identities.
- Independent review identified ERA5 content binding, child import bootstrap,
  escaped-worker cleanup and recovery self-comparison issues. All four fixed;
  regressions include real CPU child sessions and RED-to-GREEN self-comparison.
- CPU tests and historical-number recalculation only; no new GPU invocation.
- Ruling: preserve the old CP36 logical-path contract and reject relocation
  before GPU rather than rewriting checkpoint metadata. Reproduction requires
  those internal mounts. Newly generated checkpoint identities need a separately
  reviewed source lock. No acceptance is inferred for new environments.
- Publishing to the existing fork branch is explicitly authorized by the user.
  Unrelated profiling work and uv.lock are excluded.
