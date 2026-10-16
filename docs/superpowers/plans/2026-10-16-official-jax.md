# Official JAX consolidation implementation plan

> For agentic workers: use superpowers:executing-plans; implement inline and obtain a final fresh review.

**Goal:** Make the current Sugon branch a reproducible official JAX project for the weekly H100 and Hygon results.

**Architecture:** Preserve official model and HIP kernel bytes; consolidate only orchestration, assets and metrics. Keep host paths in launcher arguments, tested platform defaults in JSON, and evidence separate from executable code.

**Tech stack:** Python, JAX/Haiku/xarray, CUDA, DTK/rocBLAS/HIP, Docker.

**Spec:** ../specs/2026-10-16-official-jax.md

## Global constraints

- Pin official source to 08cf73625c9d12bd9aaa038868bcb2fe488f2a22.
- Preserve original measurement boundaries and the RNE kernel implementation.
- Do not change historical outputs, shared runtimes, GPU clocks or other workloads.
- New measurements cannot replace the original weekly numbers without explicit labeling.
- No data, checkpoint, image or compiled library in Git.

## Review focus

- Wrong timestep, initialization, grid, levels or units must fail before accuracy is reported.
- Nonfinite values and zero reference denominators must not silently pass.
- A stale source/asset/kernel library must be detected by digests.
- Reproduction must work without any previous dated experiment directory.
- A requested GPU/precision must be verified; no CPU fallback or implicit global XF32 mode.

## Tasks

- [x] 1. Archive the previous tree, vendor the official import closure, add independent metric and asset tests, extract a JAX-only frozen case.
- [x] 2. Consolidate model execution and per-platform launch configuration; retain Hygon routing/count checks, add a reproducible kernel build and device probes.
- [x] 3. Import small historical result/provenance summaries and rounding example; remove unrelated tracked implementation, tests, docs and CI assumptions.
- [x] 4. Run the CPU suite, original-output metric parity and H100/Hygon probes plus 40-step regression; record new evidence separately.
- [x] 5. Fresh review, fix important findings, check source/package scope and Git diff; prepare the verified tree for the requested project branch.

## Execution notes

User's latest instruction explicitly supplies the scope and changes the earlier proposal to keep a parallel legacy entry. Proceed with that scope; the current checkout is already an isolated linked worktree. Existing results are archived outside the repository, and previous implementation remains available at commit 39b6ee5.

Final independent review: no Critical findings. Important fixed-case grid/physical-unit validation and Minor H100 probe status inconsistency were confirmed and fixed. Added failing tests before the grid/unit fix; all eight tests then passed on both environments. Recomputed all four new predictions with the production fixed-case validator; numerical results exactly matched. Official model and HIP source bytes, device execution and timing were unchanged. H100 was busy on the attempted repeat probe; no workload was disturbed. Final verification and branch publication follow the user's already specified destination; no merge into a different branch or history rewrite is planned.
