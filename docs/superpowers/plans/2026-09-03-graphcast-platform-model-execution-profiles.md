# GraphCast Platform Model Execution Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit H100 baseline profile and one shared optimized model profile for the five domestic platforms, then validate BF16 correctness and performance.

**Architecture:** A frozen profile resolver maps `(platform, profile)` to the existing model constructor options. The formal runner accepts only valid mappings and writes the selected mapping into its report; precision and FlagGems/native routing remain independent controls.

**Tech Stack:** Python, argparse, PyTorch, pytest, existing GraphCast multi-platform launch infrastructure

**Spec:** `docs/superpowers/specs/2026-09-03-graphcast-platform-model-execution-profiles-design.md`

## Global Constraints

- H100 uses `baseline`; the five domestic platforms use `domestic-v1`.
- `domestic-v1` expands to factorized edge projection, both chunks `131072`, full-grid fast path enabled, and inline `index_add`.
- The same profile is used for FP32 and BF16 and for Native/FlagGems pairs.
- No aggregation-buffer reuse or FlagGems kernel changes are introduced.
- Existing unrelated untracked files remain untouched.

---

### Task 1: Freeze and apply the profile contract

**Files:**
- Modify: `ai4s/graphcast/tests/test_operational_40step_runner.py`
- Modify: `ai4s/graphcast/scripts/inference/run_operational_40step_inference.py`
- Modify: `ai4s/graphcast/README.md`

**Interfaces:**
- Consumes: existing `GoogleSmallConfig`, `build_model_for_run`, and `apply_edge_chunk_overrides`
- Produces: `ModelExecutionProfile` and `resolve_model_execution_profile(platform, profile)`

- [ ] **Step 1: Write failing resolver and CLI tests**

  Add literal expectations for `h100/baseline`, all five
  `domestic-v1` mappings, invalid cross-platform mappings, and FP32/BF16 profile
  equality. Require both formal CLI arguments and reject the retired low-level
  optimization flags.

- [ ] **Step 2: Run the focused test file and verify RED**

  Run `python3 -m pytest ai4s/graphcast/tests/test_operational_40step_runner.py -q`.
  Expected result: failures because the new profile type, resolver, and CLI
  arguments do not exist.

- [ ] **Step 3: Implement the minimal profile resolver and runner wiring**

  Resolve the profile before source files or the device runtime are opened,
  bind the expected backend before FlagGems/device initialization, feed its
  values to config/model construction, and record profile overrides separately
  from observed effective settings in the JSON report. Cover the complete
  `main()` wiring with a mocked-runtime integration test.

- [ ] **Step 4: Run the focused tests and verify GREEN**

  Re-run the focused command and require zero failures.

- [ ] **Step 5: Update the formal inference documentation**

  Replace the low-level command example with the required platform/profile
  arguments and document the exact allowed matrix.

### Task 2: Verify and prepare the five-platform experiment

**Files:**
- Verify: all changed Python and Markdown files
- Produce outside the repository: immutable source archive and per-platform run evidence

**Interfaces:**
- Consumes: the profile-aware formal runner and FlagGems `v5.3.5 + 81720a6d8a`
- Produces: per-platform correctness results, three-run 40-step medians, operator coverage, and BF16 MFU inputs

- [ ] **Step 1: Run all GraphCast unit tests and syntax checks**

  Run the full `ai4s/graphcast/tests` suite, Python compilation for changed
  files, and `git diff --check`; require zero new failures.

- [ ] **Step 2: Seal the tested source and dependency identities**

  Build immutable archives from the tested tracked diff, record SHA256 values,
  and verify them after transfer to each platform.

- [ ] **Step 3: Run one-step Native/FlagGems correctness gates**

  For C550, BW3000, S5000, 910C, and 810E, use identical assets and
  `domestic-v1`; require finite outputs, expected dispatch coverage, and the
  existing same-precision comparison gate.

- [ ] **Step 4: Run three fresh 40-step BF16 measurements per mode**

  Exclude synchronized warm-up, include final D2H, keep one device and one
  process, and never use profile/trace runs in the timing median.

- [ ] **Step 5: Recompute and summarize MFU**

  Use the existing measured BF16 dense-GEMM roof for each same machine/runtime,
  report Native and FlagGems medians, the 7-operator dispatch fraction, and any
  correctness qualification. Do not publish an invalid or incomplete platform.
