# GraphCast BF16 Mixed-Precision Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a non-redundant BF16 autocast mode to the existing eager, device-resident GraphCast inference loop while preserving the strict-FP32 default and producing auditable precision evidence.

**Architecture:** A small immutable precision policy selects either a no-op FP32 context or BF16 autocast. The existing rollout loop calls one shared forward helper, converts BF16 predictions to FP32 before residual reconstruction and state feedback, and reports observed dtypes; it does not duplicate the model or rollout implementation.

**Tech Stack:** Python 3.10+, PyTorch autocast, pytest, xarray, existing GraphCast compatibility modules and FlagGems dispatch audit.

**Spec:** `docs/superpowers/specs/2026-09-03-graphcast-bf16-inference-design.md`

## Global Constraints

- The CLI default remains strict FP32 and accepts one additional spelling: `--precision bf16`.
- The BF16 report labels the effective policy as `mixed_precision`; there is no `bf16-mixed` alias.
- Model parameters, initial state, forcing, feedback scales, and persistent autoregressive state remain FP32.
- Only model forward executes under BF16 autocast; its prediction is converted to FP32 before `advance_device_rollout`.
- Strict FP32 keeps TF32/HF32 or equivalent fast-FP32 controls disabled.
- Execution remains eager; no compile or graph-capture path is introduced.
- The inference FlagGems YAML and seven logical operators remain unchanged.
- Performance is not published until precision, correctness, and dispatch gates pass.

---

### Task 1: Add the minimal precision policy and forward boundary

**Files:**
- Create: `ai4s/graphcast/src/graphcast_compat/inference_precision.py`
- Create: `ai4s/graphcast/tests/test_inference_precision.py`

**Interfaces:**
- Produces: `InferencePrecisionPolicy`, `resolve_inference_precision(name: str) -> InferencePrecisionPolicy`, and `run_precision_forward(model, state, forcing, *, policy, device_type) -> PrecisionForwardResult`.
- `PrecisionForwardResult.prediction` is always FP32; `model_output_dtype` preserves the dtype returned inside autocast for evidence.

- [ ] **Step 1: Write the failing precision-policy tests**

```python
def test_fp32_policy_uses_no_autocast_and_preserves_fp32():
    policy = resolve_inference_precision("fp32")
    result = run_precision_forward(
        torch.nn.Identity(),
        torch.ones(1, 1, 1),
        torch.ones(1, 1, 1),
        policy=policy,
        device_type="cpu",
    )
    assert result.prediction.dtype == torch.float32
    assert result.model_output_dtype == "torch.float32"


def test_bf16_policy_autocasts_forward_and_returns_fp32_feedback_tensor():
    policy = resolve_inference_precision("bf16")
    model = torch.nn.Linear(2, 1).eval()
    result = run_precision_forward(
        model,
        torch.ones(1, 1, 1),
        torch.ones(1, 1, 1),
        policy=policy,
        device_type="cpu",
    )
    assert next(model.parameters()).dtype == torch.float32
    assert result.model_output_dtype == "torch.bfloat16"
    assert result.prediction.dtype == torch.float32


def test_forward_rejects_non_fp32_state_or_forcing():
    policy = resolve_inference_precision("bf16")
    with pytest.raises(TypeError, match="must remain FP32"):
        run_precision_forward(
            torch.nn.Identity(),
            torch.ones(1, 1, 1, dtype=torch.bfloat16),
            torch.ones(1, 1, 1),
            policy=policy,
            device_type="cpu",
        )


def test_resolver_rejects_unknown_precision():
    with pytest.raises(ValueError, match="unsupported inference precision"):
        resolve_inference_precision("float16")
```

The BF16 test uses a small `torch.nn.Linear` on CPU. It asserts FP32 parameters and inputs before the call, BF16 `model_output_dtype`, and an FP32 returned prediction. It skips only when installed PyTorch explicitly lacks CPU BF16 autocast support.

- [ ] **Step 2: Run the new tests and verify the missing module failure**

```bash
cd ai4s/graphcast
python -m pytest -q tests/test_inference_precision.py
```

Expected: collection fails because `graphcast_compat.inference_precision` does not exist.

- [ ] **Step 3: Implement the immutable policy and shared forward helper**

```python
@dataclass(frozen=True)
class InferencePrecisionPolicy:
    requested: Literal["fp32", "bf16"]
    effective: Literal["strict_fp32", "mixed_precision"]
    autocast_dtype: torch.dtype | None

    def context(self, device_type: str):
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=device_type, dtype=self.autocast_dtype)


@dataclass(frozen=True)
class PrecisionForwardResult:
    prediction: torch.Tensor
    model_input_dtype: str
    model_output_dtype: str


def run_precision_forward(model, state, forcing, *, policy, device_type):
    if state.dtype != torch.float32 or forcing.dtype != torch.float32:
        raise TypeError("persistent state and forcing must remain FP32")
    model_input = torch.cat((state, forcing), dim=-1)
    with policy.context(device_type):
        raw_prediction = model(model_input)
    if not raw_prediction.is_floating_point():
        raise TypeError("model prediction must be floating point")
    return PrecisionForwardResult(
        prediction=raw_prediction.to(torch.float32),
        model_input_dtype=str(model_input.dtype),
        model_output_dtype=str(raw_prediction.dtype),
    )
```

For BF16, reject a warm-up/model result that never produces BF16 output; this is the first fail-closed execution proof. Do not add hooks or another rollout loop.

- [ ] **Step 4: Run the precision tests**

Run the command from Step 2. Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add ai4s/graphcast/src/graphcast_compat/inference_precision.py ai4s/graphcast/tests/test_inference_precision.py
git commit -m "feat(graphcast): add bf16 inference precision policy"
```

### Task 2: Thread the policy through the existing rollout and report

**Files:**
- Modify: `ai4s/graphcast/scripts/inference/run_operational_40step_inference.py`
- Modify: `ai4s/graphcast/tests/test_operational_40step_runner.py`
- Modify: `docs/superpowers/specs/2026-09-03-graphcast-bf16-inference-design.md`

**Interfaces:**
- Consumes: Task 1 `InferencePrecisionPolicy`, `resolve_inference_precision`, and `run_precision_forward`.
- Produces: rollout dtype evidence in `RolloutMeasurement` and one precision metadata shape shared by FP32 and BF16 reports.

- [ ] **Step 1: Write failing runner tests**

Add parser tests for `fp32` default and explicit `bf16`. Use a spy model and monkeypatched precision context to assert warm-up and every measured call use the same policy, `advance_device_rollout` receives FP32 predictions, persistent state remains FP32, and FP32 mode never enters autocast. Add a report-construction test for requested/effective mode and observed dtype fields.

- [ ] **Step 2: Run focused tests and verify they fail**

```bash
cd ai4s/graphcast
python -m pytest -q tests/test_inference_precision.py tests/test_operational_40step_runner.py
```

Expected: BF16 parser and rollout-signature assertions fail.

- [ ] **Step 3: Modify the single existing rollout loop**

- Change precision choices to `("fp32", "bf16")`, default `fp32`.
- Resolve the policy once after `resolve_runtime_backend`.
- Continue calling `configure_float32_math(False, runtime.profile.device_type)` in both modes so any FP32 fallback remains strict.
- Pass the policy and `runtime.device.type` into `run_tensor_resident_rollout`.
- Replace both warm-up and measured `model(torch.cat((state, forcing), dim=-1))` calls with `run_precision_forward`.
- Record parameter, state, forcing, model-input, raw model-output, feedback-prediction, and final-output dtypes.
- Require all observed raw model outputs to have one stable dtype; require BF16 mode to observe `torch.bfloat16`.
- Preserve final FP32 D2H and the existing timer boundaries.

- [ ] **Step 4: Extend report metadata**

The `precision` object uses one non-duplicated schema:

```json
{
  "requested": "bf16",
  "effective": "mixed_precision",
  "autocast": {"enabled": true, "device_type": "cuda", "dtype": "torch.bfloat16"},
  "parameter_dtype": "torch.float32",
  "state_dtype": "torch.float32",
  "forcing_dtype": "torch.float32",
  "model_input_dtype": "torch.float32",
  "model_output_dtype": "torch.bfloat16",
  "feedback_dtype": "torch.float32",
  "final_output_dtype": "torch.float32",
  "float32_math_mode": "ieee_fp32"
}
```

FP32 uses the same keys with autocast disabled and all dtypes FP32. Remove the obsolete `tensor_core_enabled` Boolean because a requested flag is not hardware-path proof.

- [ ] **Step 5: Align design spelling and run focused tests**

Update the spec from public `bf16-mixed` to public `bf16`, retaining `mixed_precision` as the report classification. Run the command from Step 2; expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add ai4s/graphcast/scripts/inference/run_operational_40step_inference.py ai4s/graphcast/tests/test_operational_40step_runner.py docs/superpowers/specs/2026-09-03-graphcast-bf16-inference-design.md
git commit -m "feat(graphcast): run operational inference with bf16 autocast"
```

### Task 3: Add model-level BF16 correctness evidence

**Files:**
- Create: `ai4s/graphcast/scripts/inference/compare_bf16_inference.py`
- Create: `ai4s/graphcast/tests/test_compare_bf16_inference.py`

**Interfaces:**
- Produces: tensor metric functions and a CLI that compares FP32 Native with BF16 Native, then BF16 Native with BF16 FlagGems.
- Consumes existing prediction artifacts and report identities; it does not start a runtime or change routing.

- [ ] **Step 1: Write failing metric and identity tests**

Use deterministic small tensors to test finite checks, max absolute error, RMSE, normalized maximum error, normalized RMSE, per-lead/per-channel aggregation, and worst-case selection. Test rejection of mismatched source hashes, shapes, channel order, step counts, or precision modes.

- [ ] **Step 2: Run the new tests and verify the missing comparator failure**

```bash
cd ai4s/graphcast
python -m pytest -q tests/test_compare_bf16_inference.py
```

- [ ] **Step 3: Implement the comparator**

Compute reductions in FP64:

```python
diff = actual.to(torch.float64) - reference.to(torch.float64)
rmse = torch.sqrt(torch.mean(diff.square()))
reference_rms = torch.sqrt(torch.mean(reference.to(torch.float64).square()))
normalized_rmse = rmse / max(reference_rms, torch.finfo(torch.float64).tiny)
normalized_max = diff.abs().max() / max(reference.abs().max(), torch.finfo(torch.float64).tiny)
```

The comparator reports observations and applies a previously sealed Native FP32-versus-BF16 envelope. It never derives or changes thresholds after receiving a FlagGems result.

- [ ] **Step 4: Run comparator and existing output tests**

```bash
cd ai4s/graphcast
python -m pytest -q tests/test_compare_bf16_inference.py tests/test_compare_operational_outputs.py
```

- [ ] **Step 5: Commit Task 3**

```bash
git add ai4s/graphcast/scripts/inference/compare_bf16_inference.py ai4s/graphcast/tests/test_compare_bf16_inference.py
git commit -m "test(graphcast): add bf16 inference correctness comparator"
```

### Task 4: Verify the complete local implementation

**Files:**
- Modify only files whose demonstrated test failure requires a fix.

**Interfaces:**
- Produces: a locally validated BF16-capable runner; no device-performance claim.

- [ ] **Step 1: Run all GraphCast tests**

```bash
cd ai4s/graphcast
python -m pytest -q tests
```

- [ ] **Step 2: Run static checks**

```bash
python -m py_compile scripts/inference/run_operational_40step_inference.py scripts/inference/compare_bf16_inference.py src/graphcast_compat/inference_precision.py
ruff check scripts/inference/run_operational_40step_inference.py scripts/inference/compare_bf16_inference.py src/graphcast_compat/inference_precision.py tests/test_inference_precision.py tests/test_compare_bf16_inference.py tests/test_operational_40step_runner.py
git diff --check
```

- [ ] **Step 3: Run CPU BF16 smoke when supported**

Run a deterministic small model through both modes and verify FP32 persistent state, BF16 raw model output, FP32 feedback, finite output, and report serialization. Label it software smoke, not device evidence.

- [ ] **Step 4: Commit only demonstrated verification fixes**

Commit any required fix with a message naming its observed failure.

### Task 5: Run the six-platform correctness and performance campaign

**Files:**
- Create only attempt-specific result artifacts; do not modify source during comparable runs.

**Interfaces:**
- Consumes: the exact source revision from Tasks 1-4, existing seven-op YAML, fixed workload identities, and existing BF16 dense-GEMM protocol.
- Produces: Native/FlagGems reports, correctness evidence, dispatch evidence, BF16 roofs, three-sample E2E medians, and empirical BF16 MFU.

- [ ] **Step 1: Run capability and dtype preflight**

On H100, C550, BW3000, S5000, 910C, and 810E, verify autocast entry, raw BF16 model output, persistent FP32 state, exact device identity, and idle-device status. A failed preflight stops performance work on that platform.

- [ ] **Step 2: Establish H100 Native correctness envelope**

Run strict FP32 Native and BF16 Native with identical inputs. Collect same-input per-lead and free 40-lead comparisons, then seal the Native BF16 envelope before examining domestic FlagGems results.

- [ ] **Step 3: Run domestic correctness**

For each domestic platform, run FP32 Native, BF16 Native, and BF16 FlagGems in fresh processes. Require finite output and the frozen envelope. Require 7/7 FlagGems dispatch on C550, BW3000, S5000, and 910C; retain the explicit 810E 6/7 profile with Native `addmm`.

- [ ] **Step 4: Separate timing from profiling**

For each passing configuration, collect three fresh-process 40-lead E2E samples in counterbalanced Native/FlagGems order. Run separate profiler jobs for actual BF16 matrix-path evidence and per-core-op dtypes; never include profiler runs in timing medians.

- [ ] **Step 5: Compute path-matched BF16 MFU**

Run `tools/dense_gemm_peak` BF16 mode on the same physical device and runtime. Feed exactly three positive E2E samples and the measured BF16 roof into `calculate_path_matched_mfu.py`. Keep empirical roof and vendor theoretical peak as separate labels.

- [ ] **Step 6: Publish only verified rows**

Record source and FlagGems SHAs, runtime, device UUID, precision evidence, dispatch count, correctness status, three E2E samples, median, BF16 roof, and MFU. Failed or incomplete attempts remain visible but leave formal performance cells empty.
