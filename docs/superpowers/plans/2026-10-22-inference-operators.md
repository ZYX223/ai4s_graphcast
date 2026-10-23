# BW1100 推理记录核对与关键算子测试计划

> Execution: inline in the existing `zyx/sugon-graphcast` worktree. The user approved record reconciliation followed by operator tests and timing breakdown on 2026-10-22.

**Goal:** 核对当前设备与历史推理记录，交付真实模型形状的独立算子正确性测试及可复现耗时拆分。

**Architecture:** 保留官方模型、原始 RNE 实现和历史结果；新增独立材料生成器及算子基准。形状和调用频次由已保存的运行时日志与优化 HLO 双重核对。独立算子测量与模型端到端耗时分别报告。

**Spec:** 本轮用户确认的范围及上轮任务表：记录核对 → 关键算子测试与耗时拆分。逐时效、多案例、RNE 优化及训练不在本轮范围。

**Latest scope:** 用户进一步确认只测 RNE＋XF32（包含舍入），先做官方 JAX 整模型 profile，再完成真实形状的独立算子测试。本轮不新增 FP32 / 原始 XF32 整模型对比。

**Constraints:** 当前项目型号 BW1100；原始驱动字符串 BW1200 独立保留。固定官方提交 08cf73625c9d12bd9aaa038868bcb2fe488f2a22、现有输入/权重和 DTK 26.04 环境。设备只能选空闲卡。不得把历史耗时差当成直接测得的舍入内核耗时；不得把合成输入算子时间之和称为模型 profiler 时间。

## Tasks

- [x] 1. 生成记录核对报告及算子目录：验证源码、输入/权重、参考、历史报告对应关系；记录当前设备/权限；对 84 次 GEMM 的 15 种形状核对 HLO；人工可检查的 JSON/CSV 与来源哈希。
- [x] 2. 独立测试：CPU 测试先验证目录生成器拒绝不完整/未知权重/维度不匹配记录；新增海光 GEMM＋RNE 原生测试、JAX 图操作测试。固定随机种子，数值校验与计时分离。
- [x] 3. 运行及交付：整模型预热后捕获单步 RNE＋XF32；独立算子三轮测量、预热、同步计时，分别记录 RNE＋GEMM、预舍入 GEMM、单独权重/激活舍入和图操作。汇总历史 rollout 主机/模型调用分解与当前独立算子结果；明确测量边界。

## Review focus

- 设备商品型号与驱动设备名不一致时，分别记录，保留机器/PCI 身份，不放宽运行时检查。
- 目录必须覆盖同一步全部 84 个不同层，不能将重复步数当调用频次。
- 输入降低精度须比较独立参考，原输入不变；计时不得包含编译或首次分配。
- 独立重复形状按模型调用频次加权仅作估计，不能冒充模型实际 GPU 占比。
- Docker 无权限时保留明确失败记录，不将静态材料或历史结果描述为本次 GPU 实测。

## Logical-operator extension (user approved)

User requested the same logical/API granularity as the supplied 20261018 workbook, missing times measured on BW1100 JAX RNE+XF32, and kernel details retained as attachments.

- [x] Inspect the prior workbook: logical PyTorch/ATen operations; independent median × phase call count, not GPU-kernel-only granularity.
- [x] Export a frozen one-step JAXPR; preserve scan input/output dependencies; prune dead equations; composite Linear/LayerNorm/SiLU own their internal primitives.
- [x] Implement replay at original shapes with original constants and seeded synthetic activations, independent sampled NumPy references, synchronized timing, source/data/device/router checks.
- [x] Regression checks for composite ownership, scan liveness, literal constants, RNE ties, repeated scatter indices and large-tensor sampling. 32 container CPU tests pass; read-only review has no remaining blocking findings.
- [x] Complete final shape-case GPU measurement using capture-r3 and frozen source-final-r1; exclude preliminary or failed runs.
- [x] Deliver logical main workbook plus unchanged kernel evidence attachments, provenance and reproduction instructions; reconcile every workbook cell and original timing/oracle record.

Preliminary full-r1 was rejected: large flattened output indices overflowed under JAX x64-disabled sampling. Per-axis indexing fixes the validation path. Review additionally found scalar constants must remain compile-time constants and scan xs dependencies must be retained before liveness pruning. These are fixed in source-final-r1 with regressions; all final measurements use the corrected capture-r3. Original model and RNE numerical implementation remain unchanged.

## Progress

- Existing worktree verified; initial tracked working tree clean at ffbdd28.
- Read-only hardware probe: 16 devices, driver reports BW1200; user identifies this machine as BW1100. Original weekly device PCI 0000:88:00.0 still present.
- Runtime blocker: current user cannot access Docker socket; noninteractive sudo requires authentication. User notified asynchronously; CPU/material work continues.
- Ruling: proceed with the already authorized concrete scope without another approval cycle. No branch creation or publication is needed.
- Task 1: complete. CPU-only audit saved at outputs/inference-operators-20261022/audit-r1. Catalog tests failed before implementation (missing module), then 5 passed; 2 asset tests also passed. Full host discovery cannot import test_metrics because xarray is absent; fixed-container suite remains pending.
- Docker diagnosis: same UID/GID in successful 2026-10-16 launch records; Docker restarted 2026-10-20 08:54 Asia/Shanghai. Current socket 0660 root:docker has no per-user ACL; zhangyuxin is not in docker. Prior temporary socket permission loss is a hypothesis, not established fact.
- Docker access restored by user: account is now in docker group; existing shell uses `sg docker` to acquire supplementary group. Immutable image 919cf075… remains available. Audit r2 passed with runtime available.
- CPU/container tests: 19 passed in cpu-check-r3. GPU11 (PCI 0000:9c:00.0) selected after idle check. Original RNE/mode-switch probe passed.
- Full-model profile r1 valid; r2 rejected because an analysis script was added during execution and final code snapshot validation correctly failed. hipprof itself returned zero despite child failure; launcher now requires completed report, step audit, and valid GPU trace before accepting success. Negative check on r2 rejects it.
- Frozen source-r1 used for valid profile r3: 550 GPU kernels, 84 GEMMs, 168 RNE kernels; 969.025481 ms kernel sum. Runtime weight fingerprints and same-stream RNE/RNE/GEMM ordering confirm weight RNE 3.933102 ms and activation RNE 280.535776 ms. No compile inside measured range. No new 40-step timing/accuracy claim.
- Frozen source-r2 used for independent operator run; real model shapes, deterministic synthetic features, actual official graph indices. Added final source/data integrity verification. Read-only code review requested; Hygon profile now rejects unsupported precision modes before starting GPU work.
- Independent run operators-r1 completed: 15 GEMM shapes (84 model calls), 14 graph cases, three rounds each; source/data unchanged. Numerical checks passed. Review found no blocking numerical/timing issue; device-name guard added alongside existing PCI checks. Full-output equality is explicitly labeled execution consistency; independent FP64 oracle remains sampled.
- Final verification: 19 container CPU tests passed; final native build/small GEMM passed on the expected device; an intentionally wrong device name rejected with exit 4; JAX identity helper passed. Final code snapshot matches tested snapshot and official files remain byte-identical. Evidence copies match source SHA-256. GPU11 released after tests. Report, input/output specifications, provenance and reproduction commands completed in docs/ and evidence/20261022/; no publication requested.
