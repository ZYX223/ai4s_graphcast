# GraphCast Round-1 FlagGems inference implementation plan

Source of record: `ZYX223/ai4s_graphcast:zyx/flaggems-profile-routing` at
`242b4a33b6f502761afe8249ec0bac8e5d3ce1a2`.

1. Replace inference `default_ops`/`platform_ops` with one shared seven-operator
   allowlist plus a compact backend correctness-exclusion map. Native versus
   FlagGems remains a runtime mode; the PTG exclusion is limited to `addmm` and
   `index_add_`, as selected by the formal 40-step isolation evidence.
2. Add device-resident autoregressive helpers and a production 40-step runner.
   FlagGems/JIT warm-up is synchronized and excluded from the measured window;
   the measured window contains all 40 dependent model calls and final output
   device-to-host transfer.
3. Add behavior tests for the shared profile, recurrent state/forcing updates,
   warm-up exclusion, exactly 40 dependent calls, and report arithmetic.
4. Validate locally, then deploy an immutable patch plus the frozen
   FlagGems-v5.3.5 GraphCast patchset to isolated per-platform worktrees.
5. Run paired Native/FlagGems correctness and three performance repetitions on
   C550, BW1000, Ascend 910C, S5000, and 810E. Preserve raw logs and reports.
6. Update the GraphCast workbook Round 1 only from validated evidence. Retain
   the ordinary `torch.compile` H100 baseline and link the MFU analysis Markdown
   from a dedicated workbook sheet.
