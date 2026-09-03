# FlagGems Runtime Preparation

`third_party/` is an ignored staging area for external runtime source. The
preparation tool fetches the active manifest from
`flagos-common:zyx/flaggems-op-prcheck`, fetches the matching upstream
FlagGems branch, and composes two source trees from that same baseline:

- `official/FlagGems`: unchanged upstream source;
- `internal/FlagGems`: upstream source plus the manifest's validated overlays.

No commit is hard-coded in this repository. Each run records the resolved
commits and overlay list in `third_party/flaggems-runtime/PROVENANCE.env`.

```bash
bash tools/prepare_flaggems_runtime.sh
```

Use `--offline` to rebuild from the existing cache or `--refresh` to replace a
damaged cache. `prepare_flaggems_candidate.sh` remains as a compatibility alias.

Select a prepared runtime without modifying the Python installation:

```bash
GRAPHCAST_FLAGOS_MODE=flaggems \
GRAPHCAST_FLAGGEMS_VARIANT=internal \
  bash scripts/infer.sh
```

Set `GRAPHCAST_FLAGGEMS_VARIANT=official` for the matched upstream comparison,
or leave it as `installed` to use the FlagGems package already present in the
environment. An explicit `GRAPHCAST_FLAGGEMS_SOURCE_ROOT` takes precedence.

## Dense GEMM roof and MFU

The retained six-platform GEMM workload, hardware-path profiler launchers, and
the path-matched GraphCast MFU calculator are documented in
[`dense_gemm_peak/README.md`](dense_gemm_peak/README.md).
