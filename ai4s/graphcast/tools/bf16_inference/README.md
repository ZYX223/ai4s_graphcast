# GraphCast BF16 inference validation

These tools retain the BF16 validation workflow used for the six-platform
GraphCast campaign. They do not contain machine addresses, device allocation,
or asset locations; those remain explicit inputs to each run.

The workflow is:

1. Run a Native FP32 reference with `DIAGNOSTIC_TRAJECTORY=1`.
2. Run Native BF16 and FlagGems BF16 through
   `run_bf16_candidate_diagnostics.sh`. Each candidate lead is compared with
   the matching Native FP32 autoregressive lead by
   `run_with_trajectory_reference.py`.
3. Accept performance runs only after every lead is finite and the requested
   FlagGems operators are observed.
4. Run three fresh Native and FlagGems BF16 measurements with
   `run_bf16_performance_matrix.sh`, then calculate MFU against the
   path-matched measured BF16 dense-GEMM roof in `tools/dense_gemm_peak`.
   `summarize_campaign.py` validates those reports, their trajectory
   diagnostics, and the expected FlagGems coverage before calculating MFU.
   Its input distinguishes active dispatch coverage from coverage of the
   seven campaign core operators (for example, 810E dispatches 6/6 requested
   operators while reporting 6/7 core coverage because `addmm` stays Native).

Native and FlagGems measurements for a platform must use the same model
execution profile.  The six-platform campaign uses `baseline` for the H100
Native reference and `domestic-v1` for both modes on C550, BW3000, S5000,
910C, and 810E.  The summarizer rejects a report whose embedded profile does
not match the campaign input, so model-side optimization and FlagGems dispatch
remain separate dimensions.

When FlagGems is imported from an unpacked source artifact, its generated
`_version.py` may be absent and `flag_gems.__version__` can fall back to the
container's installed wheel metadata.  Record the immutable source artifact
SHA-256 and Git revision as the authoritative implementation identity; keep
the runtime version string only as environment metadata.

`STEPS` is any positive integer supported by the supplied forcing data. The
legacy runner filename includes “40step”, but these tools do not force a
40-step rollout.

The trajectory diagnostics report implementation drift, not forecast skill.
It records raw error, GraphCast-stddev-normalized max/RMSE, finiteness, source
hashes, and lead count. Weather RMSE/ACC additionally requires time-matched
truth and climatology and must be evaluated separately.

If nominally identical NetCDF inputs have different file hashes because of
compression or metadata encoding, run `hash_netcdf_semantics.py` on each
machine.  It hashes dimension/variable layout and canonical values while
ignoring container attributes and compression; byte-level identities must
still be retained alongside the shared semantic hash.

The shell entry points are environment-driven. At minimum,
`run_bf16_case.sh` requires the project root, immutable FlagGems source root,
asset paths, platform/backend/device, model execution profile, precision,
step count, Python executable, and a new output root. Existing output
directories are never reused.

On filesystems mounted `noexec`, use
`run_bf16_case_local_then_publish.sh`: tensor outputs are staged locally,
while Triton/FlagGems executable caches are placed under
`EXECUTABLE_CACHE_PARENT` (default `/tmp`). The staged run is checksummed
before publication and removed only after the published copy verifies.
