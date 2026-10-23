# Package validated JAX inference and training experiments

**Goal:** Publish the completed BW1100 inference changes and training diagnostics on `zyx/sugon-graphcast`.

**Architecture:** Keep the existing inference baseline entry point. Preserve byte-identical experimental sources with SHA256 provenance, concise result evidence, and an external workspace preparer. Large inputs, checkpoints, traces and dependencies remain external. The preparer freezes code and links read-only input assets into the relative layout required by the measured launchers.

**Scope:** Existing inference profile/logical operators, producer and model RNE fusion, GPU-time MFU, 38-update training, trained-checkpoint inference, cross-inference diagnosis and fixed-state replay. No new training, convergence claim, H100 measurement or Notion change.

- [x] Preserve measured source snapshots and small evidence; document current loss-only training acceptance and separate inference acceptance.
- [x] Provide new-workspace preparation, explicit asset dependencies and commands; reject overwrites and keep historic outputs intact.
- [x] Run CPU tests, snapshot checks, shell/Python validation and portable staging checks; review the exact Git diff.
Remote publication: commit and push the authorized branch, then confirm the remote commit matches; record the result in the task response.
