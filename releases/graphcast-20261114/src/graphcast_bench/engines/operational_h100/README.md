# Operational H100 engine

`ORIGINS.json` retains the original hashes and records delivery-only adapter
changes separately. The frozen model, numerical hooks and training recipe are
preserved. The delivery adapter provides portable input/output paths and
selects the frozen `shifted_mesh` inference and
`mesh-group8-m2g-save-stats-vjp-finalmesh-g2m` training configurations.

- `inference/`: independently warmed single-step check and three independent
  40-step forecasts, with Nsight timing and final-frame consistency checks.
  Thin launchers now delegate to the maintained `graphcast_bench.inference.h100`
  worker using `inference-context.json`; new runs do not execute generated Python.
- `training/`: optimized training, corresponding-update loss guards, checkpoint
  continuity checks and trace integrity helpers.
- `original_training/`: frozen model, optimizer, curriculum and runtime support
  that the optimized trainer imports and verifies.
- `frozen/`: original benchmark and pinned official model used by inference.
- `minimal-official/`: pinned official model used by the frozen training wrapper.
- `clone3_compat_exec.py`: the qualified process-local compatibility launcher.

The complete training workload carries its own parameter and optimizer state
through all twelve horizons: 16 updates at horizon 1 followed by two updates per
horizon 2–12, for 38 updates. It is a short numerical and mechanics check, not a
convergence claim. Losses are compared at corresponding updates against both the
original H100 trainer and frozen optimized H100 results.

Some internal names and historical metadata remain unchanged to preserve source
identity. Current image selection and external paths come from the versioned
delivery manifests, not the frozen historical launch configuration.
