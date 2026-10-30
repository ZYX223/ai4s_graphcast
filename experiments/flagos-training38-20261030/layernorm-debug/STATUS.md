# LayerNorm native compiler investigation

2026-10-30: User explicitly chose to leave GPU timing pending. Do not start new GPU work until resumed. The existing NP-7 checklist remains unchecked; five default-native forward timing cells remain blank.

Confirmed trigger: native XLA horizontal-loop-fusion-for-copy combines scalar512 and reshaped scale[1,1,512] copies into a multi-output copy fusion with unequal lengths. EmitTuple/GetBitcastMap calls DecomposeBitcastToTrt, asserting ReshapeIsBitcast. Pass dumps locate introduction of this fusion. Full40962x1x512 LayerNorm and minimal scalar-plus-view reproducer both exit134 with default settings.

The benchmark-only option `--xla_disable_hlo_passes=horizontal-loop-fusion-for-copy` makes the minimal reproducer and full40962x1x512 LayerNorm compile/execute and exit0. All9full LayerNorm outputs remain. This is a workaround, not an XLA source fix or evidence for all five shapes. Existing model runs and baselines are unchanged.

The all-shape benchmark was stopped with SIGTERM/exit143 after another job occupied the card. Its outputs in bench-interrupted are invalid for timing; report may retain running state because termination prevented finalization. No elapsed time from that attempt enters summary tables. Only our benchmark child was terminated; other tasks were untouched.

Future measurement uses run-bench.sh for native then fl_only, both with the same compiler option, all10captured LayerNorm cases (five forward and five backward),3rounds of5warmup+20synchronized calls. It uses a fresh bench-idle output directory; original evidence stays separate. The new resource gate checks memory and utilization twice before launch; it is not an atomic reservation. Resource-gate regression:3failures under the permissive stub, then4/4passed after rejecting busy/missing cards.

Deploy the final run-bench.sh and resource_gate.py to /tmp/zyx-training38-layernorm-debug only when resuming; container retains the earlier interrupted script, so do not reuse it blindly. Select an actually idle card through LN_CARD. Preserve precision, all saved outputs, sample seeds and reference rules. Do not merge new measurements unless both backends complete with matching case IDs/signatures, numeric checks, configs and real exit0.
