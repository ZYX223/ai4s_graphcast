# Training logical operator timing

BW1100 single card; FlagOS-container JAX native vs FlagOS JAX (fl_only), both RNE+XF32. Horizon1 useful forward+backward including loss, 2 microbatches/update. Times are independent synchronized operator wall time weighted by calls, not GPU-only whole-model time. Remat and optimizer are excluded here but included in the full-model kernel attachment.

| Logical operator | JAX ms | FlagOS fl_only ms | JAX / FlagOS | FlagTree evidence |
| --- | ---: | ---: | ---: | --- |
| Linear | 4340.534 | 4690.013 | 0.925 | 33/33 cases contain FlagTree HLO custom calls; GEMM remains rocBLAS |
| Concat | 198.526 | 376.078 | 0.528 | 12/12 cases contain FlagTree HLO custom calls |
| LayerNorm | — | 1477.281 | — | 10/10 cases contain FlagTree HLO custom calls |
| ScatterAdd / SegmentSum | 663.150 | 2289.311 | 0.290 | 9/9 cases contain FlagTree HLO custom calls |
| SiLU | 452.944 | 770.196 | 0.588 | 10/10 cases contain FlagTree HLO custom calls |
| Gather | 245.205 | 472.025 | 0.519 | 9/9 cases contain FlagTree HLO custom calls |
| Add | 221.866 | 275.970 | 0.804 | 7/7 cases contain FlagTree HLO custom calls |
| Mul | 30.007 | 24.102 | 1.245 | 13/14 cases contain FlagTree HLO custom calls |
| Sub | 16.021 | 20.754 | 0.772 | 10/10 cases contain FlagTree HLO custom calls |
| Div | 18.580 | 23.157 | 0.802 | 12/13 cases contain FlagTree HLO custom calls |
| Slice | 4.630 | 5.483 | 0.844 | 11/11 cases contain FlagTree HLO custom calls |
| DynamicSlice | 3.350 | 4.202 | 0.797 | 2/2 cases contain FlagTree HLO custom calls |
| Broadcast | 205.969 | 182.753 | 1.127 | 18/27 cases contain FlagTree HLO custom calls |
| Reshape | 13.531 | 12.657 | 1.069 | 0/6 cases contain FlagTree HLO custom calls |
| Transpose | 25.747 | 28.779 | 0.895 | 5/17 cases contain FlagTree HLO custom calls |
| Squeeze | 2.914 | 4.138 | 0.704 | 0/2 cases contain FlagTree HLO custom calls |
| ReduceSum | 138.983 | 140.461 | 0.989 | 3/8 cases contain FlagTree HLO custom calls |
| Pad | 17.227 | 15.638 | 1.102 | 11/11 cases contain FlagTree HLO custom calls |
| Split | 267.896 | 199.043 | 1.346 | 6/6 cases contain FlagTree HLO custom calls |
| IntegerPow | 8.073 | 6.257 | 1.290 | 2/4 cases contain FlagTree HLO custom calls |

Phase labels use JAX scope evidence (jvp / transpose); unmarked loss/data/gradient auxiliary operations are listed separately. Deduplicated cases can combine scopes, so phase rows are diagnostic labels, not a rigorous partition of complete forward/backward time.

One large backward Linear failed memory allocation in both sequential suites; separate fresh processes at the same80% pool completed timings and sampled comparison. Original failures and retry evidence remain separate.

Blank speedup means the complete matched GPU timing/reference set did not pass. Five forward LayerNorm cases use CPU JAX sampled correctness separately; CPU timings never substitute for missing native GPU measurements. Synthetic backward saved tensors and sampled comparison (up to258 values/output, full finite check) limit correctness scope; full training loss provides a separate model-level check.

Original historical native baseline was not modified. This native column is the native mode in the current FlagOS container, not an unlabelled reuse of vendor-JAX historical timings.
