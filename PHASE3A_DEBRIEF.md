# PHASE 3A Debrief: Adaptive `hf_exact` CUDA Graph + Canonical Q Banding Performance Characterization

## Summary
Phase 3A landed adaptive `conf_adapt + hf_exact` q>1 CUDA-graph support, added canonical q-len banding for adaptive steady decode, and completed an expanded throughput characterization up to concurrency `c=512`.

Correctness behavior remained stable enough for continued iteration, but performance findings now expose three major risks:
1. Static `k=3` gains at low concurrency do not hold at high concurrency versus non-MTP.
2. MTP codepath with `k=1` underperforms non-MTP despite semantic equivalence.
3. Adaptive canonical-banding strategy is currently too inefficient to compete with static `k=3`, even where adaptive acceptance should help.

These findings define the Phase 3B focus: performance-first triage and optimization. If unresolved, redesign is likely required.

## Checkpoint Context
1. Repository: `sglang-mtp-lm`
2. Branch: `main`
3. Phase 3A code-state commit: `bca4b465d`
4. Debrief file: `PHASE3A_DEBRIEF.md`

## What Landed in Phase 3A
1. Adaptive q>1 CUDA graph controls for `conf_adapt + hf_exact`:
   - `--enable-mtp-adaptive-hf-exact-q-len-cuda-graph`
   - `--mtp-adaptive-cuda-graph-kmax-list`
2. Adaptive canonical q-len banding controls:
   - `--enable-mtp-adaptive-hf-exact-canonical-q-banding`
   - `--mtp-adaptive-canonical-q-lens`
3. q>1 graph eligibility updates for adaptive hf-exact in `model_runner`.
4. Capture-set logic for static and adaptive q-lens in `model_runner`.
5. Strict adaptive invariants around recompute/commit layout in scheduler paths.
6. `fixed_window` path fully disabled in `sampling_params` (only `hf_exact` accepted).
7. A/B harness support for adaptive graph expectation:
   - `mtp_cudagraph_ab.py --expect-conf-adapt-hf-exact-qgt1-graph`
8. Quick-driver additive throughput metrics:
   - `wall_time_s`, `p95_latency_s`, `mean_latency_s`, `total_completion_tokens`, `completion_tokens_per_s`
9. Phase 3 workflow document:
   - `PHASE3_ADAPTIVE_CUDAGRAPH_WORKFLOW.md`

## Throughput Matrix (Current Best Combined View)
Authoritative merged artifact:
`/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3_c512_allcols_retry_20260304_060341/final_comparison_with_rooflines_plus_c1_c512_corrected_nonmtp.json`

| c | adaptive eager tok/s | adaptive graph tok/s | static k=3 eager tok/s | static k=3 graph tok/s | static k=1 graph tok/s | non-MTP graph tok/s | adaptive graph / static k=3 graph | static k=3 graph / non-MTP graph |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 53.06 | 90.03 | 142.49 | 257.58 | 136.39 | 94.77 | 0.350 | 2.718 |
| 8 | 427.17 | 757.83 | 927.41 | 1432.56 | 867.28 | 857.08 | 0.529 | 1.671 |
| 32 | 1330.34 | 1524.47 | 2416.31 | 4084.34 | 2814.57 | 2161.96 | 0.373 | 1.889 |
| 64 | 1591.23 | 1609.79 | 4305.22 | 5050.05 | 2561.38 | 2504.70 | 0.319 | 2.016 |
| 128 | 1916.41 | 1803.26 | 6374.28 | 7694.45 | 3356.75 | 4525.97 | 0.234 | 1.700 |
| 192 | 1709.79 | 1435.83 | 6701.59 | 8377.92 | 4477.57 | 6190.74 | 0.171 | 1.353 |
| 256 | 1783.66 | 1883.95 | 7714.44 | 7908.09 | 5371.20 | 8277.64 | 0.238 | 0.955 |
| 512 | 671.72 | 1925.53 | 9913.30 | 9398.76 | 6180.73 | 11747.67 | 0.205 | 0.800 |

## Data Provenance Lock
1. `static_k1_graph` uses the MTP decode path with `mtp_enabled=true`, `mtp_k=1`, static strategy.
2. `non_mtp_graph` denotes default decode requests with no `mtp_*` request fields.
3. Reported non-MTP values are from the non-MTP decode codepath across the matrix.
4. For `c=512`, an initial quick-driver attempt was invalid (driver default sent MTP fields); that value is retained only as legacy debug metadata and was replaced by a corrected true non-MTP rerun.
5. The reported `c=512` non-MTP value (`11747.67 tok/s`) is the corrected authoritative result.

## Observations (Calibrated)
1. Adaptive eager at `c=512` appears trend-inconsistent and should not drive strong conclusions by itself.
2. Adaptive graph responsiveness versus eager is established, especially at low/moderate concurrency (`c=1,8,32`), but absolute adaptive throughput remains far below static `k=3`.
3. Static `k=3` enters diminishing returns at high concurrency, but does not hard-plateau in this matrix.
4. Non-MTP continues rising through high concurrency, so there is no evidence for a universal hard wall around ~8k tok/s.
5. Low concurrency behavior remains favorable to static `k=3`: meaningful edge over `k=1` and non-MTP up to roughly `c=64`.
6. That edge declines and flips at high concurrency (`c>=256`) where non-MTP overtakes static `k=3`.

## Problem Statements for Phase 3B
### Problem 1: Static `k=3` high-concurrency edge collapse
The ~2x low-concurrency edge versus non-MTP does not persist. At high concurrency, the edge shrinks and then reverses.

### Problem 2: MTP `k=1` systematic gap vs non-MTP
Even though `k=1` MTP and non-MTP are semantically similar from a decode-policy standpoint, measured throughput diverges significantly at medium/high concurrency (`c>=64`), with non-MTP consistently faster.

### Problem 3: Adaptive canonical-banding inefficiency
Adaptive canonical q-banding is currently too expensive:
1. Even at `c=1`, adaptive is much slower than static `k=1`.
2. At higher concurrency, adaptive gains from graph/concurrency are not enough to approach static `k=3`.
3. Current recompute + scheduling overhead appears to dominate potential gains from multi-token acceptance.

## Performance Target (Next Phase)
Primary near-term target:
1. At `c=64`, raise `adaptive_graph` to approximately `static_k3_graph` throughput.
2. Preserve a practical edge versus non-MTP in `c=1..64` (targeting roughly `1.75x` to `2x` where feasible).

Context target for later comparison:
1. Evaluate against Eagle3 speculative decoding in `c=1..64` once adaptive/static baselines are stabilized.

## Phase 3B Plan (Performance-Triage First)
### A) Measurement normalization and auditability
1. Standardize all columns under one harness with explicit mode metadata per run.
2. Ensure non-MTP runs are generated only by payloads with no `mtp_*` fields.
3. Emit a run-manifest JSON for each matrix row with:
   - server flags
   - request payload mode
   - prompt count/concurrency
   - max-new-tokens
   - mem-fraction and graph settings

### B) MTP `k=1` vs non-MTP gap isolation
1. Quantify per-step host-side scheduler overhead unique to MTP path.
2. Quantify batching efficiency deltas (running req count, queue dynamics, decode step cadence).
3. Identify removable MTP bookkeeping on `k=1` path.

### C) Adaptive bottleneck localization
1. Measure frequency and impact of `mixed_bucket_defer_decode` behavior under concurrency.
2. Measure effective batch occupancy by adaptive q-bucket and graph-hit rate.
3. Profile overhead from canonical-banding recompute expansion versus accepted-token benefit.

### D) Optimization iteration
1. Prioritize changes that improve occupancy and reduce scheduler overhead before changing algorithm semantics.
2. Keep correctness gates fixed while iterating on throughput.
3. Re-run matrix at `c={1,8,32,64,128,192,256}` per iteration; use `c=512` as stress point only.

## Acceptance Criteria for Phase 3B Entry/Exit
1. Entry criteria:
   - no correctness regressions relative to current Phase 3A behavior.
2. Exit criteria (performance milestone):
   - measurable closure of adaptive-vs-static gap at `c=64`.
   - reduced `k=1` MTP vs non-MTP gap at medium/high concurrency.
   - reproducible matrix with unambiguous mode provenance for every column/row.

## Artifacts
1. Main matrix (c=1..512):
   - `/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3_c512_allcols_retry_20260304_060341/final_comparison_with_rooflines_plus_c1_c512_corrected_nonmtp.json`
2. Prior merged matrix root (c=8..256 roofline columns):
   - `/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3_roofline_cols_20260304_043119`
3. c=1 completion run root:
   - `/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3_c1_row_20260304_045250`
4. Corrected true non-MTP c=512 run summary:
   - `/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3_c512_allcols_retry_20260304_060341/results/non_mtp_graph_true_c512/summary.json`

## Assumptions and Defaults
1. FlashInfer remains required for this MTP phase.
2. `conf_adapt + hf_exact` remains the only supported adaptive mode.
3. `fixed_window` remains out of scope.
4. High-concurrency stress runs may require reduced `mem_fraction_static` for stability.
