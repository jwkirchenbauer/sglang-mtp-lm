# PHASE 3D Debrief: Low-Concurrency Robust Sweep + Eagle3 Recovery and Pareto Dataset

> Code-state anchor: `0397d8b58`
> This debrief’s results/analysis are tied to the implementation landed in that commit.

## Summary
Phase 3D completed the low-concurrency robust evaluation sweep and produced a single canonical merged dataset for throughput-latency analysis and plotting.

Primary outcomes:
1. Completed low-concurrency matrix for MTP/non-MTP strategies at `c={1,2,4,8,16}` with warmup-trimmed server/client throughput metrics.
2. Recovered and completed standalone Eagle3 runs at matching concurrency points, then merged those rows into the same Phase3D table.
3. Produced a Pareto-ready dataset and final strategy-by-strategy plot artifact for follow-on analysis.
4. Captured concrete operational lessons from control-plane failures (requeue/monitor loops, runtime code resolution, env discipline).

Final status of planned matrix rows:
1. `75/75` rows complete with `rc=0` in the merged canonical summary.
2. Coverage is `15` strategies x `5` concurrency values.

## Checkpoint Context
1. Repository: `sglang-mtp-lm`
2. Branch at debrief finalization: `main`
3. Debrief code-state anchor commit: `0397d8b58`
4. Baseline MTP/non-MTP sweep root:
   - [`outputs/phase3d_lowc_robust_20260306_035156`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3d_lowc_robust_20260306_035156)
5. Eagle3 one-shot recovery root:
   - [`outputs/phase3d_eagle3_single_20260306_185131`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3d_eagle3_single_20260306_185131)
6. Canonical merged summary root (copied/hosted with the full robust run summary set):
   - [`outputs/phase3d_lowc_robust_20260306_035156_with_eagle3_single/summary`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3d_lowc_robust_20260306_035156_with_eagle3_single/summary)

## How Phase3D Fits the Project Arc
Phase3D is a measurement-hardening phase that follows the Phase3A/3B performance characterization and workflow consolidation.

1. From Phase3A:
   - We entered with unresolved questions on static-vs-non-MTP scaling behavior and adaptive overhead profile.
2. From Phase3B:
   - We had reproducible matrix tooling but primarily high-concurrency checkpoints (`c={1,32,64,128}`).
3. Phase3D contribution:
   - Fill the low-concurrency regime with more granular points (`1,2,4,8,16`) and stronger multi-metric reporting for Pareto-frontier interpretation.
4. Cross-phase continuity:
   - Maintains the correctness-first and provenance-first discipline established in Phase2C and Phase3B (artifact roots, explicit run manifests, strict environment workflow notes).

## What Landed in Phase3D
1. Batch-sharded low-concurrency orchestration and monitoring flow:
   - [`scripts/playground/phase3d_lowc_submit.py`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/scripts/playground/phase3d_lowc_submit.py)
   - [`scripts/playground/phase3d_lowc_worker.sh`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/scripts/playground/phase3d_lowc_worker.sh)
2. Warmup-trimmed summary + Pareto point generation:
   - [`scripts/playground/phase3d_lowc_metrics_summary.py`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/scripts/playground/phase3d_lowc_metrics_summary.py)
3. Strategy-wise Pareto plotting:
   - [`scripts/playground/phase3d_pareto_plot_by_strategy.py`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/scripts/playground/phase3d_pareto_plot_by_strategy.py)
4. Standalone Eagle3 runner and merge utility:
   - [`scripts/playground/eagle3_lmeval_concurrency_runner.py`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/scripts/playground/eagle3_lmeval_concurrency_runner.py)
   - [`scripts/playground/eagle3_vs_phase3b_merge.py`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/scripts/playground/eagle3_vs_phase3b_merge.py)
5. Runtime API/tooling changes used during this phase:
   - FlashInfer draft cudagraph metadata fix for EAGLE3 draft path in [`python/sglang/srt/layers/attention/flashinfer_backend.py`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/python/sglang/srt/layers/attention/flashinfer_backend.py) (`effective_bs = bs * topk`).
   - `--disable-chat-template` option added to [`scripts/playground/mtp_lmeval_matrix_runner.py`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/scripts/playground/mtp_lmeval_matrix_runner.py) for non-chat checkpoints.
6. Workflow hardening docs:
   - [`DEBUG_ENV_WORKFLOW.md`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/DEBUG_ENV_WORKFLOW.md)
   - [`PHASE3C_EAGLE3_32B_WORKFLOW.md`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/PHASE3C_EAGLE3_32B_WORKFLOW.md)

## Evaluation Protocol and Metrics (Phase3D)
1. Task family:
   - `gsm8k_cot_singleshot`
2. Run depth:
   - `limit=512`
3. Concurrency points:
   - `c={1,2,4,8,16}`
4. Warmup trim:
   - drop first `floor(0.10 * N)` decode/request samples per case before averaging.
5. Core reported columns:
   - `server_peak_gen_tps`
   - `server_avg_gen_tps_trim10`
   - `server_avg_per_req_tps_trim10`
   - `client_avg_tpt_trim10`
   - `client_avg_per_req_tpt_trim10`
   - `Flex EM`
   - ratio columns vs non-MTP at matching `c`
6. Pareto coordinates in final table:
   - x = `pareto_x_latency_inv_tok_per_s_per_seq` (client per-request tok/s)
   - y = `pareto_y_throughput_tok_per_s_gpu` (server avg gen tok/s)

## Adaptive Q-Len Grouping Used in Phase3D
For the `conf_adapt` Phase3D runs, canonical q-len banding was disabled and adaptive grouping used native `hf_exact` q-lens.

1. Canonical banding runtime flags in these runs were:
   - `enable_mtp_adaptive_hf_exact_canonical_q_banding=False`
   - `mtp_adaptive_canonical_q_lens=[]`
2. Effective adaptive q-len capture/group sets were therefore exact ranges by `k_max`:
   - `k_max=3`: `{3,4,5}`
   - `k_max=8`: `{8,9,10,11,12,13,14,15}`
   - `k_max=16`: `{16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}`
3. `conf_adapt` shards also captured static q-lens `{3,5}` because static `k=3` q-len graph capture was co-enabled in the shard launch profile.

## Final Results Snapshot
Canonical merged source:
[`outputs/phase3d_eagle3_single_20260306_185131/summary/lowc_metrics_complete_with_eagle3.tsv`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3d_eagle3_single_20260306_185131/summary/lowc_metrics_complete_with_eagle3.tsv)

Key quantitative takeaways:
1. Low-c `k=1` parity at `c=1` is near-perfect vs non-MTP:
   - non-MTP server avg `144.24`, k=1 `142.54` (`0.988x`)
   - non-MTP per-req client `131.57`, k=1 `139.21` (`1.058x`)
2. Across `c=2..16`, `k=1` is mostly below non-MTP on server avg throughput (`0.812x..0.916x`) while remaining close in per-request client throughput (`0.853x..0.960x`).
3. Static `k=3` remains materially ahead of non-MTP in this low-c regime:
   - server avg ratio `2.364x` at `c=1` and `2.044x` at `c=16`.
4. Eagle3 standalone is faster than non-MTP at all tested low-c points:
   - server avg ratio vs non-MTP: `2.070x`, `2.027x`, `1.882x`, `1.657x`, `1.388x` for `c=1,2,4,8,16`.
5. Eagle3 vs static `k=3`:
   - competitive at `c=2..4` (`0.966x`, `0.949x` on server avg),
   - behind at `c=1,8,16` (`0.876x`, `0.862x`, `0.679x`).
6. Eagle3 accuracy stays near non-MTP and high-accuracy adaptive values:
   - Eagle3 Flex EM range `0.6074..0.6230`
   - non-MTP Flex EM range `0.6250..0.6328`
7. High-k static policies dominate throughput but collapse accuracy as expected for this checkpoint:
   - `k=16` server avg up to `8364.79`, Flex EM near `0.10`.
8. Adaptive `t=0.9` variants preserve high Flex EM but lose server throughput relative to non-MTP as concurrency rises.
9. Final merged matrix integrity:
   - `75` rows total, all `rc=0`, no missing concurrency point in any strategy.

## Pareto Plot Outputs
Primary Phase3D plot artifact:
- [`outputs/phase3d_eagle3_single_20260306_185131/summary/lowc_pareto_by_strategy_with_eagle3.png`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3d_eagle3_single_20260306_185131/summary/lowc_pareto_by_strategy_with_eagle3.png)

Supporting table used for plotting:
- [`outputs/phase3d_lowc_robust_20260306_035156_with_eagle3_single/summary/pareto_points.tsv`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3d_lowc_robust_20260306_035156_with_eagle3_single/summary/pareto_points.tsv)

Why the final summary/plot live where they do:
1. The original low-c robust root contains the completed 70 MTP/non-MTP rows.
2. Eagle3 was recovered in a separate single-attempt root after control-plane issues.
3. Final canonical merged summary root (`..._with_eagle3_single`) exists so analysis consumers can use one complete table without manually joining roots.
4. The strategy-colored Pareto PNG is written in the Eagle3 recovery summary directory because that directory already held the final merged-with-Eagle3 summary products at generation time.

## Eagle3 Recovery Postmortem (Operational + Runtime)
### What failed first
1. Batch-sharded Eagle3 job under the low-c robust root failed to produce rows.
2. Monitoring/requeue machinery complicated triage and produced repeated queue churn.
3. Worktree/editable-install switching introduced environment fragility and toolchain exposure (`C++20`/JIT mismatch in old code state).

### Concrete runtime failure signature
1. Draft CUDA graph initialization mismatch in Eagle3 path:
   - `runtime batch size 192 mismatches ... 48`
   - equivalent mismatch observed with other caps (`256 vs 64`)
2. Root cause:
   - draft wrapper metadata capture/replay planned for sequence batch while runtime used flattened draft token batch (`num_seqs * topk`).

### What fixed it
1. Switched to a single immutable Eagle3 run script with no auto-resubmit/reconcile loop.
2. Enforced standalone Eagle3 server mode with validated args and strict readiness gating.
3. Applied backend draft metadata fix (`effective_bs = bs * topk`) in flashinfer backend.
4. Re-ran one-shot Eagle3 sweep and obtained `5/5` successful rows (`c=1,2,4,8,16`).

Postmortem artifacts:
1. Failure capture:
   - [`outputs/phase3d_eagle3_single_20260306_185131/summary/eagle3_single_attempt_failure.md`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3d_eagle3_single_20260306_185131/summary/eagle3_single_attempt_failure.md)
2. Recovery note:
   - [`outputs/phase3d_eagle3_single_20260306_185131/summary/eagle3_recovery_postmortem.md`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3d_eagle3_single_20260306_185131/summary/eagle3_recovery_postmortem.md)

## Key Takeaways to Never Forget
1. Always execute the HPC startup protocol exactly:
   - interactive login shell,
   - `sslm_sgl_env`,
   - then Slurm launch (`sbatch`/launcher) from that shell.
2. Treat Eagle3 as standalone speculative decoding:
   - no MTP request semantics,
   - use dedicated Eagle3 runner/server flow.
3. For low-concurrency sweeps, explicitly reason about both client concurrency and server graph/microbatch envelopes.
4. Use job IDs and immutable run directories as source of truth for status, not inferred state from mutable launcher wrappers.
5. If a single-attempt recovery is requested, disable all auto-requeue/monitor loops and freeze one script.
6. Persist all failed artifacts; they are needed for accurate postmortem and reproducibility.
7. Keep canonical “merged” summary roots once multiple run roots are needed to complete one experimental matrix.

## Open Questions and Next Sketches
Phase3D intentionally did not include decode-path optimization work. The next actionable sketches are:

1. Low-c stability confirmation pass:
   - rerun selected anchor strategies with larger prompt counts to reduce variance (`non-MTP`, `k=1`, `k=3`, `eagle3`, `conf_adapt_k3_t09`, `conf_adapt_k8_t09`).
2. MTP `k=1` vs non-MTP divergence analysis:
   - instrument scheduler/request lifecycle deltas at `c=2..32` where crossover behavior appears.
3. Pareto frontier packaging:
   - formalize a small fixed strategy subset for repeatable publication-quality Pareto curves.
4. 32B revisit gate:
   - resume 32B matrix only after better-trained checkpoint is available; Phase3D results confirm workflow viability but not meaningful quality under early checkpoint behavior.
5. Future sweep policy tightening:
   - enforce explicit bounds on both client concurrency and server-side graph/request capacity in launch scripts to prevent ambiguity.

## Canonical Artifacts to Retain
1. Full low-c robust run root:
   - [`outputs/phase3d_lowc_robust_20260306_035156`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3d_lowc_robust_20260306_035156)
2. Eagle3 single-attempt recovery root:
   - [`outputs/phase3d_eagle3_single_20260306_185131`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3d_eagle3_single_20260306_185131)
3. Canonical merged summary root:
   - [`outputs/phase3d_lowc_robust_20260306_035156_with_eagle3_single`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3d_lowc_robust_20260306_035156_with_eagle3_single)
4. Plot export bundle:
   - [`outputs/phase3d_lowc_plot_artifacts_20260306.zip`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3d_lowc_plot_artifacts_20260306.zip)

## Assumptions and Defaults
1. Canonical Phase3D table is the merged 75-row summary with Eagle3 included.
2. This debrief is a synthesis of existing runs/artifacts only; no new benchmarks were needed.
3. Metrics interpretation uses the Phase3D warmup-trim rule (drop first 10% samples per case).

## Appendix A: Full 75-Row Matrix (Inline)
Source markdown table:
[`outputs/phase3d_eagle3_single_20260306_185131/summary/lowc_metrics_complete_with_eagle3.md`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3d_eagle3_single_20260306_185131/summary/lowc_metrics_complete_with_eagle3.md)

| strategy | c | case | rc | elapsed_s | Flex EM | decode_samples_raw | decode_samples_kept | server_peak_gen_tps | server_avg_gen_tps_trim10 | server_avg_per_req_tps_trim10 | request_samples_raw | request_samples_kept | client_avg_tpt_trim10 | client_avg_per_req_tpt_trim10 | server_avg_gen_tps_vs_non_mtp | server_avg_per_req_tps_vs_non_mtp | client_avg_tpt_vs_non_mtp | client_avg_per_req_tpt_vs_non_mtp | pareto_x_latency_inv_tok_per_s_per_seq | pareto_y_throughput_tok_per_s_gpu |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| non-MTP | 1 | non_mtp_c1 | 0 | 655 | 0.6328 | 4231 | 3808 | 157.10 | 144.24 | 144.24 | 512 | 461 | 131.91 | 131.57 | 1.000 | 1.000 | 1.000 | 1.000 | 131.57 | 144.24 |
| non-MTP | 2 | non_mtp_c2 | 0 | 182 | 0.6328 | 2142 | 1928 | 306.42 | 282.75 | 144.63 | 512 | 461 | 279.01 | 139.97 | 1.000 | 1.000 | 1.000 | 1.000 | 139.97 | 282.75 |
| non-MTP | 4 | non_mtp_c4 | 0 | 111 | 0.6328 | 1072 | 965 | 602.45 | 528.07 | 134.76 | 512 | 461 | 516.11 | 129.99 | 1.000 | 1.000 | 1.000 | 1.000 | 129.99 | 528.07 |
| non-MTP | 8 | non_mtp_c8 | 0 | 75 | 0.6270 | 539 | 486 | 1183.70 | 934.14 | 119.89 | 512 | 461 | 895.29 | 113.80 | 1.000 | 1.000 | 1.000 | 1.000 | 113.80 | 934.14 |
| non-MTP | 16 | non_mtp_c16 | 0 | 57 | 0.6250 | 274 | 247 | 2239.18 | 1497.24 | 99.45 | 512 | 461 | 1415.64 | 91.75 | 1.000 | 1.000 | 1.000 | 1.000 | 91.75 | 1497.24 |
| k=1 | 1 | mtp_k1_c1 | 0 | 592 | 0.6504 | 4302 | 3872 | 152.21 | 142.54 | 142.54 | 512 | 461 | 139.31 | 139.21 | 0.988 | 0.988 | 1.056 | 1.058 | 139.21 | 142.54 |
| k=1 | 2 | mtp_k1_c2 | 0 | 209 | 0.6504 | 2490 | 2241 | 296.59 | 243.88 | 145.80 | 512 | 461 | 239.08 | 122.57 | 0.863 | 1.008 | 0.857 | 0.876 | 122.57 | 243.88 |
| k=1 | 4 | mtp_k1_c4 | 0 | 128 | 0.6504 | 1371 | 1234 | 583.43 | 436.80 | 148.04 | 512 | 461 | 427.27 | 110.42 | 0.827 | 1.099 | 0.828 | 0.849 | 110.42 | 436.80 |
| k=1 | 8 | mtp_k1_c8 | 0 | 85 | 0.6484 | 753 | 678 | 1122.44 | 758.62 | 145.14 | 512 | 461 | 744.84 | 97.08 | 0.812 | 1.211 | 0.832 | 0.853 | 97.08 | 758.62 |
| k=1 | 16 | mtp_k1_c16 | 0 | 59 | 0.6465 | 391 | 352 | 2090.78 | 1371.32 | 141.95 | 512 | 461 | 1341.67 | 88.10 | 0.916 | 1.427 | 0.948 | 0.960 | 88.10 | 1371.32 |
| k=16 | 1 | mtp_k16_c1 | 0 | 396 | 0.0996 | 429 | 387 | 1186.72 | 1084.78 | 1084.78 | 512 | 461 | 1038.09 | 1048.14 | 7.521 | 7.521 | 7.870 | 7.966 | 1048.14 | 1084.78 |
| k=16 | 2 | mtp_k16_c2 | 0 | 59 | 0.0996 | 226 | 204 | 2289.39 | 1985.73 | 1125.90 | 512 | 461 | 1908.34 | 969.43 | 7.023 | 7.785 | 6.840 | 6.926 | 969.43 | 1985.73 |
| k=16 | 4 | mtp_k16_c4 | 0 | 44 | 0.1016 | 115 | 104 | 4245.60 | 3696.16 | 1072.79 | 512 | 461 | 3550.98 | 909.75 | 6.999 | 7.961 | 6.880 | 6.998 | 909.75 | 3696.16 |
| k=16 | 8 | mtp_k16_c8 | 0 | 37 | 0.1074 | 58 | 53 | 6893.87 | 6085.97 | 859.84 | 512 | 461 | 5869.28 | 762.80 | 6.515 | 7.172 | 6.556 | 6.703 | 762.80 | 6085.97 |
| k=16 | 16 | mtp_k16_c16 | 0 | 34 | 0.0996 | 29 | 27 | 9312.16 | 8364.79 | 618.00 | 512 | 461 | 8108.75 | 534.30 | 5.587 | 6.214 | 5.728 | 5.823 | 534.30 | 8364.79 |
| k=2 | 1 | mtp_k2_c1 | 0 | 463 | 0.5859 | 2192 | 1973 | 274.90 | 244.98 | 244.98 | 512 | 461 | 234.27 | 233.68 | 1.698 | 1.698 | 1.776 | 1.776 | 233.68 | 244.98 |
| k=2 | 2 | mtp_k2_c2 | 0 | 133 | 0.5938 | 1267 | 1141 | 539.02 | 424.67 | 260.96 | 512 | 461 | 409.81 | 209.27 | 1.502 | 1.804 | 1.469 | 1.495 | 209.27 | 424.67 |
| k=2 | 4 | mtp_k2_c4 | 0 | 88 | 0.5801 | 704 | 634 | 1040.63 | 744.18 | 269.15 | 512 | 461 | 717.98 | 185.03 | 1.409 | 1.997 | 1.391 | 1.423 | 185.03 | 744.18 |
| k=2 | 8 | mtp_k2_c8 | 0 | 62 | 0.5742 | 382 | 344 | 1933.14 | 1286.23 | 263.35 | 512 | 461 | 1252.96 | 162.46 | 1.377 | 2.197 | 1.400 | 1.428 | 162.46 | 1286.23 |
| k=2 | 16 | mtp_k2_c16 | 0 | 47 | 0.5859 | 197 | 178 | 3394.62 | 2230.10 | 238.99 | 512 | 461 | 2172.19 | 141.72 | 1.489 | 2.403 | 1.534 | 1.545 | 141.72 | 2230.10 |
| k=3 | 1 | mtp_k3_c1 | 0 | 385 | 0.5332 | 1499 | 1350 | 406.82 | 341.02 | 341.02 | 512 | 461 | 318.13 | 317.10 | 2.364 | 2.364 | 2.412 | 2.410 | 317.10 | 341.02 |
| k=3 | 2 | mtp_k3_c2 | 0 | 107 | 0.5254 | 863 | 777 | 792.68 | 593.16 | 375.78 | 512 | 461 | 559.54 | 284.76 | 2.098 | 2.598 | 2.005 | 2.034 | 284.76 | 593.16 |
| k=3 | 4 | mtp_k3_c4 | 0 | 71 | 0.5449 | 484 | 436 | 1535.95 | 1047.02 | 399.98 | 512 | 461 | 989.19 | 253.77 | 1.983 | 2.968 | 1.917 | 1.952 | 253.77 | 1047.02 |
| k=3 | 8 | mtp_k3_c8 | 0 | 53 | 0.5254 | 261 | 235 | 2815.05 | 1796.65 | 401.60 | 512 | 461 | 1720.78 | 221.72 | 1.923 | 3.350 | 1.922 | 1.948 | 221.72 | 1796.65 |
| k=3 | 16 | mtp_k3_c16 | 0 | 42 | 0.5352 | 136 | 123 | 4854.28 | 3060.68 | 375.47 | 512 | 461 | 2948.69 | 191.16 | 2.044 | 3.775 | 2.083 | 2.084 | 191.16 | 3060.68 |
| k=4 | 1 | mtp_k4_c1 | 0 | 392 | 0.4551 | 1157 | 1042 | 538.21 | 429.89 | 429.89 | 512 | 461 | 397.63 | 394.77 | 2.980 | 2.980 | 3.014 | 3.000 | 394.77 | 429.89 |
| k=4 | 2 | mtp_k4_c2 | 0 | 89 | 0.4414 | 670 | 603 | 1044.71 | 750.50 | 479.95 | 512 | 461 | 700.05 | 354.46 | 2.654 | 3.318 | 2.509 | 2.532 | 354.46 | 750.50 |
| k=4 | 4 | mtp_k4_c4 | 0 | 63 | 0.4336 | 377 | 340 | 1997.60 | 1314.11 | 518.38 | 512 | 461 | 1222.38 | 314.50 | 2.489 | 3.847 | 2.368 | 2.419 | 314.50 | 1314.11 |
| k=4 | 8 | mtp_k4_c8 | 0 | 47 | 0.4531 | 202 | 182 | 3666.87 | 2304.36 | 548.05 | 512 | 461 | 2152.37 | 275.79 | 2.467 | 4.571 | 2.404 | 2.423 | 275.79 | 2304.36 |
| k=4 | 16 | mtp_k4_c16 | 0 | 39 | 0.4414 | 105 | 95 | 6235.47 | 3822.21 | 512.93 | 512 | 461 | 3641.16 | 235.92 | 2.553 | 5.158 | 2.572 | 2.571 | 235.92 | 3822.21 |
| k=5 | 1 | mtp_k5_c1 | 0 | 372 | 0.4043 | 947 | 853 | 664.10 | 500.29 | 500.29 | 512 | 461 | 458.61 | 455.54 | 3.468 | 3.468 | 3.477 | 3.462 | 455.54 | 500.29 |
| k=5 | 2 | mtp_k5_c2 | 0 | 83 | 0.3965 | 554 | 499 | 1295.11 | 887.05 | 569.25 | 512 | 461 | 812.36 | 410.37 | 3.137 | 3.936 | 2.912 | 2.932 | 410.37 | 887.05 |
| k=5 | 4 | mtp_k5_c4 | 0 | 59 | 0.3984 | 307 | 277 | 2466.07 | 1569.94 | 658.27 | 512 | 461 | 1432.03 | 364.45 | 2.973 | 4.885 | 2.775 | 2.804 | 364.45 | 1569.94 |
| k=5 | 8 | mtp_k5_c8 | 0 | 45 | 0.4062 | 166 | 150 | 4439.47 | 2673.76 | 699.45 | 512 | 461 | 2471.71 | 317.24 | 2.862 | 5.834 | 2.761 | 2.788 | 317.24 | 2673.76 |
| k=5 | 16 | mtp_k5_c16 | 0 | 37 | 0.4180 | 85 | 77 | 7402.60 | 4437.08 | 637.78 | 512 | 461 | 4163.31 | 269.13 | 2.964 | 6.413 | 2.941 | 2.933 | 269.13 | 4437.08 |
| k=8 | 1 | mtp_k8_c1 | 0 | 361 | 0.3105 | 663 | 597 | 1060.30 | 696.76 | 696.76 | 512 | 461 | 629.30 | 630.90 | 4.831 | 4.831 | 4.771 | 4.795 | 630.90 | 696.76 |
| k=8 | 2 | mtp_k8_c2 | 0 | 70 | 0.3125 | 382 | 344 | 2053.42 | 1265.00 | 825.40 | 512 | 461 | 1164.73 | 588.07 | 4.474 | 5.707 | 4.175 | 4.202 | 588.07 | 1265.00 |
| k=8 | 4 | mtp_k8_c4 | 0 | 51 | 0.3320 | 204 | 184 | 3875.74 | 2265.08 | 943.61 | 512 | 461 | 2075.90 | 525.93 | 4.289 | 7.002 | 4.022 | 4.046 | 525.93 | 2265.08 |
| k=8 | 8 | mtp_k8_c8 | 0 | 40 | 0.3320 | 108 | 98 | 6795.63 | 3969.77 | 1052.10 | 512 | 461 | 3654.93 | 469.32 | 4.250 | 8.775 | 4.082 | 4.124 | 469.32 | 3969.77 |
| k=8 | 16 | mtp_k8_c16 | 0 | 35 | 0.3223 | 54 | 49 | 9351.47 | 6171.14 | 865.29 | 512 | 461 | 5748.61 | 372.04 | 4.122 | 8.701 | 4.061 | 4.055 | 372.04 | 6171.14 |
| conf_adapt_k16_t06 | 1 | conf_adapt_k16_t06_c1 | 0 | 473 | 0.4590 | 838 | 755 | 1326.20 | 522.63 | 522.63 | 512 | 461 | 486.08 | 504.98 | 3.623 | 3.623 | 3.685 | 3.838 | 504.98 | 522.63 |
| conf_adapt_k16_t06 | 2 | conf_adapt_k16_t06_c2 | 0 | 105 | 0.4531 | 815 | 734 | 1671.38 | 615.52 | 608.52 | 512 | 461 | 581.10 | 302.14 | 2.177 | 4.207 | 2.083 | 2.159 | 302.14 | 615.52 |
| conf_adapt_k16_t06 | 4 | conf_adapt_k16_t06_c4 | 0 | 105 | 0.4492 | 812 | 731 | 1420.28 | 618.07 | 604.71 | 512 | 461 | 578.58 | 200.30 | 1.170 | 4.487 | 1.121 | 1.541 | 200.30 | 618.07 |
| conf_adapt_k16_t06 | 8 | conf_adapt_k16_t06_c8 | 0 | 103 | 0.4570 | 795 | 716 | 1525.07 | 630.92 | 616.26 | 512 | 461 | 583.09 | 178.19 | 0.675 | 5.140 | 0.651 | 1.566 | 178.19 | 630.92 |
| conf_adapt_k16_t06 | 16 | conf_adapt_k16_t06_c16 | 0 | 112 | 0.4531 | 763 | 687 | 1622.47 | 647.53 | 610.59 | 512 | 461 | 503.02 | 166.18 | 0.432 | 6.140 | 0.355 | 1.811 | 166.18 | 647.53 |
| conf_adapt_k16_t09 | 1 | conf_adapt_k16_t09_c1 | 0 | 514 | 0.6172 | 1440 | 1296 | 1000.85 | 332.55 | 332.55 | 512 | 461 | 319.71 | 335.23 | 2.306 | 2.306 | 2.424 | 2.548 | 335.23 | 332.55 |
| conf_adapt_k16_t09 | 2 | conf_adapt_k16_t09_c2 | 0 | 149 | 0.6152 | 1406 | 1266 | 1223.59 | 371.18 | 366.89 | 512 | 461 | 362.85 | 188.43 | 1.313 | 2.537 | 1.301 | 1.346 | 188.43 | 371.18 |
| conf_adapt_k16_t09 | 4 | conf_adapt_k16_t09_c4 | 0 | 148 | 0.6191 | 1395 | 1256 | 1250.41 | 374.47 | 368.25 | 512 | 461 | 362.85 | 125.13 | 0.709 | 2.733 | 0.703 | 0.963 | 125.13 | 374.47 |
| conf_adapt_k16_t09 | 8 | conf_adapt_k16_t09_c8 | 0 | 152 | 0.6152 | 1369 | 1233 | 1248.76 | 380.56 | 372.45 | 512 | 461 | 347.20 | 113.49 | 0.407 | 3.106 | 0.388 | 0.997 | 113.49 | 380.56 |
| conf_adapt_k16_t09 | 16 | conf_adapt_k16_t09_c16 | 0 | 143 | 0.6191 | 1322 | 1190 | 1333.31 | 395.71 | 372.90 | 512 | 461 | 370.77 | 108.22 | 0.264 | 3.750 | 0.262 | 1.180 | 108.22 | 395.71 |
| conf_adapt_k3_t06 | 1 | conf_adapt_k3_t06_c1 | 0 | 448 | 0.5918 | 1681 | 1513 | 393.61 | 296.52 | 296.52 | 512 | 461 | 279.73 | 280.21 | 2.056 | 2.056 | 2.121 | 2.130 | 280.21 | 296.52 |
| conf_adapt_k3_t06 | 2 | conf_adapt_k3_t06_c2 | 0 | 165 | 0.5840 | 1616 | 1455 | 753.08 | 326.31 | 319.45 | 512 | 461 | 317.60 | 160.83 | 1.154 | 2.209 | 1.138 | 1.149 | 160.83 | 326.31 |
| conf_adapt_k3_t06 | 4 | conf_adapt_k3_t06_c4 | 0 | 160 | 0.5859 | 1558 | 1403 | 776.31 | 339.00 | 326.29 | 512 | 461 | 328.11 | 114.71 | 0.642 | 2.421 | 0.636 | 0.882 | 114.71 | 339.00 |
| conf_adapt_k3_t06 | 8 | conf_adapt_k3_t06_c8 | 0 | 147 | 0.5879 | 1381 | 1243 | 1332.98 | 377.80 | 333.69 | 512 | 461 | 361.98 | 100.80 | 0.404 | 2.783 | 0.404 | 0.886 | 100.80 | 377.80 |
| conf_adapt_k3_t06 | 16 | conf_adapt_k3_t06_c16 | 0 | 123 | 0.5840 | 1093 | 984 | 2209.63 | 465.51 | 366.58 | 512 | 461 | 445.81 | 85.20 | 0.311 | 3.686 | 0.315 | 0.929 | 85.20 | 465.51 |
| conf_adapt_k3_t09 | 1 | conf_adapt_k3_t09_c1 | 0 | 429 | 0.6387 | 2100 | 1890 | 395.55 | 244.80 | 244.80 | 512 | 461 | 233.79 | 235.26 | 1.697 | 1.697 | 1.772 | 1.788 | 235.26 | 244.80 |
| conf_adapt_k3_t09 | 2 | conf_adapt_k3_t09_c2 | 0 | 197 | 0.6328 | 2057 | 1852 | 497.56 | 263.81 | 261.54 | 512 | 461 | 258.73 | 130.56 | 0.933 | 1.808 | 0.927 | 0.933 | 130.56 | 263.81 |
| conf_adapt_k3_t09 | 4 | conf_adapt_k3_t09_c4 | 0 | 195 | 0.6309 | 2024 | 1822 | 547.56 | 267.77 | 261.92 | 512 | 461 | 261.01 | 90.37 | 0.507 | 1.944 | 0.506 | 0.695 | 90.37 | 267.77 |
| conf_adapt_k3_t09 | 8 | conf_adapt_k3_t09_c8 | 0 | 186 | 0.6348 | 1901 | 1711 | 813.10 | 284.38 | 270.84 | 512 | 461 | 274.02 | 83.06 | 0.304 | 2.259 | 0.306 | 0.730 | 83.06 | 284.38 |
| conf_adapt_k3_t09 | 16 | conf_adapt_k3_t09_c16 | 0 | 166 | 0.6367 | 1644 | 1480 | 1275.95 | 322.25 | 285.53 | 512 | 461 | 308.38 | 75.51 | 0.215 | 2.871 | 0.218 | 0.823 | 75.51 | 322.25 |
| conf_adapt_k8_t06 | 1 | conf_adapt_k8_t06_c1 | 0 | 438 | 0.5117 | 954 | 859 | 1030.11 | 484.37 | 484.37 | 512 | 461 | 452.72 | 463.56 | 3.358 | 3.358 | 3.432 | 3.523 | 463.56 | 484.37 |
| conf_adapt_k8_t06 | 2 | conf_adapt_k8_t06_c2 | 0 | 110 | 0.5059 | 921 | 829 | 1166.00 | 565.07 | 555.32 | 512 | 461 | 537.58 | 275.04 | 1.998 | 3.840 | 1.927 | 1.965 | 275.04 | 565.07 |
| conf_adapt_k8_t06 | 4 | conf_adapt_k8_t06_c4 | 0 | 109 | 0.4922 | 908 | 818 | 1447.60 | 577.31 | 563.28 | 512 | 461 | 544.41 | 189.36 | 1.093 | 4.180 | 1.055 | 1.457 | 189.36 | 577.31 |
| conf_adapt_k8_t06 | 8 | conf_adapt_k8_t06_c8 | 0 | 108 | 0.4980 | 880 | 792 | 1472.76 | 589.52 | 560.20 | 512 | 461 | 550.84 | 169.33 | 0.631 | 4.672 | 0.615 | 1.488 | 169.33 | 589.52 |
| conf_adapt_k8_t06 | 16 | conf_adapt_k8_t06_c16 | 0 | 101 | 0.5000 | 804 | 724 | 1828.92 | 644.32 | 591.65 | 512 | 461 | 592.57 | 156.14 | 0.430 | 5.949 | 0.419 | 1.702 | 156.14 | 644.32 |
| conf_adapt_k8_t09 | 1 | conf_adapt_k8_t09_c1 | 0 | 463 | 0.6172 | 1514 | 1363 | 934.62 | 329.81 | 329.81 | 512 | 461 | 316.40 | 327.25 | 2.287 | 2.287 | 2.399 | 2.487 | 327.25 | 329.81 |
| conf_adapt_k8_t09 | 2 | conf_adapt_k8_t09_c2 | 0 | 151 | 0.6172 | 1479 | 1332 | 891.06 | 365.20 | 361.78 | 512 | 461 | 356.62 | 183.77 | 1.292 | 2.501 | 1.278 | 1.313 | 183.77 | 365.20 |
| conf_adapt_k8_t09 | 4 | conf_adapt_k8_t09_c4 | 0 | 149 | 0.6191 | 1462 | 1316 | 923.59 | 369.74 | 362.26 | 512 | 461 | 359.30 | 125.06 | 0.700 | 2.688 | 0.696 | 0.962 | 125.06 | 369.74 |
| conf_adapt_k8_t09 | 8 | conf_adapt_k8_t09_c8 | 0 | 147 | 0.6133 | 1421 | 1279 | 1008.21 | 378.88 | 367.13 | 512 | 461 | 362.01 | 113.49 | 0.406 | 3.062 | 0.404 | 0.997 | 113.49 | 378.88 |
| conf_adapt_k8_t09 | 16 | conf_adapt_k8_t09_c16 | 0 | 141 | 0.6152 | 1348 | 1214 | 1258.29 | 398.63 | 376.62 | 512 | 461 | 375.64 | 107.36 | 0.266 | 3.787 | 0.265 | 1.170 | 107.36 | 398.63 |
| eagle3 | 1 | eagle3_c1 | 0 | 566 | 0.6152 | 1531 | 1378 | 406.89 | 298.62 | 298.62 | 512 | 461 | 273.13 | 273.47 | 2.070 | 2.070 | 2.071 | 2.078 | 273.47 | 298.62 |
| eagle3 | 2 | eagle3_c2 | 0 | 125 | 0.6211 | 790 | 711 | 768.39 | 573.25 | 305.44 | 512 | 461 | 560.78 | 282.53 | 2.027 | 2.112 | 2.010 | 2.019 | 282.53 | 573.25 |
| eagle3 | 4 | eagle3_c4 | 0 | 84 | 0.6230 | 393 | 354 | 1427.79 | 993.80 | 262.85 | 512 | 461 | 970.01 | 244.63 | 1.882 | 1.951 | 1.879 | 1.882 | 244.63 | 993.80 |
| eagle3 | 8 | eagle3_c8 | 0 | 63 | 0.6230 | 197 | 178 | 2339.90 | 1548.17 | 202.09 | 512 | 461 | 1494.99 | 190.10 | 1.657 | 1.686 | 1.670 | 1.670 | 190.10 | 1548.17 |
| eagle3 | 16 | eagle3_c16 | 0 | 55 | 0.6074 | 100 | 90 | 3422.32 | 2077.61 | 138.26 | 512 | 461 | 2033.23 | 131.02 | 1.388 | 1.390 | 1.436 | 1.428 | 131.02 | 2077.61 |
