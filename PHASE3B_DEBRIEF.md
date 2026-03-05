# PHASE 3B Debrief: Full `lm_eval` Matrix + Reproducible Workflow Consolidation

## Summary
Phase 3B completed the requested `lm_eval` strategy/concurrency matrix at:
- `c in {1, 32, 64, 128}`
- strategies in `{non-MTP, k=1, k=2, k=3, conf_adapt_k3_t09, conf_adapt_k8_t09}`

This phase also consolidated the test workflow into reusable scripts and a single workflow document so future runs avoid manual command chaining and shell-state failures.

## Checkpoint Context
1. Repository: `sglang-mtp-lm`
2. Branch: `main`
3. Tooling/workflow commit for this phase: `b7c25c257`
4. Debrief file: `PHASE3B_DEBRIEF.md`

## What Landed
1. Reusable lm_eval matrix runner:
   - `scripts/playground/mtp_lmeval_matrix_runner.py`
2. Reusable reduced-summary tabulator + markdown renderer:
   - `scripts/playground/mtp_lmeval_matrix_tabulate.py`
3. Reusable server launch presets:
   - `scripts/playground/mtp_phase3_server_launch.sh`
4. End-to-end reproducible HPC workflow document:
   - `PHASE3B_LMEVAL_MATRIX_WORKFLOW.md`

All script validation was run in interactive shells with `sslm_sgl_env` as the final activation step.

## Complete Results Table (Strategy x Concurrency)

| strat | c | case | Server Peak Gen TPS | Server Peak vs non-MTP | Flex EM |
|---|---:|---|---:|---:|---:|
| conf_adapt_k3_t09 | 1 | conf_adapt_k3_t09_c1 | 341.14 | 2.321 | 0.6211 |
| conf_adapt_k3_t09 | 32 | conf_adapt_k3_t09_c32 | 912.68 | 0.343 | 0.6289 |
| conf_adapt_k3_t09 | 64 | conf_adapt_k3_t09_c64 | 1417.44 | 0.379 | 0.6211 |
| conf_adapt_k3_t09 | 128 | conf_adapt_k3_t09_c128 | 2422.36 | 0.235 | 0.6211 |
| conf_adapt_k8_t09 | 1 | conf_adapt_k8_t09_c1 | 450.26 | 3.063 | 0.6016 |
| conf_adapt_k8_t09 | 32 | conf_adapt_k8_t09_c32 | 834.45 | 0.314 | 0.6133 |
| conf_adapt_k8_t09 | 64 | conf_adapt_k8_t09_c64 | 962.01 | 0.257 | 0.6055 |
| conf_adapt_k8_t09 | 128 | conf_adapt_k8_t09_c128 | 1550.70 | 0.151 | 0.6055 |
| k=1 | 1 | mtp_k1_c1 | 146.82 | 0.999 | 0.6602 |
| k=1 | 32 | mtp_k1_c32 | 3490.41 | 1.312 | 0.6367 |
| k=1 | 64 | mtp_k1_c64 | 5647.52 | 1.510 | 0.6406 |
| k=1 | 128 | mtp_k1_c128 | 7670.98 | 0.745 | 0.6367 |
| k=2 | 1 | mtp_k2_c1 | 264.58 | 1.800 | 0.5898 |
| k=2 | 32 | mtp_k2_c32 | 4486.91 | 1.686 | 0.5781 |
| k=2 | 64 | mtp_k2_c64 | 6476.65 | 1.732 | 0.5820 |
| k=2 | 128 | mtp_k2_c128 | 7019.17 | 0.681 | 0.5664 |
| k=3 | 1 | mtp_k3_c1 | 388.48 | 2.643 | 0.5273 |
| k=3 | 32 | mtp_k3_c32 | 5375.38 | 2.020 | 0.5273 |
| k=3 | 64 | mtp_k3_c64 | 6607.47 | 1.767 | 0.5195 |
| k=3 | 128 | mtp_k3_c128 | 8103.86 | 0.787 | 0.5234 |
| non-MTP | 1 | non_mtp_c1 | 147.00 | 1.000 | 0.6445 |
| non-MTP | 32 | non_mtp_c32 | 2660.64 | 1.000 | 0.6445 |
| non-MTP | 64 | non_mtp_c64 | 3738.93 | 1.000 | 0.5703 |
| non-MTP | 128 | non_mtp_c128 | 10301.44 | 1.000 | 0.6094 |

## Run Artifacts
1. Baseline matrix root (`non-MTP`, `k=1,2,3`):
   - `/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3b_lmeval_c1_32_64_128_20260305_071400`
2. Conf-adapt matrix root (`k3`, `k8`):
   - `/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3b_lmeval_confadapt_20260305_084500`
3. Final merged reduced summary:
   - `/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3b_lmeval_confadapt_20260305_084500/lmeval_matrix_summary_augmented_reduced.tsv`

## Notes
1. `conf_adapt_k8_t09` runs used `cuda-graph-max-bs=64` and `mem-fraction-static=0.82` for startup/runtime stability.
2. The included workflow/scripts are designed to preserve strict HPC process requirements:
   - interactive shells
   - explicit node shell entry
   - `sslm_sgl_env` as final activation step per working shell
