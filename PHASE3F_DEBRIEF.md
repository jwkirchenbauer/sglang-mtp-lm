# PHASE 3F Debrief: 32B Low-Concurrency Pareto Recreation

> Generated on 2026-03-13 from the Phase3F submitter.

## Summary
1. Recreated the Phase3D low-concurrency analysis shape for the 32B final checkpoint.
2. Used `chat_off` only and removed Eagle3 from the strategy set.
3. Produced the canonical Phase3D-style summary tables plus the final Pareto plot.

## Checkpoint Context
1. Checkpoint: `/capstor/scratch/cscs/jkirchen/singleshot-root/singleshot/outputs/daint_prod_large_models/daint_prod_large_models_16N64n_1f611930/step-00025040`
2. MTP mask id: `151669`
3. Runtime profile: `cg16_mem070` (`cuda_graph_max_bs=16`, `mem_fraction_static=0.70`)
4. Prompt mode: `chat_off`
5. Backend: standalone server + `lm_eval --model sglang-generate`

## Canonical Artifacts
1. Run root: [`/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3f_32b_lowc_robust_20260312_222549`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3f_32b_lowc_robust_20260312_222549)
2. Complete table: [`/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3f_32b_lowc_robust_20260312_222549/summary/lowc_metrics_complete.tsv`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3f_32b_lowc_robust_20260312_222549/summary/lowc_metrics_complete.tsv)
3. Pareto points: [`/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3f_32b_lowc_robust_20260312_222549/summary/pareto_points.tsv`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3f_32b_lowc_robust_20260312_222549/summary/pareto_points.tsv)
4. Pareto plot: [`/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3f_32b_lowc_robust_20260312_222549/summary/lowc_pareto_by_strategy_32b_no_eagle3.png`](/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3f_32b_lowc_robust_20260312_222549/summary/lowc_pareto_by_strategy_32b_no_eagle3.png)

## Notes
1. Strategy coverage matches Phase3D minus Eagle3: `non_mtp`, `static_k{1,2,3,4,5,8,16}`, `conf_adapt_k{3,8,16}_t{06,09}`.
2. Concurrency coverage is `c={1,2,4,8,16}` at `limit=512`.
3. The Phase3D summary and plotting scripts were reused unchanged by keeping the shard layout compatible.

## Final Plotting Snippets

```bash
python scripts/playground/phase3d_pareto_plot_by_strategy.py \
    --input-tsv outputs/phase3d_eagle3_single_20260306_185131/summary/lowc_metrics_complete_with_eagle3.tsv \
    --output-png outputs/phase3d_eagle3_single_20260306_185131/summary/lowc_pareto_by_strategy_8b_with_eagle3_confadapt_and_static_k1-5.png \
    --include-strategy 'k=1' \
    --include-strategy 'k=2' \
    --include-strategy 'k=3' \
    --include-strategy 'k=4' \
    --include-strategy 'k=5' \
    --include-strategy 'conf_adapt_k16_t09' \
    --include-strategy 'conf_adapt_k16_t06' \
    --include-strategy eagle3

python scripts/playground/phase3d_pareto_plot_by_strategy.py \
    --input-tsv outputs/phase3d_eagle3_single_20260306_185131/summary/lowc_metrics_complete_with_eagle3.tsv \
    --output-png outputs/phase3d_eagle3_single_20260306_185131/summary/lowc_pareto_by_strategy_8b_with_eagle3_k1_k2_k3.png \
    --include-strategy 'k=1' \
    --include-strategy 'k=2' \
    --include-strategy 'k=3' \
    --include-strategy eagle3

python scripts/playground/phase3d_pareto_plot_by_strategy.py \
    --input-tsv outputs/phase3f_32b_lowc_robust_20260312_222549/summary/lowc_metrics_complete.tsv \
    --output-png outputs/phase3f_32b_lowc_robust_20260312_222549/summary/lowc_pareto_by_strategy_32b_no_eagle3_confadapt_and_static_k1-5.png \
    --include-strategy 'k=1' \
    --include-strategy 'k=2' \
    --include-strategy 'k=3' \
    --include-strategy 'k=4' \
    --include-strategy 'k=5' \
    --include-strategy 'conf_adapt_k16_t09' \
    --include-strategy 'conf_adapt_k16_t06'

python scripts/playground/phase3d_pareto_plot_by_strategy.py \
    --input-tsv outputs/phase3f_32b_lowc_robust_20260312_222549/summary/lowc_metrics_complete.tsv \
    --output-png outputs/phase3f_32b_lowc_robust_20260312_222549/summary/lowc_pareto_by_strategy_32b_no_eagle3_k1_k2_k3.png \
    --include-strategy 'k=1' \
    --include-strategy 'k=2' \
    --include-strategy 'k=3'
```

**Output artifacts:**

- `outputs/phase3d_eagle3_single_20260306_185131/summary/lowc_pareto_by_strategy_8b_with_eagle3_confadapt_and_static_k1-5.png`
- `outputs/phase3d_eagle3_single_20260306_185131/summary/lowc_pareto_by_strategy_8b_with_eagle3_k1_k2_k3.png`
- `outputs/phase3f_32b_lowc_robust_20260312_222549/summary/lowc_pareto_by_strategy_32b_no_eagle3_confadapt_and_static_k1-5.png`
- `outputs/phase3f_32b_lowc_robust_20260312_222549/summary/lowc_pareto_by_strategy_32b_no_eagle3_k1_k2_k3.png`