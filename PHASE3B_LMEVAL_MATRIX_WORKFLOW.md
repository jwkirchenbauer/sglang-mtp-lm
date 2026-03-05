# Phase 3B lm_eval Matrix Workflow (Repeatable)

This workflow standardizes the full test loop:
1. Launch server presets.
2. Run `lm_eval` strategy/concurrency grids.
3. Tabulate reduced metrics with server peak TPS.

Use this workflow for the matrix:
- `c in {1, 32, 64, 128}`
- strategies in:
  - `non-MTP`
  - static `k=1,2,3`
  - `conf_adapt` (`k_max=3`, `threshold=0.9`)
  - `conf_adapt` (`k_max=8`, `threshold=0.9`)

## 0) Rules (do not skip)

1. Work from interactive shells only.
2. `sslm_sgl_env` must be run as the final activation step in each shell before real work.
3. For runtime testing, use `--partition=normal` with a 1-hour allocation.
4. Keep one shell per long-running server process and separate shell(s) for clients.

## 1) Allocate + enter compute node

```bash
bash -li
salloc --partition=normal --job-name=phase3b_lmeval -t60 -N1 --ntasks-per-node=1 --gpus-per-node=4 -c18
srun --jobid "$SLURM_JOB_ID" --nodes=1 --ntasks=1 --gpus-per-node=4 --pty bash -li
hostname
sslm_sgl_env
```

## 2) Common environment

```bash
cd /capstor/scratch/cscs/jkirchen/sglang-mtp-lm

export MODEL_PATH=/capstor/scratch/cscs/jkirchen/singleshot-root/singleshot/outputs/daint_prod_ift_mask_fix/daint_prod_ift_mask_fix_1N4n_9d30cad5/step-00100160
export RUN_ID=phase3b_lmeval_$(date +%Y%m%d_%H%M%S)
export RUN_ROOT=/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/$RUN_ID
mkdir -p "$RUN_ROOT"

export C_LIST="1 32 64 128"
export C_MAX=128
export LIMIT=$((2 * C_MAX))   # >= 2x top concurrency
echo "RUN_ROOT=$RUN_ROOT"
echo "LIMIT=$LIMIT"
```

## 3) Start servers (separate shells)

Open a new shell per server, then:

```bash
bash -li
srun --jobid "$SLURM_JOB_ID" --nodes=1 --ntasks=1 --gpus-per-node=4 --pty bash -li
sslm_sgl_env
cd /capstor/scratch/cscs/jkirchen/sglang-mtp-lm
```

### 3.1 Base matrix server (`non-MTP`, `k=1,2,3`)

```bash
scripts/playground/mtp_phase3_server_launch.sh \
  --mode base_matrix \
  --model-path "$MODEL_PATH" \
  --port 30130 \
  --gpu 0 \
  --log-file "$RUN_ROOT/server_base.log" \
  --metrics-dir "$RUN_ROOT/server_base_metrics"
```

### 3.2 `conf_adapt` k3 server

```bash
scripts/playground/mtp_phase3_server_launch.sh \
  --mode conf_adapt_k3_t09 \
  --model-path "$MODEL_PATH" \
  --port 30131 \
  --gpu 1 \
  --log-file "$RUN_ROOT/server_conf_k3.log" \
  --metrics-dir "$RUN_ROOT/server_conf_k3_metrics"
```

### 3.3 `conf_adapt` k8 server (lower graph BS default)

```bash
scripts/playground/mtp_phase3_server_launch.sh \
  --mode conf_adapt_k8_t09 \
  --model-path "$MODEL_PATH" \
  --port 30132 \
  --gpu 2 \
  --log-file "$RUN_ROOT/server_conf_k8.log" \
  --metrics-dir "$RUN_ROOT/server_conf_k8_metrics" \
  --mem-fraction-static 0.82
```

If `k8,c128` fails due memory, rerun that matrix with `--concurrency 1 32 64` and omit `c=128`.

## 4) Run lm_eval matrices (client shell)

In a client shell on the same node:

```bash
bash -li
srun --jobid "$SLURM_JOB_ID" --nodes=1 --ntasks=1 --gpus-per-node=4 --pty bash -li
sslm_sgl_env
cd /capstor/scratch/cscs/jkirchen/sglang-mtp-lm
```

### 4.1 Base matrix run

```bash
python scripts/playground/mtp_lmeval_matrix_runner.py \
  --model-path "$MODEL_PATH" \
  --base-url "http://127.0.0.1:30130" \
  --output-root "$RUN_ROOT/lmeval_matrix_base" \
  --task gsm8k_cot_singleshot \
  --limit "$LIMIT" \
  --concurrency $C_LIST \
  --strategies non_mtp static_k1 static_k2 static_k3
```

### 4.2 `conf_adapt` k3 run

```bash
python scripts/playground/mtp_lmeval_matrix_runner.py \
  --model-path "$MODEL_PATH" \
  --base-url "http://127.0.0.1:30131" \
  --output-root "$RUN_ROOT/lmeval_matrix_conf_k3" \
  --task gsm8k_cot_singleshot \
  --limit "$LIMIT" \
  --concurrency $C_LIST \
  --strategies conf_adapt_k3_t09
```

### 4.3 `conf_adapt` k8 run

```bash
python scripts/playground/mtp_lmeval_matrix_runner.py \
  --model-path "$MODEL_PATH" \
  --base-url "http://127.0.0.1:30132" \
  --output-root "$RUN_ROOT/lmeval_matrix_conf_k8" \
  --task gsm8k_cot_singleshot \
  --limit "$LIMIT" \
  --concurrency $C_LIST \
  --strategies conf_adapt_k8_t09
```

## 5) Tabulate reduced summary + markdown

```bash
python scripts/playground/mtp_lmeval_matrix_tabulate.py \
  --run-times "$RUN_ROOT/lmeval_matrix_base/run_times.tsv" \
             "$RUN_ROOT/lmeval_matrix_conf_k3/run_times.tsv" \
             "$RUN_ROOT/lmeval_matrix_conf_k8/run_times.tsv" \
  --server-log-by-prefix non_mtp="$RUN_ROOT/server_base.log" \
  --server-log-by-prefix mtp_k1="$RUN_ROOT/server_base.log" \
  --server-log-by-prefix mtp_k2="$RUN_ROOT/server_base.log" \
  --server-log-by-prefix mtp_k3="$RUN_ROOT/server_base.log" \
  --server-log-by-prefix conf_adapt_k3_t09="$RUN_ROOT/server_conf_k3.log" \
  --server-log-by-prefix conf_adapt_k8_t09="$RUN_ROOT/server_conf_k8.log" \
  --output-tsv "$RUN_ROOT/lmeval_matrix_summary_reduced.tsv" \
  --output-markdown "$RUN_ROOT/lmeval_matrix_summary_reduced.md" \
  --sort-mode strategy
```

Reduced columns:
- `strat`
- `c`
- `case`
- `Server Peak Gen TPS`
- `Server Peak vs non-MTP`
- `Flex EM`

## 6) Parallelization guidance

1. Running 4 servers + 4 clients in parallel speeds iteration but can distort TPS due GPU/host contention.
2. For reliable peak TPS comparisons, prefer one active benchmark stream per server/GPU.
3. If you parallelize, keep it for smoke checks; rerun final reporting sequentially.

## 7) Allocation management + cleanup

1. If a 1-hour allocation is close to expiry, request a second `normal` allocation before the first expires.
2. Stop servers after runs:

```bash
pkill -f "python -m sglang.launch_server" || true
```

3. Exit shells cleanly to release allocations.
