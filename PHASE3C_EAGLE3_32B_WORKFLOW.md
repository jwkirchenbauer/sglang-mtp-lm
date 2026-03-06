# Phase 3C Workflow: Standalone EAGLE3 + 32B TP4 MTP Viability

This workflow executes two separate workstreams:
1. Standalone EAGLE3 comparison (no MTP request fields).
2. 32B TP4 MTP server/client viability mini matrix.

For shell/allocation discipline, follow `DEBUG_ENV_WORKFLOW.md`.

## 0) Rules

1. Work from interactive shells only.
2. Run `sslm_sgl_env` as the final activation step in each working shell.
3. Use one allocation at a time.
4. Keep one shell for server and separate shell for clients.
5. Poll `/health_generate` before starting eval clients.
6. Keep artifacts under `outputs/<run_id>/...`.

## 0.1) Critical EAGLE3 guardrails (do not skip)

These items were repeatedly missed and caused avoidable delays:
1. **Set this env var in every EAGLE3 shell** (server and client):  
   `export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`
2. **Run EAGLE3 as a standalone server path**, not through MTP request semantics.
3. **Use the validated server shape below** (`speculative_eagle_topk=4`, `speculative_num_steps=3`, CUDA graphs enabled).
4. If startup fails during draft CUDA graph init with a flashinfer batch-shape mismatch,
   ensure the repository includes the fix in
   `python/sglang/srt/layers/attention/flashinfer_backend.py` that uses
   `effective_bs = bs * topk` for draft capture/replay metadata planning.

For low-concurrency sweeps (e.g. `c <= 16`), apply these additional constraints:
1. Keep `--cuda-graph-max-bs` at or below the max planned concurrency for that sweep (for `c={1,2,4,8,16}`, use `--cuda-graph-max-bs 16`).
2. Do **not** set `--max-running-requests` unless the experiment explicitly requires a server-side cap.

Recommended preflight before any EAGLE3 launch:
```bash
export MAIN_REPO=/capstor/scratch/cscs/jkirchen/sglang-mtp-lm
cd "$MAIN_REPO"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
echo "$SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"  # must print 1

# verify Python resolves sglang from the active repo install
python - <<'PY'
import importlib.util
print(importlib.util.find_spec("sglang").origin)
PY
# expected prefix: /capstor/scratch/cscs/jkirchen/sglang-mtp-lm/python/...
```

---

## 1) Allocation + node shell

```bash
bash -li
salloc --partition=normal --job-name=phase3c_eagle3_32b -t60 -N1 --ntasks-per-node=1 --gpus-per-node=4 -c18
srun --jobid "$SLURM_JOB_ID" --nodes=1 --ntasks=1 --gpus-per-node=4 --pty bash -li
hostname
sslm_sgl_env
cd /capstor/scratch/cscs/jkirchen/sglang-mtp-lm
```

---

## 2) Workstream A: Standalone EAGLE3 comparison

### 2.1 Setup

```bash
export MAIN_REPO=/capstor/scratch/cscs/jkirchen/sglang-mtp-lm
cd "$MAIN_REPO"

export EAGLE3_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
export EAGLE3_DRAFT=jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B

export RUN_ID=phase3c_eagle3_$(date +%Y%m%d_%H%M%S)
export RUN_ROOT=$MAIN_REPO/outputs/$RUN_ID
mkdir -p "$RUN_ROOT"
echo "RUN_ROOT=$RUN_ROOT"
```

### 2.2 Start EAGLE3 server (server shell)

```bash
bash -li
srun --jobid "$SLURM_JOB_ID" --nodes=1 --ntasks=1 --gpus-per-node=4 --pty bash -li
sslm_sgl_env
export MAIN_REPO=/capstor/scratch/cscs/jkirchen/sglang-mtp-lm
cd "$MAIN_REPO"

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path "$EAGLE3_MODEL" \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path "$EAGLE3_DRAFT" \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 4 \
  --speculative-num-draft-tokens 16 \
  --mem-fraction-static 0.7 \
  --cuda-graph-max-bs 128 \
  --dtype bfloat16 \
  --attention-backend flashinfer \
  --enable-metrics \
  --export-metrics-to-file \
  --export-metrics-to-file-dir "$RUN_ROOT/server_eagle3_metrics" \
  --port 30140 \
  > "$RUN_ROOT/server_eagle3.log" 2>&1
```

### 2.3 Run EAGLE3 concurrency sweep (client shell)

```bash
bash -li
srun --jobid "$SLURM_JOB_ID" --nodes=1 --ntasks=1 --gpus-per-node=4 --pty bash -li
sslm_sgl_env
export MAIN_REPO=/capstor/scratch/cscs/jkirchen/sglang-mtp-lm
cd "$MAIN_REPO"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# health poll
for i in $(seq 1 180); do
  curl -fsS http://127.0.0.1:30140/health_generate >/dev/null 2>&1 && break
  sleep 1
done

python scripts/playground/eagle3_lmeval_concurrency_runner.py \
  --model-path "$EAGLE3_MODEL" \
  --base-url "http://127.0.0.1:30140" \
  --output-root "$RUN_ROOT/lmeval_eagle3" \
  --task gsm8k_cot_singleshot \
  --limit 256 \
  --concurrency 1 32 64 128 \
  --gen-kwargs "temperature=0,top_k=1"
```

### 2.4 Tabulate EAGLE3 reduced summary

```bash
python scripts/playground/mtp_lmeval_matrix_tabulate.py \
  --run-times "$RUN_ROOT/lmeval_eagle3/run_times.tsv" \
  --server-log-by-prefix eagle3="$RUN_ROOT/server_eagle3.log" \
  --output-tsv "$RUN_ROOT/eagle3_summary_reduced.tsv" \
  --output-markdown "$RUN_ROOT/eagle3_summary_reduced.md" \
  --sort-mode concurrency
```

### 2.5 Merge with Phase3B reference

```bash
python scripts/playground/eagle3_vs_phase3b_merge.py \
  --eagle3-summary "$RUN_ROOT/eagle3_summary_reduced.tsv" \
  --phase3b-summary "/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/phase3b_lmeval_confadapt_20260305_084500/lmeval_matrix_summary_augmented_reduced.tsv" \
  --output-tsv "$RUN_ROOT/eagle3_vs_phase3b.tsv" \
  --output-markdown "$RUN_ROOT/eagle3_vs_phase3b.md"
```

### 2.6 EAGLE3 fallback (if unstable at c=128)

```bash
# restart server with smaller graph ceiling
# --cuda-graph-max-bs 64
# rerun with --concurrency 1 32 64
```

---

## 3) Workstream B: 32B TP4 MTP viability mini matrix

### 3.1 Setup

```bash
export MODEL_32B=/capstor/scratch/cscs/jkirchen/singleshot-root/singleshot/outputs/daint_prod_ift_q3-32b/daint_prod_ift_q3-32b_16N64n_270dbde6/step-00000050
export MTP_MASK_ID_32B=151669

export RUN_ID=phase3c_q3_32b_tp4_$(date +%Y%m%d_%H%M%S)
export RUN_ROOT=/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/$RUN_ID
mkdir -p "$RUN_ROOT"
echo "RUN_ROOT=$RUN_ROOT"
```

### 3.2 Start TP4 server on all 4 GPUs (server shell)

```bash
bash -li
srun --jobid "$SLURM_JOB_ID" --nodes=1 --ntasks=1 --gpus-per-node=4 --pty bash -li
sslm_sgl_env
cd /capstor/scratch/cscs/jkirchen/sglang-mtp-lm

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
  --model-path "$MODEL_32B" \
  --tp-size 4 \
  --trust-remote-code \
  --dtype bfloat16 \
  --attention-backend flashinfer \
  --disable-overlap-schedule \
  --enable-mtp-static-q-len-cuda-graph \
  --mtp-static-cuda-graph-k-list 1 3 \
  --cuda-graph-max-bs 16 \
  --mem-fraction-static 0.70 \
  --max-running-requests 16 \
  --enable-metrics \
  --export-metrics-to-file \
  --export-metrics-to-file-dir "$RUN_ROOT/server_32b_metrics" \
  --port 30150 \
  > "$RUN_ROOT/server_32b.log" 2>&1
```

### 3.3 Smoke checks at c=1 and c=8 (client shell)

```bash
bash -li
srun --jobid "$SLURM_JOB_ID" --nodes=1 --ntasks=1 --gpus-per-node=4 --pty bash -li
sslm_sgl_env
cd /capstor/scratch/cscs/jkirchen/sglang-mtp-lm

# health poll
for i in $(seq 1 240); do
  curl -fsS http://127.0.0.1:30150/health_generate >/dev/null 2>&1 && break
  sleep 1
done

python scripts/playground/mtp_quick_driver.py \
  --url http://127.0.0.1:30150 \
  --num-prompts 32 \
  --num-concurrent 1 \
  --timeout-s 180 \
  --mtp-k 3 \
  --mtp-strategy-kind static \
  --mtp-mask-id "$MTP_MASK_ID_32B" \
  --output-dir "$RUN_ROOT/smoke_c1_static_k3"

python scripts/playground/mtp_quick_driver.py \
  --url http://127.0.0.1:30150 \
  --num-prompts 32 \
  --num-concurrent 8 \
  --timeout-s 180 \
  --mtp-k 3 \
  --mtp-strategy-kind static \
  --mtp-mask-id "$MTP_MASK_ID_32B" \
  --output-dir "$RUN_ROOT/smoke_c8_static_k3"
```

### 3.4 Mini matrix (`non_mtp`, `static_k1`, `static_k3`, c={1,8})

```bash
python scripts/playground/mtp_lmeval_matrix_runner.py \
  --model-path "$MODEL_32B" \
  --base-url "http://127.0.0.1:30150" \
  --output-root "$RUN_ROOT/lmeval_matrix_32b" \
  --task gsm8k_cot_singleshot \
  --limit 32 \
  --mask-id "$MTP_MASK_ID_32B" \
  --concurrency 1 8 \
  --strategies non_mtp static_k1 static_k3
```

### 3.5 Tabulate reduced summary

```bash
python scripts/playground/mtp_lmeval_matrix_tabulate.py \
  --run-times "$RUN_ROOT/lmeval_matrix_32b/run_times.tsv" \
  --server-log-by-prefix non_mtp="$RUN_ROOT/server_32b.log" \
  --server-log-by-prefix mtp_k1="$RUN_ROOT/server_32b.log" \
  --server-log-by-prefix mtp_k3="$RUN_ROOT/server_32b.log" \
  --output-tsv "$RUN_ROOT/lmeval_matrix_32b_summary_reduced.tsv" \
  --output-markdown "$RUN_ROOT/lmeval_matrix_32b_summary_reduced.md" \
  --sort-mode strategy
```

### 3.6 32B fallback ladder

1. Lower `--cuda-graph-max-bs`: `16 -> 8`.
2. If still unstable, disable CUDA graph flags.
3. Lower `--mem-fraction-static`: `0.70 -> 0.66 -> 0.62`.
4. Re-run smoke checks then mini matrix.

---

## 4) Cleanup

```bash
pkill -f "python -m sglang.launch_server" || true
exit
exit
exit
```
