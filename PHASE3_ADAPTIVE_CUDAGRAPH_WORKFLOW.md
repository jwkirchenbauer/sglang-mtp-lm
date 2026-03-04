# Phase 3 Adaptive CUDA Graph Workflow (`hf_exact`, canonical q banding, `k=3` comparator)

This workflow implements the Phase 3 matrix for:
- static comparator: `mtp_k=3`
- adaptive target: `mtp_k=8`, `mtp_strategy=["conf_adapt", 0.9]`, `mtp_adaptive_window_mode=hf_exact`
- adaptive canonical banding target: `canonical_q_lens={11,15}` by default (configurable)

It follows the interactive environment/allocation mechanics in `DEBUG_ENV_WORKFLOW.md`.

## 0) Scope and pass criteria

Use this runbook to produce:
1. Static correctness/perf pair: eager (`k=3`) vs graph (`k=3`).
2. Adaptive correctness/perf pair: eager (`k_max=8, conf=0.9`) vs graph (`k_max=8, conf=0.9`, `hf_exact`).
3. Performance matrix at concurrencies `8, 32, 128`:
   - `static_k3_graph`
   - `adaptive_eager`
   - `adaptive_graph`

Required correctness gates:
- no scheduler exceptions
- no runtime orphan/overlap/repair signatures
- parity checks pass in `lm_eval` acceptance comparisons

## 1) Allocate and enter node (interactive only)

```bash
bash -li
sslm_sgl_env
salloc --partition=debug --job-name=phase3_adapt_graph -t29 -N1 --ntasks-per-node=1 --gpus-per-node=4 -c18
srun --jobid "$SLURM_JOB_ID" --nodes=1 --ntasks=1 --gpus-per-node=4 --pty bash -li
sslm_sgl_env
```

Optional smoke:

```bash
hostname
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
nvidia-smi -L
```

## 2) Common environment and helpers

```bash
cd /capstor/scratch/cscs/jkirchen/sglang-mtp-lm

export SGLANG_MTP_KV_LEAK_DEBUG=true
export MODEL_PATH=/capstor/scratch/cscs/jkirchen/singleshot-root/singleshot/outputs/daint_prod_ift_mask_fix/daint_prod_ift_mask_fix_1N4n_9d30cad5/step-00100160
export BASE_PORT=30000
export CAND_PORT=30001
export ADAPT_CANON_Q_LENS="11 15"
export ADAPT_CUDA_GRAPH_MAX_BS=32
export RUN_ID=phase3_adapt_graph_$(date +%Y%m%d_%H%M%S)
export RUN_ROOT=/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/$RUN_ID
mkdir -p "$RUN_ROOT"
echo RUN_ROOT=$RUN_ROOT
```

```bash
wait_health() {
  local port="$1"
  echo "Waiting for port $port"
  for i in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:${port}/health_generate" >/dev/null 2>&1; then
      echo "port $port ready at ${i}s"
      return 0
    fi
    if [ $((i % 20)) -eq 0 ]; then
      echo "port $port warming (${i}s)"
    fi
    sleep 1
  done
  echo "port $port failed to become healthy"
  return 1
}

start_server() {
  local gpu="$1"
  local port="$2"
  local mode="$3"
  local log_root="$4"
  mkdir -p "$log_root"

  local -a args=(
    --model-path "$MODEL_PATH"
    --dtype bfloat16
    --attention-backend flashinfer
    --disable-overlap-schedule
    --enable-metrics
    --export-metrics-to-file
    --export-metrics-to-file-dir "$log_root/server_metrics"
    --port "$port"
  )

  if [[ "$mode" == "eager" ]]; then
    args+=(--disable-cuda-graph)
  elif [[ "$mode" == "static_graph_k3" ]]; then
    args+=(--enable-mtp-static-q-len-cuda-graph --mtp-static-cuda-graph-k-list 3 --cuda-graph-max-bs 128)
  elif [[ "$mode" == "adaptive_graph_k8" ]]; then
    args+=(
      --enable-mtp-static-q-len-cuda-graph --mtp-static-cuda-graph-k-list 3
      --enable-mtp-adaptive-hf-exact-q-len-cuda-graph --mtp-adaptive-cuda-graph-kmax-list 8
      --enable-mtp-adaptive-hf-exact-canonical-q-banding --mtp-adaptive-canonical-q-lens ${ADAPT_CANON_Q_LENS}
      --cuda-graph-max-bs "${ADAPT_CUDA_GRAPH_MAX_BS:-32}"
    )
  else
    echo "Unknown server mode: $mode"
    return 1
  fi

  CUDA_VISIBLE_DEVICES="$gpu" python -m sglang.launch_server "${args[@]}" > "$log_root/server.log" 2>&1 &
  echo $!
}

stop_servers() {
  if [[ -n "${BASE_PID:-}" ]]; then kill "$BASE_PID" 2>/dev/null || true; wait "$BASE_PID" 2>/dev/null || true; fi
  if [[ -n "${CAND_PID:-}" ]]; then kill "$CAND_PID" 2>/dev/null || true; wait "$CAND_PID" 2>/dev/null || true; fi
  ps -fu "$USER" | grep 'sglang.launch_server' | grep -v grep || true
}
```

## 3) Static pair (`k=3`): eager vs graph

### 3.1 Start servers

```bash
export STATIC_ROOT="$RUN_ROOT/static_k3"
mkdir -p "$STATIC_ROOT"/{baseline,candidate}

BASE_PID=$(start_server 0 "$BASE_PORT" eager "$STATIC_ROOT/baseline")
CAND_PID=$(start_server 1 "$CAND_PORT" static_graph_k3 "$STATIC_ROOT/candidate")
wait_health "$BASE_PORT"
wait_health "$CAND_PORT"
```

### 3.2 q>1 graph smoke for static + adaptive

```bash
python3 scripts/playground/mtp_cudagraph_ab.py \
  --baseline-url "http://127.0.0.1:${BASE_PORT}" \
  --candidate-url "http://127.0.0.1:${CAND_PORT}" \
  --k-values 1 2 3 \
  --include-conf-adapt \
  --conf-adapt-k 8 \
  --conf-adapt-threshold 0.9 \
  --strict \
  --output-json "$STATIC_ROOT/cudagraph_ab.json" \
  > "$STATIC_ROOT/cudagraph_ab.stdout" 2>&1
```

### 3.3 Quick matrix (single + concurrent=8)

```bash
python scripts/playground/mtp_quick_driver.py \
  --url "http://127.0.0.1:${BASE_PORT}" \
  --num-prompts 32 --num-concurrent 1 --timeout-s 180 \
  --mtp-k 3 --mtp-strategy-kind static \
  --output-dir "$STATIC_ROOT/baseline_single" \
  > "$STATIC_ROOT/baseline_single.stdout" 2>&1

python scripts/playground/mtp_quick_driver.py \
  --url "http://127.0.0.1:${CAND_PORT}" \
  --num-prompts 32 --num-concurrent 1 --timeout-s 180 \
  --mtp-k 3 --mtp-strategy-kind static \
  --output-dir "$STATIC_ROOT/candidate_single" \
  > "$STATIC_ROOT/candidate_single.stdout" 2>&1

python scripts/playground/mtp_quick_driver.py \
  --url "http://127.0.0.1:${BASE_PORT}" \
  --num-prompts 32 --num-concurrent 8 --timeout-s 180 \
  --mtp-k 3 --mtp-strategy-kind static \
  --output-dir "$STATIC_ROOT/baseline_concurrent8" \
  > "$STATIC_ROOT/baseline_concurrent8.stdout" 2>&1

python scripts/playground/mtp_quick_driver.py \
  --url "http://127.0.0.1:${CAND_PORT}" \
  --num-prompts 32 --num-concurrent 8 --timeout-s 180 \
  --mtp-k 3 --mtp-strategy-kind static \
  --output-dir "$STATIC_ROOT/candidate_concurrent8" \
  > "$STATIC_ROOT/candidate_concurrent8.stdout" 2>&1
```

### 3.4 `lm_eval` matrix (single + concurrent=8)

```bash
export STATIC_LMEVAL_ROOT="$STATIC_ROOT/lmeval_4way"
mkdir -p "$STATIC_LMEVAL_ROOT"/{baseline_single,candidate_single,baseline_concurrent8,candidate_concurrent8}

CUDA_VISIBLE_DEVICES=2 lm_eval --model sglang-generate --device cuda \
  --model_args pretrained=$MODEL_PATH,base_url=http://127.0.0.1:${BASE_PORT}/generate,dtype=bfloat16,attention_backend=flashinfer,enable_metrics=true,export_metrics_to_file=true,export_metrics_to_file_dir=$STATIC_LMEVAL_ROOT/baseline_single \
  --gen_kwargs "temperature=0,top_k=1,mtp_enabled=true,mtp_k=3,mtp_mask_id=128259" \
  --tasks gsm8k_cot_singleshot --limit 32 \
  --output_path "$STATIC_LMEVAL_ROOT/baseline_single" \
  --log_samples --apply_chat_template --fewshot_as_multiturn \
  > "$STATIC_LMEVAL_ROOT/baseline_single/lm_eval.log" 2>&1

CUDA_VISIBLE_DEVICES=2 lm_eval --model sglang-generate --device cuda \
  --model_args pretrained=$MODEL_PATH,base_url=http://127.0.0.1:${CAND_PORT}/generate,dtype=bfloat16,attention_backend=flashinfer,enable_metrics=true,export_metrics_to_file=true,export_metrics_to_file_dir=$STATIC_LMEVAL_ROOT/candidate_single \
  --gen_kwargs "temperature=0,top_k=1,mtp_enabled=true,mtp_k=3,mtp_mask_id=128259" \
  --tasks gsm8k_cot_singleshot --limit 32 \
  --output_path "$STATIC_LMEVAL_ROOT/candidate_single" \
  --log_samples --apply_chat_template --fewshot_as_multiturn \
  > "$STATIC_LMEVAL_ROOT/candidate_single/lm_eval.log" 2>&1

CUDA_VISIBLE_DEVICES=2 lm_eval --model sglang-generate --device cuda \
  --model_args pretrained=$MODEL_PATH,base_url=http://127.0.0.1:${BASE_PORT}/generate,num_concurrent=8,dtype=bfloat16,attention_backend=flashinfer,enable_metrics=true,export_metrics_to_file=true,export_metrics_to_file_dir=$STATIC_LMEVAL_ROOT/baseline_concurrent8 \
  --gen_kwargs "temperature=0,top_k=1,mtp_enabled=true,mtp_k=3,mtp_mask_id=128259" \
  --tasks gsm8k_cot_singleshot --limit 32 \
  --output_path "$STATIC_LMEVAL_ROOT/baseline_concurrent8" \
  --log_samples --apply_chat_template --fewshot_as_multiturn \
  > "$STATIC_LMEVAL_ROOT/baseline_concurrent8/lm_eval.log" 2>&1

CUDA_VISIBLE_DEVICES=2 lm_eval --model sglang-generate --device cuda \
  --model_args pretrained=$MODEL_PATH,base_url=http://127.0.0.1:${CAND_PORT}/generate,num_concurrent=8,dtype=bfloat16,attention_backend=flashinfer,enable_metrics=true,export_metrics_to_file=true,export_metrics_to_file_dir=$STATIC_LMEVAL_ROOT/candidate_concurrent8 \
  --gen_kwargs "temperature=0,top_k=1,mtp_enabled=true,mtp_k=3,mtp_mask_id=128259" \
  --tasks gsm8k_cot_singleshot --limit 32 \
  --output_path "$STATIC_LMEVAL_ROOT/candidate_concurrent8" \
  --log_samples --apply_chat_template --fewshot_as_multiturn \
  > "$STATIC_LMEVAL_ROOT/candidate_concurrent8/lm_eval.log" 2>&1
```

### 3.5 Gate and stop

```bash
python scripts/playground/mtp_log_gate.py "$STATIC_ROOT/baseline/server.log" --expected-http-200 64 > "$STATIC_ROOT/baseline/gate.txt"
python scripts/playground/mtp_log_gate.py "$STATIC_ROOT/candidate/server.log" --expected-http-200 64 > "$STATIC_ROOT/candidate/gate.txt"
stop_servers
```

## 4) Adaptive pair (`k_max=8`, `conf=0.9`, `hf_exact`, canonical banding): eager vs graph

### 4.1 Start servers

```bash
export ADAPT_ROOT="$RUN_ROOT/adaptive_conf_k8_t09"
mkdir -p "$ADAPT_ROOT"/{baseline,candidate}

BASE_PID=$(start_server 0 "$BASE_PORT" eager "$ADAPT_ROOT/baseline")
CAND_PID=$(start_server 1 "$CAND_PORT" adaptive_graph_k8 "$ADAPT_ROOT/candidate")
wait_health "$BASE_PORT"
wait_health "$CAND_PORT"
```

### 4.2 Graph smoke with positive adaptive expectation

```bash
python3 scripts/playground/mtp_cudagraph_ab.py \
  --baseline-url "http://127.0.0.1:${BASE_PORT}" \
  --candidate-url "http://127.0.0.1:${CAND_PORT}" \
  --k-values 1 2 3 \
  --include-conf-adapt \
  --conf-adapt-k 8 \
  --conf-adapt-threshold 0.9 \
  --expect-conf-adapt-hf-exact-qgt1-graph \
  --strict \
  --output-json "$ADAPT_ROOT/cudagraph_ab.json" \
  > "$ADAPT_ROOT/cudagraph_ab.stdout" 2>&1
```

### 4.3 Quick matrix (single + concurrent=8)

```bash
python scripts/playground/mtp_quick_driver.py \
  --url "http://127.0.0.1:${BASE_PORT}" \
  --num-prompts 32 --num-concurrent 1 --timeout-s 180 \
  --mtp-k 8 --mtp-strategy-kind conf_adapt --conf-threshold 0.9 --adaptive-window-mode hf_exact \
  --output-dir "$ADAPT_ROOT/baseline_single" \
  > "$ADAPT_ROOT/baseline_single.stdout" 2>&1

python scripts/playground/mtp_quick_driver.py \
  --url "http://127.0.0.1:${CAND_PORT}" \
  --num-prompts 32 --num-concurrent 1 --timeout-s 180 \
  --mtp-k 8 --mtp-strategy-kind conf_adapt --conf-threshold 0.9 --adaptive-window-mode hf_exact \
  --output-dir "$ADAPT_ROOT/candidate_single" \
  > "$ADAPT_ROOT/candidate_single.stdout" 2>&1

python scripts/playground/mtp_quick_driver.py \
  --url "http://127.0.0.1:${BASE_PORT}" \
  --num-prompts 32 --num-concurrent 8 --timeout-s 180 \
  --mtp-k 8 --mtp-strategy-kind conf_adapt --conf-threshold 0.9 --adaptive-window-mode hf_exact \
  --output-dir "$ADAPT_ROOT/baseline_concurrent8" \
  > "$ADAPT_ROOT/baseline_concurrent8.stdout" 2>&1

python scripts/playground/mtp_quick_driver.py \
  --url "http://127.0.0.1:${CAND_PORT}" \
  --num-prompts 32 --num-concurrent 8 --timeout-s 180 \
  --mtp-k 8 --mtp-strategy-kind conf_adapt --conf-threshold 0.9 --adaptive-window-mode hf_exact \
  --output-dir "$ADAPT_ROOT/candidate_concurrent8" \
  > "$ADAPT_ROOT/candidate_concurrent8.stdout" 2>&1
```

### 4.4 `lm_eval` matrix (single + concurrent=8)

```bash
export ADAPT_LMEVAL_ROOT="$ADAPT_ROOT/lmeval_4way"
mkdir -p "$ADAPT_LMEVAL_ROOT"/{baseline_single,candidate_single,baseline_concurrent8,candidate_concurrent8}

CUDA_VISIBLE_DEVICES=2 lm_eval --model sglang-generate --device cuda \
  --model_args pretrained=$MODEL_PATH,base_url=http://127.0.0.1:${BASE_PORT}/generate,dtype=bfloat16,attention_backend=flashinfer,enable_metrics=true,export_metrics_to_file=true,export_metrics_to_file_dir=$ADAPT_LMEVAL_ROOT/baseline_single \
  --gen_kwargs "temperature=0,top_k=1,mtp_enabled=true,mtp_k=8,mtp_strategy=conf_adapt+0.9,mtp_adaptive_window_mode=hf_exact,mtp_mask_id=128259" \
  --tasks gsm8k_cot_singleshot --limit 32 \
  --output_path "$ADAPT_LMEVAL_ROOT/baseline_single" \
  --log_samples --apply_chat_template --fewshot_as_multiturn \
  > "$ADAPT_LMEVAL_ROOT/baseline_single/lm_eval.log" 2>&1

CUDA_VISIBLE_DEVICES=2 lm_eval --model sglang-generate --device cuda \
  --model_args pretrained=$MODEL_PATH,base_url=http://127.0.0.1:${CAND_PORT}/generate,dtype=bfloat16,attention_backend=flashinfer,enable_metrics=true,export_metrics_to_file=true,export_metrics_to_file_dir=$ADAPT_LMEVAL_ROOT/candidate_single \
  --gen_kwargs "temperature=0,top_k=1,mtp_enabled=true,mtp_k=8,mtp_strategy=conf_adapt+0.9,mtp_adaptive_window_mode=hf_exact,mtp_mask_id=128259" \
  --tasks gsm8k_cot_singleshot --limit 32 \
  --output_path "$ADAPT_LMEVAL_ROOT/candidate_single" \
  --log_samples --apply_chat_template --fewshot_as_multiturn \
  > "$ADAPT_LMEVAL_ROOT/candidate_single/lm_eval.log" 2>&1

CUDA_VISIBLE_DEVICES=2 lm_eval --model sglang-generate --device cuda \
  --model_args pretrained=$MODEL_PATH,base_url=http://127.0.0.1:${BASE_PORT}/generate,num_concurrent=8,dtype=bfloat16,attention_backend=flashinfer,enable_metrics=true,export_metrics_to_file=true,export_metrics_to_file_dir=$ADAPT_LMEVAL_ROOT/baseline_concurrent8 \
  --gen_kwargs "temperature=0,top_k=1,mtp_enabled=true,mtp_k=8,mtp_strategy=conf_adapt+0.9,mtp_adaptive_window_mode=hf_exact,mtp_mask_id=128259" \
  --tasks gsm8k_cot_singleshot --limit 32 \
  --output_path "$ADAPT_LMEVAL_ROOT/baseline_concurrent8" \
  --log_samples --apply_chat_template --fewshot_as_multiturn \
  > "$ADAPT_LMEVAL_ROOT/baseline_concurrent8/lm_eval.log" 2>&1

CUDA_VISIBLE_DEVICES=2 lm_eval --model sglang-generate --device cuda \
  --model_args pretrained=$MODEL_PATH,base_url=http://127.0.0.1:${CAND_PORT}/generate,num_concurrent=8,dtype=bfloat16,attention_backend=flashinfer,enable_metrics=true,export_metrics_to_file=true,export_metrics_to_file_dir=$ADAPT_LMEVAL_ROOT/candidate_concurrent8 \
  --gen_kwargs "temperature=0,top_k=1,mtp_enabled=true,mtp_k=8,mtp_strategy=conf_adapt+0.9,mtp_adaptive_window_mode=hf_exact,mtp_mask_id=128259" \
  --tasks gsm8k_cot_singleshot --limit 32 \
  --output_path "$ADAPT_LMEVAL_ROOT/candidate_concurrent8" \
  --log_samples --apply_chat_template --fewshot_as_multiturn \
  > "$ADAPT_LMEVAL_ROOT/candidate_concurrent8/lm_eval.log" 2>&1
```

### 4.5 Gate and stop

```bash
python scripts/playground/mtp_log_gate.py "$ADAPT_ROOT/baseline/server.log" --expected-http-200 64 > "$ADAPT_ROOT/baseline/gate.txt"
python scripts/playground/mtp_log_gate.py "$ADAPT_ROOT/candidate/server.log" --expected-http-200 64 > "$ADAPT_ROOT/candidate/gate.txt"
stop_servers
```

## 5) Performance matrix (`8, 32, 128`)

Run each configuration separately and compare `summary.json` fields:
- `completion_tokens_per_s`
- `p50_latency_s`
- `p95_latency_s`
- `mean_latency_s`

Example (`num_prompts=96`, same prompt set each run):

```bash
for conc in 8 32 128; do
  python scripts/playground/mtp_quick_driver.py \
    --url "http://127.0.0.1:${CAND_PORT}" \
    --num-prompts 96 --num-concurrent "$conc" --timeout-s 300 \
    --mtp-k 3 --mtp-strategy-kind static \
    --output-dir "$RUN_ROOT/perf/static_k3_graph_conc${conc}" \
    > "$RUN_ROOT/perf/static_k3_graph_conc${conc}.stdout" 2>&1
done
```

Repeat for:
- adaptive eager (`--mtp-k 8 --mtp-strategy-kind conf_adapt --conf-threshold 0.9 --adaptive-window-mode hf_exact`) on eager server
- adaptive graph (same decode params) on adaptive graph server with canonical q banding (`$ADAPT_CANON_Q_LENS`)

## 6) Expected artifact layout

All files remain under:

```text
outputs/<run_id>/
  static_k3/
  adaptive_conf_k8_t09/
  perf/
```

Each scenario contains:
- server logs
- gate outputs
- quick-driver `summary.json` and response jsonl
- `lm_eval` logs/results/samples
- optional `mtp_cudagraph_ab` JSON report

## 7) Cleanup

```bash
stop_servers
exit   # leave compute-node shell
exit   # release allocation shell
exit   # leave login/env shell
```
