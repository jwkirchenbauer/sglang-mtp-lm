# Phase 2C.2 4-Way Test Workflow (Quick + lm_eval)

This is the reproducible workflow for the Phase 2C.2 adaptive-concurrency closure matrix.

It is intentionally specific to this phase. For generic node/env startup mechanics, use:
`DEBUG_ENV_WORKFLOW.md`.

## Scope

- Run two 4-way matrices on one allocated node:
  - quick driver matrix:
    - baseline single
    - candidate single
    - baseline concurrent=8
    - candidate concurrent=8
  - `lm_eval` matrix:
    - baseline single
    - candidate single
    - baseline concurrent=8
    - candidate concurrent=8
- Write all artifacts only under:
  - `/capstor/scratch/cscs/jkirchen/sglang-mtp-lm/outputs/<run_id>`
- Use model read-only from:
  - `/capstor/scratch/cscs/jkirchen/singleshot-root/singleshot/outputs/daint_prod_ift_mask_fix/daint_prod_ift_mask_fix_1N4n_9d30cad5/step-00100160`

## 1) Allocate and enter node

```bash
bash -li
salloc --partition=debug --job-name=phase2c2_matrix -t29 -N1 --ntasks-per-node=1 --gpus-per-node=4 -c18
srun --jobid "$SLURM_JOB_ID" --nodes=1 --ntasks=1 --gpus-per-node=4 --pty bash -li
hostname
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
nvidia-smi -L
sslm_sgl_env
```

## 2) Common environment and helpers

```bash
cd /capstor/scratch/cscs/jkirchen/sglang-mtp-lm

export SGLANG_MTP_KV_LEAK_DEBUG=true
export MODEL_PATH=/capstor/scratch/cscs/jkirchen/singleshot-root/singleshot/outputs/daint_prod_ift_mask_fix/daint_prod_ift_mask_fix_1N4n_9d30cad5/step-00100160
export BASE_PORT=30000
export CAND_PORT=30001
export COMMON_PRETRAINED="$MODEL_PATH"
export RUN_ID=phase2c2_matrix_$(date +%Y%m%d_%H%M%S)
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

start_servers() {
  local root="$1"
  mkdir -p "$root"/baseline "$root"/candidate

  CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --dtype bfloat16 \
    --attention-backend flashinfer \
    --disable-overlap-schedule \
    --disable-cuda-graph \
    --enable-metrics \
    --export-metrics-to-file \
    --export-metrics-to-file-dir "$root/baseline/server_metrics" \
    --port "$BASE_PORT" \
    > "$root/baseline/server.log" 2>&1 &
  BASE_PID=$!

  CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --dtype bfloat16 \
    --attention-backend flashinfer \
    --disable-overlap-schedule \
    --enable-mtp-static-q-len-cuda-graph \
    --mtp-static-cuda-graph-k-list 1 2 3 4 \
    --cuda-graph-max-bs 128 \
    --enable-metrics \
    --export-metrics-to-file \
    --export-metrics-to-file-dir "$root/candidate/server_metrics" \
    --port "$CAND_PORT" \
    > "$root/candidate/server.log" 2>&1 &
  CAND_PID=$!

  export BASE_PID CAND_PID
  echo "BASE_PID=$BASE_PID CAND_PID=$CAND_PID"

  wait_health "$BASE_PORT"
  wait_health "$CAND_PORT"
}

stop_servers() {
  kill "$BASE_PID" "$CAND_PID" 2>/dev/null || true
  wait "$BASE_PID" 2>/dev/null || true
  wait "$CAND_PID" 2>/dev/null || true
  ps -fu "$USER" | grep 'sglang.launch_server' | grep -v grep || true
}
```

Notes:
- First server startup in a fresh allocation is usually slow. This is expected.
- If debug allocation is near 29 minutes, stop and restart from a fresh allocation before `lm_eval`.

## 3) Quick driver 4-way matrix

```bash
export QUICK_ROOT="$RUN_ROOT/quick_4way"
start_servers "$QUICK_ROOT"

python scripts/playground/mtp_quick_driver.py \
  --url http://127.0.0.1:${BASE_PORT} \
  --num-prompts 32 \
  --num-concurrent 1 \
  --timeout-s 180 \
  --mtp-k 8 \
  --conf-threshold 0.9 \
  --adaptive-window-mode hf_exact \
  --output-dir "$QUICK_ROOT/baseline_single" \
  > "$QUICK_ROOT/baseline_single.stdout" 2>&1
BASE_SINGLE_EXIT=$?

python scripts/playground/mtp_quick_driver.py \
  --url http://127.0.0.1:${CAND_PORT} \
  --num-prompts 32 \
  --num-concurrent 1 \
  --timeout-s 180 \
  --mtp-k 8 \
  --conf-threshold 0.9 \
  --adaptive-window-mode hf_exact \
  --output-dir "$QUICK_ROOT/candidate_single" \
  > "$QUICK_ROOT/candidate_single.stdout" 2>&1
CAND_SINGLE_EXIT=$?

python scripts/playground/mtp_quick_driver.py \
  --url http://127.0.0.1:${BASE_PORT} \
  --num-prompts 32 \
  --num-concurrent 8 \
  --timeout-s 180 \
  --mtp-k 8 \
  --conf-threshold 0.9 \
  --adaptive-window-mode hf_exact \
  --output-dir "$QUICK_ROOT/baseline_concurrent8" \
  > "$QUICK_ROOT/baseline_concurrent8.stdout" 2>&1
BASE_CONC_EXIT=$?

python scripts/playground/mtp_quick_driver.py \
  --url http://127.0.0.1:${CAND_PORT} \
  --num-prompts 32 \
  --num-concurrent 8 \
  --timeout-s 180 \
  --mtp-k 8 \
  --conf-threshold 0.9 \
  --adaptive-window-mode hf_exact \
  --output-dir "$QUICK_ROOT/candidate_concurrent8" \
  > "$QUICK_ROOT/candidate_concurrent8.stdout" 2>&1
CAND_CONC_EXIT=$?

echo BASE_SINGLE_EXIT=$BASE_SINGLE_EXIT CAND_SINGLE_EXIT=$CAND_SINGLE_EXIT BASE_CONC_EXIT=$BASE_CONC_EXIT CAND_CONC_EXIT=$CAND_CONC_EXIT
```

### Quick validation

```bash
python scripts/playground/mtp_log_gate.py "$QUICK_ROOT/baseline/server.log" --expected-http-200 64 > "$QUICK_ROOT/baseline/gate.txt"
python scripts/playground/mtp_log_gate.py "$QUICK_ROOT/candidate/server.log" --expected-http-200 64 > "$QUICK_ROOT/candidate/gate.txt"

python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["QUICK_ROOT"])
def load(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
b = load(root / "baseline_concurrent8" / "single_responses.jsonl")
c = load(root / "candidate_concurrent8" / "single_responses.jsonl")
mism = [i for i,(x,y) in enumerate(zip(b,c)) if x.get("text","") != y.get("text","")]
print(f"quick_concurrent8_parity_mismatches={len(mism)}")
if mism:
    print(f"quick_first_mismatch_idx={mism[0]}")
PY
```

Interpretation:
- Required pass signal for this phase:
  - all 4 exits are `0`
  - both `gate.txt` reports are PASS
- Quick text parity can be noisy for open-ended prompt sets; use `lm_eval` parity + metrics as the authoritative correctness gate.

```bash
stop_servers
```

## 4) lm_eval 4-way matrix

```bash
export LMEVAL_ROOT="$RUN_ROOT/lmeval_4way"
start_servers "$LMEVAL_ROOT"

mkdir -p "$LMEVAL_ROOT"/{baseline_single,candidate_single,baseline_concurrent8,candidate_concurrent8}

CUDA_VISIBLE_DEVICES=2 lm_eval --model sglang-generate \
  --device cuda \
  --model_args pretrained=$COMMON_PRETRAINED,base_url=http://127.0.0.1:${BASE_PORT}/generate,dtype=bfloat16,attention_backend=flashinfer,enable_metrics=true,export_metrics_to_file=true,export_metrics_to_file_dir=$LMEVAL_ROOT/baseline_single \
  --gen_kwargs "temperature=0,top_k=1,mtp_enabled=true,mtp_k=8,mtp_strategy=conf_adapt+0.9,mtp_adaptive_window_mode=hf_exact,mtp_mask_id=128259" \
  --tasks gsm8k_cot_singleshot \
  --limit 32 \
  --output_path "$LMEVAL_ROOT/baseline_single" \
  --log_samples --apply_chat_template --fewshot_as_multiturn \
  > "$LMEVAL_ROOT/baseline_single/lm_eval.log" 2>&1
BASE_SINGLE_EVAL_EXIT=$?

CUDA_VISIBLE_DEVICES=2 lm_eval --model sglang-generate \
  --device cuda \
  --model_args pretrained=$COMMON_PRETRAINED,base_url=http://127.0.0.1:${CAND_PORT}/generate,dtype=bfloat16,attention_backend=flashinfer,enable_metrics=true,export_metrics_to_file=true,export_metrics_to_file_dir=$LMEVAL_ROOT/candidate_single \
  --gen_kwargs "temperature=0,top_k=1,mtp_enabled=true,mtp_k=8,mtp_strategy=conf_adapt+0.9,mtp_adaptive_window_mode=hf_exact,mtp_mask_id=128259" \
  --tasks gsm8k_cot_singleshot \
  --limit 32 \
  --output_path "$LMEVAL_ROOT/candidate_single" \
  --log_samples --apply_chat_template --fewshot_as_multiturn \
  > "$LMEVAL_ROOT/candidate_single/lm_eval.log" 2>&1
CAND_SINGLE_EVAL_EXIT=$?

CUDA_VISIBLE_DEVICES=2 lm_eval --model sglang-generate \
  --device cuda \
  --model_args pretrained=$COMMON_PRETRAINED,base_url=http://127.0.0.1:${BASE_PORT}/generate,num_concurrent=8,dtype=bfloat16,attention_backend=flashinfer,enable_metrics=true,export_metrics_to_file=true,export_metrics_to_file_dir=$LMEVAL_ROOT/baseline_concurrent8 \
  --gen_kwargs "temperature=0,top_k=1,mtp_enabled=true,mtp_k=8,mtp_strategy=conf_adapt+0.9,mtp_adaptive_window_mode=hf_exact,mtp_mask_id=128259" \
  --tasks gsm8k_cot_singleshot \
  --limit 32 \
  --output_path "$LMEVAL_ROOT/baseline_concurrent8" \
  --log_samples --apply_chat_template --fewshot_as_multiturn \
  > "$LMEVAL_ROOT/baseline_concurrent8/lm_eval.log" 2>&1
BASE_CONC_EVAL_EXIT=$?

CUDA_VISIBLE_DEVICES=2 lm_eval --model sglang-generate \
  --device cuda \
  --model_args pretrained=$COMMON_PRETRAINED,base_url=http://127.0.0.1:${CAND_PORT}/generate,num_concurrent=8,dtype=bfloat16,attention_backend=flashinfer,enable_metrics=true,export_metrics_to_file=true,export_metrics_to_file_dir=$LMEVAL_ROOT/candidate_concurrent8 \
  --gen_kwargs "temperature=0,top_k=1,mtp_enabled=true,mtp_k=8,mtp_strategy=conf_adapt+0.9,mtp_adaptive_window_mode=hf_exact,mtp_mask_id=128259" \
  --tasks gsm8k_cot_singleshot \
  --limit 32 \
  --output_path "$LMEVAL_ROOT/candidate_concurrent8" \
  --log_samples --apply_chat_template --fewshot_as_multiturn \
  > "$LMEVAL_ROOT/candidate_concurrent8/lm_eval.log" 2>&1
CAND_CONC_EVAL_EXIT=$?

echo BASE_SINGLE_EVAL_EXIT=$BASE_SINGLE_EVAL_EXIT CAND_SINGLE_EVAL_EXIT=$CAND_SINGLE_EVAL_EXIT BASE_CONC_EVAL_EXIT=$BASE_CONC_EVAL_EXIT CAND_CONC_EVAL_EXIT=$CAND_CONC_EVAL_EXIT
```

### lm_eval validation

```bash
python scripts/playground/mtp_log_gate.py "$LMEVAL_ROOT/baseline/server.log" --expected-http-200 64 > "$LMEVAL_ROOT/baseline/gate.txt"
python scripts/playground/mtp_log_gate.py "$LMEVAL_ROOT/candidate/server.log" --expected-http-200 64 > "$LMEVAL_ROOT/candidate/gate.txt"

python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["LMEVAL_ROOT"])

def find_sample_file(scenario_dir: Path) -> Path:
    cands = sorted(scenario_dir.glob("**/samples_*.jsonl"))
    if not cands:
        raise RuntimeError(f"no sample jsonl under {scenario_dir}")
    return cands[0]

def find_result_file(scenario_dir: Path) -> Path:
    cands = sorted(scenario_dir.glob("**/results_*.json"))
    if not cands:
        raise RuntimeError(f"no results json under {scenario_dir}")
    return cands[0]

def load_resps(path: Path):
    vals = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        vals.append(row.get("resps", row.get("filtered_resps", row.get("response", row.get("text", "")))))
    return vals

for label, bdir, cdir in [
    ("single", "baseline_single", "candidate_single"),
    ("concurrent8", "baseline_concurrent8", "candidate_concurrent8"),
]:
    b = load_resps(find_sample_file(root / bdir))
    c = load_resps(find_sample_file(root / cdir))
    mism = [i for i,(x,y) in enumerate(zip(b,c)) if x != y]
    print(f"{label}_sample_mismatches={len(mism)}")
    if mism:
        print(f"{label}_first_mismatch_idx={mism[0]}")

for name in ["baseline_single", "candidate_single", "baseline_concurrent8", "candidate_concurrent8"]:
    result = json.loads(find_result_file(root / name).read_text(encoding="utf-8"))
    score = result["results"]["gsm8k_cot_singleshot"]["exact_match,strict-match"]
    print(f"{name}_exact_match_strict={score}")
PY
```

Expected current known-good numbers for this phase:
- `exact_match,strict-match = 0.65625` for all four scenarios.
- `single_sample_mismatches = 0`.
- `concurrent8_sample_mismatches = 0`.

```bash
stop_servers
```

## 5) Cleanup

```bash
exit   # leave sslm_sgl_env shell
exit   # leave compute-node shell
exit   # release allocation shell
```

