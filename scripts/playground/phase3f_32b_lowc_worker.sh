#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run one Phase3F 32B low-concurrency shard on a standalone TP4 server.

Required:
  --strategy <strategy>
  --shard-dir <path>
  --port <int>
  --model-path <path>
  --mask-id <int>

Optional:
  --task <name>                   (default: gsm8k_cot_singleshot)
  --limit <int>                   (default: 512)
  --concurrency <csv_or_space>    (default: 1,2,4,8,16)
  --max-gen-toks <int>            (default: 128)
  --tp-size <int>                 (default: 4)
  --dtype <dtype>                 (default: bfloat16)
  --attention-backend <name>      (default: flashinfer)
  --mem-fraction-static <float>   (default: 0.70)
  --cuda-graph-max-bs <int>       (default: 16)
  --max-running-requests <int>    (default: 16)
  --stop-token-ids <spec>         (default: 151645+151643)
  --visible-devices <csv>         (default: 0,1,2,3)
  --dry-run                       Print server/matrix commands without running them
USAGE
}

STRATEGY=""
SHARD_DIR=""
PORT=""
MODEL_PATH=""
MASK_ID=""
TASK="gsm8k_cot_singleshot"
LIMIT="512"
CONCURRENCY="1,2,4,8,16"
MAX_GEN_TOKS="128"
TP_SIZE="4"
DTYPE="bfloat16"
ATTENTION_BACKEND="flashinfer"
MEM_FRACTION_STATIC="0.70"
CUDA_GRAPH_MAX_BS="16"
MAX_RUNNING_REQUESTS="16"
STOP_TOKEN_IDS="151645+151643"
VISIBLE_DEVICES="0,1,2,3"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --strategy) STRATEGY="$2"; shift 2 ;;
    --shard-dir) SHARD_DIR="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --mask-id) MASK_ID="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --max-gen-toks) MAX_GEN_TOKS="$2"; shift 2 ;;
    --tp-size) TP_SIZE="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --attention-backend) ATTENTION_BACKEND="$2"; shift 2 ;;
    --mem-fraction-static) MEM_FRACTION_STATIC="$2"; shift 2 ;;
    --cuda-graph-max-bs) CUDA_GRAPH_MAX_BS="$2"; shift 2 ;;
    --max-running-requests) MAX_RUNNING_REQUESTS="$2"; shift 2 ;;
    --stop-token-ids) STOP_TOKEN_IDS="$2"; shift 2 ;;
    --visible-devices) VISIBLE_DEVICES="$2"; shift 2 ;;
    --dry-run) DRY_RUN="1"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$STRATEGY" || -z "$SHARD_DIR" || -z "$PORT" || -z "$MODEL_PATH" || -z "$MASK_ID" ]]; then
  echo "Missing required args." >&2
  usage
  exit 2
fi

REPO_ROOT="/capstor/scratch/cscs/jkirchen/sglang-mtp-lm"
HARNESS_ROOT="/capstor/scratch/cscs/jkirchen/lm-evaluation-harness-mtp-lm"
RUNNER_SCRIPT="$REPO_ROOT/scripts/playground/mtp_lmeval_matrix_runner.py"

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "Repo root missing: $REPO_ROOT" >&2
  exit 2
fi
if [[ ! -d "$HARNESS_ROOT" ]]; then
  echo "Harness root missing: $HARNESS_ROOT" >&2
  exit 2
fi
if [[ ! -f "$RUNNER_SCRIPT" ]]; then
  echo "Matrix runner missing: $RUNNER_SCRIPT" >&2
  exit 2
fi
if [[ ! -e "$MODEL_PATH" ]]; then
  echo "Model path missing: $MODEL_PATH" >&2
  exit 2
fi

mkdir -p "$SHARD_DIR"
RUN_OUTPUT_ROOT="$SHARD_DIR/lmeval_matrix"
METRICS_DIR="$SHARD_DIR/metrics"
SERVER_LOG="$SHARD_DIR/server.log"
ATTEMPT_LOG="$SHARD_DIR/attempt_history.log"
STATUS_FILE="$SHARD_DIR/status.txt"
PID_FILE="$SHARD_DIR/server.pid"
mkdir -p "$RUN_OUTPUT_ROOT" "$METRICS_DIR"

printf "running\n" >"$STATUS_FILE"
: >"$ATTEMPT_LOG"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/python:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
REPO_HEAD="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

CONCURRENCY_NORM="${CONCURRENCY//,/ }"
read -r -a CONCURRENCIES <<< "$CONCURRENCY_NORM"
if [[ ${#CONCURRENCIES[@]} -eq 0 ]]; then
  echo "No concurrency values parsed from: $CONCURRENCY" >&2
  printf "failed\n" >"$STATUS_FILE"
  exit 1
fi

SERVER_PID=""
cleanup_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  SERVER_PID=""
}

finalize() {
  local rc=$?
  cleanup_server
  if [[ -f "$STATUS_FILE" ]]; then
    local cur
    cur="$(cat "$STATUS_FILE" 2>/dev/null || true)"
    if [[ "$cur" == "running" ]]; then
      if [[ "$rc" -eq 0 ]]; then
        printf "ok\n" >"$STATUS_FILE"
      else
        printf "failed\n" >"$STATUS_FILE"
      fi
    fi
  fi
  exit "$rc"
}
trap finalize EXIT

wait_for_health() {
  local tries code model_code warmup_sent
  warmup_sent=0
  for tries in $(seq 1 360); do
    if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
      return 1
    fi
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health_generate" || true)
    if [[ "$code" == "200" ]]; then
      return 0
    fi

    if [[ "$code" == "503" && "$warmup_sent" == "0" ]]; then
      model_code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/model_info" || true)
      if [[ "$model_code" == "200" ]]; then
        {
          echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Sending warmup /generate probe to unblock /health_generate"
        } >>"$SERVER_LOG"
        curl -s -o /dev/null \
          -X POST "http://127.0.0.1:${PORT}/generate" \
          -H "Content-Type: application/json" \
          -d '{"text":"Warmup","sampling_params":{"temperature":0,"max_new_tokens":1}}' || true
        warmup_sent=1
      fi
    fi
    sleep 2
  done
  return 1
}

build_server_cmd() {
  local -a cmd mode_args
  mode_args=()
  if [[ "$STRATEGY" == "non_mtp" ]]; then
    mode_args=()
  elif [[ "$STRATEGY" =~ ^static_k([0-9]+)$ ]]; then
    local k="${BASH_REMATCH[1]}"
    mode_args=(
      --enable-mtp-static-q-len-cuda-graph
      --mtp-static-cuda-graph-k-list "$k"
    )
  elif [[ "$STRATEGY" =~ ^conf_adapt_k([0-9]+)_t([0-9.]+)$ ]]; then
    local kmax="${BASH_REMATCH[1]}"
    mode_args=(
      --enable-mtp-static-q-len-cuda-graph
      --mtp-static-cuda-graph-k-list 3
      --enable-mtp-adaptive-hf-exact-q-len-cuda-graph
      --mtp-adaptive-cuda-graph-kmax-list "$kmax"
    )
  else
    echo "Unsupported strategy: $STRATEGY" >&2
    return 1
  fi

  cmd=(
    python -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --port "$PORT"
    --tp-size "$TP_SIZE"
    --trust-remote-code
    --dtype "$DTYPE"
    --attention-backend "$ATTENTION_BACKEND"
    --disable-overlap-schedule
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --decode-log-interval 10
    --enable-metrics
    --export-metrics-to-file
    --export-metrics-to-file-dir "$METRICS_DIR"
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
  )
  cmd+=("${mode_args[@]}")
  printf '%s\0' "${cmd[@]}"
}

build_runner_cmd() {
  local -a cmd
  cmd=(
    python "$RUNNER_SCRIPT"
    --model-path "$MODEL_PATH"
    --base-url "http://127.0.0.1:${PORT}"
    --output-root "$RUN_OUTPUT_ROOT"
    --task "$TASK"
    --limit "$LIMIT"
    --max-gen-toks "$MAX_GEN_TOKS"
    --mask-id "$MASK_ID"
    --adaptive-window-mode hf_exact
    --base-gen-kwargs "temperature=0,top_k=1,stop_token_ids=$STOP_TOKEN_IDS"
    --concurrency "${CONCURRENCIES[@]}"
    --strategies "$STRATEGY"
    --disable-chat-template
    --resume
  )
  printf '%s\0' "${cmd[@]}"
}

start_server() {
  local -a cmd
  mapfile -d '' -t cmd < <(build_server_cmd)

  {
    echo "===== launch ====="
    echo "ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repo_head=$REPO_HEAD"
    echo "strategy=$STRATEGY port=$PORT tp_size=$TP_SIZE"
    echo "concurrency=${CONCURRENCIES[*]}"
    echo "max_running_requests=$MAX_RUNNING_REQUESTS cuda_graph_max_bs=$CUDA_GRAPH_MAX_BS mem_fraction_static=$MEM_FRACTION_STATIC"
    echo "visible_devices=$VISIBLE_DEVICES"
  } >>"$ATTEMPT_LOG"

  {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Launching server: CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES ${cmd[*]}"
  } >>"$SERVER_LOG"
  nohup env CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES" "${cmd[@]}" >>"$SERVER_LOG" 2>&1 &
  SERVER_PID="$!"
  echo "$SERVER_PID" >"$PID_FILE"

  if ! wait_for_health; then
    {
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Health check failed; server pid=${SERVER_PID}"
    } >>"$SERVER_LOG"
    cleanup_server
    return 1
  fi
  return 0
}

run_runner_once() {
  local -a cmd
  mapfile -d '' -t cmd < <(build_runner_cmd)

  {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Running matrix: ${cmd[*]}"
  } >>"$ATTEMPT_LOG"

  set +e
  (
    cd "$HARNESS_ROOT"
    "${cmd[@]}"
  )
  local rc=$?
  set -e
  return "$rc"
}

if [[ "$DRY_RUN" == "1" ]]; then
  mapfile -d '' -t server_cmd < <(build_server_cmd)
  mapfile -d '' -t runner_cmd < <(build_runner_cmd)
  echo "SERVER_CMD=CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES ${server_cmd[*]}"
  echo "RUNNER_CMD=(cd $HARNESS_ROOT && ${runner_cmd[*]})"
  printf "ok\n" >"$STATUS_FILE"
  exit 0
fi

if ! start_server; then
  printf "failed\n" >"$STATUS_FILE"
  exit 1
fi

if run_runner_once; then
  rc=0
else
  rc=$?
  echo "runner_rc=$rc" >>"$ATTEMPT_LOG"
  printf "failed\n" >"$STATUS_FILE"
  exit "$rc"
fi

cleanup_server
printf "ok\n" >"$STATUS_FILE"
exit 0
