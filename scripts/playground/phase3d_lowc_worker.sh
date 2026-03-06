#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run one Phase3D low-concurrency shard on a standalone server.

Required:
  --kind <mtp|eagle3>
  --strategy <strategy>
  --shard-dir <path>
  --port <int>

Optional:
  --model-path <path>             (required for kind=mtp)
  --mask-id <int>                 (default: 128259)
  --task <name>                   (default: gsm8k_cot_singleshot)
  --limit <int>                   (default: 512)
  --concurrency <csv_or_space>    (default: 1,2,4,8,16)
  --max-gen-toks <int>            (default: 128)
USAGE
}

KIND=""
STRATEGY=""
SHARD_DIR=""
PORT=""
MODEL_PATH=""
MASK_ID="128259"
TASK="gsm8k_cot_singleshot"
LIMIT="512"
CONCURRENCY="1,2,4,8,16"
MAX_GEN_TOKS="128"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kind) KIND="$2"; shift 2 ;;
    --strategy) STRATEGY="$2"; shift 2 ;;
    --shard-dir) SHARD_DIR="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --mask-id) MASK_ID="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --max-gen-toks) MAX_GEN_TOKS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$KIND" || -z "$STRATEGY" || -z "$SHARD_DIR" || -z "$PORT" ]]; then
  echo "Missing required args." >&2
  usage
  exit 2
fi
if [[ "$KIND" != "mtp" && "$KIND" != "eagle3" ]]; then
  echo "Invalid --kind: $KIND" >&2
  exit 2
fi
if [[ "$KIND" == "mtp" && -z "$MODEL_PATH" ]]; then
  echo "--model-path is required for kind=mtp" >&2
  exit 2
fi

MAIN_REPO_ROOT="/capstor/scratch/cscs/jkirchen/sglang-mtp-lm"
EAGLE3_REPO_ROOT="${MAIN_REPO_ROOT}/.worktrees/eagle3_80760a2"
REPO_ROOT="$MAIN_REPO_ROOT"
if [[ "$KIND" == "eagle3" ]]; then
  # EAGLE3 is pinned to the known-good pre-MTP code state.
  if [[ -d "$EAGLE3_REPO_ROOT" ]]; then
    REPO_ROOT="$EAGLE3_REPO_ROOT"
  else
    echo "WARNING: EAGLE3 worktree missing at $EAGLE3_REPO_ROOT; falling back to $MAIN_REPO_ROOT" >&2
  fi
fi
if [[ ! -d "$REPO_ROOT" ]]; then
  echo "Repo root missing: $REPO_ROOT" >&2
  exit 2
fi
REPO_EDITABLE_ROOT="${REPO_ROOT}/python"
MAIN_EDITABLE_ROOT="${MAIN_REPO_ROOT}/python"
if [[ ! -d "$REPO_EDITABLE_ROOT" ]]; then
  echo "Editable package root missing: $REPO_EDITABLE_ROOT" >&2
  exit 2
fi
if [[ ! -d "$MAIN_EDITABLE_ROOT" ]]; then
  echo "Editable package root missing: $MAIN_EDITABLE_ROOT" >&2
  exit 2
fi
cd "$REPO_ROOT"
ORIG_PYTHONPATH="${PYTHONPATH:-}"

compose_pythonpath() {
  local target_root="$1"
  local original="$2"
  local -a parts out
  local p

  out=("${target_root}/python" "${target_root}")
  IFS=':' read -r -a parts <<< "${original}"
  for p in "${parts[@]}"; do
    [[ -n "$p" ]] || continue
    case "$p" in
      "$MAIN_REPO_ROOT"|"$MAIN_REPO_ROOT/python"|"$EAGLE3_REPO_ROOT"|"$EAGLE3_REPO_ROOT/python"|"$REPO_ROOT"|"$REPO_ROOT/python")
        continue
        ;;
    esac
    out+=("$p")
  done

  local joined=""
  for p in "${out[@]}"; do
    if [[ -z "$joined" ]]; then
      joined="$p"
    else
      joined="${joined}:$p"
    fi
  done
  printf '%s\n' "$joined"
}

export PYTHONPATH="$(compose_pythonpath "$REPO_ROOT" "$ORIG_PYTHONPATH")"

source ~/.bashrc >/dev/null 2>&1 || true
if command -v sslm_sgl_env >/dev/null 2>&1; then
  sslm_sgl_env >/dev/null 2>&1 || true
fi

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
if [[ "$KIND" == "eagle3" ]]; then
  # Pre-MTP Eagle3 code path requires a C++20-capable host compiler for JIT kernels.
  if command -v g++-12 >/dev/null 2>&1; then
    export CXX="$(command -v g++-12)"
    export CUDAHOSTCXX="$CXX"
    export CMAKE_CUDA_HOST_COMPILER="$CXX"
    if [[ -n "${NVCC_PREPEND_FLAGS:-}" ]]; then
      export NVCC_PREPEND_FLAGS="-ccbin ${CXX} ${NVCC_PREPEND_FLAGS}"
    else
      export NVCC_PREPEND_FLAGS="-ccbin ${CXX}"
    fi
  fi
  if command -v gcc-12 >/dev/null 2>&1; then
    export CC="$(command -v gcc-12)"
  fi
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
REPO_HEAD="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

CONCURRENCY_NORM="${CONCURRENCY//,/ }"
read -r -a CONCURRENCIES <<< "$CONCURRENCY_NORM"
if [[ ${#CONCURRENCIES[@]} -eq 0 ]]; then
  echo "No concurrency values parsed from: $CONCURRENCY" >&2
  printf "failed\n" >"$STATUS_FILE"
  exit 1
fi
EXPECTED_CASES="${#CONCURRENCIES[@]}"
MAX_CONCURRENCY=0
for c in "${CONCURRENCIES[@]}"; do
  if [[ "$c" =~ ^[0-9]+$ ]] && (( c > MAX_CONCURRENCY )); then
    MAX_CONCURRENCY="$c"
  fi
done
SET_EXPLICIT_MAX_RUNNING_REQUESTS=0
if [[ "${PHASE3D_SET_MAX_RUNNING_REQUESTS:-0}" == "1" ]]; then
  SET_EXPLICIT_MAX_RUNNING_REQUESTS=1
fi

RESTORE_MAIN_EDITABLE=0
capture_sglang_path() {
  python - <<'PY'
import os
import sglang
print(os.path.realpath(sglang.__file__))
PY
}

install_editable_root() {
  local root="$1"
  local label="$2"
  {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] pip install -e ${root} (${label})"
  } >>"$SERVER_LOG"
  # Pip editable switches across two sglang roots can leave multiple
  # __editable__.sglang-* stubs behind. Remove them first so import
  # resolution is deterministic for the target root.
  python - <<'PY' >>"$SERVER_LOG" 2>&1 || true
import glob
import os
import site

patterns = [
    "__editable__.sglang-*.pth",
    "__editable___sglang_*_finder.py",
]
for sp in site.getsitepackages():
    for pat in patterns:
        for path in glob.glob(os.path.join(sp, pat)):
            try:
                os.remove(path)
                print(f"[editable-clean] removed {path}")
            except OSError as exc:
                print(f"[editable-clean] failed {path}: {exc}")
PY
  python -m pip uninstall -y sglang >>"$SERVER_LOG" 2>&1 || true
  python -m pip install --no-deps -e "$root" >>"$SERVER_LOG" 2>&1
}

if [[ "$KIND" == "eagle3" ]]; then
  if ! install_editable_root "$REPO_EDITABLE_ROOT" "eagle3-runtime-root"; then
    echo "Failed editable install for eagle3 root: $REPO_EDITABLE_ROOT" >&2
    printf "failed\n" >"$STATUS_FILE"
    exit 1
  fi
  ACTIVE_SGLANG_PATH="$(capture_sglang_path 2>/dev/null || true)"
  echo "runtime_sglang_path=${ACTIVE_SGLANG_PATH}" >>"$ATTEMPT_LOG"
  case "$ACTIVE_SGLANG_PATH" in
    "$REPO_EDITABLE_ROOT"/*) ;;
    *)
      echo "Runtime sglang path does not point to eagle3 repo root: ${ACTIVE_SGLANG_PATH}" >&2
      printf "failed\n" >"$STATUS_FILE"
      exit 1
      ;;
  esac
  if [[ "$REPO_ROOT" != "$MAIN_REPO_ROOT" ]]; then
    RESTORE_MAIN_EDITABLE=1
  fi
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

  if [[ "$RESTORE_MAIN_EDITABLE" -eq 1 ]]; then
    export PYTHONPATH="$(compose_pythonpath "$MAIN_REPO_ROOT" "$ORIG_PYTHONPATH")"
    if install_editable_root "$MAIN_EDITABLE_ROOT" "restore-main-root"; then
      RESTORED_SGLANG_PATH="$(capture_sglang_path 2>/dev/null || true)"
      echo "restored_runtime_sglang_path=${RESTORED_SGLANG_PATH}" >>"$ATTEMPT_LOG"
      case "$RESTORED_SGLANG_PATH" in
        "$MAIN_EDITABLE_ROOT"/*) ;;
        *)
          echo "Failed to restore editable install to main root; got ${RESTORED_SGLANG_PATH}" >>"$SERVER_LOG"
          rc=1
          ;;
      esac
    else
      echo "Failed to restore editable install to main root: $MAIN_EDITABLE_ROOT" >>"$SERVER_LOG"
      rc=1
    fi
  fi

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

    # Some servers keep /health_generate at 503 until the first real generation.
    # Trigger one tiny warmup request once /model_info is reachable.
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

start_server() {
  local cuda_graph_max_bs="$1"
  local mem_fraction_static="$2"
  local attempt_name="$3"
  local -a cmd mode_args

  {
    echo
    echo "===== ${attempt_name} ====="
    echo "ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "kind=$KIND strategy=$STRATEGY max_running_requests=$MAX_CONCURRENCY explicit_max_running_requests=$SET_EXPLICIT_MAX_RUNNING_REQUESTS cuda_graph_max_bs=$cuda_graph_max_bs mem_fraction_static=$mem_fraction_static"
    echo "toolchain CC=${CC:-unset} CXX=${CXX:-unset} CUDAHOSTCXX=${CUDAHOSTCXX:-unset}"
    echo "repo_root=$REPO_ROOT repo_head=$REPO_HEAD"
  } >>"$ATTEMPT_LOG"

  if [[ "$KIND" == "mtp" ]]; then
    mode_args=()
    if [[ "$STRATEGY" == "non_mtp" ]]; then
      mode_args=()
    elif [[ "$STRATEGY" =~ ^static_k([0-9]+)$ ]]; then
      local sk="${BASH_REMATCH[1]}"
      mode_args=(
        --enable-mtp-static-q-len-cuda-graph
        --mtp-static-cuda-graph-k-list "$sk"
      )
    elif [[ "$STRATEGY" =~ ^conf_adapt_k([0-9]+)_t[0-9.]+$ ]]; then
      local kmax="${BASH_REMATCH[1]}"
      local static_k="3"
      if [[ "$kmax" -lt 3 ]]; then
        static_k="$kmax"
      fi
      mode_args=(
        --enable-mtp-static-q-len-cuda-graph
        --mtp-static-cuda-graph-k-list "$static_k"
        --enable-mtp-adaptive-hf-exact-q-len-cuda-graph
        --mtp-adaptive-cuda-graph-kmax-list "$kmax"
      )
    else
      # Conservative fallback if an unknown MTP strategy token is used.
      mode_args=(
        --enable-mtp-static-q-len-cuda-graph
        --mtp-static-cuda-graph-k-list 1 2 3
      )
    fi

    cmd=(
      python -m sglang.launch_server
      --model-path "$MODEL_PATH"
      --port "$PORT"
      --dtype bfloat16
      --attention-backend flashinfer
      --disable-overlap-schedule
      --mem-fraction-static "$mem_fraction_static"
      --decode-log-interval 10
      --enable-metrics
      --export-metrics-to-file
      --export-metrics-to-file-dir "$METRICS_DIR"
      --cuda-graph-max-bs "$cuda_graph_max_bs"
    )
    if [[ "$SET_EXPLICIT_MAX_RUNNING_REQUESTS" -eq 1 ]]; then
      cmd+=(--max-running-requests "$MAX_CONCURRENCY")
    fi
    cmd+=("${mode_args[@]}")
  else
    cmd=(
      python -m sglang.launch_server
      --model-path meta-llama/Meta-Llama-3.1-8B-Instruct
      --port "$PORT"
      --speculative-algorithm EAGLE3
      --speculative-draft-model-path jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B
      --speculative-num-steps 3
      --speculative-eagle-topk 4
      --speculative-num-draft-tokens 16
      --dtype bfloat16
      --attention-backend flashinfer
      --disable-overlap-schedule
      --mem-fraction-static "$mem_fraction_static"
      --decode-log-interval 10
      --cuda-graph-max-bs "$cuda_graph_max_bs"
      --enable-metrics
      --export-metrics-to-file
      --export-metrics-to-file-dir "$METRICS_DIR"
    )
    if [[ "$SET_EXPLICIT_MAX_RUNNING_REQUESTS" -eq 1 ]]; then
      cmd+=(--max-running-requests "$MAX_CONCURRENCY")
    fi
  fi

  {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Launching server: ${cmd[*]}"
  } >>"$SERVER_LOG"
  nohup "${cmd[@]}" >>"$SERVER_LOG" 2>&1 &
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
  if [[ "$KIND" == "mtp" ]]; then
    cmd=(
      python scripts/playground/mtp_lmeval_matrix_runner.py
      --model-path "$MODEL_PATH"
      --base-url "http://127.0.0.1:${PORT}"
      --output-root "$RUN_OUTPUT_ROOT"
      --task "$TASK"
      --limit "$LIMIT"
      --max-gen-toks "$MAX_GEN_TOKS"
      --mask-id "$MASK_ID"
      --concurrency "${CONCURRENCIES[@]}"
      --strategies "$STRATEGY"
      --resume
    )
  else
    cmd=(
      python scripts/playground/eagle3_lmeval_concurrency_runner.py
      --model-path meta-llama/Meta-Llama-3.1-8B-Instruct
      --base-url "http://127.0.0.1:${PORT}"
      --output-root "$RUN_OUTPUT_ROOT"
      --task "$TASK"
      --limit "$LIMIT"
      --max-gen-toks "$MAX_GEN_TOKS"
      --concurrency "${CONCURRENCIES[@]}"
      --gen-kwargs "temperature=0,top_k=1"
      --resume
    )
  fi

  set +e
  "${cmd[@]}"
  local rc=$?
  set -e
  return "$rc"
}

has_any_success() {
  local run_times="$RUN_OUTPUT_ROOT/run_times.tsv"
  [[ -f "$run_times" ]] || return 1
  awk -F'\t' 'NF>=6 && $6==0 {ok=1} END{exit(ok?0:1)}' "$run_times"
}

all_cases_success() {
  local run_times="$RUN_OUTPUT_ROOT/run_times.tsv"
  [[ -f "$run_times" ]] || return 1
  awk -F'\t' -v expected="$EXPECTED_CASES" '
    NF>=6 {
      rc[$1]=$6
      seen[$1]=1
    }
    END {
      n=0
      for (k in seen) {
        n++
        if (rc[k] != 0) {
          exit 1
        }
      }
      if (n != expected) {
        exit 1
      }
      exit 0
    }
  ' "$run_times"
}

default_graph_bs=""
fallback_graph_bs=""
default_mem_frac=""
fallback_mem_frac=""
if [[ "$KIND" == "mtp" ]]; then
  default_graph_bs="$MAX_CONCURRENCY"
  if (( default_graph_bs < 16 )); then
    default_graph_bs="16"
  fi
  fallback_graph_bs="$(( default_graph_bs / 2 ))"
  if (( fallback_graph_bs < 8 )); then
    fallback_graph_bs="8"
  fi
  default_mem_frac="0.865"
  fallback_mem_frac="0.82"
  second_fallback_graph_bs="$fallback_graph_bs"
  if (( second_fallback_graph_bs > 8 )); then
    second_fallback_graph_bs="8"
  fi
  second_fallback_mem_frac="0.76"
else
  # Low-concurrency sweep policy: keep graph microbatch aligned with
  # the target request concurrency envelope instead of scaling by topk.
  default_graph_bs="$MAX_CONCURRENCY"
  if (( default_graph_bs < 8 )); then
    default_graph_bs="8"
  fi
  fallback_graph_bs="$(( default_graph_bs / 2 ))"
  if (( fallback_graph_bs < 8 )); then
    fallback_graph_bs="8"
  fi
  default_mem_frac="0.70"
  fallback_mem_frac="0.66"
fi

attempt_profiles=(
  "initial ${default_graph_bs} ${default_mem_frac}"
  "retry_same ${default_graph_bs} ${default_mem_frac}"
  "fallback_graph ${fallback_graph_bs} ${default_mem_frac}"
  "fallback_mem ${fallback_graph_bs} ${fallback_mem_frac}"
)

if [[ "$KIND" == "mtp" ]]; then
  attempt_profiles+=(
    "fallback_graph2 ${second_fallback_graph_bs} ${fallback_mem_frac}"
    "fallback_mem2 ${second_fallback_graph_bs} ${second_fallback_mem_frac}"
  )
fi

shard_ok=0
for profile in "${attempt_profiles[@]}"; do
  read -r attempt_name graph_bs mem_frac <<< "$profile"
  if ! start_server "$graph_bs" "$mem_frac" "$attempt_name"; then
    echo "attempt=${attempt_name} start_server=failed" >>"$ATTEMPT_LOG"
    continue
  fi

  if run_runner_once; then
    runner_rc=0
  else
    runner_rc=$?
  fi
  cleanup_server

  echo "attempt=${attempt_name} runner_rc=${runner_rc}" >>"$ATTEMPT_LOG"

  if all_cases_success; then
    shard_ok=1
    break
  fi

  if ! has_any_success; then
    echo "attempt=${attempt_name} no_success_rows_yet=true" >>"$ATTEMPT_LOG"
  fi
done

if [[ "$shard_ok" -eq 1 ]]; then
  printf "ok\n" >"$STATUS_FILE"
  exit 0
fi

printf "failed\n" >"$STATUS_FILE"
exit 1
