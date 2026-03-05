#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Launch a phase-3 style server preset.

Usage:
  mtp_phase3_server_launch.sh \
    --mode <base_matrix|conf_adapt_k3_t09|conf_adapt_k8_t09> \
    --model-path <path> \
    --port <int> \
    --gpu <id> \
    --log-file <path> \
    [--metrics-dir <path>] \
    [--cuda-graph-max-bs <int>] \
    [--mem-fraction-static <float>] \
    [--foreground]

Notes:
  - Run this only after activating env in the current shell via `sslm_sgl_env`.
  - Default mode-specific graph settings:
      base_matrix:       static k-list 1 2 3, cuda-graph-max-bs=128
      conf_adapt_k3_t09: static k-list 3 + adaptive kmax-list 3, cuda-graph-max-bs=128
      conf_adapt_k8_t09: static k-list 3 + adaptive kmax-list 8, cuda-graph-max-bs=64
USAGE
}

MODE=""
MODEL_PATH=""
PORT=""
GPU=""
LOG_FILE=""
METRICS_DIR=""
CUDA_GRAPH_MAX_BS=""
MEM_FRACTION_STATIC=""
FOREGROUND=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --log-file) LOG_FILE="$2"; shift 2 ;;
    --metrics-dir) METRICS_DIR="$2"; shift 2 ;;
    --cuda-graph-max-bs) CUDA_GRAPH_MAX_BS="$2"; shift 2 ;;
    --mem-fraction-static) MEM_FRACTION_STATIC="$2"; shift 2 ;;
    --foreground) FOREGROUND=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$MODE" || -z "$MODEL_PATH" || -z "$PORT" || -z "$GPU" || -z "$LOG_FILE" ]]; then
  echo "Missing required args." >&2
  usage
  exit 2
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model path not found: $MODEL_PATH" >&2
  exit 2
fi

mkdir -p "$(dirname "$LOG_FILE")"
if [[ -n "$METRICS_DIR" ]]; then
  mkdir -p "$METRICS_DIR"
fi

if [[ -z "$CUDA_GRAPH_MAX_BS" ]]; then
  case "$MODE" in
    base_matrix|conf_adapt_k3_t09) CUDA_GRAPH_MAX_BS=128 ;;
    conf_adapt_k8_t09) CUDA_GRAPH_MAX_BS=64 ;;
    *) echo "Unsupported mode: $MODE" >&2; exit 2 ;;
  esac
fi

common_args=(
  --model-path "$MODEL_PATH"
  --dtype bfloat16
  --attention-backend flashinfer
  --disable-overlap-schedule
  --enable-metrics
  --port "$PORT"
)

if [[ -n "$METRICS_DIR" ]]; then
  common_args+=(--export-metrics-to-file --export-metrics-to-file-dir "$METRICS_DIR")
fi

if [[ -n "$MEM_FRACTION_STATIC" ]]; then
  common_args+=(--mem-fraction-static "$MEM_FRACTION_STATIC")
fi

case "$MODE" in
  base_matrix)
    mode_args=(
      --enable-mtp-static-q-len-cuda-graph
      --mtp-static-cuda-graph-k-list 1 2 3
      --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    )
    ;;
  conf_adapt_k3_t09)
    mode_args=(
      --enable-mtp-static-q-len-cuda-graph
      --mtp-static-cuda-graph-k-list 3
      --enable-mtp-adaptive-hf-exact-q-len-cuda-graph
      --mtp-adaptive-cuda-graph-kmax-list 3
      --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    )
    ;;
  conf_adapt_k8_t09)
    mode_args=(
      --enable-mtp-static-q-len-cuda-graph
      --mtp-static-cuda-graph-k-list 3
      --enable-mtp-adaptive-hf-exact-q-len-cuda-graph
      --mtp-adaptive-cuda-graph-kmax-list 8
      --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    )
    ;;
  *)
    echo "Unsupported mode: $MODE" >&2
    exit 2
    ;;
esac

cmd=(python -m sglang.launch_server "${common_args[@]}" "${mode_args[@]}")

echo "Launching mode=$MODE on GPU=$GPU port=$PORT"
echo "Command: CUDA_VISIBLE_DEVICES=$GPU ${cmd[*]}"
echo "Log: $LOG_FILE"

if [[ "$FOREGROUND" -eq 1 ]]; then
  exec env CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}"
fi

nohup env CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}" >"$LOG_FILE" 2>&1 &
pid=$!
echo "PID=$pid"
echo "$pid"
