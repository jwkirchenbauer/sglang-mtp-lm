#!/usr/bin/env python3
"""Run an lm_eval matrix for MTP/non-MTP strategies against one server."""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence


DEFAULT_TASK = "gsm8k_cot_singleshot"
DEFAULT_CONCURRENCY = [1, 32, 64, 128]
DEFAULT_STRATEGIES = ["non_mtp", "static_k1", "static_k2", "static_k3"]


@dataclasses.dataclass(frozen=True)
class Strategy:
    label: str
    family: str
    k: int
    threshold: Optional[float]
    case_prefix: str


def _normalize_generate_url(url: str) -> str:
    stripped = url.rstrip("/")
    if stripped.endswith("/generate"):
        return stripped
    return f"{stripped}/generate"


def _parse_conf_threshold(raw: str) -> float:
    if "." in raw:
        return float(raw)
    digits = raw.strip()
    if not digits.isdigit():
        raise ValueError(f"Invalid threshold token: {raw!r}")
    if len(digits) == 1:
        return float(digits)
    numerator = int(digits)
    denom = 10 ** (len(digits) - 1)
    return float(numerator / denom)


def parse_strategy(spec: str) -> Strategy:
    normalized = spec.strip().lower()
    if normalized in {"non_mtp", "non-mtp"}:
        return Strategy(
            label="non-MTP",
            family="non_mtp",
            k=0,
            threshold=None,
            case_prefix="non_mtp",
        )

    m_static = re.fullmatch(r"static_k(\d+)", normalized)
    if m_static:
        k = int(m_static.group(1))
        if k < 1:
            raise ValueError(f"Invalid static strategy k={k}.")
        return Strategy(
            label=f"k={k}",
            family="static",
            k=k,
            threshold=None,
            case_prefix=f"mtp_k{k}",
        )

    m_adapt_decimal = re.fullmatch(r"conf_adapt_k(\d+)_t([0-9]+(?:\.[0-9]+)?)", normalized)
    if m_adapt_decimal:
        k = int(m_adapt_decimal.group(1))
        thr = _parse_conf_threshold(m_adapt_decimal.group(2))
        return Strategy(
            label=f"conf_adapt_k{k}_t{m_adapt_decimal.group(2)}",
            family="conf_adapt",
            k=k,
            threshold=thr,
            case_prefix=f"conf_adapt_k{k}_t{m_adapt_decimal.group(2)}",
        )

    raise ValueError(
        f"Unsupported strategy spec: {spec!r}. "
        "Use one of: non_mtp, static_k<N>, conf_adapt_k<N>_t<THRESH>."
    )


def strategy_to_gen_kwargs(strategy: Strategy, mask_id: int, window_mode: str) -> str:
    parts = ["temperature=0", "top_k=1"]
    if strategy.family == "non_mtp":
        return ",".join(parts)

    parts.extend(
        [
            "mtp_enabled=true",
            f"mtp_k={strategy.k}",
            f"mtp_mask_id={mask_id}",
        ]
    )
    if strategy.family == "conf_adapt":
        assert strategy.threshold is not None
        parts.extend(
            [
                f"mtp_strategy=conf_adapt+{strategy.threshold}",
                f"mtp_adaptive_window_mode={window_mode}",
            ]
        )
    return ",".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--base-url", type=str, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--max-gen-toks", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--attention-backend", type=str, default="flashinfer")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mask-id", type=int, default=128259)
    parser.add_argument("--adaptive-window-mode", type=str, default="hf_exact")
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=DEFAULT_CONCURRENCY,
        help="Concurrency list, e.g. --concurrency 1 32 64 128",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=DEFAULT_STRATEGIES,
        help=(
            "Strategy list. Supports: non_mtp, static_k<N>, "
            "conf_adapt_k<N>_t<THRESH>."
        ),
    )
    parser.add_argument(
        "--extra-lm-eval-arg",
        action="append",
        default=[],
        help="Pass-through extra args to lm_eval. Repeatable.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cases already marked rc=0 in run_times.tsv.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running lm_eval.",
    )
    return parser.parse_args()


def load_existing_success(run_times_path: Path) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    if not run_times_path.exists():
        return out
    for line in run_times_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        case = parts[0].strip()
        try:
            rc = int(parts[5].strip())
        except ValueError:
            continue
        if rc == 0:
            out[case] = True
    return out


def append_run_time(
    run_times_path: Path,
    *,
    case_name: str,
    concurrency: int,
    strategy: Strategy,
    elapsed_s: int,
    rc: int,
) -> None:
    line = (
        f"{case_name}\t{concurrency}\t{strategy.family}\t{strategy.k}\t{elapsed_s}\t{rc}\n"
    )
    with run_times_path.open("a", encoding="utf-8") as f:
        f.write(line)


def build_model_args(
    *,
    model_path: Path,
    generate_url: str,
    concurrency: int,
    dtype: str,
    attention_backend: str,
    max_gen_toks: int,
) -> str:
    entries = [
        f"pretrained={model_path}",
        f"base_url={generate_url}",
        f"num_concurrent={concurrency}",
        f"dtype={dtype}",
        f"attention_backend={attention_backend}",
        f"max_gen_toks={max_gen_toks}",
    ]
    return ",".join(entries)


def ensure_single_results_file(case_dir: Path) -> Optional[Path]:
    candidates = sorted(case_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    return candidates[-1]


def run_case(
    *,
    args: argparse.Namespace,
    case_name: str,
    case_dir: Path,
    strategy: Strategy,
    concurrency: int,
    generate_url: str,
) -> int:
    case_dir.mkdir(parents=True, exist_ok=True)
    model_args = build_model_args(
        model_path=args.model_path,
        generate_url=generate_url,
        concurrency=concurrency,
        dtype=args.dtype,
        attention_backend=args.attention_backend,
        max_gen_toks=int(args.max_gen_toks),
    )
    gen_kwargs = strategy_to_gen_kwargs(
        strategy=strategy,
        mask_id=int(args.mask_id),
        window_mode=str(args.adaptive_window_mode),
    )

    cmd: List[str] = [
        "lm_eval",
        "--model",
        "sglang-generate",
        "--device",
        str(args.device),
        "--model_args",
        model_args,
        "--gen_kwargs",
        gen_kwargs,
        "--tasks",
        str(args.task),
        "--limit",
        str(int(args.limit)),
        "--output_path",
        str(case_dir),
        "--log_samples",
        "--apply_chat_template",
        "--fewshot_as_multiturn",
    ]
    cmd.extend(args.extra_lm_eval_arg)

    print(f"\n=== {case_name} ===", flush=True)
    print(" ".join(shlex.quote(tok) for tok in cmd), flush=True)
    if args.dry_run:
        return 0

    log_path = case_dir / "lm_eval.log"
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if proc.returncode == 0 and ensure_single_results_file(case_dir) is None:
        print(
            f"[warn] {case_name}: lm_eval returned 0 but no results_*.json found under {case_dir}",
            file=sys.stderr,
        )
    return int(proc.returncode)


def validate_args(args: argparse.Namespace, strategies: Sequence[Strategy]) -> None:
    if int(args.limit) <= 0:
        raise ValueError("--limit must be > 0.")
    if int(args.max_gen_toks) <= 0:
        raise ValueError("--max-gen-toks must be > 0.")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    conc = [int(x) for x in args.concurrency]
    if any(x <= 0 for x in conc):
        raise ValueError("--concurrency values must be > 0.")
    if len(set(conc)) != len(conc):
        raise ValueError("--concurrency values must be unique.")
    if len(strategies) == 0:
        raise ValueError("At least one strategy must be provided.")


def main() -> int:
    args = parse_args()
    strategies = [parse_strategy(spec) for spec in args.strategies]
    validate_args(args, strategies)

    generate_url = _normalize_generate_url(str(args.base_url))
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_times_path = output_root / "run_times.tsv"
    manifest_path = output_root / "matrix_manifest.json"

    manifest = {
        "model_path": str(args.model_path),
        "generate_url": generate_url,
        "task": str(args.task),
        "limit": int(args.limit),
        "max_gen_toks": int(args.max_gen_toks),
        "dtype": str(args.dtype),
        "attention_backend": str(args.attention_backend),
        "device": str(args.device),
        "mask_id": int(args.mask_id),
        "adaptive_window_mode": str(args.adaptive_window_mode),
        "concurrency": [int(x) for x in args.concurrency],
        "strategies": [dataclasses.asdict(s) for s in strategies],
        "extra_lm_eval_arg": list(args.extra_lm_eval_arg),
        "dry_run": bool(args.dry_run),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    existing_success = load_existing_success(run_times_path) if args.resume else {}
    failures = 0

    for strategy in strategies:
        for c in [int(x) for x in args.concurrency]:
            case_name = f"{strategy.case_prefix}_c{c}"
            case_dir = output_root / case_name

            if args.resume and existing_success.get(case_name, False):
                print(f"SKIP {case_name} (already rc=0 in run_times.tsv)", flush=True)
                continue

            t0 = time.time()
            rc = run_case(
                args=args,
                case_name=case_name,
                case_dir=case_dir,
                strategy=strategy,
                concurrency=c,
                generate_url=generate_url,
            )
            elapsed = int(round(time.time() - t0))
            append_run_time(
                run_times_path,
                case_name=case_name,
                concurrency=c,
                strategy=strategy,
                elapsed_s=elapsed,
                rc=rc,
            )
            print(f"DONE {case_name} elapsed={elapsed}s rc={rc}", flush=True)
            if rc != 0:
                failures += 1

    if failures > 0:
        print(f"\nCompleted with {failures} failed case(s).", file=sys.stderr)
        return 1
    print("\nCompleted all cases successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
