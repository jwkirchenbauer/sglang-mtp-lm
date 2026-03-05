#!/usr/bin/env python3
"""Fast /generate driver for adaptive-MTP correctness debugging.

This script is intentionally lightweight so it can be used for tight
debug loops before running full lm_eval acceptance.
"""

import argparse
import json
import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests


DEFAULT_PROMPT_TEMPLATES = [
    "Solve this exactly and return only the final integer: {a}+{b}=",
    "Compute: ({a}*{b})-({c}). Return only the number.",
    "If x={a} and y={b}, what is x^2+y? Return only the number.",
    "A store sells {a} items on day one and {b} on day two. Total?",
]


def build_default_prompts(num_prompts: int) -> List[str]:
    prompts: List[str] = []
    for i in range(num_prompts):
        a = 11 + i
        b = 7 + (i % 13)
        c = 3 + (i % 5)
        template = DEFAULT_PROMPT_TEMPLATES[i % len(DEFAULT_PROMPT_TEMPLATES)]
        prompts.append(template.format(a=a, b=b, c=c))
    return prompts


def load_prompts(prompt_file: Path, num_prompts: int) -> List[str]:
    if prompt_file.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in prompt_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        prompts = [str(row.get("prompt", row.get("text", ""))) for row in rows]
    else:
        prompts = [
            line.strip()
            for line in prompt_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_file}")
    if len(prompts) >= num_prompts:
        return prompts[:num_prompts]
    expanded: List[str] = []
    while len(expanded) < num_prompts:
        expanded.extend(prompts)
    return expanded[:num_prompts]


def build_sampling_params(args: argparse.Namespace) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "temperature": 0,
        "top_k": 1,
        "max_new_tokens": int(args.max_new_tokens),
    }
    if args.disable_mtp:
        if str(args.mtp_strategy_kind) != "static":
            raise ValueError("disable_mtp cannot be combined with conf_adapt strategy.")
        return params

    params["mtp_enabled"] = True
    params["mtp_k"] = int(args.mtp_k)
    params["mtp_mask_id"] = int(args.mtp_mask_id)
    if str(args.mtp_strategy_kind) == "conf_adapt":
        params["mtp_strategy"] = ["conf_adapt", float(args.conf_threshold)]
        params["mtp_adaptive_window_mode"] = str(args.adaptive_window_mode)
    if args.stop:
        params["stop"] = list(args.stop)
    return params


def post_one(
    *,
    url: str,
    label: str,
    idx: int,
    prompt: str,
    sampling_params: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    rid = f"{label}-{idx:04d}-{uuid.uuid4().hex[:8]}"
    payload = {
        "rid": rid,
        "text": prompt,
        "sampling_params": sampling_params,
        "stream": False,
    }
    started = time.perf_counter()
    try:
        resp = requests.post(
            f"{url.rstrip('/')}/generate", json=payload, timeout=timeout_s
        )
        elapsed_s = time.perf_counter() - started
        row: Dict[str, Any] = {
            "label": label,
            "idx": idx,
            "rid": rid,
            "status_code": resp.status_code,
            "latency_s": elapsed_s,
            "ok": resp.ok,
        }
        if resp.ok:
            body = resp.json()
            row["text"] = body.get("text", "")
            row["meta_info"] = body.get("meta_info", {})
        else:
            row["error"] = resp.text
        return row
    except Exception as e:  # pylint: disable=broad-except
        return {
            "label": label,
            "idx": idx,
            "rid": rid,
            "status_code": None,
            "latency_s": time.perf_counter() - started,
            "ok": False,
            "error": repr(e),
        }


def _extract_completion_tokens(row: Dict[str, Any]) -> int:
    meta_info = row.get("meta_info")
    if not isinstance(meta_info, dict):
        return 0
    completion_tokens = meta_info.get("completion_tokens")
    if isinstance(completion_tokens, (int, float)):
        return max(0, int(completion_tokens))
    return 0


def _percentile_from_sorted(sorted_values: List[float], percentile: float) -> float:
    if len(sorted_values) == 0:
        raise ValueError("percentile requires non-empty input")
    rank = max(0, math.ceil(percentile * len(sorted_values)) - 1)
    rank = min(rank, len(sorted_values) - 1)
    return float(sorted_values[rank])


def run_endpoint(
    *,
    label: str,
    url: str,
    prompts: List[str],
    sampling_params: Dict[str, Any],
    num_concurrent: int,
    timeout_s: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    wall_time_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, int(num_concurrent))) as pool:
        futures = {
            pool.submit(
                post_one,
                url=url,
                label=label,
                idx=i,
                prompt=prompt,
                sampling_params=sampling_params,
                timeout_s=timeout_s,
            ): i
            for i, prompt in enumerate(prompts)
        }
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda x: int(x["idx"]))
    wall_time_s = time.perf_counter() - wall_time_start

    ok_rows = [x for x in rows if bool(x.get("ok"))]
    latencies = sorted([float(x["latency_s"]) for x in ok_rows])
    total_completion_tokens = int(sum(_extract_completion_tokens(x) for x in ok_rows))
    summary = {
        "label": label,
        "url": url,
        "num_requests": len(rows),
        "num_ok": len(ok_rows),
        "num_failed": len(rows) - len(ok_rows),
        "wall_time_s": float(wall_time_s),
        "p50_latency_s": (
            _percentile_from_sorted(latencies, 0.50) if len(latencies) > 0 else None
        ),
        "p95_latency_s": (
            _percentile_from_sorted(latencies, 0.95) if len(latencies) > 0 else None
        ),
        "mean_latency_s": (
            float(sum(latencies) / len(latencies)) if len(latencies) > 0 else None
        ),
        "total_completion_tokens": total_completion_tokens,
        "completion_tokens_per_s": (
            float(total_completion_tokens / wall_time_s) if wall_time_s > 0 else None
        ),
    }
    return rows, summary


def compare_rows(
    baseline_rows: List[Dict[str, Any]], candidate_rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    if len(baseline_rows) != len(candidate_rows):
        return {
            "matched": False,
            "reason": "length_mismatch",
            "baseline_len": len(baseline_rows),
            "candidate_len": len(candidate_rows),
        }

    mismatches: List[Dict[str, Any]] = []
    for b, c in zip(baseline_rows, candidate_rows):
        b_text = str(b.get("text", ""))
        c_text = str(c.get("text", ""))
        if b_text != c_text:
            mismatches.append(
                {
                    "idx": b["idx"],
                    "baseline_text": b_text,
                    "candidate_text": c_text,
                }
            )
    return {
        "matched": len(mismatches) == 0,
        "num_mismatches": len(mismatches),
        "mismatch_sample": mismatches[:10],
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", type=str, default=None, help="Single-endpoint mode.")
    parser.add_argument("--baseline-url", type=str, default=None)
    parser.add_argument("--candidate-url", type=str, default=None)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--num-concurrent", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--disable-mtp",
        action="store_true",
        help="Send baseline non-MTP sampling params (omit all mtp_* fields).",
    )
    parser.add_argument("--mtp-k", type=int, default=8)
    parser.add_argument(
        "--mtp-strategy-kind",
        type=str,
        choices=["conf_adapt", "static"],
        default="conf_adapt",
    )
    parser.add_argument("--conf-threshold", type=float, default=0.9)
    parser.add_argument("--adaptive-window-mode", type=str, default="hf_exact")
    parser.add_argument("--mtp-mask-id", type=int, default=128259)
    parser.add_argument("--stop", action="append", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / f"mtp_quick_driver_{int(time.time())}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.url is None and (args.baseline_url is None or args.candidate_url is None):
        raise ValueError("Specify either --url or both --baseline-url/--candidate-url.")
    if args.url is not None and (args.baseline_url or args.candidate_url):
        raise ValueError("Use --url OR --baseline-url/--candidate-url, not both.")

    prompts = (
        load_prompts(args.prompt_file, args.num_prompts)
        if args.prompt_file is not None
        else build_default_prompts(args.num_prompts)
    )
    sampling_params = build_sampling_params(args)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompts.json").write_text(
        json.dumps(prompts, ensure_ascii=True, indent=2), encoding="utf-8"
    )

    summaries: Dict[str, Any] = {}
    exit_code = 0

    if args.url is not None:
        rows, summary = run_endpoint(
            label="single",
            url=args.url,
            prompts=prompts,
            sampling_params=sampling_params,
            num_concurrent=args.num_concurrent,
            timeout_s=args.timeout_s,
        )
        write_jsonl(output_dir / "single_responses.jsonl", rows)
        summaries["single"] = summary
        if summary["num_failed"] > 0:
            exit_code = 1
    else:
        baseline_rows, baseline_summary = run_endpoint(
            label="baseline",
            url=args.baseline_url,
            prompts=prompts,
            sampling_params=sampling_params,
            num_concurrent=args.num_concurrent,
            timeout_s=args.timeout_s,
        )
        candidate_rows, candidate_summary = run_endpoint(
            label="candidate",
            url=args.candidate_url,
            prompts=prompts,
            sampling_params=sampling_params,
            num_concurrent=args.num_concurrent,
            timeout_s=args.timeout_s,
        )
        write_jsonl(output_dir / "baseline_responses.jsonl", baseline_rows)
        write_jsonl(output_dir / "candidate_responses.jsonl", candidate_rows)

        compare = compare_rows(baseline_rows, candidate_rows)
        summaries["baseline"] = baseline_summary
        summaries["candidate"] = candidate_summary
        summaries["compare"] = compare
        if (
            baseline_summary["num_failed"] > 0
            or candidate_summary["num_failed"] > 0
            or not compare["matched"]
        ):
            exit_code = 1

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), **summaries}, ensure_ascii=True, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
