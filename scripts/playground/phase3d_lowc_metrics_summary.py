#!/usr/bin/env python3
"""Build Phase3D low-concurrency summary tables and pareto-ready points."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


LMEVAL_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}:\d{2}:\d{2}:\d{2})")
DECODE_RE = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?: [^\]]+)?\] Decode batch, #running-req: (\d+),.*gen throughput \(token/s\): ([0-9]+(?:\.[0-9]+)?)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def parse_lmeval_ts(line: str) -> Optional[dt.datetime]:
    m = LMEVAL_TS_RE.match(line)
    if m is None:
        return None
    return dt.datetime.strptime(m.group(1), "%Y-%m-%d:%H:%M:%S")


def case_time_window(case_dir: Path) -> Tuple[Optional[dt.datetime], Optional[dt.datetime]]:
    log_path = case_dir / "lm_eval.log"
    if not log_path.exists():
        return None, None
    start: Optional[dt.datetime] = None
    end: Optional[dt.datetime] = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        ts = parse_lmeval_ts(line)
        if ts is None:
            continue
        if start is None:
            start = ts
        end = ts
    return start, end


def trim10(seq: Sequence[Tuple]) -> Sequence[Tuple]:
    if len(seq) == 0:
        return seq
    trim_n = int(math.floor(0.10 * len(seq)))
    return seq[trim_n:]


def safe_mean(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    return sum(vals) / len(vals)


def safe_ratio(numer: Optional[float], denom: Optional[float]) -> Optional[float]:
    if numer is None or denom is None or denom == 0:
        return None
    return numer / denom


def parse_decode_samples(server_log: Path) -> List[Tuple[dt.datetime, int, float]]:
    out: List[Tuple[dt.datetime, int, float]] = []
    if not server_log.exists():
        return out
    for line in server_log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = DECODE_RE.match(line)
        if m is None:
            continue
        ts = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        running = int(m.group(2))
        tps = float(m.group(3))
        out.append((ts, running, tps))
    return out


def parse_request_metrics(metrics_dir: Path) -> List[Tuple[dt.datetime, dt.datetime, int, float]]:
    rows: List[Tuple[dt.datetime, dt.datetime, int, float]] = []
    if not metrics_dir.exists():
        return rows
    for path in sorted(metrics_dir.glob("sglang-request-metrics-*.log")):
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rr = payload.get("request_received_ts")
                rs = payload.get("response_sent_to_client_ts")
                ct = payload.get("completion_tokens")
                e2e = payload.get("e2e_latency")
                if not isinstance(rr, (int, float)):
                    continue
                if not isinstance(rs, (int, float)):
                    continue
                if not isinstance(ct, (int, float)):
                    continue
                if not isinstance(e2e, (int, float)):
                    continue
                rr_dt = dt.datetime.fromtimestamp(float(rr))
                rs_dt = dt.datetime.fromtimestamp(float(rs))
                rows.append((rr_dt, rs_dt, int(ct), float(e2e)))
    rows.sort(key=lambda x: x[0])
    return rows


def find_latest_results_json(case_dir: Path) -> Optional[Path]:
    cands = sorted(case_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
    if not cands:
        return None
    return cands[-1]


def read_flex_em(results_json: Optional[Path]) -> Optional[float]:
    if results_json is None:
        return None
    payload = json.loads(results_json.read_text(encoding="utf-8"))
    task = payload.get("results", {}).get("gsm8k_cot_singleshot", {})
    flex = task.get("exact_match,flexible-extract")
    if isinstance(flex, (int, float)):
        return float(flex)
    return None


def parse_latest_run_times(run_times: Path) -> Dict[str, Dict[str, int]]:
    latest: Dict[str, Dict[str, int]] = {}
    if not run_times.exists():
        return latest
    for line in run_times.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        case = parts[0].strip()
        try:
            c = int(parts[1].strip())
            elapsed_s = int(parts[4].strip())
            rc = int(parts[5].strip())
        except ValueError:
            continue
        latest[case] = {"c": c, "elapsed_s": elapsed_s, "rc": rc}
    return latest


def infer_strategy(case_name: str, shard_name: str) -> str:
    if case_name.startswith("non_mtp_"):
        return "non-MTP"
    m_k = re.match(r"mtp_k(\d+)_c\d+$", case_name)
    if m_k:
        return f"k={int(m_k.group(1))}"
    m_conf = re.match(r"(conf_adapt_k\d+_t[0-9.]+)_c\d+$", case_name)
    if m_conf:
        return m_conf.group(1)
    if case_name.startswith("eagle3_"):
        return "eagle3"
    if shard_name == "eagle3":
        return "eagle3"
    return shard_name


def format_num(x: Optional[float], decimals: int = 4) -> str:
    if x is None:
        return ""
    return f"{x:.{decimals}f}"


def row_sort_key(row: Dict[str, object]) -> Tuple[int, str, int]:
    strategy = str(row["strategy"])
    c = int(row["c"])
    if strategy == "non-MTP":
        bucket = 0
    elif strategy.startswith("k="):
        bucket = 1
    elif strategy.startswith("conf_adapt"):
        bucket = 2
    elif strategy == "eagle3":
        bucket = 3
    else:
        bucket = 4
    return (bucket, strategy, c)


def to_markdown(rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> str:
    hdr = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [hdr, sep]
    for row in rows:
        vals = [str(row.get(col, "")) for col in columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def summarize_run(run_root: Path) -> Tuple[List[Dict[str, object]], List[str]]:
    shard_root = run_root / "shards"
    if not shard_root.exists():
        return [], [f"Missing shard directory: {shard_root}"]

    rows: List[Dict[str, object]] = []
    warnings: List[str] = []

    for shard_dir in sorted(p for p in shard_root.iterdir() if p.is_dir()):
        shard_name = shard_dir.name
        run_times = shard_dir / "lmeval_matrix" / "run_times.tsv"
        latest = parse_latest_run_times(run_times)
        decode_points = parse_decode_samples(shard_dir / "server.log")
        request_points = parse_request_metrics(shard_dir / "metrics")

        if not latest:
            warnings.append(f"No run_times rows for shard: {shard_dir}")
            continue

        for case_name, info in latest.items():
            case_dir = shard_dir / "lmeval_matrix" / case_name
            start, end = case_time_window(case_dir)
            if start is None or end is None:
                warnings.append(f"Missing lm_eval time window for case: {case_dir}")
                continue

            decode_raw = [x for x in decode_points if start <= x[0] <= end]
            decode_kept = list(trim10(decode_raw))
            server_peak = max((x[2] for x in decode_raw), default=None)
            server_avg = safe_mean(x[2] for x in decode_kept)
            server_avg_per_req = safe_mean(
                x[2] / x[1] for x in decode_kept if isinstance(x[1], int) and x[1] > 0
            )

            req_raw = [
                x
                for x in request_points
                if x[1] >= start and x[0] <= end
            ]
            req_kept = list(trim10(req_raw))
            req_kept_sorted = sorted(req_kept, key=lambda x: x[0])
            client_avg_tpt = None
            client_avg_per_req_tpt = safe_mean(
                (x[2] / x[3]) for x in req_kept_sorted if x[3] > 0
            )
            if req_kept_sorted:
                span_s = (max(x[1] for x in req_kept_sorted) - min(x[0] for x in req_kept_sorted)).total_seconds()
                if span_s > 0:
                    client_avg_tpt = sum(x[2] for x in req_kept_sorted) / span_s

            flex_em = read_flex_em(find_latest_results_json(case_dir))
            strategy = infer_strategy(case_name=case_name, shard_name=shard_name)

            rows.append(
                {
                    "strategy": strategy,
                    "c": int(info["c"]),
                    "case": case_name,
                    "rc": int(info["rc"]),
                    "elapsed_s": int(info["elapsed_s"]),
                    "Flex EM": format_num(flex_em, 4),
                    "decode_samples_raw": len(decode_raw),
                    "decode_samples_kept": len(decode_kept),
                    "server_peak_gen_tps": format_num(server_peak, 2),
                    "server_avg_gen_tps_trim10": format_num(server_avg, 2),
                    "server_avg_per_req_tps_trim10": format_num(server_avg_per_req, 2),
                    "request_samples_raw": len(req_raw),
                    "request_samples_kept": len(req_kept_sorted),
                    "client_avg_tpt_trim10": format_num(client_avg_tpt, 2),
                    "client_avg_per_req_tpt_trim10": format_num(client_avg_per_req_tpt, 2),
                    "__server_avg": server_avg,
                    "__server_avg_per_req": server_avg_per_req,
                    "__client_avg": client_avg_tpt,
                    "__client_avg_per_req": client_avg_per_req_tpt,
                }
            )

    rows.sort(key=row_sort_key)

    non_mtp_by_c: Dict[int, Dict[str, Optional[float]]] = {}
    for row in rows:
        if row["strategy"] != "non-MTP":
            continue
        c = int(row["c"])
        non_mtp_by_c[c] = {
            "server_avg": row["__server_avg"],  # type: ignore[index]
            "server_avg_per_req": row["__server_avg_per_req"],  # type: ignore[index]
            "client_avg": row["__client_avg"],  # type: ignore[index]
            "client_avg_per_req": row["__client_avg_per_req"],  # type: ignore[index]
        }

    for row in rows:
        c = int(row["c"])
        base = non_mtp_by_c.get(c, {})
        row["server_avg_gen_tps_vs_non_mtp"] = format_num(
            safe_ratio(row["__server_avg"], base.get("server_avg")), 3  # type: ignore[arg-type]
        )
        row["server_avg_per_req_tps_vs_non_mtp"] = format_num(
            safe_ratio(row["__server_avg_per_req"], base.get("server_avg_per_req")), 3  # type: ignore[arg-type]
        )
        row["client_avg_tpt_vs_non_mtp"] = format_num(
            safe_ratio(row["__client_avg"], base.get("client_avg")), 3  # type: ignore[arg-type]
        )
        row["client_avg_per_req_tpt_vs_non_mtp"] = format_num(
            safe_ratio(row["__client_avg_per_req"], base.get("client_avg_per_req")), 3  # type: ignore[arg-type]
        )
        row["pareto_x_latency_inv_tok_per_s_per_seq"] = row["client_avg_per_req_tpt_trim10"]
        row["pareto_y_throughput_tok_per_s_gpu"] = row["server_avg_gen_tps_trim10"]

    for row in rows:
        row.pop("__server_avg", None)
        row.pop("__server_avg_per_req", None)
        row.pop("__client_avg", None)
        row.pop("__client_avg_per_req", None)

    return rows, warnings


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    summary_dir = run_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows, warnings = summarize_run(run_root=run_root)
    if not rows:
        (summary_dir / "failures.md").write_text(
            "# Failures\n\n- No summary rows produced.\n",
            encoding="utf-8",
        )
        return 1

    all_columns = [
        "strategy",
        "c",
        "case",
        "rc",
        "elapsed_s",
        "Flex EM",
        "decode_samples_raw",
        "decode_samples_kept",
        "server_peak_gen_tps",
        "server_avg_gen_tps_trim10",
        "server_avg_per_req_tps_trim10",
        "request_samples_raw",
        "request_samples_kept",
        "client_avg_tpt_trim10",
        "client_avg_per_req_tpt_trim10",
        "server_avg_gen_tps_vs_non_mtp",
        "server_avg_per_req_tps_vs_non_mtp",
        "client_avg_tpt_vs_non_mtp",
        "client_avg_per_req_tpt_vs_non_mtp",
        "pareto_x_latency_inv_tok_per_s_per_seq",
        "pareto_y_throughput_tok_per_s_gpu",
    ]

    tsv_path = summary_dir / "lowc_metrics_complete.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    md_path = summary_dir / "lowc_metrics_complete.md"
    md_path.write_text(to_markdown(rows, all_columns), encoding="utf-8")

    by_c_lines: List[str] = []
    by_c_lines.append("# Low-Concurrency Summary (Pareto-Ready)\n")
    for c in sorted({int(r["c"]) for r in rows}):
        subset = [r for r in rows if int(r["c"]) == c]
        subset.sort(key=row_sort_key)
        by_c_lines.append(f"## c={c}\n")
        by_c_cols = [
            "strategy",
            "Flex EM",
            "server_avg_gen_tps_trim10",
            "client_avg_per_req_tpt_trim10",
            "client_avg_tpt_trim10",
            "server_peak_gen_tps",
            "server_avg_gen_tps_vs_non_mtp",
            "client_avg_per_req_tpt_vs_non_mtp",
        ]
        by_c_lines.append(to_markdown(subset, by_c_cols))
    (summary_dir / "lowc_metrics_by_concurrency.md").write_text(
        "\n".join(by_c_lines), encoding="utf-8"
    )

    pareto_cols = [
        "strategy",
        "c",
        "Flex EM",
        "pareto_x_latency_inv_tok_per_s_per_seq",
        "pareto_y_throughput_tok_per_s_gpu",
    ]
    pareto_path = summary_dir / "pareto_points.tsv"
    with pareto_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pareto_cols, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows({col: row.get(col, "") for col in pareto_cols} for row in rows)

    failure_lines = ["# Failures", ""]
    failed_rows = [r for r in rows if int(r["rc"]) != 0]
    if failed_rows:
        failure_lines.append("## Nonzero RC Cases")
        for row in failed_rows:
            failure_lines.append(
                f"- {row['strategy']} c={row['c']} case={row['case']} rc={row['rc']}"
            )
        failure_lines.append("")
    missing_req = [r for r in rows if int(r["request_samples_kept"]) == 0]
    if missing_req:
        failure_lines.append("## Cases With Zero Kept Request Samples")
        for row in missing_req:
            failure_lines.append(f"- {row['strategy']} c={row['c']} case={row['case']}")
        failure_lines.append("")
    if warnings:
        failure_lines.append("## Warnings")
        for item in warnings:
            failure_lines.append(f"- {item}")
        failure_lines.append("")
    if len(failure_lines) == 2:
        failure_lines.append("- none")
    (summary_dir / "failures.md").write_text("\n".join(failure_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {tsv_path}")
    print(f"Wrote: {md_path}")
    print(f"Wrote: {summary_dir / 'lowc_metrics_by_concurrency.md'}")
    print(f"Wrote: {pareto_path}")
    print(f"Wrote: {summary_dir / 'failures.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
