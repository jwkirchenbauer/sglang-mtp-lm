#!/usr/bin/env python3
"""Merge standalone EAGLE3 summary with Phase3B reference rows.

Produces a per-concurrency comparison table with:
- Eagle3 TPS
- Eagle3 Flex EM
- Phase3B non-MTP TPS
- Phase3B k=3 TPS
- Eagle3/non-MTP
- Eagle3/k=3
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eagle3-summary",
        type=Path,
        required=True,
        help="Reduced TSV from eagle3 lm_eval tabulation.",
    )
    parser.add_argument(
        "--phase3b-summary",
        type=Path,
        required=True,
        help="Phase3B reduced summary TSV.",
    )
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, default=None)
    return parser.parse_args()


def _parse_float(raw: str) -> Optional[float]:
    text = str(raw).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _format_float(value: Optional[float], digits: int) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _markdown_table(rows: Sequence[Dict[str, str]]) -> str:
    header = [
        "c",
        "Eagle3 TPS",
        "Eagle3 Flex EM",
        "Phase3B non-MTP TPS",
        "Phase3B k=3 TPS",
        "Eagle3/non-MTP",
        "Eagle3/k=3",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["c"],
                    row["Eagle3 TPS"],
                    row["Eagle3 Flex EM"],
                    row["Phase3B non-MTP TPS"],
                    row["Phase3B k=3 TPS"],
                    row["Eagle3/non-MTP"],
                    row["Eagle3/k=3"],
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    eagle_rows = _read_tsv(args.eagle3_summary)
    phase_rows = _read_tsv(args.phase3b_summary)

    eagle_by_c: Dict[int, Dict[str, Optional[float]]] = {}
    for row in eagle_rows:
        c_raw = row.get("c", "")
        try:
            c_val = int(str(c_raw).strip())
        except ValueError:
            continue
        strat = str(row.get("strat", "")).strip().lower()
        case = str(row.get("case", "")).strip().lower()
        if strat != "eagle3" and not case.startswith("eagle3_"):
            continue
        eagle_by_c[c_val] = {
            "tps": _parse_float(row.get("Server Peak Gen TPS", "")),
            "flex_em": _parse_float(row.get("Flex EM", "")),
        }

    non_mtp_by_c: Dict[int, Optional[float]] = {}
    k3_by_c: Dict[int, Optional[float]] = {}
    for row in phase_rows:
        try:
            c_val = int(str(row.get("c", "")).strip())
        except ValueError:
            continue
        strat = str(row.get("strat", "")).strip()
        tps = _parse_float(row.get("Server Peak Gen TPS", ""))
        if strat == "non-MTP":
            non_mtp_by_c[c_val] = tps
        elif strat == "k=3":
            k3_by_c[c_val] = tps

    all_c = sorted(set(eagle_by_c.keys()) | set(non_mtp_by_c.keys()) | set(k3_by_c.keys()))
    out_rows: List[Dict[str, str]] = []
    for c_val in all_c:
        eagle = eagle_by_c.get(c_val, {})
        eagle_tps = eagle.get("tps")
        eagle_flex = eagle.get("flex_em")
        non_mtp_tps = non_mtp_by_c.get(c_val)
        k3_tps = k3_by_c.get(c_val)

        ratio_non = None
        if eagle_tps is not None and non_mtp_tps is not None and non_mtp_tps > 0:
            ratio_non = eagle_tps / non_mtp_tps

        ratio_k3 = None
        if eagle_tps is not None and k3_tps is not None and k3_tps > 0:
            ratio_k3 = eagle_tps / k3_tps

        out_rows.append(
            {
                "c": str(c_val),
                "Eagle3 TPS": _format_float(eagle_tps, 2),
                "Eagle3 Flex EM": _format_float(eagle_flex, 4),
                "Phase3B non-MTP TPS": _format_float(non_mtp_tps, 2),
                "Phase3B k=3 TPS": _format_float(k3_tps, 2),
                "Eagle3/non-MTP": _format_float(ratio_non, 3),
                "Eagle3/k=3": _format_float(ratio_k3, 3),
            }
        )

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "c",
                "Eagle3 TPS",
                "Eagle3 Flex EM",
                "Phase3B non-MTP TPS",
                "Phase3B k=3 TPS",
                "Eagle3/non-MTP",
                "Eagle3/k=3",
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(out_rows)

    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(_markdown_table(out_rows), encoding="utf-8")

    print(f"Wrote merged comparison TSV: {args.output_tsv}")
    if args.output_markdown is not None:
        print(f"Wrote merged comparison markdown: {args.output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
