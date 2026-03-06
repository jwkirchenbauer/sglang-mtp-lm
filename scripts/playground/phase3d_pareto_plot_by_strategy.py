#!/usr/bin/env python3
"""Render a low-concurrency throughput-latency Pareto plot grouped by strategy."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument(
        "--title",
        type=str,
        default="Throughput-Latency Pareto Frontier",
    )
    return parser.parse_args()


def strategy_sort_key(strategy: str) -> tuple[int, str]:
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
    return (bucket, strategy)


def main() -> int:
    args = parse_args()
    rows = list(csv.DictReader(args.input_tsv.open("r", encoding="utf-8"), delimiter="\t"))

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        try:
            _ = float(row["pareto_x_latency_inv_tok_per_s_per_seq"])
            _ = float(row["pareto_y_throughput_tok_per_s_gpu"])
            _ = int(row["c"])
            _ = int(row["rc"])
        except (KeyError, TypeError, ValueError):
            continue
        grouped.setdefault(row["strategy"], []).append(row)

    strategies = sorted(grouped.keys(), key=strategy_sort_key)
    if not strategies:
        raise RuntimeError(f"No plottable rows found in {args.input_tsv}")

    # Use tab20, then fallback to hsv for larger sets.
    tab20 = list(plt.get_cmap("tab20").colors)
    if len(strategies) <= len(tab20):
        colors = tab20[: len(strategies)]
    else:
        cmap = plt.get_cmap("hsv")
        colors = [cmap(i / len(strategies)) for i in range(len(strategies))]

    fig, ax = plt.subplots(figsize=(18, 10), dpi=220)
    fig.patch.set_facecolor("#efefef")
    ax.set_facecolor("#efefef")

    all_x: list[float] = []
    all_y: list[float] = []

    for idx, strategy in enumerate(strategies):
        sr = sorted(grouped[strategy], key=lambda r: int(r["c"]))
        x = [float(r["pareto_x_latency_inv_tok_per_s_per_seq"]) for r in sr]
        y = [float(r["pareto_y_throughput_tok_per_s_gpu"]) for r in sr]
        cvals = [int(r["c"]) for r in sr]
        all_x.extend(x)
        all_y.extend(y)

        ax.plot(
            x,
            y,
            marker="o",
            markersize=7,
            linewidth=2.2,
            color=colors[idx],
            alpha=0.95,
            label=strategy,
        )
        for px, py, cval in zip(x, y, cvals):
            ax.annotate(
                f"c={cval}",
                (px, py),
                xytext=(8, 7),
                textcoords="offset points",
                fontsize=9,
                color=colors[idx],
                fontweight="bold",
            )

    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    xr = xmax - xmin
    yr = ymax - ymin
    ax.set_xlim(xmin - 0.05 * xr, xmax + 0.08 * xr)
    ax.set_ylim(max(0.0, ymin - 0.06 * yr), ymax + 0.10 * yr)

    ax.set_title(args.title, fontsize=34, fontweight="bold", pad=20)
    ax.set_xlabel(r"Latency$^{-1}$ (tok/s per sequence)", fontsize=28, fontweight="bold", labelpad=14)
    ax.set_ylabel("Throughput (tok/s/GPU)", fontsize=28, fontweight="bold", labelpad=14)
    ax.grid(True, alpha=0.25, color="#888888", linewidth=0.8)
    ax.tick_params(axis="both", labelsize=18)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.2)
        ax.spines[side].set_color("#444444")

    ax.legend(
        title="Strategy",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=12,
        title_fontsize=13,
    )

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    fig.savefig(args.output_png, bbox_inches="tight")
    plt.close(fig)

    print(args.output_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

