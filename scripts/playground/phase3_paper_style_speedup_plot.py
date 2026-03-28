#!/usr/bin/env python3
"""Render paper-style accuracy vs speedup plots from Phase3 summary TSVs."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D


CONCURRENCY_ORDER = [1, 2, 4, 8, 16]
STATIC_MARKER = "s"
ADAPTIVE_MARKERS = {3: "o", 8: "^", 16: "D"}
EAGLE3_MARKER = "*"
NON_MTP_MARKER = "X"


@dataclass(frozen=True)
class PlotPoint:
    strategy: str
    family: str
    c: int
    x: float
    y: float
    static_k: Optional[int] = None
    adaptive_kmax: Optional[int] = None
    adaptive_tau: Optional[float] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--title", type=str, required=True)
    parser.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="Optional fixed lower x-axis bound.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Optional fixed upper x-axis bound.",
    )
    parser.add_argument(
        "--include-eagle3",
        action="store_true",
        help="Include eagle3 rows if they exist in the input TSV.",
    )
    parser.add_argument(
        "--exclude-static-k",
        type=int,
        nargs="*",
        default=[],
        help="Optional list of static k values to omit from the plot.",
    )
    parser.add_argument(
        "--write-points-tsv",
        type=Path,
        default=None,
        help="Optional path to write the plotted points table.",
    )
    return parser.parse_args()


def parse_tau(token: str) -> float:
    value = int(token) / 10.0
    while value > 1.0:
        value /= 10.0
    return value


def parse_strategy(strategy: str) -> Optional[dict[str, object]]:
    if strategy == "non-MTP":
        return {"family": "non_mtp"}

    static_match = re.fullmatch(r"k=(\d+)", strategy)
    if static_match:
        return {"family": "static", "static_k": int(static_match.group(1))}

    adapt_match = re.fullmatch(r"conf_adapt_k(\d+)_t(\d+)", strategy)
    if adapt_match:
        return {
            "family": "adaptive",
            "adaptive_kmax": int(adapt_match.group(1)),
            "adaptive_tau": parse_tau(adapt_match.group(2)),
        }

    if strategy == "eagle3":
        return {"family": "eagle3"}

    return None


def load_points(
    input_tsv: Path,
    include_eagle3: bool,
    exclude_static_k: set[int],
) -> list[PlotPoint]:
    points: list[PlotPoint] = []
    with input_tsv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                rc = int(row["rc"])
                c = int(row["c"])
                x = float(row["server_avg_gen_tps_vs_non_mtp"])
                y = float(row["Flex EM"]) * 100.0
            except (KeyError, TypeError, ValueError):
                continue

            if rc != 0:
                continue

            parsed = parse_strategy(row.get("strategy", ""))
            if parsed is None:
                continue
            if parsed["family"] == "eagle3" and not include_eagle3:
                continue
            if parsed["family"] == "static" and int(parsed["static_k"]) in exclude_static_k:
                continue

            points.append(
                PlotPoint(
                    strategy=row["strategy"],
                    family=str(parsed["family"]),
                    c=c,
                    x=x,
                    y=y,
                    static_k=parsed.get("static_k"),  # type: ignore[arg-type]
                    adaptive_kmax=parsed.get("adaptive_kmax"),  # type: ignore[arg-type]
                    adaptive_tau=parsed.get("adaptive_tau"),  # type: ignore[arg-type]
                )
            )

    return points


def static_color_map(static_ks: Iterable[int]) -> dict[int, tuple[float, float, float, float]]:
    ks = sorted(set(static_ks))
    cmap = plt.get_cmap("RdYlGn_r")
    if len(ks) == 1:
        return {ks[0]: cmap(0.5)}
    return {k: cmap(idx / (len(ks) - 1)) for idx, k in enumerate(ks)}


def adaptive_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "phase3_tau",
        ["#7b63b8", "#9a88e1", "#91b5e7", "#d9ecf4"],
    )


def axis_bounds(values: list[float], *, zero_floor: bool) -> tuple[float, float]:
    if not values:
        return (0.0, 1.0)
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        pad = 0.1 * abs(vmax) if vmax else 1.0
        low = vmin - pad
        high = vmax + pad
    else:
        pad = 0.08 * (vmax - vmin)
        low = vmin - pad
        high = vmax + pad
    if zero_floor:
        low = max(0.0, low)
    return low, high


def write_points_tsv(points: list[PlotPoint], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "strategy",
                "family",
                "c",
                "x_speedup_vs_non_mtp",
                "y_flex_em_pct",
                "static_k",
                "adaptive_kmax",
                "adaptive_tau",
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for point in sorted(points, key=lambda p: (p.c, p.family, p.strategy)):
            writer.writerow(
                {
                    "strategy": point.strategy,
                    "family": point.family,
                    "c": point.c,
                    "x_speedup_vs_non_mtp": f"{point.x:.4f}",
                    "y_flex_em_pct": f"{point.y:.4f}",
                    "static_k": "" if point.static_k is None else point.static_k,
                    "adaptive_kmax": "" if point.adaptive_kmax is None else point.adaptive_kmax,
                    "adaptive_tau": "" if point.adaptive_tau is None else f"{point.adaptive_tau:.3f}",
                }
            )


def main() -> int:
    args = parse_args()
    if args.x_min is not None and args.x_max is not None and args.x_min >= args.x_max:
        raise ValueError("--x-min must be smaller than --x-max")

    points = load_points(
        args.input_tsv,
        include_eagle3=args.include_eagle3,
        exclude_static_k=set(args.exclude_static_k),
    )
    if not points:
        raise RuntimeError(f"No plottable rows found in {args.input_tsv}")

    if args.write_points_tsv is not None:
        write_points_tsv(points, args.write_points_tsv)

    x_bounds = axis_bounds([p.x for p in points], zero_floor=True)
    if args.x_min is not None or args.x_max is not None:
        x_bounds = (
            args.x_min if args.x_min is not None else x_bounds[0],
            args.x_max if args.x_max is not None else x_bounds[1],
        )
    y_bounds = axis_bounds([p.y for p in points], zero_floor=True)
    y_low = max(0.0, math.floor(y_bounds[0] / 5.0) * 5.0)
    y_high = math.ceil(y_bounds[1] / 5.0) * 5.0

    static_ks = sorted({p.static_k for p in points if p.static_k is not None})
    static_colors = static_color_map(static_ks)
    tau_values = [p.adaptive_tau for p in points if p.adaptive_tau is not None]
    tau_norm = Normalize(vmin=min(tau_values), vmax=max(tau_values))
    tau_cmap = adaptive_cmap()

    fig, axes = plt.subplots(1, len(CONCURRENCY_ORDER), figsize=(24, 6.4), dpi=220, sharey=True)
    fig.patch.set_facecolor("#ededed")

    for ax, concurrency in zip(axes, CONCURRENCY_ORDER):
        ax.set_facecolor("#ededed")
        panel_points = [p for p in points if p.c == concurrency]

        for point in panel_points:
            if point.family == "non_mtp":
                ax.scatter(
                    point.x,
                    point.y,
                    marker=NON_MTP_MARKER,
                    s=170,
                    c="#111111",
                    edgecolors="#111111",
                    linewidths=1.5,
                    zorder=4,
                )
            elif point.family == "static":
                ax.scatter(
                    point.x,
                    point.y,
                    marker=STATIC_MARKER,
                    s=215,
                    c=[static_colors[point.static_k]],  # type: ignore[index]
                    edgecolors="#3a3a3a",
                    linewidths=1.8,
                    alpha=0.95,
                    zorder=3,
                )
            elif point.family == "adaptive":
                ax.scatter(
                    point.x,
                    point.y,
                    marker=ADAPTIVE_MARKERS[point.adaptive_kmax],  # type: ignore[index]
                    s=220,
                    c=[tau_cmap(tau_norm(point.adaptive_tau))],  # type: ignore[arg-type]
                    edgecolors="#3a3a3a",
                    linewidths=1.7,
                    alpha=0.95,
                    zorder=3,
                )
            elif point.family == "eagle3":
                ax.scatter(
                    point.x,
                    point.y,
                    marker=EAGLE3_MARKER,
                    s=240,
                    c="#bf6f1a",
                    edgecolors="#3a3a3a",
                    linewidths=1.6,
                    alpha=0.95,
                    zorder=4,
                )

        ax.set_title(f"c={concurrency}", fontsize=18, fontweight="bold", pad=10)
        ax.set_xlim(*x_bounds)
        ax.set_ylim(y_low, y_high)
        ax.grid(True, alpha=0.35, color="#b7b7b7", linewidth=0.8, linestyle=(0, (1.5, 3.0)))
        ax.tick_params(axis="both", labelsize=12)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(1.2)
            ax.spines[side].set_color("#3a3a3a")

    axes[0].set_ylabel("Flex EM (%)", fontsize=18, fontweight="bold")
    fig.supxlabel("Speedup vs non-MTP (server avg gen tok/s)", fontsize=19, fontweight="bold", y=0.06)
    fig.suptitle(args.title, fontsize=24, fontweight="bold", y=0.98)

    tau_mappable = ScalarMappable(norm=tau_norm, cmap=tau_cmap)
    tau_mappable.set_array([])
    tau_cax = fig.add_axes([0.84, 0.52, 0.018, 0.28])
    tau_cbar = fig.colorbar(tau_mappable, cax=tau_cax)
    tau_cbar.set_label("Conf Adapt tau", fontsize=14, fontweight="bold")
    tau_cbar.ax.tick_params(labelsize=11)

    static_handles = [
        Line2D(
            [0],
            [0],
            marker=STATIC_MARKER,
            linestyle="None",
            markerfacecolor=static_colors[k],
            markeredgecolor="#3a3a3a",
            markeredgewidth=1.5,
            markersize=11,
            label=f"k={k}",
        )
        for k in static_ks
    ]
    adaptive_handles = [
        Line2D(
            [0],
            [0],
            marker=ADAPTIVE_MARKERS[kmax],
            linestyle="None",
            markerfacecolor="#8b8b8b",
            markeredgecolor="#3a3a3a",
            markeredgewidth=1.4,
            markersize=11,
            label=f"k_max={kmax}",
        )
        for kmax in sorted(ADAPTIVE_MARKERS)
        if any(p.adaptive_kmax == kmax for p in points)
    ]
    special_handles = [
        Line2D(
            [0],
            [0],
            marker=NON_MTP_MARKER,
            linestyle="None",
            markerfacecolor="#111111",
            markeredgecolor="#111111",
            markeredgewidth=1.4,
            markersize=11,
            label="non-MTP",
        )
    ]
    if args.include_eagle3 and any(p.family == "eagle3" for p in points):
        special_handles.append(
            Line2D(
                [0],
                [0],
                marker=EAGLE3_MARKER,
                linestyle="None",
                markerfacecolor="#bf6f1a",
                markeredgecolor="#3a3a3a",
                markeredgewidth=1.4,
                markersize=12,
                label="eagle3",
            )
        )

    legend_markers = fig.legend(
        handles=special_handles + adaptive_handles,
        title="Markers",
        loc="upper left",
        bbox_to_anchor=(0.835, 0.47),
        frameon=False,
        fontsize=11,
        title_fontsize=12,
    )
    fig.add_artist(legend_markers)

    legend_static = fig.legend(
        handles=static_handles,
        title="Static k",
        loc="upper left",
        bbox_to_anchor=(0.835, 0.24),
        frameon=False,
        fontsize=11,
        title_fontsize=12,
        ncol=1,
    )
    fig.add_artist(legend_static)

    fig.subplots_adjust(left=0.06, right=0.80, top=0.86, bottom=0.18, wspace=0.14)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, bbox_inches="tight")
    plt.close(fig)

    print(args.output_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
