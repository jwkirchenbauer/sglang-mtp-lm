#!/usr/bin/env python3
"""Render a low-concurrency throughput-latency Pareto plot grouped by strategy."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        # "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "font.serif": ["Computer Modern"],
        # "mathtext.fontset": "cm",
    }
)


C_BLUE = "#6394ED"
C_PURP = "#7763ED"
PLOT_ALPHA = 0.7
PLOT_LINEWIDTH = 2.8
PLOT_MARKER_SIZE = 76
PLOT_MARKER_EDGE_COLOR = "#111111"
PLOT_MARKER_EDGE_WIDTH = 0.95
CORELINE_ALPHA = 0.45
# CBAR_WIDTH = 0.020
CBAR_WIDTH = 0.028
RIGHT_MARGIN = 0.74

DEFAULT_LEGEND_FONTSIZE = plt.rcParams.get("legend.fontsize", "medium")
DEFAULT_LEGEND_TITLE_FONTSIZE = (
    plt.rcParams.get("legend.title_fontsize") or DEFAULT_LEGEND_FONTSIZE
)
DEFAULT_LABEL_FONTSIZE = plt.rcParams.get("axes.labelsize", "medium")
DEFAULT_TICK_FONTSIZE = plt.rcParams.get("xtick.labelsize", "medium")
DEFAULT_TITLE_FONTSIZE = plt.rcParams.get("axes.titlesize", "large")

# FIGSIZE = (6.2, 3.6)
FIGSIZE = (5, 4)
# Panel font controls
METHODS_LEGEND_FONTSIZE = DEFAULT_LEGEND_FONTSIZE
METHODS_LEGEND_TITLE_FONTSIZE = DEFAULT_LEGEND_TITLE_FONTSIZE
CONCURRENCY_LEGEND_FONTSIZE = DEFAULT_LEGEND_FONTSIZE
CONCURRENCY_LEGEND_TITLE_FONTSIZE = DEFAULT_LEGEND_TITLE_FONTSIZE
COLORBAR_LABEL_FONTSIZE = DEFAULT_LABEL_FONTSIZE
COLORBAR_CONF_TICK_FONTSIZE = DEFAULT_TICK_FONTSIZE
COLORBAR_STATIC_TICK_FONTSIZE = DEFAULT_TICK_FONTSIZE
CONCURRENCY_MARKER_SIZE = 9

# Main axes font controls
TITLE_FONTSIZE = DEFAULT_TITLE_FONTSIZE
AXIS_LABEL_FONTSIZE = DEFAULT_LABEL_FONTSIZE
AXIS_TICK_FONTSIZE = DEFAULT_TICK_FONTSIZE

X_LABELPAD = 4
Y_LABELPAD = 6

# Colorbar placement controls
PAIRED_STATIC_CBAR_Y = 0.56
PAIRED_CONF_CBAR_Y = 0.39
CONF_ONLY_CBAR_Y = 0.30
STATIC_ONLY_CBAR_Y = 0.56
PAIRED_STATIC_CBAR_HEIGHT = 0.26
PAIRED_CONF_CBAR_HEIGHT = 0.14
CONF_ONLY_CBAR_HEIGHT = None
STATIC_ONLY_CBAR_HEIGHT = 0.14

# PANEL_CENTER_X = 0.805
PANEL_CENTER_X = 0.84
# CBAR_X = PANEL_CENTER_X - (CBAR_WIDTH / 2.0)
CBAR_X = PANEL_CENTER_X - (CBAR_WIDTH / 2.0) - 0.035
METHODS_Y = 0.98
METHODS_Y_NO_CONFADAPT = 0.9
CONCURRENCY_Y = 0.38
CONCURRENCY_Y_NO_CONFADAPT = 0.53
SHOW_EMPTY_METHODS_HEADER = True
METHODS_GAP_ABOVE_STATIC_CBAR_WITH_EAGLE3 = 0.15
METHODS_GAP_ABOVE_STATIC_CBAR_WITHOUT_EAGLE3 = 0.1
THRESHOLDS = [
    "0.995",
    "0.99",
    "0.98",
    "0.97",
    "0.96",
    "0.95",
    "0.9",
    "0.87",
    "0.85",
    "0.8",
    "0.75",
    "0.7",
    "0.65",
    "0.6",
]
STATIC_COLOR_MAP = {
    1: "#76C893",
    2: "#B5D33D",
    3: "#EDBC63",
    4: "#E76F51",
    5: "#D62828",
    8: "#A61E4D",
    16: "#6A0F30",
}
NON_MTP_COLOR = "#111111"
EAGLE3_COLOR = "#5A5A5A"
CONCURRENCY_LEGEND_MARKER_COLOR = "#C7C7C7"
ADAPTIVE_LINESTYLES = {
    3: "-",
    8: "--",
    16: ":",
}
CONF_HEX_LIST = [
    "#E8F8FF",
    "#D0F0FD",
    "#BEE9FA",
    "#A0D9EF",
    "#82C1F5",
    C_BLUE,
    "#6B7CEE",
    C_PURP,
    "#6A4CD1",
    "#5E35B1",
    "#512DA8",
    "#4527A0",
    "#311B92",
]

ADAPTIVE_COLOR_RANGE_BY_COUNT = {
    1: (0.55, 0.55),
    2: (0.50, 1.00),
    3: (0.35, 1.00),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument(
        "--include-strategy",
        action="append",
        default=[],
        help="Limit the plot to one or more strategy labels. Repeat to include multiple strategies.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
    )
    return parser.parse_args()


def parse_static_k(strategy: str) -> int | None:
    match = re.fullmatch(r"k=(\d+)", strategy)
    if match is None:
        return None
    return int(match.group(1))


def build_adaptive_threshold_color_map() -> dict[str, str]:
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_conf",
        CONF_HEX_LIST,
        N=len(THRESHOLDS),
    )
    return {thr: mcolors.to_hex(cmap(i)) for i, thr in enumerate(THRESHOLDS)}


ADAPTIVE_THRESHOLD_COLOR_MAP = build_adaptive_threshold_color_map()


def parse_tau_token(strategy: str) -> str | None:
    match = re.fullmatch(r"conf_adapt_k\d+_t(\d+)", strategy)
    if match is None:
        return None
    digits = match.group(1)
    value = int(digits) / 10.0
    while value > 1.0:
        value /= 10.0
    for threshold in THRESHOLDS:
        if abs(float(threshold) - value) < 1e-9:
            return threshold
    return None


def parse_adaptive_kmax(strategy: str) -> int | None:
    match = re.fullmatch(r"conf_adapt_k(\d+)_t\d+", strategy)
    if match is None:
        return None
    return int(match.group(1))


def strategy_color(strategy: str, adaptive_color_map: dict[str, str] | None = None) -> str:
    if strategy == "non-MTP":
        return NON_MTP_COLOR
    if strategy == "eagle3":
        return EAGLE3_COLOR
    static_match = re.fullmatch(r"k=(\d+)", strategy)
    if static_match is not None:
        return STATIC_COLOR_MAP.get(int(static_match.group(1)), "#555555")
    if adaptive_color_map is not None and strategy in adaptive_color_map:
        return adaptive_color_map[strategy]
    tau = parse_tau_token(strategy)
    if tau is not None:
        return ADAPTIVE_THRESHOLD_COLOR_MAP.get(tau, C_PURP)
    return "#555555"


def strategy_linestyle(strategy: str) -> str:
    return "-"


def c_value(row: dict[str, str]) -> int | None:
    try:
        return int(row["c"])
    except (KeyError, TypeError, ValueError):
        return None


def c1_latency_inv(strategy_rows: list[dict[str, str]]) -> float | None:
    for row in strategy_rows:
        if c_value(row) == 1:
            try:
                return float(row["pareto_x_latency_inv_tok_per_s_per_seq"])
            except (KeyError, TypeError, ValueError):
                return None
    return None


def strategy_display_order(grouped: dict[str, list[dict[str, str]]]) -> list[str]:
    strategies = set(grouped)
    ordered: list[str] = []

    if "eagle3" in strategies:
        ordered.append("eagle3")
    if "non-MTP" in strategies:
        ordered.append("non-MTP")

    static_strategies = sorted(
        (strategy for strategy in strategies if parse_static_k(strategy) is not None),
        key=lambda strategy: (parse_static_k(strategy) is None, parse_static_k(strategy)),
    )
    ordered.extend(static_strategies)

    confadapt_strategies = sorted(
        (strategy for strategy in strategies if strategy.startswith("conf_adapt")),
        key=lambda strategy: (
            c1_latency_inv(grouped[strategy]) is None,
            c1_latency_inv(grouped[strategy]) or float("inf"),
            strategy,
        ),
    )
    ordered.extend(confadapt_strategies)

    ordered_set = set(ordered)
    ordered.extend(sorted(strategy for strategy in strategies if strategy not in ordered_set))
    return ordered


def present_static_ks(strategies: list[str]) -> list[int]:
    return [k for k in STATIC_COLOR_MAP if f"k={k}" in strategies]


def present_adaptive_thresholds(strategies: list[str]) -> list[str]:
    present = {parse_tau_token(strategy) for strategy in strategies}
    return [threshold for threshold in THRESHOLDS if threshold in present]


def present_adaptive_strategies(strategies: list[str]) -> list[str]:
    return [strategy for strategy in strategies if strategy.startswith("conf_adapt")]


def present_adaptive_kmaxs(strategies: list[str]) -> list[int]:
    present = {parse_adaptive_kmax(strategy) for strategy in strategies}
    return [kmax for kmax in (3, 8, 16) if kmax in present]


def rgba_with_alpha(color: str) -> tuple[float, float, float, float]:
    return mcolors.to_rgba(color, alpha=PLOT_ALPHA)


def build_present_adaptive_color_map(strategies: list[str]) -> dict[str, str]:
    adaptive_strategies = present_adaptive_strategies(strategies)
    if not adaptive_strategies:
        return {}
    start, stop = ADAPTIVE_COLOR_RANGE_BY_COUNT.get(
        len(adaptive_strategies),
        (0.0, 1.0),
    )
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "ordered_conf_present",
        CONF_HEX_LIST,
        N=256,
    )
    positions = np.linspace(start, stop, len(adaptive_strategies))
    return {
        strategy: mcolors.to_hex(cmap(position))
        for strategy, position in zip(adaptive_strategies, positions)
    }


def format_confadapt_short_label(strategy: str) -> str:
    kmax = parse_adaptive_kmax(strategy)
    tau = parse_tau_token(strategy)
    if kmax is None or tau is None:
        return strategy
    return rf"k={kmax}, $\tau$={tau}"


def add_vertical_strategy_colorbars(
    fig: plt.Figure,
    strategies: list[str],
    adaptive_color_map: dict[str, str],
) -> tuple[list[object], list[float] | None]:
    extra_artists: list[object] = []
    static_ks = present_static_ks(strategies)
    adaptive_strategies = present_adaptive_strategies(strategies)
    static_rect: list[float] | None

    if static_ks and adaptive_strategies:
        conf_height = (
            PAIRED_CONF_CBAR_HEIGHT
            if PAIRED_CONF_CBAR_HEIGHT is not None
            else max(0.20, 0.035 * len(adaptive_strategies))
        )
        static_height = (
            PAIRED_STATIC_CBAR_HEIGHT
            if PAIRED_STATIC_CBAR_HEIGHT is not None
            else max(0.14, 0.030 * len(static_ks))
        )
        static_rect = [CBAR_X, PAIRED_STATIC_CBAR_Y, CBAR_WIDTH, static_height]
        conf_rect = [CBAR_X, PAIRED_CONF_CBAR_Y, CBAR_WIDTH, conf_height]
    elif adaptive_strategies:
        conf_height = (
            CONF_ONLY_CBAR_HEIGHT
            if CONF_ONLY_CBAR_HEIGHT is not None
            else max(0.24, 0.040 * len(adaptive_strategies))
        )
        conf_rect = [CBAR_X, CONF_ONLY_CBAR_Y, CBAR_WIDTH, conf_height]
        static_rect = None
    elif static_ks:
        conf_rect = None
        static_height = (
            STATIC_ONLY_CBAR_HEIGHT
            if STATIC_ONLY_CBAR_HEIGHT is not None
            else max(0.16, 0.034 * len(static_ks))
        )
        static_rect = [CBAR_X, STATIC_ONLY_CBAR_Y, CBAR_WIDTH, static_height]
    else:
        conf_rect = None
        static_rect = None

    if conf_rect is not None:
        conf_labels = [format_confadapt_short_label(strategy) for strategy in adaptive_strategies]
        conf_colors = [rgba_with_alpha(adaptive_color_map[strategy]) for strategy in adaptive_strategies]
        ax_cb_conf = fig.add_axes(conf_rect)
        cmap_conf = mcolors.ListedColormap(conf_colors)
        norm_conf = mcolors.BoundaryNorm(range(1, len(conf_colors) + 2), cmap_conf.N)
        cb_conf = fig.colorbar(
            cm.ScalarMappable(norm=norm_conf, cmap=cmap_conf),
            cax=ax_cb_conf,
            orientation="vertical",
        )
        cb_conf.set_label("ConfAdapt", fontsize=COLORBAR_LABEL_FONTSIZE, labelpad=8)
        tick_locs = np.arange(1.5, len(conf_colors) + 1.5)
        cb_conf.set_ticks(tick_locs)
        cb_conf.ax.set_yticklabels(conf_labels, fontsize=COLORBAR_CONF_TICK_FONTSIZE)
        cb_conf.ax.yaxis.set_ticks_position("right")
        cb_conf.ax.yaxis.set_label_position("right")
        extra_artists.append(cb_conf.ax)

    if static_rect is not None:
        static_colors = [rgba_with_alpha(STATIC_COLOR_MAP[k]) for k in static_ks]
        static_labels = [str(k) for k in static_ks]
        ax_cb_static = fig.add_axes(static_rect)
        cmap_static = mcolors.ListedColormap(static_colors)
        norm_static = mcolors.BoundaryNorm(range(1, len(static_colors) + 2), cmap_static.N)
        cb_static = fig.colorbar(
            cm.ScalarMappable(norm=norm_static, cmap=cmap_static),
            cax=ax_cb_static,
            orientation="vertical",
        )
        cb_static.set_label(r"Static $k$", fontsize=COLORBAR_LABEL_FONTSIZE, labelpad=8)
        tick_locs = np.arange(1.5, len(static_colors) + 1.5)
        cb_static.set_ticks(tick_locs)
        cb_static.ax.set_yticklabels(static_labels, fontsize=COLORBAR_STATIC_TICK_FONTSIZE)
        cb_static.ax.yaxis.set_ticks_position("right")
        cb_static.ax.yaxis.set_label_position("right")
        extra_artists.append(cb_static.ax)

    return extra_artists, static_rect


def main() -> int:
    args = parse_args()
    rows = list(csv.DictReader(args.input_tsv.open("r", encoding="utf-8"), delimiter="\t"))
    include_strategy_set = set(args.include_strategy)

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

    if include_strategy_set:
        grouped = {
            strategy: strategy_rows
            for strategy, strategy_rows in grouped.items()
            if strategy in include_strategy_set
        }

    strategies = strategy_display_order(grouped)
    if not strategies:
        raise RuntimeError(f"No plottable rows found in {args.input_tsv}")

    concurrency_values = sorted(
        {
            int(row["c"])
            for strategy in strategies
            for row in grouped[strategy]
            if row.get("c")
        }
    )
    adaptive_strategies = present_adaptive_strategies(strategies)
    adaptive_color_map = build_present_adaptive_color_map(strategies)
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]
    marker_by_c = {
        c: marker_cycle[i % len(marker_cycle)]
        for i, c in enumerate(concurrency_values)
    }

    # fig, ax = plt.subplots(figsize=(12.5, 7.0), dpi=220)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    all_x: list[float] = []
    all_y: list[float] = []

    for strategy in strategies:
        sr = sorted(grouped[strategy], key=lambda r: int(r["c"]))
        x = [float(r["pareto_x_latency_inv_tok_per_s_per_seq"]) for r in sr]
        y = [float(r["pareto_y_throughput_tok_per_s_gpu"]) for r in sr]
        cvals = [int(r["c"]) for r in sr]
        all_x.extend(x)
        all_y.extend(y)
        color = strategy_color(strategy, adaptive_color_map)

        if color != NON_MTP_COLOR:
            ax.plot(
                x,
                y,
                linewidth=PLOT_MARKER_EDGE_WIDTH,
                linestyle="-",
                color=PLOT_MARKER_EDGE_COLOR,
                alpha=CORELINE_ALPHA,
                zorder=2.2,
            )
        ax.plot(
            x,
            y,
            linewidth=PLOT_LINEWIDTH,
            linestyle=strategy_linestyle(strategy),
            color=color,
            alpha=PLOT_ALPHA,
            label=strategy,
            zorder=2.5,
        )
        for px, py, cval in zip(x, y, cvals):
            ax.scatter(
                [px],
                [py],
                marker=marker_by_c.get(cval, "o"),
                s=PLOT_MARKER_SIZE,
                color=color,
                edgecolors=PLOT_MARKER_EDGE_COLOR,
                linewidths=PLOT_MARKER_EDGE_WIDTH,
                alpha=PLOT_ALPHA,
                zorder=3,
            )

    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    xr = xmax - xmin
    yr = ymax - ymin
    ax.set_xlim(xmin - 0.05 * xr, xmax + 0.08 * xr)
    ax.set_ylim(max(0.0, ymin - 0.06 * yr), ymax + 0.10 * yr)

    if args.title:
        ax.set_title(args.title, fontsize=TITLE_FONTSIZE, fontweight="normal", pad=18)
    ax.set_xlabel(r"Latency$^{-1}$ (tok/sec/sequence)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="normal", labelpad=X_LABELPAD)
    ax.set_ylabel("Throughput (tok/sec)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="normal", labelpad=Y_LABELPAD)
    ax.grid(True, linestyle=":", alpha=0.6, color="#9a9a9a", linewidth=0.8)
    ax.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.2)
        ax.spines[side].set_color("#444444")

    method_handles: list[Line2D] = []
    if "non-MTP" in strategies:
        method_handles.append(
            Line2D(
                [0],
                [0],
                color=NON_MTP_COLOR,
                linewidth=PLOT_LINEWIDTH,
                alpha=PLOT_ALPHA,
                label="non-MTP",
            )
        )
    if "eagle3" in strategies:
        method_handles.append(
            Line2D(
                [0],
                [0],
                color=EAGLE3_COLOR,
                linewidth=PLOT_LINEWIDTH,
                alpha=PLOT_ALPHA,
                label="Eagle3",
            )
        )
    cbar_artists, static_rect = add_vertical_strategy_colorbars(fig, strategies, adaptive_color_map)
    methods_gap = (
        METHODS_GAP_ABOVE_STATIC_CBAR_WITH_EAGLE3
        if "eagle3" in strategies
        else METHODS_GAP_ABOVE_STATIC_CBAR_WITHOUT_EAGLE3
    )
    methods_y = (
        static_rect[1] + static_rect[3] + methods_gap
        if static_rect is not None
        else (METHODS_Y if adaptive_strategies else METHODS_Y_NO_CONFADAPT)
    )

    method_legend = None
    if method_handles:
        method_legend = fig.legend(
            handles=method_handles,
            title="Methods",
            loc="upper center",
            bbox_to_anchor=(
                PANEL_CENTER_X,
                methods_y,
            ),
            bbox_transform=fig.transFigure,
            frameon=False,
            fontsize=METHODS_LEGEND_FONTSIZE,
            title_fontsize=METHODS_LEGEND_TITLE_FONTSIZE,
        )
    elif SHOW_EMPTY_METHODS_HEADER:
        method_legend = fig.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    linestyle="None",
                    linewidth=0.0,
                    alpha=0.0,
                    label=" ",
                )
            ],
            title="Methods",
            loc="upper center",
            bbox_to_anchor=(
                PANEL_CENTER_X,
                methods_y,
            ),
            bbox_transform=fig.transFigure,
            frameon=False,
            fontsize=METHODS_LEGEND_FONTSIZE,
            title_fontsize=METHODS_LEGEND_TITLE_FONTSIZE,
            handlelength=0.0,
            handletextpad=0.0,
            borderaxespad=0.0,
        )
        for text in method_legend.get_texts():
            text.set_alpha(0.0)

    concurrency_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_by_c[c],
            color="none",
            markerfacecolor=CONCURRENCY_LEGEND_MARKER_COLOR,
            markeredgecolor=PLOT_MARKER_EDGE_COLOR,
            markersize=CONCURRENCY_MARKER_SIZE,
            linestyle="None",
            alpha=PLOT_ALPHA,
            label=f"c={c}",
        )
        for c in concurrency_values
    ]
    concurrency_legend = fig.legend(
        handles=concurrency_handles[::-1],
        title="Concurrency",
        loc="upper center",
        bbox_to_anchor=(
            PANEL_CENTER_X,
            CONCURRENCY_Y if adaptive_strategies else CONCURRENCY_Y_NO_CONFADAPT,
        ),
        bbox_transform=fig.transFigure,
        frameon=False,
        fontsize=CONCURRENCY_LEGEND_FONTSIZE,
        title_fontsize=CONCURRENCY_LEGEND_TITLE_FONTSIZE,
    )

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.10, right=RIGHT_MARGIN, bottom=0.14, top=0.96)
    extra_artists = [concurrency_legend, *cbar_artists]
    if method_legend is not None:
        extra_artists.append(method_legend)
    pdf_output = (
        args.output_png.with_suffix(".pdf")
        if args.output_png.suffix.lower() == ".png"
        else args.output_png.parent / f"{args.output_png.name}.pdf"
    )
    fig.savefig(
        args.output_png,
        bbox_inches="tight",
        bbox_extra_artists=tuple(extra_artists),
    )
    fig.savefig(
        pdf_output,
        bbox_inches="tight",
        bbox_extra_artists=tuple(extra_artists),
    )
    plt.close(fig)

    print(args.output_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
