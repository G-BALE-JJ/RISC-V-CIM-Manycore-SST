#!/usr/bin/env python3
"""Render three focused Row Engine cycle figures from one result artifact."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from plot_cycle_breakdown_1024x4096 import (
    COLORS,
    critical_path_from_ticks,
    cycle_offset_from_ticks,
    number,
    read_summary,
    save_source_data,
)


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Arial", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.9


def finish(fig, output_dir, stem):
    base = output_dir / stem
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def annotate_title(fig, title, subtitle=None):
    fig.suptitle(title, x=0.07, y=0.965, ha="left", fontsize=19,
                 fontweight="bold", color=COLORS["ink"])
    if subtitle:
        fig.text(0.07, 0.905, subtitle, ha="left", fontsize=10, color=COLORS["muted"])


def draw_stage_windows(result, output_dir):
    stages = result["stage_cycles"]
    windows = result["actual_stage_windows_cycles"]
    timeline = result["timeline_ticks"]
    observed = result["issue_to_completion_observed_cycles"]
    overlap = max(0, stages["sequential_active"] - windows["pipeline"])
    overlap_fraction = overlap / stages["sequential_active"]

    fig = plt.figure(figsize=(10.5, 5.4), facecolor="white")
    grid = fig.add_gridspec(
        1, 2, width_ratios=(14, 3.8), left=0.10, right=0.97,
        bottom=0.17, top=0.78, wspace=0.10,
    )
    ax = fig.add_subplot(grid[0])
    ax_active = fig.add_subplot(grid[1], sharey=ax)
    annotate_title(
        fig,
        "1024 x 4096 Softmax: causal Row Engine pipeline",
    )
    stage_rows = [
        ("Max", "first_max_start", "last_max_done", windows["max"], stages["max_active"], COLORS["max"]),
        ("Exp + sum", "first_exp_sum_start", "last_exp_sum_done", windows["exp_sum"], stages["exp_sum_active"], COLORS["exp"]),
        ("Normalize", "first_normalize_start", "last_normalize_done", windows["normalize"], stages["normalize_active"], COLORS["norm"]),
    ]
    for y, (label, start_event, end_event, window, active, color) in zip([2, 1, 0], stage_rows):
        start = cycle_offset_from_ticks(timeline, start_event)
        end = cycle_offset_from_ticks(timeline, end_event)
        ax.barh(y, end - start, left=start, height=0.52, color=color, edgecolor="white")
        ax.text((start + end) / 2, y, f"{label} window: {window:,} cycles",
                ha="center", va="center", color="white", fontsize=12, fontweight="bold")
        ax_active.text(0.90, y, f"active {active:,}", transform=ax_active.get_yaxis_transform(),
                       ha="right", va="center", fontsize=10, color=color)
    ax.axvline(observed, color=COLORS["ink"], linewidth=1.2, linestyle="--")
    ax.text(observed - 600, 2.38, f"actual completion {observed:,}",
            ha="right", va="bottom", fontsize=11, color=COLORS["ink"])
    ax.annotate(f"pipeline overlap = {overlap:,} cycles ({overlap_fraction:.1%} of active work)",
                xy=(30000, 2.7), xytext=(30000, 3.08), ha="center", fontsize=11,
                arrowprops={"arrowstyle": "-[,widthB=10.5,lengthB=0.6", "lw": 1.1,
                            "color": COLORS["muted"]})
    ax.hlines(-0.62, 0, observed, color=COLORS["transport"], linewidth=5)
    ax.plot(observed, -0.62, marker="o", color=COLORS["gap"], ms=5)
    ax.text(observed - 600, -0.62, "last output ACK + worker completion",
            ha="right", va="center", fontsize=10, color=COLORS["muted"])
    ax.set_xlim(-1200, observed * 1.04)
    ax.set_ylim(-1.03, 3.28)
    ax.set_yticks([2, 1, 0], ["Max", "Exp + sum", "Normalize"])
    ax.set_xlabel("Cycles from descriptor acceptance", fontsize=12)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.65)
    ax_active.set_xlim(0, 1)
    ax_active.set_xticks([])
    ax_active.set_yticks([])
    ax_active.text(0.90, 2.82, "Active cycles", transform=ax_active.get_yaxis_transform(),
                   ha="right", va="center", fontsize=10, color=COLORS["muted"], fontweight="bold")
    for spine in ax_active.spines.values():
        spine.set_visible(False)
    finish(fig, output_dir, "01_stage_pipeline_1024x4096")


def draw_critical_path(result, output_dir):
    observed = result["issue_to_completion_observed_cycles"]
    segments = critical_path_from_ticks(result["timeline_ticks"])
    colors = ["#A9A9A9", "#BFC4C8", COLORS["exp"], "#6F8DAD", COLORS["gap"]]
    dominant_index = max(range(len(segments)), key=lambda index: segments[index][1])
    dominant_label, dominant_cycles = segments[dominant_index]

    fig, ax = plt.subplots(figsize=(10.5, 6.0), facecolor="white")
    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.08, top=0.78)
    annotate_title(
        fig,
        "1024 x 4096 Softmax: observed issue-to-completion critical path",
        f"Five causal subsegments sum exactly to the measured {observed:,}-cycle latency",
    )
    ax.set_axis_off()
    ax.text(0.04, 0.87, "Observed critical path", transform=ax.transAxes,
            fontsize=11, color=COLORS["muted"], fontweight="bold")
    ax.text(0.04, 0.78, f"{observed:,} cycles", transform=ax.transAxes,
            fontsize=25, color=COLORS["ink"], fontweight="bold")
    ax.text(0.58, 0.87, "Dominant subtask", transform=ax.transAxes,
            fontsize=11, color=COLORS["muted"], fontweight="bold")
    ax.text(0.58, 0.78,
            f"{dominant_label}: {dominant_cycles:,} ({dominant_cycles / observed:.1%})",
            transform=ax.transAxes, fontsize=17, color=COLORS["exp"], fontweight="bold")

    bar_x, bar_y, bar_w, bar_h = 0.04, 0.67, 0.92, 0.055
    left = bar_x
    for (_, cycles), color in zip(segments, colors):
        width = bar_w * cycles / observed
        ax.add_patch(Rectangle((left, bar_y), width, bar_h, transform=ax.transAxes,
                               facecolor=color, edgecolor="white", linewidth=0.4))
        left += width
    ax.text(bar_x, bar_y - 0.045,
            "Descriptor accepted", transform=ax.transAxes, fontsize=10, color=COLORS["muted"])
    ax.text(bar_x + bar_w, bar_y - 0.045, "Actual accelerator completion",
            transform=ax.transAxes, ha="right", fontsize=10, color=COLORS["gap"], fontweight="bold")

    header_y = 0.53
    ax.text(0.05, header_y, "Step", transform=ax.transAxes, fontsize=10, color=COLORS["muted"], fontweight="bold")
    ax.text(0.13, header_y, "Critical-path subtask", transform=ax.transAxes, fontsize=10, color=COLORS["muted"], fontweight="bold")
    ax.text(0.78, header_y, "Span", transform=ax.transAxes, ha="right", fontsize=10, color=COLORS["muted"], fontweight="bold")
    ax.text(0.95, header_y, "Share", transform=ax.transAxes, ha="right", fontsize=10, color=COLORS["muted"], fontweight="bold")

    row_y = 0.47
    row_gap = 0.072
    for index, ((label, cycles), color) in enumerate(zip(segments, colors), start=1):
        y = row_y - (index - 1) * row_gap
        ax.add_patch(Rectangle((0.04, y - 0.020), 0.008, 0.039, transform=ax.transAxes,
                               facecolor=color, edgecolor="none"))
        ax.text(0.06, y, f"{index:02d}", transform=ax.transAxes, va="center", fontsize=10, color=COLORS["muted"])
        ax.text(0.13, y, label, transform=ax.transAxes, va="center", fontsize=12,
                color=COLORS["ink"], fontweight="bold" if index - 1 == dominant_index else "normal")
        ax.text(0.78, y, f"{cycles:,}", transform=ax.transAxes, ha="right", va="center", fontsize=12,
                color=color if color != "#BFC4C8" else COLORS["muted"], fontweight="bold")
        ax.text(0.95, y, f"{cycles / observed:.1%}", transform=ax.transAxes, ha="right", va="center", fontsize=11,
                color=COLORS["muted"])
        ax.plot([0.04, 0.96], [y - 0.035, y - 0.035], transform=ax.transAxes, color=COLORS["grid"], linewidth=0.6)
    finish(fig, output_dir, "02_critical_path_1024x4096")


def draw_optimization_boundary(result, summary, output_dir):
    stages = result["stage_cycles"]
    segments = critical_path_from_ticks(result["timeline_ticks"])
    observed = result["issue_to_completion_observed_cycles"]
    pipeline_cycles = dict(segments)["DMA-fed row pipeline"]
    values = [stages["max_active"], stages["exp_sum_active"], stages["normalize_active"]]
    labels = ["Max", "Exp + sum", "Normalize"]
    colors = [COLORS["max"], COLORS["exp"], COLORS["norm"]]
    total_active = sum(values)
    two_x_target = stages["exp_sum_active"] / 2

    fig, ax = plt.subplots(figsize=(10.5, 5.4), facecolor="white")
    fig.subplots_adjust(left=0.18, right=0.94, bottom=0.25, top=0.76)
    annotate_title(
        fig,
        "1024 x 4096 Softmax: bottleneck and optimization target",
        "EXP/SUM accounts for two-thirds of active service demand; the DMA-fed row pipeline dominates the measured path",
    )
    y_positions = [2, 1, 0]
    ax.barh(y_positions, values, color=colors, height=0.58)
    for y, value in zip(y_positions, values):
        label_x = value * 0.80 if value > 30000 else value * 0.50
        ax.text(label_x, y, f"{value:,} ({value / total_active:.1%})",
                ha="center", va="center", fontsize=10.5, color="white", fontweight="bold")
    ax.axvline(two_x_target, color=COLORS["target"], linewidth=1.6, linestyle="--")
    ax.text(two_x_target + 900, 2.42, "Near-term target\n2x EXP throughput",
            fontsize=10.5, color=COLORS["target"], va="top", fontweight="bold")
    ax.set_xlim(0, max(values) * 1.27)
    ax.set_ylim(-0.65, 2.55)
    ax.set_yticks(y_positions, labels)
    ax.set_xlabel("Aggregate active service cycles", fontsize=12)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.65)
    ax.text(0, -0.42,
            f"Measured path: row pipeline {pipeline_cycles:,}/{observed:,} cycles ({pipeline_cycles / observed:.1%}).  "
            f"Transport: {int(number(summary['dma_read_issue_count_sum'])):,} reads + "
            f"{int(number(summary['dma_write_issue_count_sum'])):,} writes; "
            f"max NoC port utilization {number(summary['noc_max_port_util_pct']):.2f}%.",
            fontsize=9.5, color=COLORS["ink"], va="top")
    finish(fig, output_dir, "03_optimization_boundary_1024x4096")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    summary = read_summary(args.run_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    segments = critical_path_from_ticks(result["timeline_ticks"])
    save_source_data(args.output_dir / "cycle_breakdown_1024x4096_source_data.csv",
                     result, summary, segments)
    draw_stage_windows(result, args.output_dir)
    draw_critical_path(result, args.output_dir)
    draw_optimization_boundary(result, summary, args.output_dir)
    print(f"saved split figures in {args.output_dir}")


if __name__ == "__main__":
    main()
