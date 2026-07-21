#!/usr/bin/env python3
"""Render a cycle-breakdown figure from a causal Row Engine result artifact."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Arial", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.9

COLORS = {
    "max": "#3775BA",
    "exp": "#0F4D92",
    "norm": "#42949E",
    "transport": "#A9A9A9",
    "gap": "#B64342",
    "ink": "#272727",
    "muted": "#767676",
    "grid": "#D9D9D9",
    "target": "#2E9E44",
}


def read_summary(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty summary: {path}")
    return rows[-1]


def number(value):
    return float(value) if value else 0.0


def cycle_offset_from_ticks(timeline_ticks, event, clock_hz=2_300_000_000,
                            ticks_per_second=1_000_000_000_000):
    """Convert an event timestamp to cycles from descriptor acceptance."""
    return math.ceil(
        (timeline_ticks[event] - timeline_ticks["descriptor_accept"])
        * clock_hz / ticks_per_second
    )


def critical_path_from_ticks(timeline_ticks, clock_hz=2_300_000_000,
                             ticks_per_second=1_000_000_000_000):
    """Build non-overlapping path segments from one cumulatively rounded origin."""
    endpoints = [
        ("Band to first worker", "first_worker_dispatch"),
        ("First input DMA", "first_input_dma_ready"),
        ("DMA-fed row pipeline", "last_compute_done"),
        ("Final output DMA", "final_output_dma_ack"),
        ("Completion delivery", "accelerator_complete"),
    ]
    previous = 0
    segments = []
    for label, event in endpoints:
        endpoint = cycle_offset_from_ticks(
            timeline_ticks, event, clock_hz, ticks_per_second
        )
        segments.append((label, endpoint - previous))
        previous = endpoint
    return segments


def save_source_data(path, result, summary, segments):
    actual_windows = result["actual_stage_windows_cycles"]
    stages = result["stage_cycles"]
    overlap = max(0, stages["sequential_active"] - actual_windows["pipeline"])
    rows = [
        ("configuration", "shape", f"{result['rows']}x{result['cols']}", ""),
        ("configuration", "physical_sfus", result["physical_sfus_with_stats"], ""),
        ("latency", "analytical_compute_cycles", result["analytical_compute_cycles"], "cycles"),
        ("latency", "actual_accelerator_completion", result["accelerator_latency_cycles"], "cycles"),
        ("latency", "kernel_window", result["kernel_window_cycles"], "cycles"),
        ("stage", "max_active", stages["max_active"], "cycles"),
        ("stage", "exp_sum_active", stages["exp_sum_active"], "cycles"),
        ("stage", "normalize_active", stages["normalize_active"], "cycles"),
        ("stage_window", "max", actual_windows["max"], "cycles"),
        ("stage_window", "exp_sum", actual_windows["exp_sum"], "cycles"),
        ("stage_window", "normalize", actual_windows["normalize"], "cycles"),
        ("stage_window", "pipeline", actual_windows["pipeline"], "cycles"),
        ("stage", "overlap_cycles", overlap, "cycles"),
        ("transport", "dma_read_issues", summary["dma_read_issue_count_sum"], "count"),
        ("transport", "dma_write_issues", summary["dma_write_issue_count_sum"], "count"),
        ("transport", "noc_max_port_util", summary["noc_max_port_util_pct"], "percent"),
    ]
    rows.extend(("critical_path", label, cycles, "cycles") for label, cycles in segments)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "metric", "value", "unit"])
        writer.writerows(rows)


def panel_label(ax, label):
    ax.text(-0.065, 1.04, label, transform=ax.transAxes, fontsize=15,
            fontweight="bold", ha="left", va="bottom", color=COLORS["ink"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    summary = read_summary(args.run_summary)
    stages = result["stage_cycles"]
    actual_windows = result["actual_stage_windows_cycles"]
    observed = result["issue_to_completion_observed_cycles"]

    critical_segments = critical_path_from_ticks(result["timeline_ticks"])
    if sum(cycles for _, cycles in critical_segments) != observed:
        raise ValueError("critical-path segments do not sum to observed latency")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_source_data(args.output_dir / "cycle_breakdown_1024x4096_source_data.csv",
                     result, summary, critical_segments)

    fig = plt.figure(figsize=(13.333, 7.5), facecolor="white")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.14, hspace=0.38, wspace=0.55)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.8, 1), height_ratios=(1.35, 1))
    ax_timeline = fig.add_subplot(grid[0, :])
    ax_path = fig.add_subplot(grid[1, 0])
    ax_targets = fig.add_subplot(grid[1, 1])

    fig.suptitle("1024 x 4096 Softmax Row Engine: cycle breakdown and optimization boundary",
                 x=0.04, y=0.965, ha="left", fontsize=19, fontweight="bold", color=COLORS["ink"])
    fig.text(0.04, 0.915,
             "16 physical SFUs | 4 row contexts/SFU | 1 row per DMA | 2.3 GHz | 1200 GB/s NoC",
             ha="left", fontsize=9.5, color=COLORS["muted"])

    tick_data = result["timeline_ticks"]
    stage_rows = [
        ("Max", "first_max_start", "last_max_done", actual_windows["max"], COLORS["max"]),
        ("Exp + sum", "first_exp_sum_start", "last_exp_sum_done", actual_windows["exp_sum"], COLORS["exp"]),
        ("Normalize", "first_normalize_start", "last_normalize_done", actual_windows["normalize"], COLORS["norm"]),
    ]
    ypos = [2, 1, 0]
    for y, (label, start_event, end_event, window, color) in zip(ypos, stage_rows):
        start = cycle_offset_from_ticks(tick_data, start_event)
        end = cycle_offset_from_ticks(tick_data, end_event)
        ax_timeline.barh(y, end - start, left=start, height=0.5, color=color, edgecolor="white")
        ax_timeline.text((start + end) / 2, y, f"{label} window: {window:,} cycles",
                         ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    overlap = max(0, stages["sequential_active"] - actual_windows["pipeline"])
    overlap_fraction = overlap / stages["sequential_active"]
    ax_timeline.axvline(observed, color=COLORS["ink"], linewidth=1.1, linestyle="--")
    ax_timeline.text(observed - 500, 2.35,
                     f"actual completion\n{observed:,}", ha="right", fontsize=10,
                     color=COLORS["ink"], va="top")
    ax_timeline.annotate(f"pipeline overlap = {overlap:,} cycles ({overlap_fraction:.1%})",
                         xy=(30000, 2.72), xytext=(30000, 3.12), ha="center", fontsize=10,
                         arrowprops={"arrowstyle": "-[,widthB=10.5,lengthB=0.6", "lw": 1.1,
                                     "color": COLORS["muted"]})
    ax_timeline.hlines(-0.62, 0, observed, color=COLORS["transport"], linewidth=5)
    ax_timeline.plot(observed, -0.62, marker="o", color=COLORS["gap"], ms=5)
    ax_timeline.text(observed - 500, -0.62, "output ACK + worker completion",
                     ha="right", va="center", fontsize=9, color=COLORS["muted"])
    ax_timeline.set_xlim(-1200, observed * 1.04)
    ax_timeline.set_ylim(-1.02, 3.32)
    ax_timeline.set_yticks(ypos, [row[0] for row in stage_rows])
    ax_timeline.set_xlabel("Cycles from descriptor acceptance")
    ax_timeline.set_title("Measured causal stage windows (stage activity is not additive latency)",
                          loc="left", x=0.06, y=1.02, fontsize=12, pad=12, color=COLORS["ink"])
    ax_timeline.text(1.0, 1.02,
                     "active cycles: Max 16,384 | Exp+sum 65,536 | Normalize 16,384",
                     transform=ax_timeline.transAxes, ha="right", va="bottom", fontsize=8.5,
                     color=COLORS["muted"])
    ax_timeline.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    panel_label(ax_timeline, "a")

    # Panel B: exact, non-overlapping path segments sum to observed latency.
    left = 0
    path_colors = [COLORS["transport"], "#BFC4C8", COLORS["exp"], "#6F8DAD", COLORS["gap"]]
    for (label, cycles), color in zip(critical_segments, path_colors):
        ax_path.barh(0, cycles, left=left, color=color, edgecolor="white", height=0.55)
        if cycles >= 900:
            ax_path.text(left + cycles / 2, 0, f"{cycles / 1000:.1f}k", ha="center", va="center",
                         fontsize=9, color="white" if color in {COLORS["gap"], "#4D6E91"} else COLORS["ink"],
                         fontweight="bold")
        left += cycles
    ax_path.set_xlim(0, observed * 1.03)
    ax_path.set_yticks([])
    ax_path.set_xlabel("Observed issue-to-completion latency (cycles)")
    ax_path.set_title(f"Critical path = {observed:,} cycles", loc="left", x=0.08, y=1.02, fontsize=12, pad=10)
    ax_path.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax_path.legend([Patch(facecolor=color) for color in path_colors],
                   [label for label, _ in critical_segments],
                   ncol=2, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.32), handlelength=1.1)
    panel_label(ax_path, "b")

    active_values = [stages["max_active"], stages["exp_sum_active"], stages["normalize_active"]]
    active_labels = ["Max", "Exp + sum", "Normalize"]
    active_colors = [COLORS["max"], COLORS["exp"], COLORS["norm"]]
    ax_targets.barh([2, 1, 0], active_values, color=active_colors, height=0.55)
    for y, value in zip([2, 1, 0], active_values):
        ax_targets.text(value + 1500, y, f"{value:,}", va="center", fontsize=9,
                        color=COLORS["ink"], fontweight="bold")
    ax_targets.set_yticks([2, 1, 0], active_labels)
    ax_targets.set_xlim(0, max(active_values) * 1.22)
    ax_targets.set_xlabel("Aggregate active service cycles")
    ax_targets.set_title("EXP/SUM service demand is 4x either vector stage",
                         loc="left", x=0.02, y=1.02, fontsize=11, pad=10)
    ax_targets.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    panel_label(ax_targets, "c")

    stem = args.output_dir / "cycle_breakdown_1024x4096"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    print(f"saved {stem.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
