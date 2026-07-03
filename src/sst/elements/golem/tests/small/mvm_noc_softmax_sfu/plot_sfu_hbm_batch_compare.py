#!/usr/bin/env python3
"""Collect and plot SFU HBM primitive batch/non-batch comparison results."""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path


PASS_RE = re.compile(r"\[SOFTMAX\].*mode=sfu-primitive-hbm-stream.*PASS")
KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
TIME_RE = re.compile(r"^\s*([0-9.]+)\s*(us|ms|s)\s*$")


@dataclass
class Row:
    run_id: str
    mode: str
    total_elems: int
    chunk_elems: int
    chunks: int
    processed_elems: int
    hbm_read_bytes: int
    hbm_write_bytes: int
    dma_read_issue_count: int
    dma_write_issue_count: int
    dma_wait_count: int
    sfu_ops_issued: int
    sfu_primitive_elems: int
    simulated_time_us: float
    wall_time_sec: float

    @property
    def hbm_stream_bytes(self) -> int:
        return self.hbm_read_bytes + self.hbm_write_bytes

    @property
    def issue_reduction_vs_chunks(self) -> float:
        if self.sfu_ops_issued <= 0:
            return 0.0
        return self.chunks / self.sfu_ops_issued


def parse_time_to_us(value: str) -> float:
    match = TIME_RE.match(value.strip())
    if not match:
        return 0.0
    number = float(match.group(1))
    unit = match.group(2)
    if unit == "us":
        return number
    if unit == "ms":
        return number * 1000.0
    return number * 1_000_000.0


def parse_mode(run_id: str) -> str:
    if "_nonbatch_" in run_id:
        return "non-batch"
    if "_batch_" in run_id:
        return "batch"
    return "unknown"


def read_run_summary(root: Path) -> dict[str, dict[str, str]]:
    path = root / "stats" / "run_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing run summary: {path}")
    with path.open(newline="") as handle:
        return {row["run_id"]: row for row in csv.DictReader(handle)}


def read_metric_sum(path: Path, metric: str) -> int:
    if not path.exists():
        return 0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("metric") == metric:
                return int(float(row.get("sum", "0") or 0))
    return 0


def read_sfu_stat_sum(path: Path, metric: str) -> int:
    if not path.exists():
        return 0
    total = 0
    with path.open(errors="replace") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) < 7 or parts[1] != metric:
                continue
            total += int(float(parts[6]))
    return total


def parse_pass_line(stdout_dir: Path) -> dict[str, str] | None:
    if not stdout_dir.exists():
        return None
    for path in sorted(stdout_dir.glob("stdout-*")):
        with path.open(errors="replace") as handle:
            for line in handle:
                if PASS_RE.search(line):
                    return dict(KV_RE.findall(line))
    return None


def collect_rows(root: Path) -> list[Row]:
    rows: list[Row] = []
    for run_id, summary in sorted(read_run_summary(root).items()):
        mode = parse_mode(run_id)
        if mode == "unknown":
            continue
        pass_kv = parse_pass_line(root / "stdout" / "overlap0" / run_id)
        if not pass_kv:
            continue
        stats_dir = root / "stats" / "overlap0" / run_id
        dma_summary = stats_dir / "dma_summary.csv"
        stats_selfcom = stats_dir / "stats_selfcom.txt"
        rows.append(
            Row(
                run_id=run_id,
                mode=mode,
                total_elems=int(pass_kv["total_elems"]),
                chunk_elems=int(pass_kv["chunk_elems"]),
                chunks=int(pass_kv["chunks"]),
                processed_elems=int(pass_kv["processed_elems"]),
                hbm_read_bytes=int(pass_kv["hbm_read_bytes"]),
                hbm_write_bytes=int(pass_kv["hbm_write_bytes"]),
                dma_read_issue_count=read_metric_sum(dma_summary, "read_issue_count"),
                dma_write_issue_count=read_metric_sum(dma_summary, "write_issue_count"),
                dma_wait_count=read_metric_sum(dma_summary, "wait_count"),
                sfu_ops_issued=read_sfu_stat_sum(stats_selfcom, "sfu_ops_issued"),
                sfu_primitive_elems=read_sfu_stat_sum(stats_selfcom, "sfu_primitive_elems"),
                simulated_time_us=parse_time_to_us(summary.get("simulated_time", "")),
                wall_time_sec=float(summary.get("wall_time_sec", 0) or 0),
            )
        )
    rows.sort(key=lambda row: (row.total_elems, row.mode))
    if not rows:
        raise RuntimeError(f"no completed batch comparison rows found under {root}")
    return rows


def write_csv(rows: list[Row], path: Path) -> None:
    names = [
        "run_id",
        "mode",
        "total_elems",
        "chunk_elems",
        "chunks",
        "processed_elems",
        "hbm_read_bytes",
        "hbm_write_bytes",
        "hbm_stream_bytes",
        "dma_read_issue_count",
        "dma_write_issue_count",
        "dma_wait_count",
        "sfu_ops_issued",
        "sfu_primitive_elems",
        "issue_reduction_vs_chunks",
        "simulated_time_us",
        "wall_time_sec",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in names})


def write_notes(rows: list[Row], path: Path) -> None:
    lines = [
        "# SFU HBM batch comparison notes",
        "",
        "- Workload: single `EXP` primitive over HBM C-region stream.",
        "- Fixed chunk size: `chunk_elems=256`.",
        "- Claim: batch descriptors reduce SFU/RoCC issue count without changing HBM bytes or DMA request count.",
        "",
        "| elems | mode | chunks | SFU issue | DMA R/W | HBM bytes R/W | sim time (us) | wall (s) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.total_elems} | {row.mode} | {row.chunks} | "
            f"{row.sfu_ops_issued} | {row.dma_read_issue_count}/{row.dma_write_issue_count} | "
            f"{row.hbm_read_bytes}/{row.hbm_write_bytes} | "
            f"{row.simulated_time_us:.3f} | {row.wall_time_sec:.0f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- For 1024 elements, batch changes SFU issue count from 4 to 1 while DMA read/write stays 4/4.",
        "- For 4096 elements, batch changes SFU issue count from 16 to 1 while DMA read/write stays 16/16.",
        "- Simulated time is dominated by the same HBM/DMA stream in this small single-op setup; batch mainly proves the control-event reduction path.",
    ]
    path.write_text("\n".join(lines) + "\n")


def plot(rows: list[Row], out_prefix: Path) -> None:
    mpl_config = out_prefix.parent / ".mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )

    elems = sorted({row.total_elems for row in rows})
    modes = ["non-batch", "batch"]
    colors = {"non-batch": "#4c78a8", "batch": "#e68632"}
    width = 0.34
    x = list(range(len(elems)))

    by_key = {(row.total_elems, row.mode): row for row in rows}

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    fig.suptitle("Batched SFU primitive descriptors reduce control issues", fontweight="bold")

    def grouped_bars(ax, metric: str, ylabel: str, title: str, annotate: bool = False) -> None:
        for offset, mode in [(-width / 2, modes[0]), (width / 2, modes[1])]:
            values = [getattr(by_key[(elem, mode)], metric) for elem in elems]
            bars = ax.bar(
                [pos + offset for pos in x],
                values,
                width=width,
                color=colors[mode],
                label=mode,
            )
            if annotate:
                for bar, value in zip(bars, values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{value:g}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                    )
        ax.set_xticks(x, [str(elem) for elem in elems])
        ax.set_xlabel("elements per op")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    grouped_bars(
        axes[0][0],
        "sfu_ops_issued",
        "SFU issues",
        "Control issue count",
        annotate=True,
    )
    grouped_bars(
        axes[0][1],
        "dma_read_issue_count",
        "DMA read issues",
        "HBM DMA requests unchanged",
        annotate=True,
    )

    ax = axes[1][0]
    for mode in modes:
        ax.plot(
            elems,
            [by_key[(elem, mode)].wall_time_sec for elem in elems],
            marker="o",
            color=colors[mode],
            label=mode,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("elements per op")
    ax.set_ylabel("wall time (s)")
    ax.set_title("Wall-clock runtime")
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[1][1]
    for mode in modes:
        ax.plot(
            elems,
            [by_key[(elem, mode)].simulated_time_us for elem in elems],
            marker="s",
            color=colors[mode],
            label=mode,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("elements per op")
    ax.set_ylabel("simulated time (us)")
    ax.set_title("Simulated time")
    ax.grid(True, which="both", alpha=0.25)

    axes[0][0].legend(loc="upper left")
    fig.savefig(f"{out_prefix}.svg", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_prefix}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    out_dir = args.out_dir or args.sweep_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(args.sweep_root)
    csv_path = out_dir / "sfu_hbm_batch_compare_source.csv"
    notes_path = out_dir / "sfu_hbm_batch_compare_notes.md"
    out_prefix = out_dir / "sfu_hbm_batch_compare"
    write_csv(rows, csv_path)
    write_notes(rows, notes_path)
    plot(rows, out_prefix)
    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {notes_path}")
    print(f"[OK] wrote {out_prefix}.svg/.pdf/.png/.tiff")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
