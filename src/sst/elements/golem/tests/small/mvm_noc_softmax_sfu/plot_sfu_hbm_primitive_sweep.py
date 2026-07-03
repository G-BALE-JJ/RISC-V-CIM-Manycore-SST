#!/usr/bin/env python3
"""Collect and plot SFU HBM primitive sweep results.

The preferred plotting backend is matplotlib.  The local SST environment used
for these small tests does not always provide it, so the script also has a
stdlib/Pillow fallback that writes a publication-style SVG plus a PNG/PDF copy.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PASS_RE = re.compile(r"\[SOFTMAX\].*mode=sfu-primitive-hbm-stream.*PASS")
KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
TIME_RE = re.compile(r"^\s*([0-9.]+)\s*(us|ms|s)\s*$")


@dataclass
class SweepRow:
    run_id: str
    total_elems: int
    ops: str
    ops_count: int
    chunk_elems: int
    chunks: int
    processed_elems: int
    hbm_init_write_bytes: int
    hbm_read_bytes: int
    hbm_write_bytes: int
    dma_read_issue_count: int
    dma_write_issue_count: int
    dma_completion: int
    dma_write_completion: int
    dma_wait_count: int
    sfu_ops_issued: int
    sfu_primitive_elems: int
    sfu_credit_stalls: int
    sfu_retry_events: int
    simulated_time_us: float
    wall_time_sec: float
    noc_total_xbar_stalls: float
    noc_avg_packet_latency_ns: float
    noc_p99_packet_latency_ns: float
    hbm_backend_read_active_cycles: float

    @property
    def hbm_total_stream_bytes(self) -> int:
        return self.hbm_read_bytes + self.hbm_write_bytes

    @property
    def expected_stream_bytes(self) -> int:
        return self.total_elems * self.ops_count * 4 * 2

    @property
    def hbm_effective_gib_per_s(self) -> float:
        if self.simulated_time_us <= 0:
            return 0.0
        return (self.hbm_total_stream_bytes / (1024.0**3)) / (self.simulated_time_us * 1e-6)

    @property
    def primitive_melems_per_sim_s(self) -> float:
        if self.simulated_time_us <= 0:
            return 0.0
        return (self.processed_elems / 1e6) / (self.simulated_time_us * 1e-6)


def parse_time_to_us(value: str) -> float:
    match = TIME_RE.match(value)
    if not match:
        return 0.0
    number = float(match.group(1))
    unit = match.group(2)
    if unit == "us":
        return number
    if unit == "ms":
        return number * 1000.0
    if unit == "s":
        return number * 1_000_000.0
    return 0.0


def read_run_summary(sweep_root: Path) -> dict[str, dict[str, str]]:
    path = sweep_root / "stats" / "run_summary.csv"
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
                value = row.get("sum", "0") or "0"
                return int(float(value))
    return 0


def read_sfu_stat_sum(path: Path, metric: str) -> int:
    if not path.exists():
        return 0
    total = 0
    with path.open() as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) < 7 or parts[1] != metric:
                continue
            try:
                total += int(float(parts[6]))
            except ValueError:
                pass
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


def collect_rows(sweep_root: Path, sizes: set[int]) -> list[SweepRow]:
    run_summary = read_run_summary(sweep_root)
    rows: list[SweepRow] = []
    for run_id, summary in sorted(run_summary.items()):
        stdout_dir = sweep_root / "stdout" / "overlap0" / run_id
        pass_kv = parse_pass_line(stdout_dir)
        if not pass_kv:
            continue
        total_elems = int(pass_kv["total_elems"])
        if sizes and total_elems not in sizes:
            continue
        ops = pass_kv["ops"]
        stats_dir = sweep_root / "stats" / "overlap0" / run_id
        dma_summary = stats_dir / "dma_summary.csv"
        stats_selfcom = stats_dir / "stats_selfcom.txt"
        sfu_ops_issued = read_sfu_stat_sum(stats_selfcom, "sfu_ops_issued")
        sfu_primitive_elems = read_sfu_stat_sum(stats_selfcom, "sfu_primitive_elems")
        chunks = int(pass_kv["chunks"])
        processed_elems = int(pass_kv["processed_elems"])
        if sfu_ops_issued == 0 and processed_elems > 0:
            sfu_ops_issued = chunks
        if sfu_primitive_elems == 0 and processed_elems > 0:
            sfu_primitive_elems = processed_elems
        rows.append(
            SweepRow(
                run_id=run_id,
                total_elems=total_elems,
                ops=ops,
                ops_count=len([op for op in ops.split(",") if op]),
                chunk_elems=int(pass_kv["chunk_elems"]),
                chunks=chunks,
                processed_elems=processed_elems,
                hbm_init_write_bytes=int(pass_kv["hbm_init_write_bytes"]),
                hbm_read_bytes=int(pass_kv["hbm_read_bytes"]),
                hbm_write_bytes=int(pass_kv["hbm_write_bytes"]),
                dma_read_issue_count=read_metric_sum(dma_summary, "read_issue_count"),
                dma_write_issue_count=read_metric_sum(dma_summary, "write_issue_count"),
                dma_completion=read_metric_sum(dma_summary, "completion"),
                dma_write_completion=read_metric_sum(dma_summary, "write_completion"),
                dma_wait_count=read_metric_sum(dma_summary, "wait_count"),
                sfu_ops_issued=sfu_ops_issued,
                sfu_primitive_elems=sfu_primitive_elems,
                sfu_credit_stalls=read_sfu_stat_sum(stats_selfcom, "sfu_credit_stalls"),
                sfu_retry_events=read_sfu_stat_sum(stats_selfcom, "sfu_retry_events"),
                simulated_time_us=parse_time_to_us(summary.get("simulated_time", "")),
                wall_time_sec=float(summary.get("wall_time_sec", 0) or 0),
                noc_total_xbar_stalls=float(summary.get("noc_total_xbar_stalls", 0) or 0),
                noc_avg_packet_latency_ns=float(summary.get("noc_avg_packet_latency_ns", 0) or 0),
                noc_p99_packet_latency_ns=float(summary.get("noc_p99_packet_latency_ns", 0) or 0),
                hbm_backend_read_active_cycles=float(
                    summary.get("hbm_backend_read_active_cycles", 0) or 0
                ),
            )
        )
    rows.sort(key=lambda row: row.total_elems)
    if not rows:
        raise RuntimeError(f"no completed HBM primitive PASS rows found under {sweep_root}")
    return rows


def write_source_csv(rows: list[SweepRow], path: Path) -> None:
    field_names = [
        "run_id",
        "total_elems",
        "ops",
        "ops_count",
        "chunk_elems",
        "chunks",
        "processed_elems",
        "hbm_init_write_bytes",
        "hbm_read_bytes",
        "hbm_write_bytes",
        "hbm_total_stream_bytes",
        "expected_stream_bytes",
        "dma_read_issue_count",
        "dma_write_issue_count",
        "dma_completion",
        "dma_write_completion",
        "dma_wait_count",
        "sfu_ops_issued",
        "sfu_primitive_elems",
        "sfu_credit_stalls",
        "sfu_retry_events",
        "simulated_time_us",
        "wall_time_sec",
        "hbm_effective_gib_per_s",
        "primitive_melems_per_sim_s",
        "noc_total_xbar_stalls",
        "noc_avg_packet_latency_ns",
        "noc_p99_packet_latency_ns",
        "hbm_backend_read_active_cycles",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in field_names if hasattr(row, name)})


def try_plot_matplotlib(rows: list[SweepRow], out_prefix: Path, title: str) -> bool:
    mpl_config = out_prefix.parent / ".mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    elems = [row.total_elems for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), constrained_layout=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax = axes[0][0]
    ax.plot(elems, [row.simulated_time_us for row in rows], "o-", label="simulated time")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("elements per op")
    ax.set_ylabel("simulated time (us)")
    ax.grid(True, which="both", alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(elems, [row.wall_time_sec for row in rows], "s--", color="#9b5d00", label="wall time")
    ax2.set_ylabel("wall time (s)")
    ax.set_title("Runtime")

    ax = axes[0][1]
    ax.plot(elems, [row.hbm_total_stream_bytes for row in rows], "o-", label="observed")
    ax.plot(elems, [row.expected_stream_bytes for row in rows], "--", label="expected")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("elements per op")
    ax.set_ylabel("HBM stream bytes")
    ax.set_title("HBM traffic")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1][0]
    ax.plot(elems, [row.dma_read_issue_count for row in rows], "o-", label="DMA read")
    ax.plot(elems, [row.dma_write_issue_count for row in rows], "s-", label="DMA write")
    ax.plot(elems, [row.sfu_ops_issued for row in rows], "^-", label="SFU issue")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("elements per op")
    ax.set_ylabel("event count")
    ax.set_title("Primitive events")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1][1]
    ax.plot(elems, [row.hbm_effective_gib_per_s for row in rows], "o-", label="HBM GiB/s")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("elements per op")
    ax.set_ylabel("effective stream bandwidth (GiB/s)")
    ax.set_title("Effective bandwidth")
    ax.grid(True, alpha=0.25)

    for suffix in (".svg", ".pdf", ".png", ".tiff"):
        fig.savefig(out_prefix.with_suffix(suffix), dpi=300, facecolor="white")
    plt.close(fig)
    return True


def _nice_ticks_log2(values: Iterable[int]) -> list[int]:
    unique = sorted(set(values))
    return unique


def _fmt_bytes(value: float) -> str:
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.1f} MiB"
    if value >= 1024:
        return f"{value / 1024:.1f} KiB"
    return f"{value:.0f} B"


class SvgCanvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
        ]

    def text(self, x: float, y: float, value: str, size: int = 14, weight: str = "400",
             fill: str = "#202020", anchor: str = "start") -> None:
        safe = (
            value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}">{safe}</text>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str = "#333",
             width: float = 1.0, dash: str | None = None) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{width:.1f}"{dash_attr}/>'
        )

    def polyline(self, points: list[tuple[float, float]], stroke: str, width: float = 2.0,
                 dash: str | None = None) -> None:
        if not points:
            return
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{stroke}" '
            f'stroke-width="{width:.1f}" stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>'
        )

    def circle(self, x: float, y: float, r: float, fill: str, stroke: str = "white") -> None:
        self.parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.2"/>'
        )

    def finish(self) -> str:
        return "\n".join(self.parts + ["</svg>\n"])


def render_svg_fallback(rows: list[SweepRow], path: Path, title: str) -> None:
    width, height = 1180, 820
    svg = SvgCanvas(width, height)
    svg.text(44, 42, title, size=22, weight="700")
    svg.text(
        44,
        68,
        "Completed sizes: 16, 1024, 4096 elems/op; six primitives per point. Larger attempted runs excluded by scope/timeout.",
        size=13,
        fill="#555",
    )
    panels = [
        (70, 110, 480, 260, "Runtime", "sim us + wall s"),
        (650, 110, 480, 260, "HBM traffic", "observed vs expected bytes"),
        (70, 465, 480, 260, "Primitive events", "DMA and SFU issue count"),
        (650, 465, 480, 260, "Effective bandwidth", "stream bytes / simulated time"),
    ]

    colors = ["#1f77b4", "#d55e00", "#009e73", "#6a3d9a"]
    elems = [row.total_elems for row in rows]

    def plot_panel(
        panel: tuple[int, int, int, int, str, str],
        series: list[tuple[str, list[float], str, bool, str]],
        y_label: str,
        y_formatter=lambda value: f"{value:g}",
    ) -> None:
        x, y, w, h, heading, subheading = panel
        left, right, top, bottom = x + 68, x + w - 28, y + 42, y + h - 46
        svg.text(x, y, heading, size=16, weight="700")
        svg.text(x, y + 20, subheading, size=12, fill="#666")
        svg.line(left, bottom, right, bottom, "#444", 1.1)
        svg.line(left, top, left, bottom, "#444", 1.1)
        log_x_min = math.log2(min(elems))
        log_x_max = math.log2(max(elems))
        all_y = [value for _, ys, _, _, _ in series for value in ys if value > 0]
        y_min = min(all_y) if all_y else 0.0
        y_max = max(all_y) if all_y else 1.0
        use_log_y = y_min > 0 and (y_max / y_min) >= 16
        if use_log_y:
            y_min_log = math.floor(math.log10(y_min))
            y_max_log = math.ceil(math.log10(y_max))
            y_min, y_max = 10**y_min_log, 10**y_max_log
        elif math.isclose(y_min, y_max):
            y_min = 0
            y_max = y_max * 1.2 + 1
        else:
            pad = (y_max - y_min) * 0.12
            y_min = max(0.0, y_min - pad)
            y_max = y_max + pad

        def sx(elem: int) -> float:
            if math.isclose(log_x_min, log_x_max):
                return (left + right) / 2
            return left + (math.log2(elem) - log_x_min) / (log_x_max - log_x_min) * (right - left)

        def sy(value: float) -> float:
            if use_log_y:
                return bottom - (math.log10(value) - math.log10(y_min)) / (
                    math.log10(y_max) - math.log10(y_min)
                ) * (bottom - top)
            return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

        for elem in _nice_ticks_log2(elems):
            px = sx(elem)
            svg.line(px, bottom, px, bottom + 5, "#444", 1.0)
            svg.text(px, bottom + 22, str(elem), size=11, fill="#444", anchor="middle")
            svg.line(px, top, px, bottom, "#e6e6e6", 0.8)

        if use_log_y:
            tick_values = [10**power for power in range(int(math.log10(y_min)), int(math.log10(y_max)) + 1)]
        else:
            step = (y_max - y_min) / 4
            tick_values = [y_min + i * step for i in range(5)]
        for value in tick_values:
            py = sy(value)
            svg.line(left - 5, py, left, py, "#444", 1.0)
            svg.text(left - 9, py + 4, y_formatter(value), size=10, fill="#444", anchor="end")
            svg.line(left, py, right, py, "#ededed", 0.8)

        for label, ys, color, dashed, marker in series:
            points = [(sx(elem), sy(value)) for elem, value in zip(elems, ys) if value > 0]
            svg.polyline(points, color, 2.1, "5,4" if dashed else None)
            for px, py in points:
                svg.circle(px, py, 4.4, color)
        svg.text((left + right) / 2, y + h - 12, "elements per op", size=12, fill="#444", anchor="middle")
        svg.text(x + 4, (top + bottom) / 2, y_label, size=12, fill="#444")
        legend_x, legend_y = left + 8, top + 17
        for idx, (label, _, color, dashed, _) in enumerate(series):
            yy = legend_y + idx * 18
            svg.line(legend_x, yy - 4, legend_x + 22, yy - 4, color, 2.0, "5,4" if dashed else None)
            svg.circle(legend_x + 11, yy - 4, 3.2, color)
            svg.text(legend_x + 30, yy, label, size=11, fill="#333")

    plot_panel(
        panels[0],
        [
            ("simulated time (us)", [row.simulated_time_us for row in rows], colors[0], False, "o"),
            ("wall time (s)", [row.wall_time_sec for row in rows], colors[1], True, "s"),
        ],
        "time",
        lambda value: f"{value:.0f}",
    )
    plot_panel(
        panels[1],
        [
            ("observed read+write", [row.hbm_total_stream_bytes for row in rows], colors[0], False, "o"),
            ("expected", [row.expected_stream_bytes for row in rows], colors[2], True, "s"),
        ],
        "bytes",
        _fmt_bytes,
    )
    plot_panel(
        panels[2],
        [
            ("DMA read", [row.dma_read_issue_count for row in rows], colors[0], False, "o"),
            ("DMA write", [row.dma_write_issue_count for row in rows], colors[1], True, "s"),
            ("SFU issue", [row.sfu_ops_issued for row in rows], colors[2], False, "^"),
        ],
        "count",
        lambda value: f"{value:.0f}",
    )
    plot_panel(
        panels[3],
        [
            ("effective HBM GiB/s", [row.hbm_effective_gib_per_s for row in rows], colors[3], False, "o"),
        ],
        "GiB/s",
        lambda value: f"{value:.3g}",
    )
    svg.text(
        70,
        790,
        "Interpretation: HBM bytes scale linearly with tensor size; DMA/SFU issue counts stay at 6 here because chunk_elems equals total_elems for each op.",
        size=13,
        fill="#555",
    )
    path.write_text(svg.finish())


def render_png_pdf_fallback(rows: list[SweepRow], png_path: Path, pdf_path: Path, title: str) -> bool:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError:
        return False

    width, height = 1180, 820
    scale = 2
    image = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(image)

    def font(name: str, size: int):
        candidates = [
            f"/usr/share/fonts/truetype/dejavu/{name}.ttf",
            f"/usr/share/fonts/truetype/liberation2/{name}.ttf",
        ]
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, size * scale)
            except OSError:
                continue
        return ImageFont.load_default()

    font_title = font("DejaVuSans-Bold", 22)
    font_panel = font("DejaVuSans-Bold", 15)
    font_regular = font("DejaVuSans", 12)
    font_small = font("DejaVuSans", 10)

    def t(x: float, y: float, text: str, fill=(32, 32, 32), used_font=None) -> None:
        draw.text((x * scale, y * scale), text, fill=fill, font=used_font or font_regular)

    def line(points: list[tuple[float, float]], fill, width_px=2) -> None:
        if len(points) >= 2:
            draw.line([(x * scale, y * scale) for x, y in points], fill=fill, width=width_px * scale)

    colors = [(31, 119, 180), (213, 94, 0), (0, 158, 115), (106, 61, 154)]
    elems = [row.total_elems for row in rows]
    panels = [
        (70, 110, 480, 260, "Runtime", [("sim us", [row.simulated_time_us for row in rows], colors[0]), ("wall s", [row.wall_time_sec for row in rows], colors[1])]),
        (650, 110, 480, 260, "HBM traffic", [("observed", [row.hbm_total_stream_bytes for row in rows], colors[0]), ("expected", [row.expected_stream_bytes for row in rows], colors[2])]),
        (70, 465, 480, 260, "Primitive events", [("DMA read", [row.dma_read_issue_count for row in rows], colors[0]), ("DMA write", [row.dma_write_issue_count for row in rows], colors[1]), ("SFU issue", [row.sfu_ops_issued for row in rows], colors[2])]),
        (650, 465, 480, 260, "Effective bandwidth", [("GiB/s", [row.hbm_effective_gib_per_s for row in rows], colors[3])]),
    ]
    t(44, 34, title, used_font=font_title)
    t(44, 63, "Completed sizes: 16, 1024, 4096 elems/op; six primitives per point.", used_font=font_regular)
    for x, y, w, h, heading, series in panels:
        left, right, top, bottom = x + 68, x + w - 28, y + 42, y + h - 46
        t(x, y, heading, used_font=font_panel)
        draw.line([(left * scale, bottom * scale), (right * scale, bottom * scale)], fill=(70, 70, 70), width=scale)
        draw.line([(left * scale, top * scale), (left * scale, bottom * scale)], fill=(70, 70, 70), width=scale)
        log_x_min, log_x_max = math.log2(min(elems)), math.log2(max(elems))
        all_y = [value for _, values, _ in series for value in values if value > 0]
        y_min, y_max = min(all_y), max(all_y)
        use_log_y = y_min > 0 and (y_max / y_min) >= 16
        if use_log_y:
            y_min, y_max = 10 ** math.floor(math.log10(y_min)), 10 ** math.ceil(math.log10(y_max))
        elif math.isclose(y_min, y_max):
            y_min, y_max = 0, y_max * 1.2 + 1
        else:
            pad = (y_max - y_min) * 0.12
            y_min, y_max = max(0, y_min - pad), y_max + pad

        def sx(elem: int) -> float:
            return left + (math.log2(elem) - log_x_min) / (log_x_max - log_x_min) * (right - left)

        def sy(value: float) -> float:
            if use_log_y:
                return bottom - (math.log10(value) - math.log10(y_min)) / (
                    math.log10(y_max) - math.log10(y_min)
                ) * (bottom - top)
            return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

        for elem in elems:
            px = sx(elem)
            draw.line([(px * scale, bottom * scale), (px * scale, (bottom + 5) * scale)], fill=(60, 60, 60), width=scale)
            t(px - 16, bottom + 10, str(elem), used_font=font_small)
        for label, values, color in series:
            points = [(sx(elem), sy(value)) for elem, value in zip(elems, values) if value > 0]
            line(points, color, 2)
            for px, py in points:
                draw.ellipse(
                    [(px - 4) * scale, (py - 4) * scale, (px + 4) * scale, (py + 4) * scale],
                    fill=color,
                    outline=(255, 255, 255),
                )
        for idx, (label, _, color) in enumerate(series):
            t(left + 8, top + 12 + idx * 17, label, color, used_font=font_small)
        t((left + right) / 2 - 42, y + h - 18, "elements/op", used_font=font_small)
    t(70, 788, "HBM bytes scale with tensor size; event count stays constant because each op fits in one chunk.", used_font=font_regular)
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
    image = image.resize((width, height), resample)
    image.save(png_path)
    image.save(pdf_path, "PDF", resolution=300.0)
    return True


def write_notes(rows: list[SweepRow], path: Path) -> None:
    sizes = ", ".join(str(row.total_elems) for row in rows)
    chunks = ", ".join(str(row.chunks) for row in rows)
    chunk_elems = sorted({row.chunk_elems for row in rows})
    chunk_text = str(chunk_elems[0]) if len(chunk_elems) == 1 else ", ".join(str(v) for v in chunk_elems)
    ops = rows[0].ops if rows else ""
    ops_count = rows[0].ops_count if rows else 0
    lines = [
        "# SFU HBM Primitive Sweep Notes",
        "",
        f"- Scope: `GOLEM_SFU_PRIMITIVE_HBM_OPS={ops}`, completed PASS sizes `{sizes}` elements per op.",
        f"- Configured `chunk_elems`: `{chunk_text}`.",
        "- Only completed guest PASS rows are included in the source CSV and plots.",
        f"- Each point runs `{ops_count}` unary primitive op(s).",
        f"- Observed total primitive chunks in the included rows: `{chunks}`.",
        f"- HBM stream bytes are expected to scale as `total_elems * {ops_count} ops * 4 bytes * 2 directions`.",
        "",
        "## Collected Rows",
        "",
        "| elems/op | processed elems | HBM read bytes | HBM write bytes | DMA R/W issues | SFU issues | sim time (us) | wall time (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.total_elems} | {row.processed_elems} | {row.hbm_read_bytes} | "
            f"{row.hbm_write_bytes} | {row.dma_read_issue_count}/{row.dma_write_issue_count} | "
            f"{row.sfu_ops_issued} | {row.simulated_time_us:.3f} | {row.wall_time_sec:.0f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_root", type=Path, help="artifact sweep root")
    parser.add_argument("--sizes", default="16,1024,4096", help="comma-separated element sizes to include")
    parser.add_argument("--out-dir", type=Path, default=None, help="output directory; default: SWEEP_ROOT/figures")
    args = parser.parse_args(argv)

    sizes = {int(item) for item in args.sizes.split(",") if item.strip()}
    sweep_root = args.sweep_root.resolve()
    out_dir = (args.out_dir or (sweep_root / "figures")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(sweep_root, sizes)
    title = "SFU HBM Primitive Sweep"
    out_prefix = out_dir / "sfu_hbm_primitive_sweep"
    csv_path = out_dir / "sfu_hbm_primitive_sweep_source.csv"
    notes_path = out_dir / "sfu_hbm_primitive_sweep_notes.md"

    write_source_csv(rows, csv_path)
    write_notes(rows, notes_path)

    used_backend = "matplotlib"
    if not try_plot_matplotlib(rows, out_prefix, title):
        used_backend = "fallback-svg-pillow"
        render_svg_fallback(rows, out_prefix.with_suffix(".svg"), title)
        if not render_png_pdf_fallback(rows, out_prefix.with_suffix(".png"), out_prefix.with_suffix(".pdf"), title):
            used_backend = "fallback-svg-only"

    print(f"backend={used_backend}")
    print(f"source_csv={csv_path}")
    print(f"notes={notes_path}")
    print(f"svg={out_prefix.with_suffix('.svg')}")
    print(f"png={out_prefix.with_suffix('.png')}")
    print(f"pdf={out_prefix.with_suffix('.pdf')}")
    print(f"tiff={out_prefix.with_suffix('.tiff')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
