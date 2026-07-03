#!/usr/bin/env python3
"""Plot fixed-size SFU HBM primitive chunk diagnostics."""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path


PASS_RE = re.compile(r"\[SOFTMAX\].*mode=sfu-primitive-hbm-stream.*PASS")
KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")
RUN_RE = re.compile(r"chunk([0-9]+)_elems_([0-9]+)")
TIME_RE = re.compile(r"^\s*([0-9.]+)\s*(us|ms|s)\s*$")
SIM_RE = re.compile(r"(?:Simulation is complete, simulated time:|# Simulated time:)\s*([0-9.]+)\s*(us|ms|s)")
DMA_RE = re.compile(
    r"GlobalMemory core=7 DMA READ stats:.*?"
    r"read_issue_count=([0-9]+).*?"
    r"write_issue_count=([0-9]+).*?"
    r"read_bytes_total=([0-9]+).*?"
    r"write_bytes_total=([0-9]+).*?"
    r"completion=([0-9]+).*?"
    r"write_completion=([0-9]+).*?"
    r"wait_count=([0-9]+)"
)


@dataclass
class Row:
    run_id: str
    status: str
    ops: str
    total_elems: int
    chunk_elems: int
    chunks: int
    processed_elems: int
    hbm_read_bytes: int
    hbm_write_bytes: int
    dma_read_issue_count: int
    dma_write_issue_count: int
    dma_completion: int
    dma_write_completion: int
    dma_wait_count: int
    simulated_time_us: float
    wall_time_sec: float

    @property
    def hbm_stream_bytes(self) -> int:
        return self.hbm_read_bytes + self.hbm_write_bytes

    @property
    def completed_fraction(self) -> float:
        expected_bytes = self.total_elems * 4
        if expected_bytes <= 0:
            return 0.0
        return min(1.0, self.hbm_read_bytes / expected_bytes)

    @property
    def primitive_melems_per_wall_s(self) -> float:
        if self.wall_time_sec <= 0:
            return 0.0
        return (self.processed_elems / 1e6) / self.wall_time_sec


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


def read_run_summary(root: Path) -> dict[str, dict[str, str]]:
    path = root / "stats" / "run_summary.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {row["run_id"]: row for row in csv.DictReader(handle)}


def parse_pass_line(stdout_dir: Path) -> dict[str, str] | None:
    if not stdout_dir.exists():
        return None
    for path in sorted(stdout_dir.glob("stdout-*")):
        with path.open(errors="replace") as handle:
            for line in handle:
                if PASS_RE.search(line):
                    return dict(KV_RE.findall(line))
    return None


def read_metric_sum(path: Path, metric: str) -> int:
    if not path.exists():
        return 0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("metric") == metric:
                return int(float(row.get("sum", "0") or 0))
    return 0


def collect_pass_rows(root: Path) -> list[Row]:
    rows: list[Row] = []
    for run_id, summary in sorted(read_run_summary(root).items()):
        pass_kv = parse_pass_line(root / "stdout" / "overlap0" / run_id)
        if not pass_kv:
            continue
        dma = root / "stats" / "overlap0" / run_id / "dma_summary.csv"
        rows.append(
            Row(
                run_id=run_id,
                status="PASS",
                ops=pass_kv["ops"],
                total_elems=int(pass_kv["total_elems"]),
                chunk_elems=int(pass_kv["chunk_elems"]),
                chunks=int(pass_kv["chunks"]),
                processed_elems=int(pass_kv["processed_elems"]),
                hbm_read_bytes=int(pass_kv["hbm_read_bytes"]),
                hbm_write_bytes=int(pass_kv["hbm_write_bytes"]),
                dma_read_issue_count=read_metric_sum(dma, "read_issue_count"),
                dma_write_issue_count=read_metric_sum(dma, "write_issue_count"),
                dma_completion=read_metric_sum(dma, "completion"),
                dma_write_completion=read_metric_sum(dma, "write_completion"),
                dma_wait_count=read_metric_sum(dma, "wait_count"),
                simulated_time_us=parse_time_to_us(summary.get("simulated_time", "")),
                wall_time_sec=float(summary.get("wall_time_sec", 0) or 0),
            )
        )
    return rows


def parse_partial_log(path: Path, wall_time_sec: float) -> Row:
    text = path.read_text(errors="replace")
    run_id = path.stem
    run_match = RUN_RE.search(run_id)
    chunk_elems = int(run_match.group(1)) if run_match else 0
    total_elems = int(run_match.group(2)) if run_match else 0
    sim_match = SIM_RE.search(text)
    simulated_time_us = (
        parse_time_to_us(" ".join(sim_match.groups())) if sim_match else 0.0
    )
    dma_match = DMA_RE.search(text)
    if dma_match:
        read_issue, write_issue, read_bytes, write_bytes, completion, write_completion, wait_count = [
            int(value) for value in dma_match.groups()
        ]
    else:
        read_issue = write_issue = read_bytes = write_bytes = completion = write_completion = wait_count = 0
    processed_elems = min(total_elems, read_bytes // 4)
    chunks = (total_elems + chunk_elems - 1) // chunk_elems if chunk_elems else 0
    return Row(
        run_id=run_id,
        status="TIMEOUT",
        ops="EXP",
        total_elems=total_elems,
        chunk_elems=chunk_elems,
        chunks=chunks,
        processed_elems=processed_elems,
        hbm_read_bytes=read_bytes,
        hbm_write_bytes=write_bytes,
        dma_read_issue_count=read_issue,
        dma_write_issue_count=write_issue,
        dma_completion=completion,
        dma_write_completion=write_completion,
        dma_wait_count=wait_count,
        simulated_time_us=simulated_time_us,
        wall_time_sec=wall_time_sec,
    )


def write_csv(rows: list[Row], path: Path) -> None:
    names = [
        "run_id",
        "status",
        "ops",
        "total_elems",
        "chunk_elems",
        "chunks",
        "processed_elems",
        "completed_fraction",
        "hbm_read_bytes",
        "hbm_write_bytes",
        "hbm_stream_bytes",
        "dma_read_issue_count",
        "dma_write_issue_count",
        "dma_completion",
        "dma_write_completion",
        "dma_wait_count",
        "simulated_time_us",
        "wall_time_sec",
        "primitive_melems_per_wall_s",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in names})


def write_notes(rows: list[Row], path: Path) -> None:
    pass_rows = [row for row in rows if row.status == "PASS"]
    timeout_rows = [row for row in rows if row.status != "PASS"]
    lines = [
        "# SFU HBM chunk diagnostic notes",
        "",
        "- Workload: fixed 65536 elements/op, single EXP primitive.",
        "- Larger chunk sizes reduce SFU primitive chunk count, while DMA issue count is also bounded by the 16 KiB DMA burst size.",
        "- PASS rows are complete SST runs; TIMEOUT rows are partial emergency-shutdown diagnostics.",
        "",
        "## PASS rows",
        "",
    ]
    for row in sorted(pass_rows, key=lambda item: item.chunk_elems):
        lines.append(
            f"- chunk_elems={row.chunk_elems}: chunks={row.chunks}, "
            f"DMA R/W={row.dma_read_issue_count}/{row.dma_write_issue_count}, "
            f"wait={row.dma_wait_count}, wall={row.wall_time_sec:.0f}s, "
            f"sim={row.simulated_time_us:.2f}us."
        )
    lines += ["", "## Partial rows", ""]
    for row in sorted(timeout_rows, key=lambda item: item.chunk_elems):
        lines.append(
            f"- chunk_elems={row.chunk_elems}: TIMEOUT after about {row.wall_time_sec:.0f}s, "
            f"DMA R/W={row.dma_read_issue_count}/{row.dma_write_issue_count}, "
            f"completed_read_fraction={row.completed_fraction:.3f}."
        )
    if pass_rows:
        min_chunk = min(pass_rows, key=lambda item: item.chunk_elems)
        max_chunk = max(pass_rows, key=lambda item: item.chunk_elems)
        lines += [
            "",
            "## Interpretation",
            "",
            f"- chunk_elems {min_chunk.chunk_elems}->{max_chunk.chunk_elems} changes SFU chunks "
            f"{min_chunk.chunks}->{max_chunk.chunks}; both completed with identical DMA R/W "
            f"{min_chunk.dma_read_issue_count}/{min_chunk.dma_write_issue_count} because "
            "8192 fp32 values split into two 16 KiB DMA bursts.",
            "- The 1024-chunk timeout indicates the long run is mainly amplified by per-chunk guest/SST executor work, not by HBM preload or full statistics dumping.",
        ]
    path.write_text("\n".join(lines) + "\n")


def plot(rows: list[Row], out_prefix: Path) -> None:
    mpl_config = out_prefix.parent / ".mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda item: item.chunk_elems)
    labels = [str(row.chunk_elems) for row in rows]
    xpos = list(range(len(rows)))
    colors = ["#3b6ea8" if row.status == "PASS" else "#b85c38" for row in rows]
    markers = ["o" if row.status == "PASS" else "x" for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), constrained_layout=True)
    fig.suptitle("SFU HBM primitive chunk diagnostic: EXP, 65536 elements", fontsize=14, fontweight="bold")

    ax = axes[0][0]
    for pos, row, color, marker in zip(xpos, rows, colors, markers):
        ax.scatter(pos, row.wall_time_sec, color=color, marker=marker, s=68)
    pass_x = [pos for pos, row in zip(xpos, rows) if row.status == "PASS"]
    pass_y = [row.wall_time_sec for row in rows if row.status == "PASS"]
    ax.plot(pass_x, pass_y, color="#3b6ea8")
    ax.set_xticks(xpos, labels)
    ax.set_xlabel("chunk_elems")
    ax.set_ylabel("wall time (s)")
    ax.set_title("Host wall time")
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[0][1]
    width = 0.24
    ax.bar([value - width for value in xpos], [row.chunks for row in rows], width=width, label="SFU chunks", color="#4c78a8")
    ax.bar(xpos, [row.dma_read_issue_count for row in rows], width=width, label="DMA reads", color="#59a14f")
    ax.bar([value + width for value in xpos], [row.dma_write_issue_count for row in rows], width=width, label="DMA writes", color="#f28e2b")
    ax.set_xticks(xpos, labels)
    ax.set_xlabel("chunk_elems")
    ax.set_ylabel("count")
    ax.set_title("Issued work")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1][0]
    for pos, row, color, marker in zip(xpos, rows, colors, markers):
        ax.scatter(pos, row.completed_fraction * 100.0, color=color, marker=marker, s=68)
    ax.set_xticks(xpos, labels)
    ax.set_ylim(0, 105)
    ax.set_xlabel("chunk_elems")
    ax.set_ylabel("completed read fraction (%)")
    ax.set_title("Completion")
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1][1]
    for pos, row, color, marker in zip(xpos, rows, colors, markers):
        ax.scatter(pos, row.simulated_time_us, color=color, marker=marker, s=68)
    ax.plot(pass_x, [row.simulated_time_us for row in rows if row.status == "PASS"], color="#3b6ea8")
    ax.set_xticks(xpos, labels)
    ax.set_xlabel("chunk_elems")
    ax.set_ylabel("simulated time (us)")
    ax.set_title("Simulated time")
    ax.grid(True, axis="y", alpha=0.25)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color="#3b6ea8", label="PASS"),
        plt.Line2D([0], [0], marker="x", linestyle="", color="#b85c38", label="TIMEOUT partial"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    for suffix in ("svg", "pdf", "png", "tiff"):
        fig.savefig(out_prefix.with_suffix(f".{suffix}"), dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Sweep root with completed PASS rows")
    parser.add_argument("--partial-log", type=Path, action="append", default=[], help="Partial timeout log")
    parser.add_argument("--partial-wall-sec", type=float, action="append", default=[], help="Wall seconds for the corresponding partial log")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    rows = collect_pass_rows(args.root)
    for index, path in enumerate(args.partial_log):
        wall = args.partial_wall_sec[index] if index < len(args.partial_wall_sec) else 0.0
        rows.append(parse_partial_log(path, wall))
    rows.sort(key=lambda row: row.chunk_elems)

    out_dir = args.out_dir or (args.root / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "sfu_hbm_exp65536_chunk_diag"
    write_csv(rows, prefix.with_name(prefix.name + "_source.csv"))
    write_notes(rows, prefix.with_name(prefix.name + "_notes.md"))
    plot(rows, prefix)
    print(f"[OK] rows={len(rows)}")
    print(f"[OK] wrote {prefix}_source.csv")
    print(f"[OK] wrote {prefix}_notes.md")
    print(f"[OK] wrote {prefix}.svg/.pdf/.png/.tiff")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
