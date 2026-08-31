#!/usr/bin/env python3
"""Publication-style E3 cycle waterfall for the multicore Attention optimizations."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch


# Keep SVG text editable for later PPT/Illustrator adjustment.
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


STAGES = [
    ("初始 QK/KV 基线", 2_241_546, None, "baseline"),
    ("K/V 并行 DMA", 2_179_516, -62_030, "movement"),
    ("Q matrix broadcast", 1_936_507, -243_009, "layout"),
    ("QK dataflow transpose", 1_782_364, -154_143, "layout"),
    ("KV tile rotation", 1_195_277, -587_087, "topology"),
    ("V tile reuse", 965_933, -229_344, "reuse"),
    ("K/V double buffer", 849_298, -116_635, "pipeline"),
    ("PV input pipeline", 799_873, -49_425, "pipeline"),
    ("Softmax/PV overlap", 790_516, -9_357, "pipeline"),
    ("PV restore pipeline", 759_338, -31_178, "pipeline"),
    ("PV output pipeline", 729_683, -29_655, "pipeline"),
    ("PV early compute (G5)", 699_750, -29_933, "final"),
]

COLORS = {
    "baseline": "#4D4D4D",
    "movement": "#4C78A8",
    "layout": "#59A14F",
    "topology": "#B279A2",
    "reuse": "#F28E2B",
    "pipeline": "#76B7B2",
    "final": "#1B6E5B",
}


def compact_cycles(value):
    return f"{value / 1_000_000:.2f}M"


def make_figure(output_stem: Path):
    for previous_stage, current_stage in zip(STAGES, STAGES[1:]):
        previous_value = previous_stage[1]
        current_value = current_stage[1]
        delta = current_stage[2]
        assert delta == current_value - previous_value, (previous_stage, current_stage)
    previous = STAGES[0][1]
    y_positions = list(range(len(STAGES) - 1, -1, -1))
    fig, ax = plt.subplots(figsize=(13.2, 7.2), constrained_layout=False)
    fig.subplots_adjust(left=0.30, right=0.97, top=0.84, bottom=0.16)

    # Each optimization is a horizontal waterfall segment from the new total
    # to the previous total; the baseline/final states are full reference bars.
    for y, (label, value, delta, family) in zip(y_positions, STAGES):
        if delta is None:
            left, width = 0, value
        else:
            left, width = value, previous - value
        ax.barh(
            y,
            width,
            left=left,
            height=0.58,
            color=COLORS[family],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.text(2_410_000, y, f"{value:,}", va="center", ha="right", fontsize=9.2, color="#272727", zorder=5)
        if delta is not None:
            if width >= 110_000:
                ax.text(left + width / 2, y, f"−{abs(delta) / 1_000:.0f}k", va="center", ha="center", fontsize=8.5, color="white", fontweight="bold", zorder=5)
            else:
                ax.text(left + width / 2, y + 0.34, f"−{abs(delta) / 1_000:.0f}k", va="bottom", ha="center", fontsize=8.1, color=COLORS[family], fontweight="bold", zorder=5)
        previous = value

    labels = [item[0] for item in STAGES]
    ax.set_yticks(y_positions, labels)
    ax.set_xlim(0, 2_430_000)
    ax.set_xlabel("E3 accelerator cycles（越低越好）", labelpad=9)
    ax.set_title(
        "多核 Fused Attention：E3 各阶段优化周期瀑布图",
        loc="left",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )
    ax.text(
        0,
        1.025,
        "B=1, H=1, S=1024, D=128; 4 Manager + 16 Worker; all formal points: 0 mismatch",
        transform=ax.transAxes,
        fontsize=10,
        color="#536372",
        va="bottom",
    )
    ax.grid(axis="x", color="#D9DEE3", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(lambda value, _: f"{value / 1_000_000:.1f}M")

    # Reference guides make the total reduction visible without adding a second panel.
    ax.axvline(STAGES[0][1], color="#7A7A7A", linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)
    ax.axvline(STAGES[-1][1], color=COLORS["final"], linestyle=(0, (4, 3)), linewidth=1.1, zorder=1)
    ax.text(STAGES[0][1], len(STAGES) - 0.3, "初始 2.242M", ha="right", va="bottom", fontsize=9, color="#4D4D4D")
    ax.text(STAGES[-1][1], -0.85, "正式 G5：699,750", ha="left", va="top", fontsize=9, color=COLORS["final"])

    total_drop = STAGES[0][1] - STAGES[-1][1]
    drop_fraction = total_drop / STAGES[0][1]
    ax.annotate(
        f"累计减少 {total_drop:,} cycles\n下降 {drop_fraction:.2%}（剩余 {1-drop_fraction:.2%}）",
        xy=(STAGES[-1][1], y_positions[-1]),
        xytext=(1_500_000, -0.65),
        fontsize=10,
        color=COLORS["final"],
        ha="center",
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#EEF7F3", edgecolor=COLORS["final"], linewidth=0.9),
        arrowprops=dict(arrowstyle="-", color=COLORS["final"], linewidth=0.9),
    )

    legend_items = [
        Patch(facecolor=COLORS["movement"], label="数据搬运"),
        Patch(facecolor=COLORS["layout"], label="QK 数据布局"),
        Patch(facecolor=COLORS["topology"], label="HBM 访问拓扑"),
        Patch(facecolor=COLORS["reuse"], label="数据复用"),
        Patch(facecolor=COLORS["pipeline"], label="流水与重叠"),
    ]
    ax.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.53, -0.105), ncol=5, frameon=False, fontsize=9)

    # Keep the formal baseline and correctness gate visible in the figure itself.
    fig.text(
        0.30,
        0.035,
        "Formal result: 131,072 outputs checked; 0 mismatch. The waterfall includes only accepted sequential A/B points.",
        fontsize=9,
        color="#536372",
    )
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    return output_stem


if __name__ == "__main__":
    output = Path(__file__).with_name("attention_cycles_waterfall")
    make_figure(output)
    print(output.with_suffix(".svg"))
