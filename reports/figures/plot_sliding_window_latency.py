#!/usr/bin/env python3
"""Plot absolute latency for V1, Triton, and V5 sliding-window benchmarks."""

from __future__ import annotations

import csv
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sparseattn-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]

# Edit these definitions for the paper figure.
INPUT_CSV = REPO_ROOT / "experiments/legacy/cmake_sparse_compare/sliding_window_perf_matrix.csv"
OUTPUT_PDF = REPO_ROOT / "results/plots/sliding_window_latency.pdf"
OUTPUT_PNG = REPO_ROOT / "results/plots/sliding_window_latency.png"
INCLUDE_NON_OK = False
TITLE: str | None = None

# Display order and names for the three implementations.
# Keys must stay aligned with CSV latency columns: triton -> triton_ms, v1 -> v1_ms, v5 -> v5_ms.
PLOT_IMPLS = ["triton", "v1", "v5"]
IMPL_LABELS = {
    "triton": "triton",
    "v1": "cutlass",
    "v5": "sptc",
}

# Exact configs to plot, in display order:
#   (head_dim, block_m, window_size, ctx)
# Leave empty to plot every valid row in the CSV.
SELECTED_CONFIGS: list[tuple[int, int, int, int]] = [
    # (64, 32, 32, 32768),
    # (64, 32, 32, 65536),
    # (64, 64, 64, 32768),
    # (64, 64, 64, 65536),
]


def parse_csv(path: Path) -> dict[tuple[int, int, int, int], dict[str, float | int | str]]:
    rows: dict[tuple[int, int, int, int], dict[str, float | int | str]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if not INCLUDE_NON_OK and (
                raw["v1_status"] != "OK"
                or raw["triton_status"] != "OK"
                or raw["v5_status"] != "OK"
            ):
                continue

            v1_ms = float(raw["v1_ms"])
            triton_ms = float(raw["triton_ms"])
            v5_ms = float(raw["v5_ms"])
            if v1_ms <= 0.0 or triton_ms <= 0.0 or v5_ms <= 0.0:
                continue

            key = (
                int(raw["head_dim"]),
                int(raw["block_m"]),
                int(raw["window_size"]),
                int(raw["ctx"]),
            )
            rows[key] = {
                "dtype": raw["dtype"],
                "head_dim": key[0],
                "block_m": key[1],
                "window_size": key[2],
                "ctx": key[3],
                "v1_ms": v1_ms,
                "triton_ms": triton_ms,
                "v5_ms": v5_ms,
            }
    return rows


def select_rows(
    rows_by_config: dict[tuple[int, int, int, int], dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    if not SELECTED_CONFIGS:
        return [
            rows_by_config[key]
            for key in sorted(rows_by_config, key=lambda x: (x[0], x[1], x[2], x[3]))
        ]

    missing = [config for config in SELECTED_CONFIGS if config not in rows_by_config]
    if missing:
        formatted = ", ".join(str(config) for config in missing)
        raise ValueError(f"Selected configs not found in {INPUT_CSV}: {formatted}")
    return [rows_by_config[config] for config in SELECTED_CONFIGS]


def format_ctx(ctx: int) -> str:
    if ctx % 1024 == 0:
        return f"{ctx // 1024}K"
    return str(ctx)


def format_config_label(row: dict[str, float | int | str]) -> str:
    return (
        f"d={row['head_dim']}, B={row['block_m']}\n"
        f"W={row['window_size']}, L={format_ctx(int(row['ctx']))}"
    )


def annotate_bars(ax: plt.Axes, bars, labels: list[str]) -> None:
    for bar, label in zip(bars, labels):
        if not label:
            continue
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.2,
            rotation=90,
        )


def plot_latency(rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        raise ValueError("No valid rows to plot.")

    labels = [format_config_label(row) for row in rows]
    latency_by_impl = {
        impl: [float(row[f"{impl}_ms"]) for row in rows]
        for impl in PLOT_IMPLS
    }
    triton_ms = [float(row["triton_ms"]) for row in rows]
    speedup_labels_by_impl = {
        impl: [f"{triton / latency:.2f}x" for triton, latency in zip(triton_ms, latencies)]
        for impl, latencies in latency_by_impl.items()
    }
    speedup_labels_by_impl["triton"] = [""] * len(rows)
    x = list(range(len(rows)))
    width = 0.24

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig_width = max(6.8, 0.82 * len(rows) + 1.9)
    fig, ax = plt.subplots(figsize=(fig_width, 3.25), constrained_layout=True)

    colors = {
        "v1": "#4C78A8",
        "triton": "#F58518",
        "v5": "#54A24B",
    }
    hatches = {
        "v1": "",
        "triton": "//",
        "v5": "..",
    }
    offsets = {
        impl: (idx - (len(PLOT_IMPLS) - 1) / 2.0) * width
        for idx, impl in enumerate(PLOT_IMPLS)
    }
    for impl in PLOT_IMPLS:
        values = latency_by_impl[impl]
        bars = ax.bar(
            [pos + offsets[impl] for pos in x],
            values,
            width,
            label=IMPL_LABELS[impl],
            color=colors[impl],
            edgecolor="black",
            linewidth=0.45,
            hatch=hatches[impl],
        )
        annotate_bars(ax, bars, speedup_labels_by_impl[impl])

    ymax = max(max(values) for values in latency_by_impl.values())
    ax.set_ylim(0.0, ymax * 1.28)
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    if TITLE:
        ax.set_title(TITLE, pad=20)

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    if OUTPUT_PNG is not None:
        OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_PNG, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    global INCLUDE_NON_OK, INPUT_CSV, OUTPUT_PDF, OUTPUT_PNG
    parser = argparse.ArgumentParser(description="Plot SparseAttn sliding-window latency CSV.")
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-pdf", type=Path, default=OUTPUT_PDF)
    parser.add_argument("--output-png", type=Path, default=OUTPUT_PNG)
    parser.add_argument("--include-non-ok", action="store_true")
    args = parser.parse_args()
    INPUT_CSV = args.input_csv.resolve()
    OUTPUT_PDF = args.output_pdf.resolve()
    OUTPUT_PNG = args.output_png.resolve()
    INCLUDE_NON_OK = args.include_non_ok
    rows_by_config = parse_csv(INPUT_CSV)
    rows = select_rows(rows_by_config)
    plot_latency(rows)
    print(f"Wrote {OUTPUT_PDF}")
    if OUTPUT_PNG is not None:
        print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
