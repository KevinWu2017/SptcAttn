#!/usr/bin/env python3
"""Plot absolute latency for sparse-attention benchmark matrix."""

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
INPUT_CSV = REPO_ROOT / "reports/figures/data/sparse_latency_matrix_a100.csv"
OUTPUT_PDF = REPO_ROOT / "results/plots/sparse_latency.pdf"
OUTPUT_PNG = REPO_ROOT / "results/plots/sparse_latency.png"
INCLUDE_NON_OK = False
TITLE: str | None = None
SPEEDUP_BASELINE = "triton"

# Display order and names. Keys are plot-local aliases; values map to CSV implementation names.
PLOT_IMPLS = ["triton", "v1", "v5"]
CSV_IMPL_NAMES = {
    "dense": "dense",
    "triton": "minference_triton",
    "v1": "minference_cutlass",
    "v5": "v5",
}
IMPL_LABELS = {
    "dense": "dense",
    "triton": "triton",
    "v1": "cutlass",
    "v5": "sptc",
}

# Exact configs to plot, in display order:
#   (precision, head_dim, block_m, block_n, ctx, nnz_s, causal)
# Leave empty to plot every config that has at least one implementation in PLOT_IMPLS.
SELECTED_CONFIGS: list[tuple[str, int, int, int, int, int, bool]] = [
    ("fp16", 64, 32, 32, 32768, 512, True),
    ("fp16", 64, 32, 32, 32768, 512, False),
    # ("fp16", 64, 64, 64, 65536, 1024, True),
    # ("fp16", 64, 64, 64, 65536, 1024, False),
]


ConfigKey = tuple[str, int, int, int, int, int, bool]
BaseConfigKey = tuple[str, int, int, int, int, int]


def parse_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {raw!r}")


def parse_csv(path: Path) -> dict[ConfigKey, dict[str, dict[str, float | int | str]]]:
    rows: dict[ConfigKey, dict[str, dict[str, float | int | str]]] = {}
    csv_impl_to_alias = {csv_name: alias for alias, csv_name in CSV_IMPL_NAMES.items()}

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            alias = csv_impl_to_alias.get(raw["implementation"])
            if alias is None:
                continue
            if not INCLUDE_NON_OK and raw["status"] != "OK":
                continue
            if raw["avg_ms"] == "":
                continue

            avg_ms = float(raw["avg_ms"])
            if avg_ms <= 0.0:
                continue

            precision = raw["precision"].lower()
            key = (
                precision,
                int(raw["head_dim"]),
                int(raw["block_m"]),
                int(raw["block_n"]),
                int(raw["ctx"]),
                int(raw["nnz_s"]),
                parse_bool(raw["causal"]),
            )
            rows.setdefault(key, {})[alias] = {
                "precision": precision,
                "head_dim": key[1],
                "block_m": key[2],
                "block_n": key[3],
                "ctx": key[4],
                "nnz_s": key[5],
                "causal": key[6],
                "avg_ms": avg_ms,
            }
    return rows


def has_plotted_impl(row_by_impl: dict[str, dict[str, float | int | str]]) -> bool:
    return any(impl in row_by_impl for impl in PLOT_IMPLS)


def select_rows(
    rows_by_config: dict[ConfigKey, dict[str, dict[str, float | int | str]]],
) -> list[tuple[ConfigKey, dict[str, dict[str, float | int | str]]]]:
    if not SELECTED_CONFIGS:
        complete_keys = [key for key, value in rows_by_config.items() if has_plotted_impl(value)]
        return [
            (key, rows_by_config[key])
            for key in sorted(complete_keys, key=lambda x: (x[1], x[2], x[3], x[4], x[5], x[6]))
        ]

    missing = [
        config
        for config in SELECTED_CONFIGS
        if config not in rows_by_config or not has_plotted_impl(rows_by_config[config])
    ]
    if missing:
        formatted = ", ".join(str(config) for config in missing)
        raise ValueError(f"Selected configs missing plotted implementations in {INPUT_CSV}: {formatted}")
    return [(config, rows_by_config[config]) for config in SELECTED_CONFIGS]


def strip_causal(config: ConfigKey) -> BaseConfigKey:
    precision, head_dim, block_m, block_n, ctx, nnz_s, _ = config
    return precision, head_dim, block_m, block_n, ctx, nnz_s


def group_rows_by_mode(
    rows: list[tuple[ConfigKey, dict[str, dict[str, float | int | str]]]]
) -> tuple[
    list[BaseConfigKey],
    dict[bool, dict[BaseConfigKey, dict[str, dict[str, float | int | str]]]],
]:
    base_configs: list[BaseConfigKey] = []
    by_mode: dict[bool, dict[BaseConfigKey, dict[str, dict[str, float | int | str]]]] = {
        True: {},
        False: {},
    }
    for config, row_by_impl in rows:
        base_config = strip_causal(config)
        if base_config not in base_configs:
            base_configs.append(base_config)
        by_mode[config[-1]][base_config] = row_by_impl
    return base_configs, by_mode


def format_ctx(ctx: int) -> str:
    if ctx % 1024 == 0:
        return f"{ctx // 1024}K"
    return str(ctx)


def format_config_label(config: BaseConfigKey) -> str:
    _, head_dim, block_m, block_n, ctx, nnz_s = config
    block = f"B={block_m}" if block_m == block_n else f"BM={block_m}, BN={block_n}"
    return f"d={head_dim}, {block}\nL={format_ctx(ctx)}, S={nnz_s}"


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


def build_latency_by_impl(
    base_configs: list[BaseConfigKey],
    rows_by_base: dict[BaseConfigKey, dict[str, dict[str, float | int | str]]],
) -> dict[str, list[float | None]]:
    latency_by_impl = {
        impl: [
            None
            if base_config not in rows_by_base or impl not in rows_by_base[base_config]
            else float(rows_by_base[base_config][impl]["avg_ms"])
            for base_config in base_configs
        ]
        for impl in PLOT_IMPLS
    }
    return latency_by_impl


def build_speedup_labels(
    latency_by_impl: dict[str, list[float | None]]
) -> dict[str, list[str]]:
    baseline_ms = latency_by_impl[SPEEDUP_BASELINE]
    speedup_labels_by_impl = {}
    for impl, latencies in latency_by_impl.items():
        labels_for_impl = []
        for baseline, latency in zip(baseline_ms, latencies):
            if impl == SPEEDUP_BASELINE or baseline is None or latency is None:
                labels_for_impl.append("")
            else:
                labels_for_impl.append(f"{baseline / latency:.2f}x")
        speedup_labels_by_impl[impl] = labels_for_impl
    return speedup_labels_by_impl


def draw_mode_subplot(
    ax: plt.Axes,
    base_configs: list[BaseConfigKey],
    rows_by_base: dict[BaseConfigKey, dict[str, dict[str, float | int | str]]],
    mode_label: str,
    show_xticklabels: bool,
) -> None:
    latency_by_impl = build_latency_by_impl(base_configs, rows_by_base)
    speedup_labels_by_impl = build_speedup_labels(latency_by_impl)

    x = list(range(len(base_configs)))
    width = 0.24
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
        bar_positions = [pos + offsets[impl] for pos, value in zip(x, values) if value is not None]
        present_values = [value for value in values if value is not None]
        present_labels = [
            label
            for value, label in zip(values, speedup_labels_by_impl[impl])
            if value is not None
        ]
        if not present_values:
            continue
        bars = ax.bar(
            bar_positions,
            present_values,
            width,
            label=IMPL_LABELS[impl],
            color=colors[impl],
            edgecolor="black",
            linewidth=0.45,
            hatch=hatches[impl],
        )
        annotate_bars(ax, bars, present_labels)

    present_values = [
        value for values in latency_by_impl.values() for value in values if value is not None
    ]
    ymax = max(present_values) if present_values else 1.0
    ax.set_ylim(0.0, ymax * 1.28)
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    if show_xticklabels:
        ax.set_xticklabels([format_config_label(config) for config in base_configs])
    else:
        ax.set_xticklabels([])
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(mode_label, loc="left", fontsize=9, pad=3)


def plot_latency(rows: list[tuple[ConfigKey, dict[str, dict[str, float | int | str]]]]) -> None:
    if not rows:
        raise ValueError("No valid rows to plot.")

    base_configs, rows_by_mode = group_rows_by_mode(rows)

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

    fig_width = max(6.8, 0.82 * len(base_configs) + 1.9)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(fig_width, 5.6),
        sharex=True,
        constrained_layout=True,
    )

    draw_mode_subplot(axes[0], base_configs, rows_by_mode[True], "Causal", False)
    draw_mode_subplot(axes[1], base_configs, rows_by_mode[False], "Non-causal", True)

    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(PLOT_IMPLS), frameon=False, loc="upper center")
    if TITLE:
        fig.suptitle(TITLE)

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    if OUTPUT_PNG is not None:
        OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_PNG, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    global INCLUDE_NON_OK, INPUT_CSV, OUTPUT_PDF, OUTPUT_PNG
    parser = argparse.ArgumentParser(description="Plot SparseAttn kernel latency CSV.")
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
