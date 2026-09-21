#!/usr/bin/env python3
"""Plot end-to-end profile latency from MInference profile CSV outputs."""

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
INPUT_CSV = (
    REPO_ROOT
    / "baselines/minference/experiments/profile_e2e_results/"
    / "gradientai_Llama-3-8B-Instruct-Gradient-1048k_20260520_105150.csv"
)
OUTPUT_PDF = REPO_ROOT / "results/plots/e2e_profile_latency.pdf"
OUTPUT_PNG = REPO_ROOT / "results/plots/e2e_profile_latency.png"
INCLUDE_NON_OK = False
TITLE: str | None = None

# Leave empty to plot all contexts/cases found in the CSV.
SELECTED_CONTEXTS: list[int] = [
    # 10000,
]
SELECTED_CASES: list[str] = [
    # "dense",
    # "a_shape",
    # "triton/bs32",
    # "cutlass/bs32",
    # "v5/bs32",
]

# If this baseline exists for a context, non-baseline bars are annotated with
# baseline_latency / method_latency. Set to None to disable annotations.
SPEEDUP_BASELINE: str | None = "dense"

# Optional display names. Unknown future backends/cases are formatted automatically.
CASE_LABELS = {
    "dense": "dense",
    "a_shape": "A-shape",
    "triton/bs32": "triton-b32",
    "triton/bs64": "triton-b64",
    "cutlass/bs32": "cutlass-b32",
    "cutlass/bs64": "cutlass-b64",
    "v5/bs32": "sptc-b32",
    "v5/bs64": "sptc-b64",
}

# Preferred order for known cases. Future cases are appended alphabetically.
CASE_ORDER = [
    "dense",
    "a_shape",
    "triton/bs32",
    "triton/bs64",
    "cutlass/bs32",
    "cutlass/bs64",
    "v5/bs32",
    "v5/bs64",
]


def normalize_case(row: dict[str, str]) -> str:
    attn_type = row["attn_type"].strip()
    backend = row["backend"].strip()
    block_size = row["block_size"].strip()
    case_name = row["case_name"].strip()

    if attn_type == "minference":
        if block_size:
            return f"{backend}/bs{block_size}"
        return backend or case_name
    return case_name or attn_type


def parse_csv(path: Path) -> dict[int, dict[str, dict[str, float | str]]]:
    rows: dict[int, dict[str, dict[str, float | str]]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if not INCLUDE_NON_OK and raw["status"].lower() != "ok":
                continue
            if raw["mean_sec"] == "":
                continue

            ctx = int(raw["context_window"])
            case = normalize_case(raw)
            mean_sec = float(raw["mean_sec"])
            std_sec = 0.0 if raw["std_sec"] == "" else float(raw["std_sec"])
            rows.setdefault(ctx, {})[case] = {
                "mean_sec": mean_sec,
                "std_sec": std_sec,
                "status": raw["status"],
            }
    return rows


def select_contexts(rows_by_ctx: dict[int, dict[str, dict[str, float | str]]]) -> list[int]:
    if SELECTED_CONTEXTS:
        missing = [ctx for ctx in SELECTED_CONTEXTS if ctx not in rows_by_ctx]
        if missing:
            raise ValueError(f"Selected contexts not found in {INPUT_CSV}: {missing}")
        return SELECTED_CONTEXTS
    return sorted(rows_by_ctx)


def select_cases(
    rows_by_ctx: dict[int, dict[str, dict[str, float | str]]],
    contexts: list[int],
) -> list[str]:
    available = {case for ctx in contexts for case in rows_by_ctx.get(ctx, {})}
    if SELECTED_CASES:
        missing = [case for case in SELECTED_CASES if case not in available]
        if missing:
            raise ValueError(f"Selected cases not found in {INPUT_CSV}: {missing}")
        return SELECTED_CASES

    known = [case for case in CASE_ORDER if case in available]
    unknown = sorted(available - set(CASE_ORDER))
    return known + unknown


def format_ctx(ctx: int) -> str:
    if ctx % 1000 == 0:
        return f"{ctx // 1000}K"
    return str(ctx)


def format_case_label(case: str) -> str:
    if case in CASE_LABELS:
        return CASE_LABELS[case]
    return case.replace("/", "-")


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
            fontsize=7.0,
            rotation=90,
        )


def plot_latency(
    rows_by_ctx: dict[int, dict[str, dict[str, float | str]]],
    contexts: list[int],
    cases: list[str],
) -> None:
    if not contexts or not cases:
        raise ValueError("No valid E2E rows to plot.")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    x = list(range(len(contexts)))
    group_width = 0.82
    bar_width = min(0.12, group_width / max(1, len(cases)))
    offsets = {
        case: (idx - (len(cases) - 1) / 2.0) * bar_width
        for idx, case in enumerate(cases)
    }

    cmap = plt.get_cmap("tab20")
    colors = {case: cmap(idx % cmap.N) for idx, case in enumerate(cases)}

    fig_width = max(6.8, 1.15 * len(contexts) + 0.22 * len(cases) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_width, 3.45), constrained_layout=True)

    ymax = 0.0
    for case in cases:
        positions = []
        means = []
        stds = []
        labels = []
        for xpos, ctx in zip(x, contexts):
            record = rows_by_ctx.get(ctx, {}).get(case)
            if record is None:
                continue
            mean_sec = float(record["mean_sec"])
            std_sec = float(record["std_sec"])
            positions.append(xpos + offsets[case])
            means.append(mean_sec)
            stds.append(std_sec)
            ymax = max(ymax, mean_sec + std_sec)

            if SPEEDUP_BASELINE is None or case == SPEEDUP_BASELINE:
                labels.append("")
            else:
                baseline = rows_by_ctx.get(ctx, {}).get(SPEEDUP_BASELINE)
                if baseline is None:
                    labels.append("")
                else:
                    labels.append(f"{float(baseline['mean_sec']) / mean_sec:.2f}x")

        if not means:
            continue

        bars = ax.bar(
            positions,
            means,
            bar_width,
            yerr=stds,
            capsize=2.0,
            label=format_case_label(case),
            color=colors[case],
            edgecolor="black",
            linewidth=0.45,
        )
        annotate_bars(ax, bars, labels)

    ax.set_ylim(0.0, ymax * 1.25 if ymax > 0.0 else 1.0)
    ax.set_ylabel("E2E latency (s)")
    ax.set_xticks(x)
    ax.set_xticklabels([format_ctx(ctx) for ctx in contexts])
    ax.set_xlabel("Context window")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=min(4, max(1, len(cases))), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    if TITLE:
        ax.set_title(TITLE, pad=18)

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    if OUTPUT_PNG is not None:
        OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_PNG, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    global INCLUDE_NON_OK, INPUT_CSV, OUTPUT_PDF, OUTPUT_PNG
    parser = argparse.ArgumentParser(description="Plot SparseAttn E2E profile CSV.")
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-pdf", type=Path, default=OUTPUT_PDF)
    parser.add_argument("--output-png", type=Path, default=OUTPUT_PNG)
    parser.add_argument("--include-non-ok", action="store_true")
    args = parser.parse_args()
    INPUT_CSV = args.input_csv.resolve()
    OUTPUT_PDF = args.output_pdf.resolve()
    OUTPUT_PNG = args.output_png.resolve()
    INCLUDE_NON_OK = args.include_non_ok
    rows_by_ctx = parse_csv(INPUT_CSV)
    contexts = select_contexts(rows_by_ctx)
    cases = select_cases(rows_by_ctx, contexts)
    plot_latency(rows_by_ctx, contexts, cases)
    print(f"Wrote {OUTPUT_PDF}")
    if OUTPUT_PNG is not None:
        print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
