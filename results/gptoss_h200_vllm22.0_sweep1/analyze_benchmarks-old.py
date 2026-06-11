#!/usr/bin/env python3
"""
Benchmark Analysis Script
Parses LLM inference benchmark results and generates performance charts.

Usage:
    python analyze_benchmarks.py [--results-dir ./results]

Expected folder structure:
    results/
        gptoss_fp4_h200_tp2_conc64_isl1024_osl1024/
            gptoss_fp4_h200_tp2_conc64_isl1024_osl1024.json
        ...
"""

import os
import re
import json
import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ── Optional: nicer fonts ──────────────────────────────────────────────────────
try:
    import matplotlib.font_manager as fm
    plt.rcParams["font.family"] = "DejaVu Sans"
except Exception:
    pass


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_tp(name: str) -> int:
    """Pull the tensor-parallel degree from the folder / file name."""
    m = re.search(r"_tp(\d+)", name)
    if not m:
        raise ValueError(f"Could not find '_tp<N>' in name: {name!r}")
    return int(m.group(1))


def load_results(results_dir: str) -> list[dict]:
    base = Path(results_dir)
    rows = []

    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue

        # Same-named JSON inside the folder
        json_path = folder / f"{folder.name}.json"
        if not json_path.exists():
            print(f"  [skip] No JSON found in {folder.name}")
            continue

        with open(json_path) as fh:
            data = json.load(fh)

        tp = extract_tp(folder.name)

        duration            = float(data["duration"])
        total_input_tokens  = float(data["total_input_tokens"])
        total_output_tokens = float(data["total_output_tokens"])
        max_concurrency     = float(data["max_concurrency"])
        median_e2el_ms      = float(data["median_e2el_ms"])
        p99_e2el_ms         = float(data["p99_e2el_ms"])

        total_token     = total_input_tokens + total_output_tokens
        tokpersec       = total_token / duration
        tokpersecpergpu = tokpersec / tp
        tokenperuser    = tokpersec / max_concurrency
        median_e2el_s   = median_e2el_ms / 1000.0
        p99_e2el_s      = p99_e2el_ms    / 1000.0

        rows.append({
            "TP":                       tp,
            "Concurrency":              int(max_concurrency),
            "Duration":                 round(duration, 2),
            "Total Token":              int(total_token),
            "Tok/sec":                  round(tokpersec, 2),
            "Tok/sec/gpu":              round(tokpersecpergpu, 2),
            "Tok/sec/user":             round(tokenperuser, 2),
            "P99_E2E_Latency (sec)":    round(p99_e2el_s, 3),
            "Median_E2E_Latency (sec)": round(median_e2el_s, 3),
        })

    rows.sort(key=lambda r: (r["TP"], r["Concurrency"]))
    return rows


# ── Table printer ──────────────────────────────────────────────────────────────

COLUMNS = [
    "TP", "Concurrency", "Duration", "Total Token",
    "Tok/sec", "Tok/sec/gpu", "Tok/sec/user",
    "P99_E2E_Latency (sec)", "Median_E2E_Latency (sec)",
]

def print_table(rows: list[dict]) -> None:
    col_widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in COLUMNS}
    sep   = "+-" + "-+-".join("-" * col_widths[c] for c in COLUMNS) + "-+"
    hdr   = "| " + " | ".join(c.ljust(col_widths[c]) for c in COLUMNS) + " |"

    print(sep)
    print(hdr)
    print(sep)
    for r in rows:
        print("| " + " | ".join(str(r[c]).ljust(col_widths[c]) for c in COLUMNS) + " |")
    print(sep)


# ── Colour helpers ─────────────────────────────────────────────────────────────

def tp_colour_map(rows):
    """Return a stable colour per TP value."""
    tps = sorted(set(r["TP"] for r in rows))
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {tp: palette[i % len(palette)] for i, tp in enumerate(tps)}


# ── Chart factory ──────────────────────────────────────────────────────────────

def make_chart(
    rows: list[dict],
    x_key: str,
    y_key: str = "Tok/sec/gpu",
    title: str = "",
    xlabel: str = "",
    output_path: str = "",
) -> None:
    tp_col = tp_colour_map(rows)
    tp_data: dict[int, list] = {}
    for r in rows:
        tp_data.setdefault(r["TP"], []).append(r)
    for tp in tp_data:
        tp_data[tp].sort(key=lambda r: r[x_key])

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3d4d")

    ax.tick_params(colors="#c0c4d4", labelsize=9)
    ax.xaxis.label.set_color("#c0c4d4")
    ax.yaxis.label.set_color("#c0c4d4")
    ax.title.set_color("#e8ecf4")
    ax.grid(color="#2a2d3d", linewidth=0.7, linestyle="--")

    for tp, group in sorted(tp_data.items()):
        xs    = [r[x_key]  for r in group]
        ys    = [r[y_key]  for r in group]
        concs = [r["Concurrency"] for r in group]
        color = tp_col[tp]

        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=7,
                color=color, label=f"TP={tp}")

        for x, y, conc in zip(xs, ys, concs):
            ax.annotate(
                f"TP{tp}/C{conc}",
                xy=(x, y),
                xytext=(5, 6),
                textcoords="offset points",
                fontsize=7.5,
                color=color,
                alpha=0.9,
            )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(y_key, fontsize=10)

    legend = ax.legend(
        title="Tensor Parallel",
        facecolor="#252837",
        edgecolor="#3a3d4d",
        labelcolor="#c0c4d4",
        title_fontsize=9,
        fontsize=9,
    )
    legend.get_title().set_color("#8b90a8")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {output_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyse LLM benchmark results.")
    parser.add_argument(
        "--results-dir", default="./results",
        help="Path to the results folder (default: ./results)"
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Directory where charts are saved (default: current dir)"
    )
    args = parser.parse_args()

    print(f"\nScanning: {args.results_dir}\n")
    rows = load_results(args.results_dir)

    if not rows:
        print("No data found. Check --results-dir path and folder structure.")
        return

    # ── Table ──────────────────────────────────────────────────────────────────
    print_table(rows)
    print()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Chart 1: Tok/sec/gpu  vs  Tok/sec/user ────────────────────────────────
    print("Generating charts …")
    make_chart(
        rows,
        x_key="Tok/sec/user",
        y_key="Tok/sec/gpu",
        title="Throughput Efficiency  –  Tok/sec/GPU vs Tok/sec/User",
        xlabel="Tok/sec/user",
        output_path=str(out / "chart1_tokpergpu_vs_tokperuser.png"),
    )

    # ── Chart 2: Tok/sec/gpu  vs  Median E2E Latency ─────────────────────────
    make_chart(
        rows,
        x_key="Median_E2E_Latency (sec)",
        y_key="Tok/sec/gpu",
        title="Throughput vs Median E2E Latency",
        xlabel="Median E2E Latency (sec)",
        output_path=str(out / "chart2_tokpergpu_vs_median_latency.png"),
    )

    # ── Chart 3: Tok/sec/gpu  vs  P99 E2E Latency ────────────────────────────
    make_chart(
        rows,
        x_key="P99_E2E_Latency (sec)",
        y_key="Tok/sec/gpu",
        title="Throughput vs P99 E2E Latency",
        xlabel="P99 E2E Latency (sec)",
        output_path=str(out / "chart3_tokpergpu_vs_p99_latency.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
