#!/usr/bin/env python3
"""
Plot sweep results CSV produced by benchmarks/sweep_ecg.py.

Generates:
- f1_vs_energy.png           (F1 vs estimated energy savings %)
- recall_vs_precision.png    (Recall vs Precision; points colored by gate_temperature)
- f1_heatmap_threshold_temp.png (optional poor-man "heatmap" scatter of F1 vs (thr,temp))

Usage
-----
python -m benchmarks.plot_sweep --csv results/sweep.csv --out results/plots_sweep
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from typing import List, Dict

import matplotlib.pyplot as plt


def read_rows(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # cast numeric fields where relevant
            for k in [
                "activation_threshold",
                "gate_temperature",
                "target_activation_rate",
                "refractory",
                "precision",
                "recall",
                "f1",
                "estimated_energy_savings_pct",
            ]:
                if k in row and row[k] != "":
                    row[k] = float(row[k])
            rows.append(row)
    return rows


def plot_f1_vs_energy(rows: List[Dict], out_dir: str) -> str:
    xs = [row["estimated_energy_savings_pct"] for row in rows]
    ys = [row["f1"] for row in rows]
    labels = [
        f"thr={row['activation_threshold']:.2f}, T={row['gate_temperature']:.2f}, "
        f"p*={row['target_activation_rate']:.2f}, ref={int(row['refractory'])}"
        for row in rows
    ]

    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, s=40)
    plt.xlabel("Estimated Energy Savings (%)")
    plt.ylabel("F1 Score")
    plt.title("F1 vs. Energy Savings")
    plt.grid(True, alpha=0.3)

    # Annotate top-5 by F1
    top_idx = sorted(range(len(rows)), key=lambda i: ys[i], reverse=True)[:5]
    for i in top_idx:
        plt.annotate(labels[i], (xs[i], ys[i]), textcoords="offset points", xytext=(6, 6), fontsize=8)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "f1_vs_energy.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_recall_vs_precision(rows: List[Dict], out_dir: str) -> str:
    xs = [row["precision"] for row in rows]
    ys = [row["recall"] for row in rows]
    temps = [row["gate_temperature"] for row in rows]

    # size by target_activation_rate
    sizes = [200 * float(row["target_activation_rate"]) for row in rows]

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(xs, ys, s=sizes, c=temps)
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("Recall vs. Precision (color = gate_temperature, size ~ target)")
    plt.grid(True, alpha=0.3)
    cbar = plt.colorbar(sc)
    cbar.set_label("gate_temperature")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "recall_vs_precision.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_f1_scatter_threshold_temp(rows: List[Dict], out_dir: str) -> str:
    thr = [row["activation_threshold"] for row in rows]
    temp = [row["gate_temperature"] for row in rows]
    f1 = [row["f1"] for row in rows]

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(thr, temp, s=[200 * v for v in f1], c=f1)
    plt.xlabel("activation_threshold")
    plt.ylabel("gate_temperature")
    plt.title("F1 scatter over (threshold, temperature)")
    plt.grid(True, alpha=0.3)
    cbar = plt.colorbar(sc)
    cbar.set_label("F1")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "f1_scatter_threshold_temp.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot sweep results CSV")
    ap.add_argument("--csv", required=True, help="Path to CSV from sweep_ecg.py")
    ap.add_argument("--out", required=True, help="Output directory for plots")
    args = ap.parse_args()

    rows = read_rows(args.csv)
    p1 = plot_f1_vs_energy(rows, args.out)
    p2 = plot_recall_vs_precision(rows, args.out)
    p3 = plot_f1_scatter_threshold_temp(rows, args.out)

    print("✅ Saved plots:")
    print("  ", p1)
    print("  ", p2)
    print("  ", p3)


if __name__ == "__main__":
    main()
