#!/usr/bin/env python3
"""
Plot best trade-offs from a Sundew sweep.

Usage
-----
python -m benchmarks.plot_best_tradeoffs --csv results/sweep_cm.csv \
  --out results/plots/best_tradeoffs.png --top-n 10 --sort f1,precision

Notes
-----
- Expects columns produced by `benchmarks/sweep_ecg.py`, including:
  f1, precision, recall, estimated_energy_savings_pct (or energy_savings_pct),
  fp, tn (for FP rate), and identification fields like preset, activation_threshold,
  gate_temperature, target_activation_rate, refractory, etc.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


NUMERIC_FIELDS = {
    "activation_threshold",
    "gate_temperature",
    "target_activation_rate",
    "refractory",
    "precision",
    "recall",
    "f1",
    "activation_rate",
    "ema_activation_rate",
    "avg_processing_time",
    "total_energy_spent",
    "energy_remaining",
    "threshold_final",
    "baseline_energy_cost",
    "actual_energy_cost",
    "estimated_energy_savings_pct",
    "energy_savings_pct",
    "total_inputs",
    "activations",
    "tp",
    "fp",
    "fn",
    "tn",
}


def _read_rows(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            for k in list(row.keys()):
                if k in NUMERIC_FIELDS and row[k] != "":
                    try:
                        row[k] = float(row[k])
                    except ValueError:
                        pass
            rows.append(row)
    return rows


def _rank(rows: List[Dict], sort_fields: List[str], top_n: int) -> List[Dict]:
    valid = {"f1", "precision", "recall", "estimated_energy_savings_pct", "energy_savings_pct"}
    for s in sort_fields:
        if s not in valid:
            raise ValueError(f"Unsupported sort field '{s}'. Use one of: {sorted(valid)}")

    def key(r: Dict):
        return tuple(r.get(s, 0.0) for s in sort_fields)

    ranked = sorted(rows, key=key, reverse=True)
    return ranked[:top_n]


def _fp_rate(row: Dict) -> float:
    fp = row.get("fp", 0.0)
    tn = row.get("tn", 0.0)
    denom = fp + tn
    return (fp / denom) if denom > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot best trade-offs (F1 vs Energy Savings)")
    ap.add_argument("--csv", required=True, help="Sweep CSV (e.g., results/sweep_cm.csv)")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--top-n", type=int, default=10, help="Top-N to plot after sorting")
    ap.add_argument("--sort", default="f1,precision", help="Comma-separated sort fields (desc)")
    args = ap.parse_args()

    rows = _read_rows(args.csv)
    if not rows:
        raise RuntimeError(f"No rows found in {args.csv}")

    sort_fields = [s.strip() for s in args.sort.split(",") if s.strip()]
    best = _rank(rows, sort_fields, args.top_n)

    # X = energy savings %, Y = F1
    xs = []
    ys = []
    labels = []

    def get_savings(row: Dict) -> float:
        return float(row.get("estimated_energy_savings_pct",
                             row.get("energy_savings_pct", 0.0)))

    for r in best:
        xs.append(get_savings(r))
        ys.append(r.get("f1", 0.0))

        # Build a compact label
        thr = r.get("activation_threshold", "")
        T = r.get("gate_temperature", "")
        targ = r.get("target_activation_rate", "")
        ref = r.get("refractory", "")
        fpr = _fp_rate(r)

        labels.append(f"thr={thr:.2f}, T={T:.2f}, p*={targ:.3f}, ref={int(ref)} | FP%={100.0*fpr:.2f}")

    # Plot
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys)

    # Annotate each point
    for x, y, lab in zip(xs, ys, labels):
        plt.annotate(lab, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)

    plt.xlabel("Estimated Energy Savings (%)")
    plt.ylabel("F1 score")
    plt.title("Best Trade-offs: F1 vs Energy Savings (labels show FP-rate)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print(f"✅ saved plot: {args.out}")


if __name__ == "__main__":
    main()
