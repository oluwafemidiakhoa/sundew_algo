#!/usr/bin/env python3
"""
Threshold-only curve for ECG:
- Fix gate_temperature (T), target_activation_rate (p*), and refractory.
- Sweep activation_threshold over a range.
- Write CSV and a simple PNG plot of F1 / Precision / Recall vs. threshold.
- Also records energy-savings % per point.

Usage (from repo root)
----------------------
python -m benchmarks.curve_threshold ^
  --csv "data/MIT-BIH Arrhythmia Database.csv" ^
  --preset ecg_v1 --limit 50000 ^
  --thr-min 0.50 --thr-max 0.70 --thr-step 0.01 ^
  --gate-temperature 0.10 --target 0.15 --refractory 0 ^
  --out-csv results/curve_thr.csv ^
  --out-png results/curve_thr.png
"""

from __future__ import annotations

import argparse
import csv
import math
from typing import List, Dict, Any

import matplotlib.pyplot as plt  # std plotting; used only if --out-png

from benchmarks.run_ecg import run as run_ecg  # unified ECG runner


def frange(start: float, stop: float, step: float) -> List[float]:
    vals = []
    x = start
    # include stop (with tolerance) to avoid floating-point drift
    while x <= stop + 1e-12:
        vals.append(round(x, 6))
        x += step
    return vals


def main() -> None:
    ap = argparse.ArgumentParser(description="F1/Precision/Recall vs activation_threshold for Sundew ECG.")
    ap.add_argument("--csv", required=True, help="Path to ECG CSV (e.g., data/MIT-BIH Arrhythmia Database.csv)")
    ap.add_argument("--preset", default="ecg_v1", help="Config preset to start from (default: ecg_v1)")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of samples")
    ap.add_argument("--thr-min", type=float, required=True, help="Min activation_threshold (e.g., 0.50)")
    ap.add_argument("--thr-max", type=float, required=True, help="Max activation_threshold (e.g., 0.70)")
    ap.add_argument("--thr-step", type=float, required=True, help="Step size (e.g., 0.01)")
    ap.add_argument("--gate-temperature", type=float, required=True, help="Fixed temperature T (e.g., 0.10)")
    ap.add_argument("--target", type=float, required=True, help="Fixed p* target activation rate (e.g., 0.15)")
    ap.add_argument("--refractory", type=int, default=0, help="Refractory (samples). 0 disables. (default: 0)")
    ap.add_argument("--out-csv", required=True, help="Output CSV file for the curve results")
    ap.add_argument("--out-png", default=None, help="Optional PNG path for F1/P/R vs threshold")
    args = ap.parse_args()

    thresholds = frange(args.thr_min, args.thr_max, args.thr_step)
    rows: List[Dict[str, Any]] = []

    print(f"▶️  Threshold curve: {len(thresholds)} points "
          f"(thr ∈ [{args.thr_min:.3f}, {args.thr_max:.3f}] step {args.thr_step:.3f}), "
          f"T={args.gate_temperature:.3f}, p*={args.target:.3f}, ref={args.refractory}")

    for i, thr in enumerate(thresholds, start=1):
        overrides = dict(
            activation_threshold=float(thr),
            gate_temperature=float(args.gate_temperature),
            target_activation_rate=float(args.target),
        )
        out = run_ecg(
            csv_path=args.csv,
            preset=args.preset,
            limit=args.limit,
            save_path=None,  # not saving per-run JSONs for this curve
            refractory=args.refractory,
            overrides=overrides,
        )

        rep = out.get("report", {})
        cnt = out.get("counts", {})

        row = dict(
            activation_threshold=thr,
            gate_temperature=args.gate_temperature,
            target_activation_rate=args.target,
            refractory=args.refractory,
            f1=cnt.get("f1"),
            precision=cnt.get("precision"),
            recall=cnt.get("recall"),
            tp=cnt.get("tp"),
            fp=cnt.get("fp"),
            fn=cnt.get("fn"),
            tn=cnt.get("tn"),
            energy_savings_pct=rep.get("estimated_energy_savings_pct"),
            activations=rep.get("activations"),
            total_inputs=rep.get("total_inputs"),
        )
        rows.append(row)

        print(f"[{i:3d}/{len(thresholds)}] thr={thr:.3f}  "
              f"F1={row['f1']:.3f}  P={row['precision']:.3f}  R={row['recall']:.3f}  "
              f"Savings={row['energy_savings_pct']:.1f}%")

    # Write CSV
    fieldnames = [
        "activation_threshold", "gate_temperature", "target_activation_rate", "refractory",
        "f1", "precision", "recall", "tp", "fp", "fn", "tn",
        "energy_savings_pct", "activations", "total_inputs"
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"✅ Curve CSV saved to {args.out_csv}")

    # Plot (optional)
    if args.out_png:
        xs = [r["activation_threshold"] for r in rows]
        f1 = [r["f1"] for r in rows]
        pr = [r["precision"] for r in rows]
        rc = [r["recall"] for r in rows]

        plt.figure(figsize=(8, 5))
        plt.plot(xs, f1, marker="o", label="F1")
        plt.plot(xs, pr, marker="s", label="Precision")
        plt.plot(xs, rc, marker="^", label="Recall")
        plt.xlabel("activation_threshold")
        plt.ylabel("score")
        plt.title("F1 / Precision / Recall vs activation_threshold")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_png, dpi=160)
        print(f"✅ Curve PNG saved to {args.out_png}")


if __name__ == "__main__":
    main()
