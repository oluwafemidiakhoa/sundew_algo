#!/usr/bin/env python3
"""
Grid sweep over {activation_threshold} × {gate_temperature} × {target_activation_rate} × {refractory}
for the ECG CSV. Writes a single CSV with classification/energy metrics, now including tp/fp/fn/tn.

Usage
-----
python -m benchmarks.sweep_ecg ^
  --csv "data/MIT-BIH Arrhythmia Database.csv" ^
  --out results/sweep.csv --preset ecg_v1 --limit 50000

Tip: Keep refractory small (0) for MIT-BIH unless you have a reason; large values crush recall.
"""

from __future__ import annotations

import argparse
import csv
from typing import Dict, Iterable, Tuple, List, Any

from benchmarks.run_ecg import run as run_ecg


def params_grid() -> Iterable[Tuple[float, float, float, int]]:
    # Tune as desired
    thresholds = [0.55, 0.60, 0.65, 0.70]
    temps      = [0.05, 0.10, 0.15]
    targets    = [0.10, 0.12, 0.15]
    refractory = [0, 150, 200]
    for thr in thresholds:
        for T in temps:
            for p in targets:
                for ref in refractory:
                    yield thr, T, p, ref


def sweep(csv_path: str, out_csv: str, preset: str, limit: int | None) -> None:
    rows: List[Dict[str, Any]] = []
    grid = list(params_grid())
    print(f"▶️  Sweep size: {len(grid)} runs")

    for i, (thr, T, pstar, ref) in enumerate(grid, start=1):
        overrides = dict(
            activation_threshold=float(thr),
            gate_temperature=float(T),
            target_activation_rate=float(pstar),
        )
        out = run_ecg(
            csv_path=csv_path,
            preset=preset,
            limit=limit,
            save_path=None,
            refractory=ref,
            overrides=overrides,
        )

        rep = out.get("report", {})
        cnt = out.get("counts", {})

        row = dict(
            preset=preset,
            activation_threshold=thr,
            gate_temperature=T,
            target_activation_rate=pstar,
            refractory=ref,
            # classification
            f1=cnt.get("f1"),
            precision=cnt.get("precision"),
            recall=cnt.get("recall"),
            tp=cnt.get("tp"),
            fp=cnt.get("fp"),
            fn=cnt.get("fn"),
            tn=cnt.get("tn"),
            # energy / activations
            energy_savings_pct=rep.get("estimated_energy_savings_pct"),
            activations=rep.get("activations"),
            total_inputs=rep.get("total_inputs"),
        )
        rows.append(row)

        print(f"[{i:4d}/{len(grid)}] thr={thr:.3f} T={T:.3f} p*={pstar:.3f} ref={ref:3d}  "
              f"F1={row['f1']:.3f}  P={row['precision']:.3f}  R={row['recall']:.3f}  "
              f"Savings={row['energy_savings_pct']:.1f}%")

    fieldnames = [
        "preset", "activation_threshold", "gate_temperature", "target_activation_rate", "refractory",
        "f1", "precision", "recall", "tp", "fp", "fn", "tn",
        "energy_savings_pct", "activations", "total_inputs"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n✅ Sweep results saved to {out_csv}")


def main() -> None:
    ap = argparse.ArgumentParser(description="ECG parameter sweep with confusion-matrix columns.")
    ap.add_argument("--csv", required=True, help="Path to ECG CSV")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--preset", default="ecg_v1", help="Config preset (default: ecg_v1)")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on samples (e.g., 50000)")
    args = ap.parse_args()

    sweep(args.csv, args.out, args.preset, args.limit)


if __name__ == "__main__":
    main()
