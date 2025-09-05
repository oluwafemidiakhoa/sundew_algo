#!/usr/bin/env python3
"""
Re-run a selected configuration from a sweep CSV on the ECG dataset.

- Reads a row from results/sweep.csv (or best.csv / best_strict.csv).
- Extracts: activation_threshold, gate_temperature, target_activation_rate, refractory.
- Calls benchmarks.run_ecg.run(...) with those values (plus optional preset/overrides/limit).
- Saves a JSON and optionally runs the classification evaluator.

Usage
-----
# Re-run row 3 from the strict-best file, save JSON, then evaluate
python -m benchmarks.rerun_from_row ^
  --sweep results\best_strict.csv ^
  --row-index 3 ^
  --data "data\MIT-BIH Arrhythmia Database.csv" ^
  --save results\repro_row3.json ^
  --eval

# Re-run by matching values instead of index
python -m benchmarks.rerun_from_row ^
  --sweep results\sweep.csv ^
  --match "activation_threshold=0.58,gate_temperature=0.08,target_activation_rate=0.15,refractory=0" ^
  --data "data\MIT-BIH Arrhythmia Database.csv" ^
  --save results\repro_thr058_T008_p015.json

# Limit samples and override preset on the fly
python -m benchmarks.rerun_from_row ^
  --sweep results\best.csv ^
  --row-index 1 ^
  --data "data\MIT-BIH Arrhythmia Database.csv" ^
  --preset ecg_v1 ^
  --limit 50000 ^
  --save results\repro_best1.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

from benchmarks.run_ecg import run as run_ecg  # our unified runner


NUMERIC_FIELDS = {"activation_threshold", "gate_temperature", "target_activation_rate", "refractory"}


def _read_rows(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # cast common numeric parameters
            for k in list(row.keys()):
                if k in NUMERIC_FIELDS and row[k] != "":
                    try:
                        row[k] = float(row[k]) if k != "refractory" else int(float(row[k]))
                    except ValueError:
                        pass
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def _parse_match(s: str) -> Dict[str, float | int]:
    """
    Parse --match "activation_threshold=0.58,gate_temperature=0.08,target_activation_rate=0.15,refractory=0"
    """
    out: Dict[str, float | int] = {}
    for pair in s.split(","):
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        k, v = k.strip(), v.strip()
        if k not in NUMERIC_FIELDS:
            raise ValueError(f"Unsupported match key '{k}'. Use one of: {sorted(NUMERIC_FIELDS)}")
        # refractory is int; others are float
        if k == "refractory":
            out[k] = int(float(v))
        else:
            out[k] = float(v)
    missing = [k for k in ("activation_threshold", "gate_temperature", "target_activation_rate", "refractory") if k not in out]
    if missing:
        raise ValueError(f"--match is missing: {missing}")
    return out


def _select_row(rows: List[Dict], row_index: Optional[int], match: Optional[Dict]) -> Dict:
    if row_index is not None:
        if row_index < 1 or row_index > len(rows):
            raise IndexError(f"--row-index {row_index} out of range 1..{len(rows)}")
        return rows[row_index - 1]

    assert match is not None
    # Exact numeric match (within tolerance for floats)
    def close(a, b, tol=1e-9):
        return abs(float(a) - float(b)) <= tol

    for r in rows:
        ok = (
            close(r.get("activation_threshold"), match["activation_threshold"]) and
            close(r.get("gate_temperature"), match["gate_temperature"]) and
            close(r.get("target_activation_rate"), match["target_activation_rate"]) and
            int(r.get("refractory", 0)) == int(match["refractory"])
        )
        if ok:
            return r
    raise RuntimeError("No row matched the provided --match values.")


def _build_overrides(row: Dict, extra_overrides: Optional[str]) -> Dict:
    """
    Create SundewConfig overrides from row + CLI extras.
    """
    overrides: Dict = {
        "activation_threshold": float(row["activation_threshold"]),
        "gate_temperature": float(row["gate_temperature"]),
        "target_activation_rate": float(row["target_activation_rate"]),
    }
    if extra_overrides:
        for pair in extra_overrides.split(","):
            if "=" not in pair:
                continue
            k, v = pair.split("=", 1)
            k = k.strip()
            v = v.strip()
            try:
                if "." in v or "e" in v.lower():
                    overrides[k] = float(v)
                else:
                    overrides[k] = int(v)
            except ValueError:
                if v.lower() in ("true", "false"):
                    overrides[k] = (v.lower() == "true")
                else:
                    overrides[k] = v
    return overrides


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-run Sundew from a selected sweep row.")
    ap.add_argument("--sweep", required=True, help="Sweep CSV (e.g., results/sweep.csv or results/best.csv)")
    ap.add_argument("--row-index", type=int, default=None, help="1-based row index to re-run")
    ap.add_argument("--match", type=str, default=None,
                    help='Alternative to --row-index. Example: "activation_threshold=0.58,gate_temperature=0.08,target_activation_rate=0.15,refractory=0"')
    ap.add_argument("--data", required=True, help="Path to ECG CSV (e.g., data/MIT-BIH Arrhythmia Database.csv)")
    ap.add_argument("--preset", default=None, help="Override preset name (else use row['preset'] if present, or 'tuned_v2')")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of samples to process")
    ap.add_argument("--save", type=str, default=None, help="Optional JSON output path")
    ap.add_argument("--eval", action="store_true", help="Run benchmarks.eval_classification on the saved JSON")
    ap.add_argument("--overrides", type=str, default=None, help='Extra Sundew overrides, e.g. "min_threshold=0.2,max_threshold=0.9"')

    args = ap.parse_args()
    if (args.row_index is None) == (args.match is None):
        ap.error("Specify exactly one of --row-index or --match")

    rows = _read_rows(args.sweep)
    match_dict = _parse_match(args.match) if args.match else None
    row = _select_row(rows, args.row_index, match_dict)

    # Decide preset
    preset = args.preset or row.get("preset") or "tuned_v2"

    # Refractory from row (int)
    refractory = int(float(row.get("refractory", 0)))

    # Build overrides (row -> overrides + any CLI extras)
    overrides = _build_overrides(row, args.overrides)

    # Run
    out = run_ecg(
        csv_path=args.data,
        preset=preset,
        limit=args.limit,
        save_path=args.save,
        refractory=refractory,
        overrides=overrides,
    )

    # Print a compact summary
    rep = out.get("report", {})
    cnt = out.get("counts", {})
    print("\n=== Re-run Summary ===")
    print(f"preset                 : {preset}")
    print(f"activation_threshold   : {overrides['activation_threshold']}")
    print(f"gate_temperature       : {overrides['gate_temperature']}")
    print(f"target_activation_rate : {overrides['target_activation_rate']}")
    print(f"refractory             : {refractory}")
    print(f"precision              : {cnt.get('precision')}")
    print(f"recall                 : {cnt.get('recall')}")
    print(f"f1                     : {cnt.get('f1')}")
    print(f"energy_savings_pct     : {rep.get('estimated_energy_savings_pct')}")

    # Optional evaluator
    if args.eval:
        if not args.save:
            print("\n[warn] --eval requested but no --save path provided; evaluator needs a JSON.")
            return
        try:
            # lazy import to avoid hard dependency if user doesn't want this
            from benchmarks.eval_classification import main as eval_main
            # Simulate CLI: we want to call the evaluator programmatically
            sys.argv = ["eval_classification.py", "--json", args.save]
            print("\n--- Evaluator ---")
            eval_main()
        except Exception as e:
            print(f"\n[warn] evaluator failed: {e}")


if __name__ == "__main__":
    main()
