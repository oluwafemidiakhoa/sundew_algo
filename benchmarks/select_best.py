#!/usr/bin/env python3
"""
Robust selector for Sundew sweep CSVs.

- Accepts multiple savings column names or computes from baseline/actual energy
- Supports confusion-matrix constraints (max-fn, max-fp-rate) when tp/fp/fn/tn exist
- Adds --describe to summarize metric distributions pre-filter
- Can emit a Markdown research update block for your repo

Usage (loose):
  python -m benchmarks.select_best ^
    --csv results\\sweep_cm.csv ^
    --out-csv results\\best_loose.csv ^
    --out-md results\\best_loose.md ^
    --min-savings 85 --sort f1,precision --top-n 30

Usage (with CM constraints + research note):
  python -m benchmarks.select_best ^
    --csv results\\sweep_cm.csv ^
    --out-csv results\\best_by_counts.csv ^
    --out-md results\\best_by_counts.md ^
    --research-md results\\updates\\2025-09-ecg-mitbih.md ^
    --dataset-name "MIT-BIH Arrhythmia Database" ^
    --dataset-notes "CSV from PhysioNet; ~50k rows; binary abnormal-beat labels; ecg_v1 sweep." ^
    --min-savings 88 --max-fn 9000 --max-fp-rate 0.08 ^
    --sort f1,precision --top-n 20
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from statistics import median
from typing import Dict, List, Tuple, Optional

# Accept these savings aliases; compute if absent
SAVINGS_ALIASES = [
    "estimated_energy_savings_pct",
    "savings_pct",
    "energy_savings",
    "energy_savings_pct",
]

NUMERIC_FIELDS_BASE = {
    "activation_threshold", "gate_temperature", "target_activation_rate", "refractory",
    "precision", "recall", "f1", "activation_rate", "ema_activation_rate",
    "avg_processing_time", "total_energy_spent", "energy_remaining",
    "threshold_final", "baseline_energy_cost", "actual_energy_cost",
    "total_inputs", "activations",
    "tp", "fp", "fn", "tn",
}
NUMERIC_FIELDS = set(NUMERIC_FIELDS_BASE) | set(SAVINGS_ALIASES)

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

def _ensure_savings_pct(row: Dict) -> float:
    # 1) direct column if present
    for k in SAVINGS_ALIASES:
        v = row.get(k, None)
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return float(v)  # if string with numeric contents
        except Exception:
            pass
    # 2) compute from baseline/actual if both present
    b = row.get("baseline_energy_cost", None)
    a = row.get("actual_energy_cost", None)
    if isinstance(b, (int, float)) and isinstance(a, (int, float)) and b and math.isfinite(b):
        return max(-100.0, min(100.0, 100.0 * (b - a) / b))
    # 3) fallback 0
    return 0.0

def _fp_rate(row: Dict) -> Optional[float]:
    tn = row.get("tn"); fp = row.get("fp")
    if isinstance(tn, (int, float)) and isinstance(fp, (int, float)) and (tn + fp) > 0:
        return float(fp) / float(tn + fp)
    return None

def _has_cm(row: Dict) -> bool:
    return all(isinstance(row.get(k), (int, float)) for k in ("tp","fp","fn","tn"))

def _passes_constraints(
    row: Dict,
    min_savings: float,
    min_precision: float,
    min_recall: float,
    min_f1: float,
    max_fn: Optional[int],
    max_fp_rate: Optional[float],
) -> bool:
    if _ensure_savings_pct(row) < min_savings:
        return False
    if row.get("precision", 0.0) < min_precision:
        return False
    if row.get("recall", 0.0) < min_recall:
        return False
    if row.get("f1", 0.0) < min_f1:
        return False
    if max_fn is not None and _has_cm(row):
        if row.get("fn", 0.0) > max_fn:
            return False
    if max_fp_rate is not None and _has_cm(row):
        fpr = _fp_rate(row)
        if fpr is not None and fpr > max_fp_rate:
            return False
    return True

def _rank_key_factory(sort_fields: List[str]):
    valid = {"f1", "precision", "recall", "estimated_energy_savings_pct", "savings_pct",
             "energy_savings_pct", "energy_savings", "activation_rate"}
    for s in sort_fields:
        if s not in valid:
            raise ValueError(f"Unsupported sort field '{s}'. Use one of: {sorted(valid)}")
    def key(row: Dict):
        vals = []
        for f in sort_fields:
            if f in SAVINGS_ALIASES:
                vals.append(_ensure_savings_pct(row))
            else:
                vals.append(row.get(f, 0.0))
        return tuple(vals)
    return key

def _write_csv(path: str, rows: List[Dict], fields: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fields}
            out["estimated_energy_savings_pct"] = f"{_ensure_savings_pct(r):.6g}"
            if "fp_rate" in fields:
                fpr = _fp_rate(r)
                out["fp_rate"] = f"{fpr:.6g}" if fpr is not None else ""
            w.writerow(out)

def _write_md_table(f, rows: List[Dict], fields: List[str]) -> None:
    f.write("| " + " | ".join(fields) + " |\n")
    f.write("|" + "|".join(["---"] * len(fields)) + "|\n")
    def fmt(k, r):
        if k in ("estimated_energy_savings_pct","savings_pct","energy_savings","energy_savings_pct"):
            return f"{_ensure_savings_pct(r):.3f}"
        v = r.get(k, "")
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)
    for r in rows:
        f.write("| " + " | ".join(fmt(k, r) for k in fields) + " |\n")

def _write_md(path: str, rows: List[Dict], fields: List[str], meta: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Best Sundew Configurations (Sweep Selection)\n\n")
        f.write("**Constraints:**\n\n")
        f.write(f"- Min energy savings: **{meta['min_savings']}%**\n")
        f.write(f"- Min precision: **{meta['min_precision']}**\n")
        f.write(f"- Min recall: **{meta['min_recall']}**\n")
        f.write(f"- Min F1: **{meta['min_f1']}**\n")
        if meta.get("max_fn") is not None:
            f.write(f"- Max FN: **{meta['max_fn']}**\n")
        if meta.get("max_fp_rate") is not None:
            f.write(f"- Max FP rate: **{meta['max_fp_rate']}**\n")
        f.write(f"- Sort: **{', '.join(meta['sort_fields'])}** (desc)\n")
        f.write(f"- Top-N: **{meta['top_n']}**\n")
        f.write("\n---\n\n**Columns shown:**\n\n")
        f.write(", ".join(fields) + "\n\n---\n\n")
        _write_md_table(f, rows, fields)

def _write_research_update(path: str, rows: List[Dict], meta: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Research Update — {meta.get('dataset_name','Dataset')}\n\n")
        if meta.get("dataset_notes"):
            f.write(f"> {meta['dataset_notes']}\n\n")
        f.write("**Selection criteria:**\n\n")
        f.write(f"- Savings ≥ {meta['min_savings']}%\n")
        f.write(f"- Precision ≥ {meta['min_precision']}, Recall ≥ {meta['min_recall']}, F1 ≥ {meta['min_f1']}\n")
        if meta.get("max_fn") is not None:
            f.write(f"- FN ≤ {meta['max_fn']}\n")
        if meta.get("max_fp_rate") is not None:
            f.write(f"- FP rate ≤ {meta['max_fp_rate']}\n")
        f.write("\n**Top configurations:**\n\n")
        fields = [
            "preset","activation_threshold","gate_temperature","target_activation_rate","refractory",
            "precision","recall","f1","estimated_energy_savings_pct","activation_rate","threshold_final",
            "tp","fp","fn","tn"
        ]
        _write_md_table(f, rows, fields)
        f.write("\n*Generated automatically from sweep results.*\n")

def _describe(rows: List[Dict]) -> str:
    keys = ["f1","precision","recall","activation_rate"] + list(SAVINGS_ALIASES) + \
           ["baseline_energy_cost","actual_energy_cost","tp","fp","fn","tn"]
    lines = []
    for k in keys:
        vals = [r.get(k) if k not in SAVINGS_ALIASES else _ensure_savings_pct(r) for r in rows]
        vals = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
        if not vals:
            continue
        lines.append(f"{k:28s} min={min(vals):.4g}  med={median(vals):.4g}  max={max(vals):.4g}")
    return "\n".join(lines)

def select_best(
    csv_in: str,
    csv_out: str,
    md_out: Optional[str],
    research_md: Optional[str],
    dataset_name: Optional[str],
    dataset_notes: Optional[str],
    min_savings: float,
    min_precision: float,
    min_recall: float,
    min_f1: float,
    max_fn: Optional[int],
    max_fp_rate: Optional[float],
    sort_fields: List[str],
    top_n: int,
    describe: bool,
) -> Tuple[str, Optional[str], Optional[str]]:
    rows = _read_rows(csv_in)
    if not rows:
        raise RuntimeError(f"No rows found in {csv_in}")

    if describe:
        print("=== Distribution (pre-filter) ===")
        print(_describe(rows))

    filt = [
        r for r in rows
        if _passes_constraints(r, min_savings, min_precision, min_recall, min_f1, max_fn, max_fp_rate)
    ]
    if not filt:
        raise RuntimeError(
            "No configurations satisfied the constraints. "
            "Try lowering --min-savings/precision/recall/f1 or relaxing FN/FP limits."
        )

    keyf = _rank_key_factory(sort_fields)
    ranked = sorted(filt, key=keyf, reverse=True)
    best = ranked[:top_n]

    fields = [
        "preset","activation_threshold","gate_temperature","target_activation_rate","refractory",
        "precision","recall","f1","estimated_energy_savings_pct","activation_rate",
        "threshold_final","total_inputs","activations","tp","fp","fn","tn","fp_rate"
    ]
    _write_csv(csv_out, best, fields)

    md_path = None
    if md_out:
        _write_md(
            md_out, best, fields,
            meta=dict(
                min_savings=min_savings, min_precision=min_precision, min_recall=min_recall, min_f1=min_f1,
                max_fn=max_fn, max_fp_rate=max_fp_rate, sort_fields=sort_fields, top_n=top_n,
            ),
        )
        md_path = md_out

    research_path = None
    if research_md:
        _write_research_update(
            research_md, best,
            meta=dict(
                dataset_name=dataset_name or "Dataset",
                dataset_notes=dataset_notes or "",
                min_savings=min_savings, min_precision=min_precision, min_recall=min_recall, min_f1=min_f1,
                max_fn=max_fn, max_fp_rate=max_fp_rate, sort_fields=sort_fields, top_n=top_n,
            ),
        )
        research_path = research_md

    return csv_out, md_path, research_path

def main() -> None:
    ap = argparse.ArgumentParser(description="Select best Sundew configs from sweep CSV (robust)")
    ap.add_argument("--csv", required=True, help="Input sweep CSV")
    ap.add_argument("--out-csv", required=True, help="Output CSV with top-N rows")
    ap.add_argument("--out-md", default=None, help="Optional Markdown summary path")
    ap.add_argument("--research-md", default=None, help="Optional research update block path")
    ap.add_argument("--dataset-name", default=None, help="Dataset name for research update")
    ap.add_argument("--dataset-notes", default=None, help="Dataset notes for research update")

    ap.add_argument("--min-savings", type=float, default=80.0, help="Minimum energy savings (%)")
    ap.add_argument("--min-precision", type=float, default=0.0, help="Minimum precision")
    ap.add_argument("--min-recall", type=float, default=0.0, help="Minimum recall")
    ap.add_argument("--min-f1", type=float, default=0.0, help="Minimum F1")
    ap.add_argument("--max-fn", type=int, default=None, help="Maximum false negatives (requires tp/fp/fn/tn)")
    ap.add_argument("--max-fp-rate", type=float, default=None, help="Maximum FP rate in [0,1] (requires tp/fp/fn/tn)")

    ap.add_argument("--sort", default="f1", help="Comma-separated ranking fields (desc). "
                    "Options: f1,precision,recall,estimated_energy_savings_pct,savings_pct,energy_savings_pct,energy_savings,activation_rate")
    ap.add_argument("--top-n", type=int, default=10, help="Number of rows to return")
    ap.add_argument("--describe", action="store_true", help="Print distribution summary pre-filter")

    args = ap.parse_args()
    sort_fields = [s.strip() for s in args.sort.split(",") if s.strip()]

    csv_out, md_out, research_out = select_best(
        csv_in=args.csv,
        csv_out=args.out_csv,
        md_out=args.out_md,
        research_md=args.research_md,
        dataset_name=args.dataset_name,
        dataset_notes=args.dataset_notes,
        min_savings=args.min_savings,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
        min_f1=args.min_f1,
        max_fn=args.max_fn,
        max_fp_rate=args.max_fp_rate,
        sort_fields=sort_fields,
        top_n=args.top_n,
        describe=args.describe,
    )

    print(f"✅ Wrote: {csv_out}")
    if md_out:
        print(f"✅ Wrote: {md_out}")
    if research_out:
        print(f"✅ Wrote: {research_out}")

if __name__ == "__main__":
    main()
