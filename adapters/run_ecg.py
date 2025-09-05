#!/usr/bin/env python3
"""
Run Sundew on an ECG CSV (e.g., MIT-BIH) and save results.

Example
-------
python -m benchmarks.run_ecg --csv "data/MIT-BIH Arrhythmia Database.csv" \
    --preset tuned_v2 --limit 50000 --save results/real_ecg_run.json

Notes
-----
- Pure stdlib (no pandas). Reads CSV streaming.
- Expects at least one numeric signal column (we'll try common names).
- If a label column exists, we keep it in the per-step log for later eval.
- Basic features:
    * magnitude         -> scaled |signal| into [0,100]
    * anomaly_score     -> sigmoid(|zscore| / 3)
    * context_relevance -> EMA of anomaly (slow context)
    * urgency           -> normalized derivative magnitude in [0,1]
- Optional refractory window to reduce multiple detections on the same beat.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Tuple

from sundew import SundewAlgorithm
from sundew.config_presets import get_preset


# ---------------------- Streaming stats (for z-score, EMA) ----------------------

class RunningStats:
    """Welford’s algorithm for running mean/std."""
    __slots__ = ("n", "mean", "M2")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.M2 / max(1, self.n - 1)

    @property
    def std(self) -> float:
        v = self.variance
        return math.sqrt(v) if v > 1e-12 else 0.0


class EMA:
    __slots__ = ("alpha", "y")

    def __init__(self, alpha: float, init: float = 0.0) -> None:
        self.alpha = float(alpha)
        self.y = float(init)

    def update(self, x: float) -> float:
        self.y = (1 - self.alpha) * self.y + self.alpha * x
        return self.y


# ---------------------- CSV parsing helpers ----------------------

COMMON_SIGNAL_KEYS = [
    "ml2", "signal", "ecg", "value", "val", "amplitude", "lead", "lead1", "lead2",
]

COMMON_LABEL_KEYS = [
    "label", "annotation", "ann", "y", "class", "arrhythmia", "beat_type",
]


def _best_key(header: List[str], candidates: List[str]) -> Optional[str]:
    lower = {h.lower(): h for h in header}
    for k in candidates:
        if k in lower:
            return lower[k]
    # fuzzy contains
    for h in header:
        hlow = h.lower()
        for k in candidates:
            if k in hlow:
                return h
    return None


def _label_to_binary(v: str) -> int:
    """
    Map raw label to 0/1.
    - Numeric: any nonzero -> 1
    - Strings: treat typical arrhythmia symbols or non-'N' as positive
    Customize as needed for your CSV.
    """
    if v is None:
        return 0
    s = str(v).strip()
    if s == "":
        return 0
    # numeric?
    try:
        f = float(s)
        return 1 if abs(f) > 1e-9 else 0
    except ValueError:
        pass
    # common ECG codes: N normal; others like V, S, F, A often arrhythmic
    if s.upper() in ("N", "NORMAL"):
        return 0
    return 1


def ecg_events_from_csv(path: str) -> Iterable[Dict]:
    """
    Yields dicts with 'signal' and optional 'label' from CSV.
    We don't enforce a specific schema; we try to pick reasonable columns.
    """
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if not header:
            raise ValueError("CSV appears to have no header.")

        key_signal = _best_key(header, COMMON_SIGNAL_KEYS)
        if key_signal is None:
            # fallback: first numeric-looking column
            key_signal = header[0]

        key_label = _best_key(header, COMMON_LABEL_KEYS)

        for row in reader:
            # parse signal
            raw = row.get(key_signal, "")
            try:
                sig = float(raw)
            except ValueError:
                # non-numeric row; skip
                continue

            item = {"signal": sig}
            if key_label is not None:
                item["label"] = _label_to_binary(row.get(key_label))
            yield item


# ---------------------- Feature engineering ----------------------

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def make_feature_stream(rows: Iterable[Dict], max_abs_for_scale: float = 3.0) -> Iterable[Dict]:
    """
    Convert raw ECG rows into Sundew-compatible lightweight features.
    Returns dicts with:
      - magnitude (0..100)
      - anomaly_score (0..1)
      - context_relevance (0..1)
      - urgency (0..1)
      - label (optional)
    """
    stats = RunningStats()
    ema_anom = EMA(alpha=0.02, init=0.0)
    prev_sig: Optional[float] = None

    # derive a robust scale for magnitude (use running std, clamp by max_abs_for_scale)
    for row in rows:
        sig = float(row["signal"])
        stats.update(sig)
        sd = stats.std or 1.0

        # z-score and anomaly
        z = (sig - stats.mean) / sd
        anom = sigmoid(abs(z) / max_abs_for_scale)  # smooth in [0,1]
        ctx = ema_anom.update(anom)

        # derivative / urgency
        if prev_sig is None:
            deriv = 0.0
        else:
            deriv = sig - prev_sig
        prev_sig = sig
        # normalized derivative via tanh
        urg = math.tanh(abs(deriv) / (5.0 * sd + 1e-6))
        urg = max(0.0, min(1.0, urg))

        # magnitude -> scale |z| into 0..100 for interpretability
        mag = max(0.0, min(100.0, 100.0 * (abs(z) / (6.0 + 1e-9))))  # ~6σ maps near 100

        out = {
            "magnitude": mag,
            "anomaly_score": anom,
            "context_relevance": ctx,
            "urgency": urg,
        }
        if "label" in row:
            out["label"] = int(row["label"])
        yield out


# ---------------------- Main runner ----------------------

def run(csv_path: str,
        preset: str = "tuned_v2",
        limit: Optional[int] = None,
        save_path: Optional[str] = None,
        refractory: int = 0,
        overrides: Optional[Dict] = None) -> Dict:
    """
    Execute Sundew on a CSV stream.
    Returns a dict with:
      - config, report, counts, and per-step (subset) if saved.
    """
    cfg = get_preset(preset, overrides=overrides or {})
    algo = SundewAlgorithm(cfg)

    y_true: List[int] = []
    y_pred: List[int] = []

    cooldown = 0
    processed = 0

    for i, ev in enumerate(make_feature_stream(ecg_events_from_csv(csv_path))):
        if limit and i >= limit:
            break

        gt = int(ev.get("label", 0))
        y_true.append(gt)

        if cooldown > 0:
            # pay only evaluation cost: emulate by forcing no-activation
            # (we could optionally call algo.process with a bypass; here we skip)
            y_pred.append(0)
            cooldown -= 1
            continue

        res = algo.process(ev)
        pred = 1 if res is not None else 0
        y_pred.append(pred)

        if pred == 1 and refractory > 0:
            cooldown = refractory

        processed += 1

    # Metrics (confusion if labels present)
    tp = fp = fn = tn = 0
    if any(y_true):
        for t, p in zip(y_true, y_pred):
            if t == 1 and p == 1:
                tp += 1
            elif t == 0 and p == 1:
                fp += 1
            elif t == 1 and p == 0:
                fn += 1
            else:
                tn += 1

    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = (2 * prec * rec) / max(1e-12, (prec + rec)) if (tp + fp + fn) > 0 else 0.0

    report = algo.report()
    out = {
        "config": cfg.__dict__,
        "report": report,
        "counts": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1,
            "total_inputs": len(y_true),
            "activations": sum(y_pred),
        },
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"✅ Saved results to {save_path}")

        # Also print a compact summary to stdout
        for k, v in report.items():
            if isinstance(v, float):
                print(f"{k:25s} : {v}")
            else:
                print(f"{k:25s} : {v}")

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Sundew on an ECG CSV and save results.")
    ap.add_argument("--csv", required=True, help="Path to ECG CSV (e.g., MIT-BIH export).")
    ap.add_argument("--preset", default="tuned_v2", help="Config preset name.")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit of samples to process.")
    ap.add_argument("--save", type=str, default=None, help="Optional JSON output path.")
    ap.add_argument("--refractory", type=int, default=0,
                   help="Samples to suppress further activations after a detection.")
    args = ap.parse_args()

    run(
        csv_path=args.csv,
        preset=args.preset,
        limit=args.limit,
        save_path=args.save,
        refractory=args.refractory,
    )


if __name__ == "__main__":
    main()
