# adapters/ecg_adapter.py
# Robust ECG CSV adapter for Sundew
from __future__ import annotations

import csv
import math
import statistics
from collections import deque
from typing import Dict, Generator, Iterable, Optional, Tuple, List

# --- Beat symbol → urgency mapping (tweak as needed) ---
# MIT-BIH typical symbols:
# N  (normal), L, R, e, j, A, a, J, S (supraventricular), V (ventricular),
# F (fusion), Q (unknown)
URGENCY_MAP = {
    "N": 0.10, "L": 0.10, "R": 0.10, "e": 0.10, "j": 0.10,  # normal-ish
    "A": 0.60, "a": 0.60, "J": 0.60, "S": 0.60,             # supraventricular
    "V": 0.90,                                              # ventricular ectopic
    "F": 0.80,                                              # fusion
    "Q": 0.20                                               # unknown / other
}

# Heuristics for column names
CAND_SIGNAL = ["lead1", "ml2", "mlii", "v1", "v2", "val", "signal", "ecg", "ecg1", "lead", "ch1"]
CAND_LABELS = ["label", "symbol", "ann", "beat", "class", "y"]
CAND_TS     = ["timestamp", "time", "t", "sec", "seconds"]


def _norm(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch == "_")


def _pick_column(header: List[str], candidates: List[str]) -> Optional[str]:
    norm = {h: _norm(h) for h in header}
    cand = set(candidates)
    for h, n in norm.items():
        if n in cand:
            return h
    # fuzzy contains match
    for h, n in norm.items():
        for c in candidates:
            if c in n:
                return h
    return None


def _rolling_stats(window: deque[float]) -> Tuple[float, float]:
    if not window:
        return 0.0, 1.0
    mu = statistics.fmean(window)
    sd = statistics.pstdev(window) or 1.0
    return mu, sd


def _to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _clean_label(x: str) -> str:
    if x is None:
        return "N"
    x = x.strip()
    # Some exports use integers; map a couple of common ones
    if x.isdigit():
        # user can refine this map; keep conservative defaults
        m = {"0": "N", "1": "S", "2": "V", "3": "F"}
        return m.get(x, "Q")
    # Take first char if a string like "N|0"
    if len(x) > 1 and "|" in x:
        x = x.split("|", 1)[0]
    return x[:1] if x else "N"


def ecg_events_from_csv(
    path: str,
    win: int = 200,
    fs_hz: float = 360.0,            # MIT-BIH default
    mag_clip: float = 3.0,
    anomaly_clip: float = 3.0,
) -> Generator[Dict, None, None]:
    """
    Yields Sundew-friendly event dicts:
      {
        "timestamp", "magnitude", "anomaly_score",
        "context_relevance", "urgency", "label"
      }
    The 'magnitude' is in 0..100 (Sundew core normalizes by /100).
    Others are in 0..1.
    """
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if not header:
            raise ValueError("CSV appears to have no header row.")

        sig_col = _pick_column(header, CAND_SIGNAL)
        lbl_col = _pick_column(header, CAND_LABELS)
        ts_col  = _pick_column(header, CAND_TS)

        if sig_col is None:
            raise ValueError(f"Could not find an ECG signal column. "
                             f"Tried {CAND_SIGNAL}. Got columns: {header}")
        if lbl_col is None:
            # not fatal; assume all normal
            lbl_col = None

        buf: deque[float] = deque(maxlen=win)
        last_event_ts: Optional[float] = None
        i = 0

        for row in reader:
            x = _to_float(row.get(sig_col, "0"))
            lbl = _clean_label(row.get(lbl_col)) if lbl_col else "N"

            # time
            if ts_col:
                ts = _to_float(row.get(ts_col, "0"))
            else:
                ts = i / fs_hz

            # rolling z-score features
            buf.append(x)
            mu, sd = _rolling_stats(buf)
            z = (x - mu) / sd

            magnitude = min(1.0, abs(z) / mag_clip)
            anomaly   = min(1.0, abs(z) / anomaly_clip)
            urgency   = URGENCY_MAP.get(lbl, 0.2)

            # time since last important event (S/V/F) → context
            if last_event_ts is None:
                dt = 0.0
            else:
                dt = max(0.0, ts - last_event_ts)
            context = 1.0 - math.exp(-dt / 5.0)
            if lbl in ("S", "V", "F"):
                last_event_ts = ts

            yield {
                "timestamp": ts,
                "magnitude": magnitude * 100.0,  # Sundew expects 0..100 before normalization
                "anomaly_score": anomaly,
                "context_relevance": context,
                "urgency": urgency,
                "label": lbl,
            }
            i += 1
