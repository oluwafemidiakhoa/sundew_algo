# benchmarks/eval_classification.py
from __future__ import annotations
import argparse, json


def prf1(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1)


def main():
    ap = argparse.ArgumentParser(description="Evaluate y_true vs y_pred from a Sundew real-data run")
    ap.add_argument("--json", required=True, help="Path to results/real_ecg_run.json")
    args = ap.parse_args()

    with open(args.json, "r") as f:
        out = json.load(f)
    y_true = out["y_true"]
    y_pred = out["y_pred"]

    metrics = prf1(y_true, y_pred)

    print("=== Event Detection Quality ===")
    for k, v in metrics.items():
        print(f"{k:16s}: {v:.6f}" if isinstance(v, float) else f"{k:16s}: {v}")

    print("\n=== Energy/Activation Summary ===")
    rep = out["report"]
    for k, v in rep.items():
        print(f"{k:28s}: {v}" if not isinstance(v, float) else f"{k:28s}: {v:.6f}")


if __name__ == "__main__":
    main()
