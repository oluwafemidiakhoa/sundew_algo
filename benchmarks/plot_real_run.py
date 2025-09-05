# benchmarks/plot_real_run.py
from __future__ import annotations
import argparse, json, os
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default="results/plots_real")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.json, "r") as f:
        out = json.load(f)

    # simple series we stored? we didn’t log per-step values, so here:
    # Option A: extend SundewAlgorithm to log threshold/ema over time (like your single-run plotter).
    # Quick hack: just bar plot for activation vs truth
    y_true = out["y_true"]
    y_pred = out["y_pred"]
    n = min(len(y_true), 500)

    fig = plt.figure(figsize=(12, 3))
    plt.plot(range(n), y_true[:n], lw=1, label="Ground truth (important)")
    plt.plot(range(n), y_pred[:n], lw=1, label="Sundew activation")
    plt.title("ECG: Ground Truth vs Sundew Activation (first 500)")
    plt.xlabel("Event index"); plt.ylabel("Binary")
    plt.legend()
    p = os.path.join(args.out, "ecg_truth_vs_activation.png")
    plt.tight_layout(); plt.savefig(p, dpi=150)
    print(f"📈 saved {p}")

if __name__ == "__main__":
    main()
