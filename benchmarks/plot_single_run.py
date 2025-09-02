"""
Single-run visualization for Sundew with config presets.
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

try:
    from sundew import SundewAlgorithm
    from sundew.demo import synth_event
    from sundew.config_presets import get_preset, list_presets
except ModuleNotFoundError:
    import sys
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from sundew import SundewAlgorithm
    from sundew.demo import synth_event
    from sundew.config_presets import get_preset, list_presets


def main():
    ap = argparse.ArgumentParser(description="Single-run visualization with presets")
    ap.add_argument("--preset", type=str, default="tuned_v2",
                    help=f"Config preset to use. Options: {list_presets()}")
    ap.add_argument("--events", type=int, default=200, help="Number of events to simulate")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--out", type=str, default="results/plots", help="Output directory for PNGs")
    ap.add_argument("--savecsv", type=str, default="", help="Optional CSV path to save per-step time series")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = get_preset(args.preset)
    setattr(cfg, "preset_name", args.preset)
    cfg.rng_seed = args.seed

    algo = SundewAlgorithm(cfg)

    steps, thresholds, energies, emas, activations = [], [], [], [], []

    for i in range(args.events):
        x = synth_event(i)
        res = algo.process(x)
        steps.append(i + 1)
        thresholds.append(algo.threshold)
        e_val = getattr(getattr(algo, "energy", None), "value", None)
        energies.append(e_val if e_val is not None else float(getattr(algo, "energy", 0.0)))
        emas.append(algo.metrics.ema_activation_rate)
        activations.append(1 if res is not None else 0)

    if args.savecsv:
        csv_path = Path(args.savecsv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "threshold", "energy", "ema_activation", "activated", "preset", "seed"])
            for s, thr, en, ema, act in zip(steps, thresholds, energies, emas, activations):
                w.writerow([s, thr, en, ema, act, args.preset, args.seed])
        print(f"📄 saved timeseries CSV: {csv_path.resolve()}")

    # 1) Threshold & EMA
    fig, ax1 = plt.subplots()
    ax1.plot(steps, thresholds, label="threshold")
    ax1.set_xlabel("Event")
    ax1.set_ylabel("Threshold")
    ax2 = ax1.twinx()
    ax2.plot(steps, emas, label="ema_activation", linestyle="--")
    ax2.set_ylabel("EMA Activation Rate")
    ax1.set_title(f"Threshold & EMA over time — preset={args.preset}, seed={args.seed}")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    p1 = outdir / f"single_run_threshold_ema_{args.preset}.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    # 2) Energy
    plt.figure()
    plt.plot(steps, energies)
    plt.xlabel("Event")
    plt.ylabel("Energy")
    plt.title(f"Energy over time — preset={args.preset}, seed={args.seed}")
    plt.grid(True, alpha=0.3)
    p2 = outdir / f"single_run_energy_{args.preset}.png"
    plt.savefig(p2, dpi=160, bbox_inches="tight")
    plt.close()

    # 3) Activation spikes (fixed: no use_line_collection)
    plt.figure()
    plt.stem(steps, activations)
    plt.xlabel("Event")
    plt.ylabel("Activation (0/1)")
    plt.title(f"Activation spikes — preset={args.preset}, seed={args.seed}")
    plt.grid(True, alpha=0.3)
    p3 = outdir / f"single_run_activations_{args.preset}.png"
    plt.savefig(p3, dpi=160, bbox_inches="tight")
    plt.close()

    print("✅ saved plots:")
    print("   ", p1.resolve())
    print("   ", p2.resolve())
    print("   ", p3.resolve())


if __name__ == "__main__":
    main()
