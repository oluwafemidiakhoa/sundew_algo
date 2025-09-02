"""
Run multiple Sundew presets and write a single combined CSV.

Examples (from repo root, Windows):
  python benchmarks\run_presets.py --events 300 --repeats 3 ^
    --presets baseline tuned_v2 aggressive energy_saver ^
    --out results\grid_multi.csv --logdir results\runs_multi

PowerShell:
  python benchmarks/run_presets.py --events 300 --repeats 3 `
    --presets baseline tuned_v2 aggressive energy_saver `
    --out results/grid_multi.csv --logdir results/runs_multi

The combined CSV can be plotted directly with:
  python benchmarks\plot_grid.py --csv results\grid_multi.csv --out results\plots_multi
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# ---- Robust imports (works even if package not installed) ----
try:
    from sundew import SundewAlgorithm, SundewConfig
    from sundew.demo import synth_event
    from sundew.config_presets import get_preset, list_presets
except ModuleNotFoundError:
    import sys
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from sundew import SundewAlgorithm, SundewConfig
    from sundew.demo import synth_event
    from sundew.config_presets import get_preset, list_presets


FIELDS = [
    "preset",
    "seed",
    "total_inputs",
    "activations",
    "activation_rate",
    "ema_activation_rate",
    "avg_processing_time",
    "total_energy_spent",
    "energy_remaining",
    "threshold",
    "baseline_energy_cost",
    "actual_energy_cost",
    "estimated_energy_savings_pct",
]


def run_once(n_events: int, cfg: SundewConfig, seed: int) -> Dict[str, Any]:
    cfg.rng_seed = seed
    algo = SundewAlgorithm(cfg)
    for i in range(n_events):
        algo.process(synth_event(i))
    rep = algo.report(assumed_baseline_per_event=15.0)
    rep.update({
        "preset": getattr(cfg, "preset_name", "custom"),
        "seed": seed,
    })
    return rep


def main():
    ap = argparse.ArgumentParser(description="Run multiple Sundew presets and aggregate results")
    ap.add_argument("--presets", nargs="+", required=True,
                    help=f"One or more preset names. Available: {list_presets()}")
    ap.add_argument("--events", type=int, default=300, help="Events per run")
    ap.add_argument("--repeats", type=int, default=3, help="Seeds per preset")
    ap.add_argument("--seed-base", type=int, default=42, help="Base seed used for repeats")
    ap.add_argument("--out", type=str, default="results/grid_multi.csv", help="Combined CSV output path")
    ap.add_argument("--logdir", type=str, default="", help="Optional directory to dump per-run JSONs")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.logdir:
        Path(args.logdir).mkdir(parents=True, exist_ok=True)

    print("▶️  Running presets:", ", ".join(args.presets))
    print("   events:", args.events, "| repeats:", args.repeats)
    print("📄 Combined CSV:", out_path.resolve())
    if args.logdir:
        print("🗂️  Per-run logs:", Path(args.logdir).resolve())

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for preset in args.presets:
            # Build config for this preset
            cfg = get_preset(preset)
            setattr(cfg, "preset_name", preset)

            print(f"\n💡 preset: {preset}")
            for r in range(args.repeats):
                seed = args.seed_base + r
                rep = run_once(args.events, cfg, seed)
                writer.writerow({k: rep.get(k, "") for k in FIELDS})

                if args.logdir:
                    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                    js_out = Path(args.logdir) / f"{preset}_seed{seed}_{stamp}.json"
                    with open(js_out, "w") as jf:
                        json.dump(rep, jf, indent=2)

                print(f"   ✓ seed {seed}: "
                      f"act_rate={rep['activation_rate']:.3f}, "
                      f"savings={rep['estimated_energy_savings_pct']:.1f}%, "
                      f"thr={rep['threshold']:.3f}")

    print(f"\n✅ wrote CSV: {out_path} (exists: {out_path.exists()})")


if __name__ == "__main__":
    main()
