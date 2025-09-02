"""
Grid benchmark for Sundew Algorithm with config presets.

Examples (Windows, from repo root):
  python benchmarks\grid_search.py --preset tuned_v2 --events 300 --repeats 3 --out results\grid.csv --logdir results\runs
"""

from __future__ import annotations
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# ---- Robust imports ----
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", type=str, default="tuned_v2",
                    help=f"Config preset to use. Options: {list_presets()}")
    ap.add_argument("--events", type=int, default=300,
                    help="Number of events per run")
    ap.add_argument("--repeats", type=int, default=3,
                    help="Number of seeds per preset")
    ap.add_argument("--out", type=str, default="results/grid.csv")
    ap.add_argument("--logdir", type=str, default="")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.logdir:
        Path(args.logdir).mkdir(parents=True, exist_ok=True)

    # Build config from preset
    cfg = get_preset(args.preset)
    setattr(cfg, "preset_name", args.preset)

    print("▶️  Running grid for preset:", args.preset)
    print("📍 Writing CSV:", out_path.resolve())

    fields = [
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

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for s in range(args.repeats):
            seed = 42 + s
            rep = run_once(args.events, cfg, seed)
            w.writerow({k: rep.get(k, "") for k in fields})
            if args.logdir:
                stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                js_out = Path(args.logdir) / f"{args.preset}_seed{seed}_{stamp}.json"
                with open(js_out, "w") as jf:
                    json.dump(rep, jf, indent=2)

    print(f"✅ wrote CSV: {out_path} (exists: {out_path.exists()})")


if __name__ == "__main__":
    main()
