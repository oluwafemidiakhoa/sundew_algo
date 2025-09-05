# src/sundew/cli.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    # Prefer package-relative imports (installed or editable)
    from .core import SundewAlgorithm, SundewConfig
    from .demo import synth_event
except Exception:
    # Fallback if module layout differs in dev
    from sundew.core import SundewAlgorithm, SundewConfig
    from sundew.demo import synth_event


def _stdout_supports_unicode() -> bool:
    """Return True if stdout can encode a typical emoji; else False (Windows legacy codepages)."""
    enc = getattr(sys.stdout, "encoding", None) or ""
    try:
        "🌿".encode(enc or "utf-8", errors="strict")
        return True
    except Exception:
        return False


EMOJI_OK = _stdout_supports_unicode()
BULLET = "🌿" if EMOJI_OK else "[sundew]"
CHECK = "✅" if EMOJI_OK else "[ok]"
PAUSE = "⏸" if EMOJI_OK else "[idle]"
FLAG_DONE = "🏁" if EMOJI_OK else "[done]"
DISK = "💾" if EMOJI_OK else "[saved]"


def _energy_float(algo) -> float:
    """Return energy as float regardless of whether algo.energy is a number or an object."""
    e = getattr(algo, "energy", 0.0)
    v = getattr(e, "value", e)
    try:
        return float(v)
    except Exception:
        return 0.0


def run_demo(n_events: int, temperature: float) -> dict:
    """Tiny inline demo for reliability across environments. Returns summary dict."""
    cfg = SundewConfig(gate_temperature=temperature)
    algo = SundewAlgorithm(cfg)

    print(f"{BULLET} Sundew Algorithm — Demo")
    print("=" * 60)
    print(f"Initial threshold: {algo.threshold:.3f} | Energy: {_energy_float(algo):.1f}\n")

    processed = []
    for i in range(n_events):
        x = synth_event(i)
        res = algo.process(x)
        if res is None:
            print(f"{i+1:02d}. {x['type']:<15} {PAUSE} dormant | energy {_energy_float(algo):6.1f} | thr {algo.threshold:.3f}")
        else:
            processed.append(res)
            print(
                f"{i+1:02d}. {x['type']:<15} {CHECK} processed "
                f"(sig={res.significance:.3f}, {res.processing_time:.3f}s, ΔE≈{res.energy_consumed:.1f}) | "
                f"energy {_energy_float(algo):6.1f} | thr {algo.threshold:.3f}"
            )

    print(f"\n{FLAG_DONE} Final Report")
    report = algo.report()
    for k, v in report.items():
        if isinstance(v, float):
            if "pct" in k:
                print(f"  {k:30s}: {v:7.2f}%")
            else:
                print(f"  {k:30s}: {v:10.3f}")
        else:
            print(f"  {k:30s}: {v}")

    return {
        "config": cfg.__dict__,
        "report": report,
        "processed_events": [getattr(r, "__dict__", {}) for r in processed],
    }


def main(argv: list[str] | None = None) -> int:
    # NOTE: tests look for "Sundew Algorithm CLI" or "Sundew Algorithm" in help text
    ap = argparse.ArgumentParser(description="Sundew Algorithm CLI")
    ap.add_argument("--demo", action="store_true", help="Run the interactive demo")
    ap.add_argument("--events", type=int, default=40, help="Number of demo events")
    ap.add_argument("--temperature", type=float, default=0.1, help="Gating temperature (0=hard)")
    ap.add_argument("--save", type=str, default="", help="Optional: save demo results to JSON path")
    args = ap.parse_args(argv)

    if args.demo:
        out = run_demo(args.events, args.temperature)
        if args.save:
            path = Path(args.save)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            print(f"\n{DISK} Results saved to {path}")
        return 0

    # No subcommand → print help and exit successfully (tests expect rc==0)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
