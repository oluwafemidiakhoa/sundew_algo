# src/sundew/cli.py
from __future__ import annotations
import argparse
import json
import sys
from typing import Any, Dict
from sundew.config_presets import list_presets, get_preset
from .config import SundewConfig


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", default="tuned_v2", help="Preset name (see list-presets)")
    p.add_argument(
        "--overrides",
        default=None,
        help="JSON dict of field overrides, e.g. '{\"target_activation_rate\":0.2}'",
    )


def _parse_overrides(s: str | None) -> Dict[str, Any]:
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception as e:
        raise SystemExit(f"Invalid JSON for --overrides: {e}")


def cmd_list_presets(_: argparse.Namespace) -> int:
    for name in list_presets():
        print(name)
    return 0


def cmd_print_config(args: argparse.Namespace) -> int:
    ov = _parse_overrides(args.overrides)
    cfg: SundewConfig = get_preset(args.preset, ov if ov else None)
    print(json.dumps(cfg.__dict__, indent=2, sort_keys=True, default=str))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """
    Minimal demo 'run' that just instantiates the config/preset and prints the
    selected activation threshold; your richer demo/benchmark lives elsewhere.
    """
    ov = _parse_overrides(args.overrides)
    cfg: SundewConfig = get_preset(args.preset, ov if ov else None)
    print(f"[sundew] preset={args.preset} thr0={cfg.activation_threshold:.3f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sundew", description="Sundew Algorithm CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # list-presets
    p_list = sub.add_parser("list-presets", help="List available presets")
    p_list.set_defaults(func=cmd_list_presets)

    # print-config
    p_cfg = sub.add_parser("print-config", help="Print a preset (with optional overrides) as JSON")
    _add_common_args(p_cfg)
    p_cfg.set_defaults(func=cmd_print_config)

    # run (lightweight)
    p_run = sub.add_parser("run", help="Lightweight instantiation (for smoke testing)")
    _add_common_args(p_run)
    p_run.set_defaults(func=cmd_run)

    return ap


def main(argv: list[str] | None = None) -> None:
    ap = build_parser()
    ns = ap.parse_args(argv)
    rc = ns.func(ns)
    sys.exit(rc)


if __name__ == "__main__":
    main()
