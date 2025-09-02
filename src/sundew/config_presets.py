"""
Named configuration presets for Sundew.

Usage
-----
from sundew.config_presets import get_preset, list_presets
cfg = get_preset("tuned_v2")         # returns a SundewConfig
algo = SundewAlgorithm(cfg)

You can also override any field ad-hoc:
cfg = get_preset("tuned_v2", overrides=dict(target_activation_rate=0.30, gate_temperature=0.15))

To see available presets:
print(list_presets())
"""

from __future__ import annotations
from typing import Dict, Any, Callable
from dataclasses import replace

from .config import SundewConfig


# ---------- Baseline (former numbers) ----------
def _baseline() -> SundewConfig:
    """
    Former defaults used earlier in the project and in your first plots.
    Good to reproduce the "conservative / under-activating" behavior.
    """
    return SundewConfig(
        # Activation & adaptation
        activation_threshold=0.70,
        target_activation_rate=0.25,
        ema_alpha=0.10,

        # Controller (former)
        adapt_kp=0.06,
        adapt_ki=0.01,
        error_deadband=0.010,
        integral_clamp=0.50,
        adapt_lr=0.02,  # legacy (unused in PI but kept)

        # Threshold bounds (former)
        min_threshold=0.30,
        max_threshold=0.95,

        # Energy pressure (former strong)
        energy_pressure=0.15,

        # Gating (harder by default)
        gate_temperature=0.00,

        # Energy model
        max_energy=100.0,
        dormant_tick_cost=0.5,
        dormancy_regen=(1.0, 3.0),
        eval_cost=0.6,
        base_processing_cost=10.0,

        # Weights
        w_magnitude=0.30,
        w_anomaly=0.40,
        w_context=0.20,
        w_urgency=0.10,

        rng_seed=42,
    )


# ---------- Tuned v1 (intermediate, PI + softer pressure) ----------
def _tuned_v1() -> SundewConfig:
    """
    First PI iteration that reduced threshold pegging and improved activation rate.
    """
    return SundewConfig(
        activation_threshold=0.70,
        target_activation_rate=0.25,
        ema_alpha=0.10,

        adapt_kp=0.06,
        adapt_ki=0.01,
        error_deadband=0.010,
        integral_clamp=0.50,

        min_threshold=0.20,
        max_threshold=0.90,

        energy_pressure=0.05,   # softer than baseline
        gate_temperature=0.10,  # a little softness helps hit target

        max_energy=100.0,
        dormant_tick_cost=0.5,
        dormancy_regen=(1.0, 3.0),
        eval_cost=0.6,
        base_processing_cost=10.0,

        w_magnitude=0.30,
        w_anomaly=0.40,
        w_context=0.20,
        w_urgency=0.10,

        rng_seed=42,
    )


# ---------- Tuned v2 (current recommended defaults from your latest plots) ----------
def _tuned_v2() -> SundewConfig:
    """
    Current recommended settings from your latest experiments:
    - higher gains, smaller deadband
    - softer energy pressure
    - tighter max_threshold
    """
    return SundewConfig(
        activation_threshold=0.70,
        target_activation_rate=0.25,
        ema_alpha=0.10,

        adapt_kp=0.08,          # ↑ from 0.06
        adapt_ki=0.02,          # ↑ from 0.01
        error_deadband=0.005,   # ↓ from 0.01
        integral_clamp=0.50,

        min_threshold=0.20,     # ↓ from 0.30
        max_threshold=0.90,     # ↓ from 0.95

        energy_pressure=0.03,   # ↓ from 0.05
        gate_temperature=0.10,

        max_energy=100.0,
        dormant_tick_cost=0.5,
        dormancy_regen=(1.0, 3.0),
        eval_cost=0.6,
        base_processing_cost=10.0,

        w_magnitude=0.30,
        w_anomaly=0.40,
        w_context=0.20,
        w_urgency=0.10,

        rng_seed=42,
    )


# ---------- Aggressive (hit target faster, more activations; less savings) ----------
def _aggressive() -> SundewConfig:
    return replace(
        _tuned_v2(),
        adapt_kp=0.12,
        adapt_ki=0.04,
        error_deadband=0.003,
        energy_pressure=0.02,
        gate_temperature=0.15,   # softer gate to allow borderline events
        max_threshold=0.88,
    )


# ---------- Conservative (maximize savings; will under-activate if stream is quiet) ----------
def _conservative() -> SundewConfig:
    return replace(
        _tuned_v2(),
        adapt_kp=0.05,
        adapt_ki=0.01,
        error_deadband=0.010,
        energy_pressure=0.05,
        gate_temperature=0.05,
        min_threshold=0.25,
        max_threshold=0.92,
    )


# ---------- High-temperature (probe/explore more; useful for anomaly-heavy streams) ----------
def _high_temp() -> SundewConfig:
    return replace(
        _tuned_v2(),
        gate_temperature=0.20,
        energy_pressure=0.025,
    )


# ---------- Low-temperature (nearly hard gate; sharper selectivity) ----------
def _low_temp() -> SundewConfig:
    return replace(
        _tuned_v2(),
        gate_temperature=0.00,
        energy_pressure=0.035,
    )


# ---------- Energy saver (prioritize battery; accept lower activation) ----------
def _energy_saver() -> SundewConfig:
    return replace(
        _tuned_v2(),
        energy_pressure=0.08,
        adapt_kp=0.06,
        adapt_ki=0.01,
        max_threshold=0.92,
        gate_temperature=0.05,
    )


# ---------- Higher target (e.g., 0.30) ----------
def _target_0p30() -> SundewConfig:
    return replace(
        _tuned_v2(),
        target_activation_rate=0.30,
    )


# Map of preset name -> builder
_PRESETS: Dict[str, Callable[[], SundewConfig]] = {
    "baseline": _baseline,
    "tuned_v1": _tuned_v1,
    "tuned_v2": _tuned_v2,        # current recommended
    "aggressive": _aggressive,
    "conservative": _conservative,
    "high_temp": _high_temp,
    "low_temp": _low_temp,
    "energy_saver": _energy_saver,
    "target_0p30": _target_0p30,
}


def list_presets() -> list[str]:
    """Return a sorted list of available preset names."""
    return sorted(_PRESETS.keys())


def get_preset(name: str, overrides: Dict[str, Any] | None = None) -> SundewConfig:
    """
    Return a SundewConfig for the named preset. Optionally override fields:

        cfg = get_preset("tuned_v2", overrides=dict(target_activation_rate=0.30))

    Raises KeyError if the preset name is unknown.
    """
    try:
        cfg = _PRESETS[name]()  # build
    except KeyError as e:
        raise KeyError(f"Unknown preset '{name}'. Available: {list_presets()}") from e

    if overrides:
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise AttributeError(f"SundewConfig has no field '{k}'")
            setattr(cfg, k, v)
    return cfg
