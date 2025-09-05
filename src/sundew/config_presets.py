# src/sundew/config_presets.py
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

from dataclasses import replace
from typing import Any, Callable, Dict

from .config import SundewConfig


# ------------------------------------------------------------------------------#
# Baseline (former numbers)
# ------------------------------------------------------------------------------#

def _baseline() -> SundewConfig:
    """
    Former defaults used earlier in the project and in the first plots.
    Conservative and prone to under-activation (maximizes savings).
    """
    return SundewConfig(
        # Activation & adaptation
        activation_threshold=0.70,
        target_activation_rate=0.25,
        ema_alpha=0.10,

        # Controller (initial PI-style numbers kept for reproducibility)
        adapt_kp=0.06,
        adapt_ki=0.01,
        error_deadband=0.010,
        integral_clamp=0.50,
        adapt_lr=0.02,  # legacy (not used when PI is active, but preserved)

        # Threshold bounds
        min_threshold=0.30,
        max_threshold=0.95,

        # Energy pressure (stronger -> more conservative)
        energy_pressure=0.15,

        # Gating
        gate_temperature=0.00,  # hard

        # Energy model
        max_energy=100.0,
        dormant_tick_cost=0.5,
        dormancy_regen=(1.0, 3.0),
        eval_cost=0.6,
        base_processing_cost=10.0,

        # Significance weights
        w_magnitude=0.30,
        w_anomaly=0.40,
        w_context=0.20,
        w_urgency=0.10,

        # Extras
        rng_seed=42,
        probe_every=200,
        refractory=0,
    )


# ------------------------------------------------------------------------------#
# Tuned v1 (PI + softer pressure)
# ------------------------------------------------------------------------------#

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
        gate_temperature=0.10,  # soft-ish

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
        probe_every=200,
        refractory=0,
    )


# ------------------------------------------------------------------------------#
# Tuned v2 (current recommended general-use defaults)
# ------------------------------------------------------------------------------#

def _tuned_v2() -> SundewConfig:
    """
    Recommended general-purpose settings (from your latest experiments):
    - slightly higher gains, smaller deadband
    - softer energy pressure
    - tighter max_threshold to avoid hard-pegging
    """
    return SundewConfig(
        activation_threshold=0.70,
        target_activation_rate=0.25,
        ema_alpha=0.10,

        adapt_kp=0.08,          # up from 0.06
        adapt_ki=0.02,          # up from 0.01
        error_deadband=0.005,   # down from 0.01
        integral_clamp=0.50,

        min_threshold=0.20,     # down from 0.30
        max_threshold=0.90,     # down from 0.95

        energy_pressure=0.03,   # softer conservation pressure
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
        probe_every=200,
        refractory=0,
    )


# ------------------------------------------------------------------------------#
# ECG-focused preset (improve recall on arrhythmias)
# ------------------------------------------------------------------------------#

def _ecg_v1() -> SundewConfig:
    """
    ECG-oriented trade-off:
    - Lower starting threshold and slightly softer gate to raise recall
    - Slightly faster controller
    - Bias significance toward anomaly/context (ECG morphology deviations)
    """
    return SundewConfig(
        activation_threshold=0.60,
        target_activation_rate=0.12,   # allow more duty than generic tuned_v2
        ema_alpha=0.08,

        adapt_kp=0.09,
        adapt_ki=0.02,
        error_deadband=0.005,
        integral_clamp=0.50,

        min_threshold=0.45,
        max_threshold=0.95,

        energy_pressure=0.08,       # don’t clamp too early in low energy
        gate_temperature=0.12,      # admit borderline beats

        max_energy=100.0,
        dormant_tick_cost=0.5,
        dormancy_regen=(1.0, 3.0),
        eval_cost=0.6,
        base_processing_cost=10.0,

        w_magnitude=0.20,
        w_anomaly=0.50,
        w_context=0.20,
        w_urgency=0.10,

        rng_seed=42,
        probe_every=200,
        refractory=0,
    )


# ------------------------------------------------------------------------------#
# Frozen best trade-off from your MIT-BIH sweep
# ------------------------------------------------------------------------------#

def _ecg_mitbih_best() -> SundewConfig:
    """
    Frozen trade-off from MIT-BIH sweep:
    - modest starting threshold
    - slightly soft gate to admit borderline beats
    - a bit more duty (target ~0.14) to lift recall
    """
    return SundewConfig(
        activation_threshold=0.65,
        target_activation_rate=0.14,
        ema_alpha=0.08,

        adapt_kp=0.09,
        adapt_ki=0.02,
        error_deadband=0.005,
        integral_clamp=0.50,

        min_threshold=0.45,
        max_threshold=0.90,

        energy_pressure=0.07,
        gate_temperature=0.14,

        max_energy=100.0,
        dormant_tick_cost=0.5,
        dormancy_regen=(1.0, 3.0),
        eval_cost=0.6,
        base_processing_cost=10.0,

        w_magnitude=0.20,
        w_anomaly=0.50,
        w_context=0.20,
        w_urgency=0.10,

        rng_seed=42,
        probe_every=200,
        refractory=0,
    )


# ------------------------------------------------------------------------------#
# Variants
# ------------------------------------------------------------------------------#

def _aggressive() -> SundewConfig:
    """Faster to hit target; more activations; lower energy savings."""
    return replace(
        _tuned_v2(),
        adapt_kp=0.12,
        adapt_ki=0.04,
        error_deadband=0.003,
        energy_pressure=0.02,
        gate_temperature=0.15,
        max_threshold=0.88,
    )


def _conservative() -> SundewConfig:
    """Maximize savings (will under-activate in quiet streams)."""
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


def _high_temp() -> SundewConfig:
    """Probe/explore more (useful for anomaly-heavy streams)."""
    return replace(
        _tuned_v2(),
        gate_temperature=0.20,
        energy_pressure=0.025,
    )


def _low_temp() -> SundewConfig:
    """Nearly hard gate; sharper selectivity."""
    return replace(
        _tuned_v2(),
        gate_temperature=0.00,
        energy_pressure=0.035,
    )


def _energy_saver() -> SundewConfig:
    """Prioritize battery; accept lower activation rate."""
    return replace(
        _tuned_v2(),
        energy_pressure=0.08,
        adapt_kp=0.06,
        adapt_ki=0.01,
        max_threshold=0.92,
        gate_temperature=0.05,
    )


def _target_0p30() -> SundewConfig:
    """Convenience preset for a higher target activation rate."""
    return replace(
        _tuned_v2(),
        target_activation_rate=0.30,
    )


# ------------------------------------------------------------------------------#
# Registry
# ------------------------------------------------------------------------------#

_PRESETS: Dict[str, Callable[[], SundewConfig]] = {
    "baseline": _baseline,
    "tuned_v1": _tuned_v1,
    "tuned_v2": _tuned_v2,        # current general recommendation
    "ecg_v1": _ecg_v1,            # ECG-focused preset (generic)
    "ecg_mitbih_best": _ecg_mitbih_best,  # frozen best from MIT-BIH sweep
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
