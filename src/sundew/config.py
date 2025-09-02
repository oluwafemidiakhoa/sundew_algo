from dataclasses import dataclass
from typing import Tuple

@dataclass
class SundewConfig:
    # -------- Activation & adaptation targets --------
    activation_threshold: float = 0.7
    target_activation_rate: float = 0.25     # desired fraction of events processed
    ema_alpha: float = 0.1                   # smoothing for activation-rate estimate

    # -------- PI controller (new) --------
    # Proportional gain: how strongly to react to current error (ema - target)
    adapt_kp: float = 0.08
    # Integral gain: how strongly to react to accumulated error over time
    adapt_ki: float = 0.01
    # Ignore tiny errors to avoid jitter
    error_deadband: float = 0.01
    # Prevent runaway integral
    integral_clamp: float = 0.5


    # Legacy scalar step (kept for compatibility, not used by the new controller)
    adapt_lr: float = 0.02

    # Threshold bounds
    min_threshold: float = 0.20
    max_threshold: float = 0.90

    # Energy pressure: how much low energy pushes the threshold UP (stricter)
    # Softer than before to avoid dominating the controller
    energy_pressure: float = 0.05

    # Gating
    gate_temperature: float = 0.1           # small softness helps hit target more smoothly

    # -------- Energy model --------
    max_energy: float = 100.0
    dormant_tick_cost: float = 0.5
    dormancy_regen: Tuple[float, float] = (1.0, 3.0)
    eval_cost: float = 0.6
    base_processing_cost: float = 10.0

    # -------- Significance weights (must sum to 1.0) --------
    w_magnitude: float = 0.30
    w_anomaly: float = 0.40
    w_context: float = 0.20
    w_urgency: float = 0.10

    # -------- Simulation --------
    rng_seed: int = 42
1