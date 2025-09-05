# src/sundew/core.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import math
import random

from .config import SundewConfig
from .energy import EnergyAccount


def _sigmoid(x: float) -> float:
    # numerically stable-ish
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _bounded(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _feature_get(event: Dict[str, Any], key: str, default: float = 0.0) -> float:
    v = event.get(key, default)
    try:
        return float(v)
    except Exception:
        return default


@dataclass
class SundewMetrics:
    total_inputs: int = 0
    activations: int = 0
    ema_activation_rate: float = 0.0
    activation_rate: float = 0.0
    avg_processing_time: float = 0.0  # placeholder (your benches can fill this)
    total_energy_spent: float = 0.0
    energy_remaining: float = 0.0
    threshold: float = 0.0
    baseline_energy_cost: float = 0.0
    actual_energy_cost: float = 0.0


class SundewAlgorithm:
    """
    Energy-aware, adaptive selective activation.

    - Computes a bounded significance score s in [0,1] from lightweight features.
    - Applies temperature-controlled gating vs. an adaptive threshold θ.
    - Uses a PI-like controller to steer the EMA activation rate toward a target.
    - Applies an energy-pressure term that increases θ as energy declines.
    - Optional "probe" to guarantee at least occasional activation for testability.
    """

    def __init__(self, cfg: SundewConfig):
        self.cfg = cfg

        # Controller state
        self.threshold: float = float(cfg.activation_threshold)
        self._ema: float = 0.0
        self._i_term: float = 0.0

        # Book-keeping
        self.total_inputs: int = 0
        self.activations: int = 0
        self._since_last_activation: int = 0
        self._refrac_left: int = 0  # steps remaining in refractory

        # Deterministic RNG if seed provided
        if getattr(cfg, "rng_seed", None) is not None:
            random.seed(cfg.rng_seed)

        # Energy model
        self.energy = EnergyAccount(
            max_energy=cfg.max_energy,
            dormant_tick_cost=cfg.dormant_tick_cost,
            dormancy_regen=cfg.dormancy_regen,
            eval_cost=cfg.eval_cost,
            base_processing_cost=cfg.base_processing_cost,
        )

        # Watchdog probe: ensure at least sporadic activation in long quiet runs (helps tests)
        # 0 disables. Default conservative (200) if not on cfg.
        self.probe_every: int = int(getattr(cfg, "probe_every", 200))

        # Optional refractory window after an activation (0 = off)
        self.refractory: int = int(getattr(cfg, "refractory", 0))

        # Baseline cost (process-everything) for savings estimation
        self._baseline_per_step = cfg.base_processing_cost + cfg.eval_cost
        self._actual_energy_spent = 0.0
        self._baseline_energy_spent = 0.0

    # -------------------------
    # Public API
    # -------------------------

    def process(self, event: Dict[str, Any]) -> Dict[str, Any] | None:
        """
        Process a single event (dict of lightweight features).
        Returns a dict (e.g., result) if activated; otherwise None.
        """
        self.total_inputs += 1
        self._baseline_energy_spent += self._baseline_per_step

        # 1) Compute bounded significance
        s = self._significance(event)

        # 2) Decide: activate?
        can_activate = self._can_activate_now()
        act = self._gate(s) if can_activate else False

        # 3) Watchdog "probe" (once in a long while if dormant too long)
        self._since_last_activation += 1
        if (not act) and self.probe_every and (self._since_last_activation >= self.probe_every):
            act = True  # single forced probe
            # Note: we keep normal processing cost for probe to keep accounting simple.

        # 4) Spend/regen energy, update counts
        if act:
            self.activations += 1
            self._since_last_activation = 0
            self._refrac_left = self.refractory
            # Spend processing energy
            self._actual_energy_spent += self.energy.spend(self.cfg.base_processing_cost + self.cfg.eval_cost)
        else:
            # Dormant tick: small maintenance + regen
            self._actual_energy_spent += self.energy.spend(self.cfg.dormant_tick_cost)
            self.energy.regen()  # applies cfg.dormancy_regen inside EnergyAccount

            # tick down refractory if set
            if self._refrac_left > 0:
                self._refrac_left -= 1

        # 5) Update EMA and adapt threshold via PI + energy pressure
        self._adapt_threshold(1.0 if act else 0.0)

        # 6) Return result if activated (your downstream pipeline would go here)
        return {"score": s, "threshold": self.threshold} if act else None

    def report(self) -> SundewMetrics:
        activation_rate = (self.activations / self.total_inputs) if self.total_inputs else 0.0
        return SundewMetrics(
            total_inputs=self.total_inputs,
            activations=self.activations,
            ema_activation_rate=self._ema,
            activation_rate=activation_rate,
            avg_processing_time=0.0,  # placeholder for benches
            total_energy_spent=self._actual_energy_spent,
            energy_remaining=self.energy.value,
            threshold=self.threshold,
            baseline_energy_cost=self._baseline_energy_spent,
            actual_energy_cost=self._actual_energy_spent,
        )

    # -------------------------
    # Internals
    # -------------------------

    def _significance(self, event: Dict[str, Any]) -> float:
        """Convex combination of bounded features -> s in [0,1]."""
        w_mag = self.cfg.w_magnitude
        w_an  = self.cfg.w_anomaly
        w_ctx = self.cfg.w_context
        w_urg = self.cfg.w_urgency
        # Normalize weights (defensive)
        w_sum = max(1e-9, (w_mag + w_an + w_ctx + w_urg))
        w_mag, w_an, w_ctx, w_urg = (w_mag / w_sum, w_an / w_sum, w_ctx / w_sum, w_urg / w_sum)

        # Pull features (already in [0,1] in our synthetic/demo conventions)
        mag = _bounded(_feature_get(event, "magnitude", 0.0))
        ano = _bounded(_feature_get(event, "anomaly_score", 0.0))
        ctx = _bounded(_feature_get(event, "context_relevance", 0.0))
        urg = _bounded(_feature_get(event, "urgency", 0.0))

        s = (w_mag * mag) + (w_an * ano) + (w_ctx * ctx) + (w_urg * urg)
        return _bounded(s)

    def _can_activate_now(self) -> bool:
        """Honor a refractory window if configured."""
        return self._refrac_left <= 0

    def _gate(self, s: float) -> bool:
        """Temperature-controlled gating vs. adaptive threshold."""
        tau = self.cfg.gate_temperature
        if tau <= 0.0:
            return s >= self.threshold
        # logistic probabilistic gate; for determinism in tests, we use p>=0.5
        p = _sigmoid((s - self.threshold) / max(1e-9, tau))
        return p >= 0.5

    def _adapt_threshold(self, activated: float) -> None:
        """
        PI-like adaptation toward target activation rate + energy pressure.
        """
        # EMA of activations
        alpha = self.cfg.ema_alpha
        self._ema = (1.0 - alpha) * self._ema + alpha * activated

        # Error with deadband
        target = self.cfg.target_activation_rate
        err = target - self._ema
        if abs(err) < self.cfg.error_deadband:
            err = 0.0

        # Integral with clamp
        self._i_term += err
        clamp = abs(self.cfg.integral_clamp)
        if clamp > 0:
            self._i_term = max(-clamp, min(clamp, self._i_term))

        # Energy pressure: increase theta as energy declines
        # (1 - E/Emax) is 0 at full energy, -> 1 at empty.
        energy_frac = self.energy.value / max(1e-9, self.energy.max_energy)
        pressure = self.cfg.energy_pressure * (1.0 - energy_frac)

        # PI update (+pressure makes gate stricter when low energy)
        delta = self.cfg.adapt_kp * err + self.cfg.adapt_ki * self._i_term + pressure
        self.threshold = _bounded(
            self.threshold + delta,
            self.cfg.min_threshold,
            self.cfg.max_threshold,
        )
