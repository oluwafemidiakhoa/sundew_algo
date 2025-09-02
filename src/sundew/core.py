from __future__ import annotations

import math
import time
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .config import SundewConfig
from .energy import EnergyAccount, clamp
from .gating import significance_score, gate_probability


@dataclass
class ProcessingResult:
    input_data: Dict
    significance: float
    insights: str
    processing_time: float
    energy_consumed: float
    activated: bool
    timestamp: str


@dataclass
class SundewMetrics:
    total_inputs: int = 0
    activations: int = 0
    total_processing_time: float = 0.0
    total_energy_spent: float = 0.0
    ema_activation_rate: float = 0.0

    def as_dict(self) -> Dict:
        return {
            "total_inputs": self.total_inputs,
            "activations": self.activations,
            "activation_rate": (self.activations / self.total_inputs) if self.total_inputs else 0.0,
            "ema_activation_rate": self.ema_activation_rate,
            "avg_processing_time": (self.total_processing_time / max(1, self.activations)),
            "total_energy_spent": self.total_energy_spent,
        }


class SundewAlgorithm:
    """Bio-inspired selective activation for AI systems."""

    def __init__(self, config: SundewConfig = SundewConfig()):
        self.cfg = config
        self.threshold = self.cfg.activation_threshold
        self.energy = EnergyAccount(self.cfg.max_energy, self.cfg.max_energy)
        self.state = "dormant"
        self.metrics = SundewMetrics()
        self.history: List[ProcessingResult] = []

        # --- PI controller state ---
        self._integral_error: float = 0.0

        random.seed(self.cfg.rng_seed)

        # Validate significance weights sum to 1.0
        s = (
            self.cfg.w_magnitude
            + self.cfg.w_anomaly
            + self.cfg.w_context
            + self.cfg.w_urgency
        )
        if not math.isclose(s, 1.0, rel_tol=1e-6):
            raise ValueError(f"Significance weights must sum to 1.0, got {s:.3f}")

    # ---------- Core utilities ----------

    def evaluate_significance(self, x: Dict) -> float:
        return significance_score(
            x,
            self.cfg.w_magnitude,
            self.cfg.w_anomaly,
            self.cfg.w_context,
            self.cfg.w_urgency,
        )

    def _decide_activation(self, sig: float) -> bool:
        p = gate_probability(sig, self.threshold, self.cfg.gate_temperature)
        return random.random() < p

    # ---------- PI Adaptation (with deadband + anti-windup) ----------

    def _adapt_threshold(self) -> None:
        """
        PI control toward target activation rate with energy pressure:
          - error = (ema - target), positive when over-activating
          - proportional term reacts to current error
          - integral term accumulates error over time (anti-windup applied)
          - energy pressure raises threshold when battery is low
        """
        target = self.cfg.target_activation_rate
        ema = self.metrics.ema_activation_rate

        # Positive when over target (means "be stricter")
        error = ema - target

        # Deadband: ignore tiny deviations to avoid jitter
        if abs(error) < self.cfg.error_deadband:
            error_for_integrator = 0.0
            p_term = 0.0
        else:
            error_for_integrator = error
            p_term = self.cfg.adapt_kp * error

        # Anti-windup: if we're at a bound and the integral would drive further out, don't integrate
        at_min = math.isclose(self.threshold, self.cfg.min_threshold, rel_tol=0.0, abs_tol=1e-12)
        at_max = math.isclose(self.threshold, self.cfg.max_threshold, rel_tol=0.0, abs_tol=1e-12)
        pushing_lower = (error_for_integrator < 0.0) and at_min
        pushing_higher = (error_for_integrator > 0.0) and at_max
        if not (pushing_lower or pushing_higher):
            self._integral_error += error_for_integrator
            # Clamp integral
            self._integral_error = clamp(
                self._integral_error, -self.cfg.integral_clamp, self.cfg.integral_clamp
            )

        i_term = self.cfg.adapt_ki * self._integral_error

        # Energy pressure: lower energy -> raise threshold (be stricter), softened
        energy_frac = self.energy.value / self.energy.max_value
        pressure = (1.0 - energy_frac) * self.cfg.energy_pressure

        # Combine
        delta = p_term + i_term + pressure

        # Anti-windup scaling near bounds (smooth landing)
        margin = min(self.threshold - self.cfg.min_threshold,
                     self.cfg.max_threshold - self.threshold)
        if margin < 0.1:
            scale = max(0.25, margin / 0.1)
            delta *= scale

        # Apply & clamp
        self.threshold = clamp(
            self.threshold + delta,
            self.cfg.min_threshold,
            self.cfg.max_threshold,
        )

    # ---------- Processing ----------

    def _deep_process(self, x: Dict, sig: float) -> Tuple[str, float, float]:
        start = time.time()
        complexity_mult = 1.0 + 1.5 * sig
        energy_cost = self.cfg.base_processing_cost * complexity_mult
        self.metrics.total_energy_spent += self.energy.spend(energy_cost)

        delay = 0.05 + 0.25 * sig
        time.sleep(delay)

        band = "CRITICAL" if sig >= 0.9 else ("HIGH" if sig >= 0.7 else "MODERATE")
        insight = (
            f"Deep analysis of {x.get('type','unknown')} "
            f"(mag={x.get('magnitude',0):.1f}) — {band} ({sig:.3f})"
        )
        return insight, energy_cost, time.time() - start

    def _dormancy_tick(self) -> None:
        lo, hi = self.cfg.dormancy_regen
        self.energy.tick(random.uniform(lo, hi), self.cfg.dormant_tick_cost)

    def process(self, x: Dict) -> Optional[ProcessingResult]:
        self.metrics.total_inputs += 1
        self.state = "evaluating"

        # lightweight evaluation cost
        self.metrics.total_energy_spent += self.energy.spend(self.cfg.eval_cost)

        sig = self.evaluate_significance(x)
        activate = self._decide_activation(sig)

        # EMA update
        alpha = self.cfg.ema_alpha
        self.metrics.ema_activation_rate = (
            (1 - alpha) * self.metrics.ema_activation_rate + alpha * (1.0 if activate else 0.0)
        )

        if not activate:
            self.state = "dormant"
            self._dormancy_tick()
            self._adapt_threshold()
            return None

        self.state = "processing"
        insight, energy_spent, proc_time = self._deep_process(x, sig)
        self.metrics.activations += 1
        self.metrics.total_processing_time += proc_time

        result = ProcessingResult(
            input_data=x,
            significance=sig,
            insights=insight,
            processing_time=proc_time,
            energy_consumed=energy_spent + self.cfg.eval_cost,
            activated=True,
            timestamp=datetime.utcnow().isoformat(),
        )
        self.history.append(result)

        self._adapt_threshold()
        self.state = "dormant"
        return result

    # ---------- Reporting ----------

    def report(self, assumed_baseline_per_event: float = 15.0) -> Dict:
        m = self.metrics.as_dict()
        m.update(
            {
                "energy_remaining": self.energy.value,
                "threshold": self.threshold,
            }
        )
        baseline_cost = self.metrics.total_inputs * assumed_baseline_per_event
        actual_cost = self.metrics.total_energy_spent
        savings = (baseline_cost - actual_cost) / max(1e-9, baseline_cost)
        savings = max(-1.0, min(1.0, savings))
        m.update(
            {
                "baseline_energy_cost": baseline_cost,
                "actual_energy_cost": actual_cost,
                "estimated_energy_savings_pct": 100.0 * savings,
            }
        )
        return m
