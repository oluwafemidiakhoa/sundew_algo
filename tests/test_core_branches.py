import numpy as np
import pytest

from sundew import SundewAlgorithm
from sundew.config_presets import get_preset

def tiny_stream(n=200, hi_prob=0.15, seed=123):
    rng = np.random.default_rng(seed)
    # Create a simple significance-like scalar with occasional spikes
    base = rng.normal(0.0, 0.4, size=n)
    spikes = rng.random(n) < hi_prob
    base[spikes] += rng.uniform(0.9, 1.3, size=spikes.sum())  # force activations
    # Represent as dicts if your algorithm expects dict inputs
    for v in base:
        yield {"x": float(v)}

def run_for(cfg, n=200, seed=123):
    algo = SundewAlgorithm(cfg)
    out = {"activated": 0, "energy_spent": 0.0}
    for ev in tiny_stream(n=n, seed=seed):
        # Adjust call signature to your public API (process/process_input/step)
        res = algo.process_input(ev) if hasattr(algo, "process_input") else algo.process(ev)
        # Expect res to include activation boolean/flag; adapt if your API differs
        if isinstance(res, dict) and res.get("activated"):
            out["activated"] += 1
        # Energy accounting attribute names may differ—adapt to yours
        if hasattr(algo, "energy_spent"):
            out["energy_spent"] = algo.energy_spent
    return out

def test_core_energy_pressure_and_gate_paths():
    # Softer gate to exercise gate_temperature path
    cfg = get_preset("tuned_v2", overrides=dict(gate_temperature=0.15, energy_pressure=0.04))
    stats = run_for(cfg, n=300)
    assert stats["activated"] >= 1  # we crossed the gate at least once

def test_core_integral_clamp_and_bounds():
    # Push the controller harder to touch integral clamp / threshold bounds
    cfg = get_preset("tuned_v2", overrides=dict(adapt_kp=0.12, adapt_ki=0.05, min_threshold=0.2, max_threshold=0.9))
    stats = run_for(cfg, n=350)
    assert stats["activated"] >= 1

def test_ecg_best_smoke():
    # New frozen preset should instantiate and run
    cfg = get_preset("ecg_mitbih_best")
    stats = run_for(cfg, n=200)
    assert stats["activated"] >= 0  # smoke: main loop executes without raising
