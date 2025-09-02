import json, random, time
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List
from .core import SundewAlgorithm, ProcessingResult
from .config import SundewConfig

EVENT_TYPES = [
    {"type": "environmental", "anomaly_bias": (0.0, 0.4), "urgency_bias": (0.1, 0.6)},
    {"type": "security", "anomaly_bias": (0.4, 0.9), "urgency_bias": (0.3, 0.8)},
    {"type": "health_monitor", "anomaly_bias": (0.2, 0.7), "urgency_bias": (0.2, 0.7)},
    {"type": "system_alert", "anomaly_bias": (0.3, 0.8), "urgency_bias": (0.2, 0.9)},
    {"type": "emergency", "anomaly_bias": (0.8, 1.0), "urgency_bias": (0.9, 1.0)},
]

def synth_event(i: int) -> Dict:
    kind = random.choice(EVENT_TYPES)
    return {
        "id": f"event_{i:05d}",
        "type": kind["type"],
        "magnitude": random.uniform(0, 100),
        "anomaly_score": random.uniform(*kind["anomaly_bias"]),
        "context_relevance": random.uniform(0, 1),
        "urgency": random.uniform(*kind["urgency_bias"]),
        "timestamp": time.time(),
    }

def run_demo(n_events: int = 40, temperature: float = 0.1) -> Dict:
    cfg = SundewConfig(gate_temperature=temperature)
    algo = SundewAlgorithm(cfg)
    processed: List[ProcessingResult] = []

    print("🌿 Sundew Algorithm — Demo")
    print("=" * 60)
    print(f"Initial threshold: {algo.threshold:.3f} | Energy: {algo.energy.value:.1f}\n")

    for i in range(n_events):
        x = synth_event(i)
        res = algo.process(x)
        if res is None:
            print(f"{i+1:02d}. {x['type']:<15} ⏸ dormant | energy {algo.energy.value:6.1f} | thr {algo.threshold:.3f}")
        else:
            processed.append(res)
            print(
              f"{i+1:02d}. {x['type']:<15} ✅ processed (sig={res.significance:.3f}, "
              f"{res.processing_time:.3f}s, ΔE≈{res.energy_consumed:.1f}) | energy {algo.energy.value:6.1f} | thr {algo.threshold:.3f}"
            )

    print("\n🏁 Final Report")
    report = algo.report()
    for k, v in report.items():
        if isinstance(v, float):
            if "pct" in k: print(f"  {k:30s}: {v:7.2f}%")
            else:         print(f"  {k:30s}: {v:10.3f}")
        else:
            print(f"  {k:30s}: {v}")

    return {
        "config": asdict(cfg),
        "report": report,
        "processed_events": [asdict(r) for r in processed],
        "generated_at": datetime.utcnow().isoformat(),
    }
