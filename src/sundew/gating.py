import math
import random

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def significance_score(x, w_mag, w_ano, w_ctx, w_urg) -> float:
    mag = clamp(float(x.get("magnitude", 0.0)) / 100.0, 0.0, 1.0)
    ano = clamp(float(x.get("anomaly_score", 0.0)), 0.0, 1.0)
    ctx = clamp(float(x.get("context_relevance", 0.0)), 0.0, 1.0)
    urg = clamp(float(x.get("urgency", 0.0)), 0.0, 1.0)
    sig = w_mag*mag + w_ano*ano + w_ctx*ctx + w_urg*urg
    sig += random.uniform(-0.03, 0.03)
    return clamp(sig, 0.0, 1.0)

def gate_probability(sig: float, threshold: float, temperature: float) -> float:
    if temperature <= 0.0:  # hard gate
        return 1.0 if sig >= threshold else 0.0
    t = max(1e-6, temperature)
    z = (sig - threshold) / t
    return 1.0 / (1.0 + math.exp(-z))
