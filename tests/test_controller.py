import math
from sundew import SundewAlgorithm, SundewConfig

def test_threshold_moves_down_when_under_target_with_PI():
    cfg = SundewConfig(
        activation_threshold=0.7,
        target_activation_rate=0.25,
        ema_alpha=1.0,        # trust last measurement
        adapt_kp=0.06,
        adapt_ki=0.02,
        energy_pressure=0.0,  # isolate controller behavior
        min_threshold=0.2,
        max_threshold=0.9,
        gate_temperature=0.1,
    )
    algo = SundewAlgorithm(cfg)

    # Strongly under target
    algo.metrics.ema_activation_rate = 0.0
    t0 = algo.threshold
    for _ in range(5):
        algo._adapt_threshold()
    t1 = algo.threshold
    assert t1 < t0, f"threshold should go DOWN when under-target: {t0} -> {t1}"

def test_threshold_moves_up_when_over_target_with_PI():
    cfg = SundewConfig(
        activation_threshold=0.7,
        target_activation_rate=0.25,
        ema_alpha=1.0,
        adapt_kp=0.06,
        adapt_ki=0.02,
        energy_pressure=0.0,
        min_threshold=0.2,
        max_threshold=0.9,
        gate_temperature=0.1,
    )
    algo = SundewAlgorithm(cfg)

    # Strongly over target
    algo.metrics.ema_activation_rate = 0.9
    t0 = algo.threshold
    for _ in range(5):
        algo._adapt_threshold()
    t1 = algo.threshold
    assert t1 > t0, f"threshold should go UP when over-target: {t0} -> {t1}"

def test_energy_pressure_increases_threshold_when_low_energy():
    cfg = SundewConfig(
        activation_threshold=0.7,
        target_activation_rate=0.25,
        ema_alpha=1.0,
        adapt_kp=0.0,         # isolate energy pressure
        adapt_ki=0.0,
        energy_pressure=0.05,
        min_threshold=0.2,
        max_threshold=0.9,
    )
    algo = SundewAlgorithm(cfg)

    # Neutralize control term
    algo.metrics.ema_activation_rate = cfg.target_activation_rate

    # High energy -> no change
    algo.energy.value = cfg.max_energy
    t0 = algo.threshold
    algo._adapt_threshold()
    t1 = algo.threshold
    assert abs(t1 - t0) < 1e-9, f"high energy should not change threshold: {t0} -> {t1}"

    # Low energy -> threshold increases
    algo.threshold = t0
    algo.energy.value = 0.2 * cfg.max_energy
    algo._adapt_threshold()
    t2 = algo.threshold
    assert t2 > t0, f"low energy should increase threshold: {t0} -> {t2}"
