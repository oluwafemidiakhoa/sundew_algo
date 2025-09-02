from sundew.gating import gate_probability

def test_gate_hard_soft():
    assert gate_probability(0.8, 0.7, 0.0) == 1.0
    assert gate_probability(0.6, 0.7, 0.0) == 0.0
    p = gate_probability(0.7, 0.7, 0.1)
    assert 0.45 < p < 0.55  # roughly centered
