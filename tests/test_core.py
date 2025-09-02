from sundew import SundewAlgorithm, SundewConfig

def test_runs_and_reports():
    algo = SundewAlgorithm(SundewConfig(gate_temperature=0.0))
    # feed a few benign events
    for i in range(10):
        algo.process({"magnitude": 10*i, "anomaly_score": 0.2, "context_relevance": 0.3, "urgency": 0.1})
    r = algo.report()
    assert "estimated_energy_savings_pct" in r
    assert 0.0 <= r["threshold"] <= 1.0
