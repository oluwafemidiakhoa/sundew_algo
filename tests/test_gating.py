from sundew.energy import EnergyAccount

def test_spend_and_tick():
    e = EnergyAccount(50.0, 100.0)
    e.spend(10.0)
    assert e.value == 40.0
    e.tick(regen=5.0, keepalive_cost=1.0)
    assert 43.9 <= e.value <= 44.1  # allow float wiggle
