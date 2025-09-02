from dataclasses import dataclass

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

@dataclass
class EnergyAccount:
    value: float
    max_value: float

    def spend(self, amount: float) -> float:
        self.value = clamp(self.value - amount, 0.0, self.max_value)
        return amount

    def tick(self, regen: float, keepalive_cost: float) -> None:
        self.value = clamp(self.value + regen - keepalive_cost, 0.0, self.max_value)
