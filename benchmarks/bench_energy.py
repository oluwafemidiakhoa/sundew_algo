import json, time
from sundew import SundewAlgorithm, SundewConfig
from sundew.demo import synth_event

def run(n=500, temp=0.1, baseline_cost=15.0):
    algo = SundewAlgorithm(SundewConfig(gate_temperature=temp))
    for i in range(n):
        algo.process(synth_event(i))
    return algo.report(assumed_baseline_per_event=baseline_cost)

if __name__ == "__main__":
    start = time.time()
    rep = run()
    rep["wall_time_s"] = time.time() - start
    print(json.dumps(rep, indent=2))
