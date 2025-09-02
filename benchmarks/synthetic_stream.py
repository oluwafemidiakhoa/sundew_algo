import random, json
EVENT_TYPES = ["environmental","security","health_monitor","system_alert","emergency"]

def gen_event(i):
    kind = random.choice(EVENT_TYPES)
    return {"id": i, "type": kind,
            "magnitude": random.uniform(0, 100),
            "anomaly_score": random.uniform(0, 1),
            "context_relevance": random.uniform(0, 1),
            "urgency": random.uniform(0, 1)}

if __name__ == "__main__":
    data = [gen_event(i) for i in range(500)]
    with open("synthetic_events.json", "w") as f:
        json.dump(data, f)
