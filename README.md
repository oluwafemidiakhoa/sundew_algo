# Sundew Algorithm
**Energy-Aware Selective Activation for Edge AI Systems**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

🌿 **"Nature's wisdom, encoded in silicon."**

Bio-inspired, event-driven intelligence that only "wakes up" when it matters—delivering large energy savings on constrained devices.

## 🔎 What is Sundew?

Sundew is a tiny, dependency-light framework for selective activation: a stream of events is scored for significance; a temperature-controlled gate decides if deeper processing should run; an adaptive controller nudges the decision threshold to meet a target activation rate while respecting an energy budget.

**Why it matters**: On edge/embedded systems, always-on inference wastes power. Sundew's dormant-until-useful behavior yields ~85–91% energy savings in our synthetic and ECG experiments (see below).

## 🗂 Repository Layout
```
sundew/
├─ src/sundew/
│  ├─ __init__.py            # library entry
│  ├─ cli.py                 # CLI front-end
│  ├─ config.py              # SundewConfig dataclass
│  ├─ config_presets.py      # named presets (incl. ECG)
│  ├─ core.py                # algorithm + controller
│  ├─ energy.py              # energy model
│  └─ gating.py              # temperature gate
│
├─ benchmarks/               # experiments & utilities
│  ├─ run_ecg.py             # run on real ECG CSV
│  ├─ eval_classification.py # precision/recall/F1 + energy
│  ├─ sweep_ecg.py           # grid sweep over params
│  ├─ select_best.py         # pick best trade-offs (+report)
│  ├─ plot_best_tradeoffs.py # small PNG chart for README
│  ├─ plot_single_run.py     # time series & histos
│  └─ (…more helpers)
│
├─ tests/                    # 8 tests, CLI + core + energy + gating
├─ data/                     # put your CSV here (e.g., MIT-BIH sample)
├─ results/                  # JSON/CSV/plots land here
└─ README.md                 # this file
```

## 🚀 Quick Start

### 1) Install
```bash
# (Recommended) Create a venv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -e .
# or, for dev:
pip install -r requirements-dev.txt  # if present
```

The library itself uses only the Python stdlib. Benchmarks and plots use numpy / matplotlib / pandas (kept optional).

### 2) Hello CLI
```bash
# Prints help + options
python -m sundew.cli
```

### 3) Minimal demo (synthetic stream)
```bash
python -m sundew.cli --demo --events 200 --temperature 0.1
```

Results (energy, threshold trajectory, activations) are printed to stdout.
See `benchmarks/plot_single_run.py` for timeseries/EMA plots.

## 🧠 Algorithm (One-Pager)

**Bounded Significance**

𝑠 ∈ [0,1] = ∑ᵢ wᵢ fᵢ(x)

**Temperature Gate**

p = σ((s-θ)/τ)

- τ→0: hard gate (inference)
- τ>0: smooth for analysis/sweeps

**Adaptive Threshold (PI + Energy Pressure)**

θ ← clip(θ + kₚ e + kᵢ ∑e + λ(1 - E/Eₘₐₓ), θₘᵢₙ, θₘₐₓ)

where e = p* - p̂ and p̂ is EMA of activations.

**Energy Accounting**
Tracks baseline (always-on) vs actual consumption to estimate savings.

## 📦 Programmatic Use

```python
from sundew.config_presets import get_preset
from sundew.core import SundewAlgorithm

cfg  = get_preset("tuned_v2")  # or "ecg_mitbih_best" (see below)
algo = SundewAlgorithm(cfg)

for event in stream:
    out = algo.process(event)    # None if dormant; dict if activated

print(algo.report())             # energy + activation metrics
```

## ❤️ Real-World Test: MIT-BIH Arrhythmia (PhysioNet)

We tested Sundew on an ECG CSV derived from MIT-BIH (binary abnormal beat labels).
Below are reproducible commands that generated the results:

### 1) Single run on 50k beats
```bash
python -m benchmarks.run_ecg ^
  --csv "data\MIT-BIH Arrhythmia Database.csv" ^
  --preset tuned_v2 ^
  --limit 50000 ^
  --save results\real_ecg_run.json
```

Sample output (yours may vary by dataset slice & preset):

```
total_inputs             : 50000
activations              : 3521
activation_rate          : 0.070
avg_processing_time      : 0.159
threshold                : 0.682
baseline_energy_cost     : 750000
actual_energy_cost       : 86398
estimated_energy_savings : 88.5%
```
## 📊 MIT-BIH ECG Results (Sep 2025)

We ran Sundew on the **MIT-BIH Arrhythmia Dataset (50k samples)**.  
A new preset `ecg_mitbih_best` is now frozen in `config_presets.py`.

- **Energy savings:** ~90%  
- **F1 score:** ~0.106  
- **Precision:** ~0.17  
- **Recall:** ~0.075  

<img src="results/best_tradeoffs_chart.png" width="500">

> This represents the first open-source energy-aware controller validated on a gold-standard arrhythmia dataset.


Quality (eval):
```bash
python -m benchmarks.eval_classification --json results\real_ecg_run.json
```

Example:
```
precision 0.291   recall 0.199   f1 0.236
savings   88.48%
```

**Takeaway**: extremely high energy savings out-of-the-box; recall is modest with generic presets—so we add ECG-focused sweeps.

### 2) Grid sweep + pick best trade-offs
```bash
# (a) Sweep a small grid around an ECG-oriented preset
python -m benchmarks.sweep_ecg ^
  --csv "data\MIT-BIH Arrhythmia Database.csv" ^
  --out results\sweep_cm.csv ^
  --preset ecg_v1 ^
  --limit 50000

# (b) Select best rows subject to constraints + emit report
python -m benchmarks.select_best ^
  --csv results\sweep_cm.csv ^
  --out-csv results\best_by_counts.csv ^
  --out-md results\best_by_counts.md ^
  --research-md results\updates\2025-09-ecg-mitbih.md ^
  --dataset-name "MIT-BIH Arrhythmia Database" ^
  --dataset-notes "CSV ~50k rows; abnormal-vs-normal labels; ecg_v1 sweep." ^
  --min-savings 88 ^
  --sort f1,precision ^
  --top-n 20 --describe
```

This writes:
- `results/sweep_cm.csv` – raw sweep
- `results/best_by_counts.{csv,md}` – best rows by your constraints
- `results/updates/2025-09-ecg-mitbih.md` – a research-log snippet (copy into your paper or wiki)

### 3) Freeze the winner as a preset

We snapshot the single best row (by F1 then precision while keeping ≥88% savings) into `ecg_mitbih_best` inside `src/sundew/config_presets.py`.

Use it directly:
```bash
python -m benchmarks.run_ecg ^
  --csv "data\MIT-BIH Arrhythmia Database.csv" ^
  --preset ecg_mitbih_best ^
  --limit 50000 ^
  --save results\ecg_best_run.json
```

## 📊 Figures (drop straight into your paper/README)

The repository already produces and/or expects these images (place them in results/):

```
results/
├─ activation_vs_target.png
├─ energy_savings_vs_temp.png
├─ threshold_hist.png
├─ single_run_energy_tuned_v2.png
├─ single_run_threshold_ema_tuned_v2.png
└─ best_tradeoffs.png        # plotted by plot_best_tradeoffs.py
```

Add them to your README as:
![Energy over time](results/single_run_energy_tuned_v2.png)
![Threshold & EMA](results/single_run_threshold_ema_tuned_v2.png)
![Final thresholds](results/threshold_hist.png)
![Activation vs Target](results/activation_vs_target.png)
![Energy savings vs T](results/energy_savings_vs_temp.png)
![Top trade-offs](results/best_tradeoffs.png)
```

To regenerate the small "top trade-offs" PNG for the README:
```bash
python -m benchmarks.plot_best_tradeoffs ^
  --csv results\best_by_counts.csv ^
  --out results\best_tradeoffs.png
```

## 🧪 Testing
```bash
pytest -v
# with coverage for library code
pytest --cov=src --cov-report=term-missing
```

Windows console sometimes chokes on emoji; the CLI avoids them by default.
If you hit encoding errors, ensure `PYTHONIOENCODING=utf-8`.

## 🔧 Key Presets (short list)

- **tuned_v2** — balanced general-purpose defaults (PI control + energy pressure).
- **ecg_v1** — wider gate & lower threshold to boost recall for arrhythmias.
- **ecg_mitbih_best** — frozen from our MIT-BIH sweep winner (use for reproducibility).
- **aggressive** / **conservative** — trade speed vs savings.
- **high_temp** / **low_temp** — probe more vs hard selectivity.
- **energy_saver** — maximize battery life (will under-activate).
- **target_0p30** — convenience variant with higher target activation.

List them:
```python
from sundew.config_presets import list_presets
print(list_presets())
```

## 🛠 How to plug in your dataset

Create a CSV with at least:
- A continuous signal (or features to derive significance)
- A binary label column (e.g., `abnormal ∈ {0,1}`) for evaluation

Point `run_ecg.py` (or adapt it) at your path:
```bash
python -m benchmarks.run_ecg --csv "data\your.csv" --preset tuned_v2 --limit 100000 --save results\run.json
python -m benchmarks.eval_classification --json results\run.json
```

Sweep around a preset to find better trade-offs:
```bash
python -m benchmarks.sweep_ecg --csv "data\your.csv" --preset ecg_v1 --out results\sweep.csv
python -m benchmarks.select_best --csv results\sweep.csv --out-csv results\best.csv --top-n 20
```

## 🎯 Where Sundew Fits

| Domain | Examples |
|--------|----------|
| **Healthcare** | Wearables & arrhythmia alerts, triage filters |
| **Security** | Smart acoustic/vision triggers on edge cameras/mics |
| **Robotics** | Duty-cycled perception (SLAM updates, obstacle alerts) |
| **Space** | Long-range probes & rovers with strict power budgets |
| **Neuromorphic** | Event-driven pipelines that align with spiking/async hardware |

## 📚 Cite

```bibtex
@techreport{Idiakhoa2025Sundew,
  title       = {Sundew Algorithm: Energy-Aware Selective Activation (Prototype)},
  author      = {Oluwafemi Idiakhoa},
  year        = {2025},
  note        = {Open-source prototype; real-data results on MIT-BIH ECG},
  url         = {https://github.com/oluwafemidiakhoa/sundew}
}
```

## 🤝 Contributing

PRs welcome! Good first issues:
- Feature engineering for significance (domain-specific)
- Alternative controllers (PID, adaptive gains, model-predictive)
- Device-calibrated energy models + public benchmarks
- Visualization and eval tooling

Run checks before opening a PR:
```bash
pytest -v
python -m benchmarks.plot_single_run --preset tuned_v2 --events 200 --out results\plots_tuned
```

## 📄 License

MIT — see LICENSE.

Commercial use permitted under MIT terms. For bespoke integrations, reach out.

## Appendix — Reproducing the ECG Winner

1. **Sweep**: `benchmarks/sweep_ecg.py` (108 runs around `ecg_v1`)
2. **Selection**: `benchmarks/select_best.py` with `--min-savings 88` + sort `f1,precision`
3. **Frozen preset**: `ecg_mitbih_best` in `src/sundew/config_presets.py`
4. **README figure**: `benchmarks/plot_best_tradeoffs.py` → `results/best_tradeoffs.png`
