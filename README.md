\# Sundew Algorithm

\*\*Energy-Aware Selective Activation for Edge AI Systems\*\*



\[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

\[!\[Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

\[!\[Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)



<p align="center">

&nbsp; <a href="results/plots/best\_tradeoffs.png">

&nbsp;   <img alt="ECG MIT-BIH Best Trade-off Plot" src="https://img.shields.io/badge/ECG%20MIT--BIH-best%20trade--off-6C63FF?logo=heartbeat\&logoColor=white">

&nbsp; </a>

&nbsp; <a href="results/updates/2025-09-ecg-mitbih.md">

&nbsp;   <img alt="Research Update" src="https://img.shields.io/badge/Research%20Update-2025--09-success?logo=readthedocs\&logoColor=white">

&nbsp; </a>

&nbsp; <a href="#-presets">

&nbsp;   <img alt="Preset: ecg\_mitbih\_best" src="https://img.shields.io/badge/Preset-ecg\_\_mitbih\_\_best-00b894?logo=python\&logoColor=white">

&nbsp; </a>

</p>



> ### ⭐ Featured Preset: `ecg\_mitbih\_best`

> \*\*Frozen from MIT-BIH sweep\*\* for a practical precision–recall–savings trade-off.  

> - `activation\_threshold = 0.65` · `gate\_temperature = 0.15` · `target\_activation\_rate = 0.10` · `refractory = 0`  

> - See \*\*plot\*\*: \[best\_tradeoffs.png](results/plots/best\_tradeoffs.png)  

> - See \*\*notes \& settings\*\*: \[2025-09 ECG MIT-BIH Update](results/updates/2025-09-ecg-mitbih.md)



\#### Quick start (ECG best)

```bash

python -m benchmarks.run\_ecg \\

&nbsp; --csv "data/MIT-BIH Arrhythmia Database.csv" \\

&nbsp; --preset ecg\_mitbih\_best \\

&nbsp; --limit 50000 \\

&nbsp; --save results/real\_ecg\_best.json

```



> \*"Nature's wisdom, encoded in silicon."\*



A bio-inspired, event-driven intelligence system designed for resource-constrained AI applications. Sundew implements adaptive selective activation to optimize energy consumption while maintaining processing quality on edge devices.



\## Overview



\### ✅ Real-Dataset Validation (MIT-BIH Arrhythmia)



We evaluated \*\*Sundew\*\* on the \*\*MIT-BIH Arrhythmia Database\*\* (PhysioNet; ~50k rows; binary abnormal-beat labels) using preset \*\*`ecg\_v1`\*\* and a 108-point grid sweep over threshold/temperature/targets.



\- \*\*Energy savings (median across sweep):\*\* ~90.8% (min 90.18%, max 91.31%)

\- \*\*Top configuration selection:\*\* constrained by savings ≥ 88%, FN ≤ 9000, FP-rate ≤ 0.08; ranked by \*\*F1, then precision\*\*

\- Artifacts:

&nbsp; - CSV: \[`results/best\_by\_counts.csv`](results/best\_by\_counts.csv)

&nbsp; - Markdown: \[`results/best\_by\_counts.md`](results/best\_by\_counts.md)

&nbsp; - Research note: \[`results/updates/2025-09-ecg-mitbih.md`](results/updates/2025-09-ecg-mitbih.md)



> Reproduce:

> ```bash

> python -m benchmarks.sweep\_ecg --csv "data/MIT-BIH Arrhythmia Database.csv" \\

>   --out results/sweep\_cm.csv --preset ecg\_v1 --limit 50000

>

> python -m benchmarks.select\_best \\

>   --csv results/sweep\_cm.csv \\

>   --out-csv results/best\_by\_counts.csv \\

>   --out-md results/best\_by\_counts.md \\

>   --research-md results/updates/2025-09-ecg-mitbih.md \\

>   --dataset-name "MIT-BIH Arrhythmia Database" \\

>   --dataset-notes "CSV from PhysioNet; ~50k rows; binary abnormal-beat labels; ecg\_v1 sweep." \\

>   --min-savings 88 --max-fn 9000 --max-fp-rate 0.08 \\

>   --sort f1,precision --top-n 20 --describe

> ```



The Sundew Algorithm addresses a critical challenge in edge AI: \*\*when to process events\*\* in energy-constrained environments. Instead of processing every input, Sundew intelligently selects which events deserve computational attention based on their significance and available energy resources.



\### Key Capabilities



\- \*\*Bounded Significance Scoring\*\*: Convex combination of lightweight feature extractors

\- \*\*Temperature-Controlled Gating\*\*: Soft gating during analysis, hard thresholding at inference

\- \*\*Adaptive Thresholding\*\*: PI-style controller with energy-aware pressure adjustment

\- \*\*Energy Accounting\*\*: Transparent comparison between baseline and actual energy consumption

\- \*\*Minimal Dependencies\*\*: Pure Python implementation with minimal overhead



\### Use Cases



| Domain | Application Examples |

|--------|---------------------|

| \*\*Healthcare\*\* | Continuous patient monitoring, wearable devices |

| \*\*Security\*\* | Smart surveillance, anomaly detection systems |

| \*\*Robotics\*\* | Duty-cycled perception, autonomous navigation |

| \*\*Aerospace\*\* | Remote sensing, space-constrained operations |

| \*\*Neuromorphic\*\* | Event-driven computing architectures |



\## Installation \& Quick Start



\### Prerequisites

\- Python 3.8 or higher

\- Virtual environment (recommended)



\### Installation



```bash

\# Create and activate virtual environment

python -m venv sundew-env

source sundew-env/bin/activate  # On Windows: sundew-env\\Scripts\\activate



\# Install package

pip install -U pip

pip install -e .

```



\### Quick Demo



```bash

\# Run demonstration with synthetic data

python -m sundew.cli --demo --events 40 --temperature 0.1 --save results.json

```



\*\*Temperature Parameter Guide:\*\*

\- `--temperature 0`: Hard gating (production/inference mode)

\- `--temperature 0.1-0.3`: Soft gating (analysis and exploration)



\## Algorithm Design



\### Core Mechanism



The Sundew Algorithm operates through four main stages:



\#### 1. Significance Scoring

Events are assigned bounded significance scores using a weighted combination of features:



```

s = Σ wᵢ fᵢ(x)    where s ∈ \[0,1]

```



\#### 2. Temperature-Controlled Gating

Activation probability is computed using a temperature-controlled sigmoid:



```

p = σ((s - θ) / τ)

activate ~ Bernoulli(p)

```



Where:

\- `τ → 0`: Hard thresholding (inference mode)

\- `τ > 0`: Soft gating (exploration mode)



\#### 3. Adaptive Threshold Control

The activation threshold adapts using PI control with energy pressure:



```

θ ← clip(θ + η(p\* - p̂) + λ(1 - E/Eₘₐₓ))

```



Where:

\- `p\*`: Target activation rate

\- `p̂`: Exponential moving average of recent activations

\- `E/Eₘₐₓ`: Normalized energy reserve



\#### 4. Energy Accounting

The system tracks and compares actual energy consumption against a baseline (always-on) scenario, providing transparent energy savings metrics.



\## Usage



\### Programmatic API



```python

from sundew import SundewAlgorithm, SundewConfig

from sundew.demo import synth\_event



\# Configure algorithm

config = SundewConfig(

&nbsp;   gate\_temperature=0.0,        # Hard gating for inference

&nbsp;   target\_activation\_rate=0.25, # Process ~25% of events

&nbsp;   energy\_pressure\_weight=0.1,  # Energy-aware adaptation

&nbsp;   threshold\_learning\_rate=0.01 # Adaptation speed

)



\# Initialize algorithm

algorithm = SundewAlgorithm(config)



\# Process events

for i in range(100):

&nbsp;   event = synth\_event(i)

&nbsp;   result = algorithm.process(event)

&nbsp;   

&nbsp;   if result is not None:

&nbsp;       print(f"Event {i} processed: {result}")

&nbsp;   else:

&nbsp;       print(f"Event {i} skipped")



\# Generate performance report

print(algorithm.report())

```



\### Configuration Presets



The system includes several pre-tuned configurations:



\- \*\*`baseline`\*\*: Conservative settings for stable operation

\- \*\*`tuned\_v2`\*\*: Balanced performance and efficiency

\- \*\*`aggressive`\*\*: Maximum selectivity for extreme energy constraints

\- \*\*`energy\_saver`\*\*: Optimized for battery-powered devices



\## Benchmarking \& Analysis



\### Single Run Analysis



Generate detailed time-series analysis and CSV output:



```bash

python benchmarks/plot\_single\_run.py \\

&nbsp; --preset tuned\_v2 \\

&nbsp; --events 400 \\

&nbsp; --out results/plots\_tuned \\

&nbsp; --savecsv results/single\_run.csv

```



\### Multi-Preset Comparison



Compare multiple configurations across repeated runs:



```bash

\# Generate comparison data

python benchmarks/run\_presets.py \\

&nbsp; --events 300 \\

&nbsp; --repeats 3 \\

&nbsp; --presets baseline tuned\_v2 aggressive energy\_saver \\

&nbsp; --out results/comparison.csv



\# Plot comparative results

python benchmarks/plot\_grid.py \\

&nbsp; --csv results/comparison.csv \\

&nbsp; --out results/comparison\_plots

```



\### Key Performance Metrics



| Metric | Description |

|--------|-------------|

| `total\_inputs` | Total events processed |

| `activations` | Number of events that triggered processing |

| `activation\_rate` | Proportion of events processed |

| `ema\_activation\_rate` | Smoothed activation rate |

| `actual\_energy\_cost` | Measured energy consumption |

| `baseline\_energy\_cost` | Energy cost of always-on processing |

| `estimated\_energy\_savings\_pct` | Relative energy savings |



\## Project Structure



```

sundew\_algo/

├── src/sundew/              # Core implementation

│   ├── core.py             # Main algorithm

│   ├── config\_presets.py   # Pre-defined configurations

│   └── demo.py             # Synthetic data generator

├── benchmarks/             # Performance evaluation

│   ├── grid\_search.py      # Parameter optimization

│   ├── run\_presets.py      # Multi-config benchmarking

│   ├── plot\_grid.py        # Comparative visualization

│   └── plot\_single\_run.py  # Single-run analysis

├── results/                # Output data and plots

├── tests/                  # Unit test suite

├── .github/workflows/      # CI/CD configuration

├── CITATION.cff           # Citation metadata

└── CONTRIBUTING.md        # Contribution guidelines

```



\## Contributing



We welcome contributions in the following areas:



\- \*\*Algorithm Improvements\*\*: New control strategies, optimization techniques

\- \*\*Energy Models\*\*: More sophisticated energy consumption models

\- \*\*Benchmarking\*\*: Additional datasets and evaluation metrics

\- \*\*Visualization\*\*: Enhanced plotting and analysis tools

\- \*\*Documentation\*\*: Tutorials, examples, and guides



Please see \[CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.



\## Citation



If you use Sundew in your research, please cite:



```bibtex

@software{Idiakhoa2025Sundew,

&nbsp; author       = {Idiakhoa, Oluwafemi},

&nbsp; title        = {Sundew Algorithm: Bio-Inspired Event-Driven Intelligence},

&nbsp; year         = {2025},

&nbsp; url          = {https://github.com/oluwafemidiakhoa/sundew\_algo},

&nbsp; note         = {Open-source prototype, MIT License}

}

```



Citation metadata is also available in \[CITATION.cff](CITATION.cff).



\## License



This project is licensed under the MIT License - see the \[LICENSE](LICENSE) file for details.



\## Contact



\*\*Oluwafemi Idiakhoa\*\*

\- Email: oluwafemidiakhoa@gmail.com

\- ORCID: \[0009-0008-7911-1171](https://orcid.org/0009-0008-7911-1171)



---

!\[Best trade-offs (F1 vs Energy Savings)](results/plots/best\_tradeoffs.png)



\*Inspired by nature, engineered for efficiency\*

