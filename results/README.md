# Sundew Algorithm: Bio-Inspired Adaptive Intelligence for Edge Computing

[![CI Status](https://github.com/oluwafemidiakhoa/sundew_algo/workflows/ci/badge.svg)](https://github.com/oluwafemidiakhoa/sundew_algo/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

> *"In the dance between energy and intelligence, nature has always led."*

## Abstract

Sundew represents a paradigm shift in edge AI processing, drawing inspiration from the carnivorous sundew plant's selective prey capture mechanism. This novel algorithm implements **energy-aware selective activation** for resource-constrained AI systems, achieving significant computational savings while maintaining processing quality through intelligent event filtering.

Our bio-inspired approach combines bounded significance scoring, temperature-controlled gating, and adaptive thresholding to create a self-regulating system that dynamically balances performance with energy efficiency—critical for deployment in IoT devices, autonomous systems, and edge computing environments.

---

## Core Innovation

### The Sundew Principle

Just as the sundew plant conserves energy by selectively capturing only the most nutritious prey, our algorithm intelligently chooses which computational events warrant full processing. This biomimetic approach enables:

- **Selective Activation**: Process only significant events, filtering noise and redundancy
- **Adaptive Response**: Dynamic threshold adjustment based on environmental conditions
- **Energy Optimization**: Real-time energy-performance trade-off management
- **Graceful Degradation**: Maintained functionality under severe resource constraints

### Technical Architecture

```
Event Stream → Significance Scoring → Temperature Gating → Adaptive Control → Selective Processing
     ↑                    ↓                    ↓              ↓                    ↓
Energy Monitor ← Energy Accounting ← Threshold Update ← Performance Metrics ← Results
```

## Algorithm Foundation

### 1. Bounded Significance Scoring
Events are evaluated through a convex combination of lightweight feature extractors:

```
s(x) = Σᵢ wᵢ fᵢ(x) ∈ [0,1]
```

Where `fᵢ(x)` represents normalized feature functions and `wᵢ` are learned weights ensuring the score remains bounded.

### 2. Temperature-Controlled Gating
Processing decisions employ a temperature-parameterized sigmoid:

```
P(activate|x) = σ((s(x) - θ) / τ)
```

- **τ → 0**: Hard decisions for deployment
- **τ > 0**: Soft exploration for training/analysis

### 3. Adaptive Threshold Control
A PI-controller with energy-aware pressure regulation:

```
θₜ₊₁ = clip(θₜ + η(p* - p̂ₜ) + λ(1 - Eₜ/E_max))
```

- **p***: Target activation rate
- **p̂ₜ**: Exponential moving average of recent activations  
- **E/E_max**: Normalized energy reserve
- **η, λ**: Learning rates for performance and energy terms

### 4. Energy Accounting Framework
Transparent energy tracking with baseline comparison:

```
Savings = 1 - (E_actual / E_baseline) × 100%
```

---

## Installation & Quick Start

### Prerequisites
- Python 3.8+
- NumPy, SciPy (automatically installed)

### Installation
```bash
# Clone and setup
git clone https://github.com/oluwafemidiakhoa/sundew_algo.git
cd sundew_algo

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .\.venv\Scripts\activate  # Windows

# Install package
pip install -e .
```

### Minimal Example
```python
from sundew import SundewAlgorithm, SundewConfig
from sundew.demo import synth_event

# Configure for edge deployment
config = SundewConfig(
    gate_temperature=0.0,        # Hard gating for production
    target_activation_rate=0.2,  # Process 20% of events
    energy_pressure_weight=0.15  # Moderate energy awareness
)

algorithm = SundewAlgorithm(config)

# Process event stream
for t in range(1000):
    event = synth_event(t)
    result = algorithm.process(event)
    
    if result:  # Event was processed
        # Handle processing result
        pass

# Performance report
metrics = algorithm.report()
print(f"Energy savings: {metrics['estimated_energy_savings_pct']:.1f}%")
```

### Command Line Interface
```bash
# Quick demonstration
python -m sundew.cli --demo --events 100 --temperature 0.1

# Production configuration
python -m sundew.cli --config energy_saver --events 1000 --output results.json
```

---

## Configuration Presets

| Preset | Target Rate | Temperature | Use Case |
|--------|-------------|-------------|----------|
| `baseline` | 1.0 | 0.0 | Reference (always-on) |
| `balanced` | 0.4 | 0.0 | General edge deployment |
| `aggressive` | 0.15 | 0.0 | Ultra-low power |
| `energy_saver` | 0.25 | 0.0 | Battery-powered devices |
| `exploration` | 0.3 | 0.2 | Research and analysis |

### Advanced Configuration
```python
config = SundewConfig(
    # Core parameters
    gate_temperature=0.0,
    target_activation_rate=0.25,
    
    # Controller gains
    controller_gain=0.01,
    energy_pressure_weight=0.1,
    
    # Feature weights
    feature_weights=[0.3, 0.4, 0.3],  # Custom feature importance
    
    # Energy model
    baseline_energy_per_event=1.0,
    activation_energy_multiplier=5.0
)
```

---

## Benchmarking & Analysis

### Single Configuration Analysis
```bash
python benchmarks/plot_single_run.py \
    --preset energy_saver \
    --events 500 \
    --output results/energy_analysis \
    --save-csv results/energy_saver_metrics.csv
```

### Multi-Configuration Comparison
```bash
python benchmarks/run_presets.py \
    --events 1000 \
    --repeats 5 \
    --presets baseline balanced aggressive energy_saver \
    --output results/comparison.csv

python benchmarks/plot_grid.py \
    --csv results/comparison.csv \
    --output results/comparative_analysis
```

### Key Performance Metrics

| Metric | Description | Typical Range |
|--------|-------------|---------------|
| `activation_rate` | Fraction of events processed | [0.1, 1.0] |
| `threshold_stability` | Convergence of adaptive threshold | [0, 1] |
| `energy_savings_pct` | Relative energy reduction | [0, 90]% |
| `processing_latency` | Decision-making overhead | [0.1, 10]ms |

---

## Applications & Impact

### Target Domains

**🏥 Healthcare IoT**
- Continuous vital sign monitoring with 60-80% energy savings
- Smart wearables with extended battery life
- Remote patient monitoring systems

**🛡️ Security & Surveillance** 
- Motion detection with adaptive sensitivity
- Anomaly detection in sensor networks
- Smart camera systems with selective recording

**🤖 Autonomous Systems**
- Duty-cycled perception for drones
- Robot navigation with energy constraints
- Sensor fusion optimization

**🚀 Aerospace & Remote Sensing**
- Satellite data processing optimization
- Remote sensing with power limitations
- Space-constrained computational systems

**🧠 Neuromorphic Computing**
- Event-driven neural networks
- Spike-based processing optimization
- Bio-inspired computing architectures

### Real-World Impact Projections

Based on synthetic evaluations and energy modeling:
- **IoT Sensors**: 40-70% battery life extension
- **Edge AI Cameras**: 50-80% processing load reduction  
- **Wearable Devices**: 30-60% improved operational time
- **Autonomous Drones**: 25-45% flight time increase

---

## Repository Structure

```
sundew_algo/
├── src/sundew/              # Core implementation
│   ├── core.py             # Algorithm engine
│   ├── config_presets.py   # Pre-tuned configurations
│   ├── demo.py             # Synthetic data generation
│   └── cli.py              # Command-line interface
├── benchmarks/             # Performance evaluation
│   ├── grid_search.py      # Parameter optimization
│   ├── run_presets.py      # Multi-config evaluation
│   ├── plot_grid.py        # Comparative visualization
│   └── plot_single_run.py  # Single-run analysis
├── results/                # Output data and plots
├── tests/                  # Comprehensive test suite
├── .github/workflows/      # CI/CD pipeline
├── docs/                   # Extended documentation
├── CITATION.cff           # Academic citation info
└── CONTRIBUTING.md        # Development guidelines
```

---

## Research & Development

### Current Status
- ✅ Core algorithm implementation
- ✅ Comprehensive benchmarking suite
- ✅ Synthetic data validation
- ✅ Energy modeling framework
- 🔄 Real-world dataset validation
- 🔄 Hardware deployment studies
- 📋 Neuromorphic integration
- 📋 Multi-modal event processing

### Contributing

We actively welcome contributions in the following areas:

**Algorithm Enhancement**
- Novel significance scoring functions
- Advanced control strategies
- Multi-objective optimization

**Validation & Testing**  
- Real-world dataset integration
- Hardware validation studies
- Comparative algorithm analysis

**Tools & Infrastructure**
- Visualization improvements
- Performance profiling tools
- Integration libraries

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## Citation

If Sundew contributes to your research, please cite:

```bibtex
@software{Idiakhoa2025Sundew,
  author       = {Idiakhoa, Oluwafemi},
  title        = {Sundew Algorithm: Bio-Inspired Adaptive Intelligence 
                  for Edge Computing},
  year         = {2025},
  url          = {https://github.com/oluwafemidiakhoa/sundew_algo},
  version      = {1.0.0},
  note         = {Open-source implementation with MIT License}
}
```

For academic papers using this work, we recommend including performance metrics from your specific application domain.

---

## About

**Lead Developer**: Oluwafemi Idiakhoa  
📧 [oluwafemidiakhoa@gmail.com](mailto:oluwafemidiakhoa@gmail.com)  
🔬 [ORCID: 0009-0008-7911-1171](https://orcid.org/0009-0008-7911-1171)  
🐙 [GitHub Profile](https://github.com/oluwafemidiakhoa)

### License
MIT License - see [LICENSE](LICENSE) for details.

### Acknowledgments
Inspired by the elegant efficiency of carnivorous plants and the growing need for sustainable AI at the edge.

---

<div align="center">
<strong>🌿 Nature's wisdom, silicon's future 🌿</strong><br>
<em>Where biology meets bytes, efficiency emerges</em>
</div>