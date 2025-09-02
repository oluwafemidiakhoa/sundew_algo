🤝 Contributing to Sundew

Thank you for your interest in contributing to the Sundew Algorithm project!
This repo explores bio-inspired, event-driven intelligence with adaptive thresholds and energy-aware gating.

📋 How to Contribute
1. Clone & Setup
git clone https://github.com/<your-username>/sundew.git
cd sundew
python -m venv .venv
.venv\Scripts\activate   # Windows
# or source .venv/bin/activate on Linux/Mac

python -m pip install -e .

2. Run Demos
# Basic demo
python -m sundew.cli --demo

# With custom events
python -m sundew.cli --events 100 --temperature 0.1

🔬 Running Benchmarks
Grid Search (multi-preset sweeps)
python benchmarks/grid_search.py --events 300 --repeats 3 --out results/grid.csv --logdir results/runs
python benchmarks/plot_grid.py --csv results/grid.csv --out results/plots


This produces:

results/grid.csv (raw data)

Plots in results/plots/

Single Run (deep dive)
python benchmarks/plot_single_run.py --preset tuned_v2 --events 400 \
  --out results/plots_tuned --savecsv results/runs_tuned/single_run_tuned_v2.csv


This produces:

Per-event CSV

Plots of threshold adaptation, energy, and activations

⚙️ Adding New Presets

Presets live in src/sundew/config_presets.py.

Example:

"my_experiment": SundewConfig(
    activation_threshold=0.65,
    target_activation_rate=0.20,
    adapt_kp=0.08,
    adapt_ki=0.02,
    error_deadband=0.005,
    energy_pressure=0.03,
)


After adding, you can run:

python benchmarks/plot_single_run.py --preset my_experiment --events 300 --out results/plots_mine

🧪 Writing Tests

Tests go under tests/. Use pytest:

pytest -v

🖼️ Results Documentation

Every run produces:

CSVs under results/runs*/

Plots under results/plots*/

See results/README.md
 for interpretation tips.

📜 Code Style

Follow PEP 8 (run flake8).

Keep functions small and interpretable.

Use docstrings to explain algorithms, not just parameters.

🚀 Submitting Pull Requests

Fork the repo & create a branch:

git checkout -b feature/my-improvement


Commit with clear messages.

Run tests and benchmarks.

Submit a Pull Request with:

What you changed

Why it matters

Any plots or metrics from your experiments

💡 Ideas for Contributions

New presets for different environments (e.g., ultra-low energy IoT, high-urgency monitoring).

Improved visualizations (interactive dashboards, seaborn styling).

Extensions: multi-tier gating, spiking neural network integration, real sensor datasets.

🙏 Acknowledgements

Inspired by the sundew plant’s selective activation strategy.
This repo is for research & educational purposes