# 🤝 Contributing to Sundew

Thanks for your interest in improving **Sundew** — a bio-inspired, event-driven AI framework.

Repo: <https://github.com/oluwafemidiakhoa/sundew_algo>

---

## 📦 Setup

```bat
git clone https://github.com/oluwafemidiakhoa/sundew_algo
cd sundew_algo
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e .
Run a quick demo:

bat
Copy code
python -m sundew.cli --demo --events 40 --temperature 0.1
🔬 Benchmarks
Single-run (time-series visual + CSV)
bat
Copy code
python benchmarks\plot_single_run.py --preset tuned_v2 --events 400 ^
  --out results\plots_tuned ^
  --savecsv results\runs_tuned\single_run_tuned_v2.csv
Multi-preset sweep (combined CSV)
bat
Copy code
python benchmarks\run_presets.py --events 300 --repeats 3 ^
  --presets baseline tuned_v2 aggressive energy_saver ^
  --out results\grid_multi.csv --logdir results\runs_multi

python benchmarks\plot_grid.py --csv results\grid_multi.csv --out results\plots_multi
Artifacts land under results/.

⚙️ Add a Config Preset
Edit src/sundew/config_presets.py:

python
Copy code
"my_experiment": SundewConfig(
    activation_threshold=0.65,
    target_activation_rate=0.20,
    adapt_kp=0.08,
    adapt_ki=0.02,
    error_deadband=0.005,
    energy_pressure=0.03,
),
Test it:

bat
Copy code
python benchmarks\plot_single_run.py --preset my_experiment --events 300 --out results\plots_mine
🧪 Tests
bat
Copy code
python -m pip install -r requirements-dev.txt  # if present
pytest -q
🧹 Style
PEP 8; type hints where practical.

Keep control logic small & well-commented.

Prefer deterministic RNG paths for benchmarks (rng_seed).

🚀 Pull Requests
Create a feature branch:

bat
Copy code
git checkout -b feature/<short-name>
Add tests/plots if relevant.

Update results/README.md if you add new figures.

Open a PR with a clear description and sample outputs.

💡 Ideas
Additional presets (ultra-low-power IoT / high-urgency ops).

Alternative controllers (PID variants, energy-budget MPC).

Real-world datasets & adapters.

Visualization polish / dashboards.

🙏 Citation
If this helps your work, please cite using CITATION.cff or the GitHub “Cite this repository” button