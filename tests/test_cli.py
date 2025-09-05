import subprocess
import sys
import os

def test_cli_help_runs():
    # Run the CLI with no args: should print help and exit 0
    cmd = [sys.executable, "-m", "sundew.cli"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "Sundew Algorithm CLI" in proc.stdout or "Sundew Algorithm" in proc.stdout

def test_cli_demo_runs_quick():
    # Run a tiny demo to ensure wiring works (very small to keep CI fast)
    cmd = [sys.executable, "-m", "sundew.cli", "--demo", "--events", "5", "--temperature", "0.1"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "Final Report" in proc.stdout
