import sys
import subprocess
import os
import shlex
import pytest

PY = sys.executable

def run(cmd):
    # cross-platform small wrapper
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def test_cli_help_ok():
    r = run([PY, "-m", "sundew.cli", "--help"])
    assert r.returncode == 0
    assert "usage" in r.stdout.lower()

def test_cli_list_presets_ok():
    r = run([PY, "-m", "sundew.cli", "list-presets"])
    assert r.returncode == 0
    assert "tuned_v2" in r.stdout  # at least one well-known name

def test_cli_dry_run_ok(tmp_path):
    # If your CLI has a dry-run or print-config path, use it; otherwise adapt to a no-op that doesn’t touch files.
    out = tmp_path / "noop.json"
    r = run([PY, "-m", "sundew.cli", "print-config", "--preset", "tuned_v2"])
    # If your CLI subcommand differs, swap to whatever prints config; the goal is to exercise arg parsing branches.
    assert r.returncode == 0
    assert "tuned_v2" in r.stdout
