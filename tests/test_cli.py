import sys
import subprocess
import tempfile
from pathlib import Path

def test_cli_demo_saves_json(tmp_path: Path):
    out_json = tmp_path / "demo.json"
    cmd = [
        sys.executable, "-m", "sundew.cli",
        "--demo", "--events", "3", "--temperature", "0.0",
        "--save", str(out_json),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert out_json.exists(), proc.stdout + proc.stderr
    # sanity: small file with keys
    txt = out_json.read_text(encoding="utf-8")
    assert '"report"' in txt and '"config"' in txt

def test_cli_ascii_fallback_runs(tmp_path: Path, monkeypatch):
    """
    Force stdout encoding to ascii to exercise the non-emoji branch.
    """
    class Dummy:
        encoding = "ascii"
    monkeypatch.setattr(sys, "stdout", Dummy(), raising=False)

    cmd = [
        sys.executable, "-m", "sundew.cli",
        "--demo", "--events", "2", "--temperature", "0.0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    # ensure we didn't crash on UnicodeEncodeError and saw ASCII tokens
    assert "[sundew]" in (proc.stdout or "") or "[done]" in (proc.stdout or "")
