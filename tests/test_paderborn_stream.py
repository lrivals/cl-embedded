"""Tests du streaming Paderborn via sensor_stream.py."""

import pytest
from pathlib import Path
import subprocess
import sys


class TestPaderbornStream:

    def test_paderborn_stream_dryrun_runs(self, tmp_path):
        """sensor_stream.py --dataset paderborn --dry-run ne lève pas d'exception."""
        result = subprocess.run(
            [
                sys.executable, "scripts/sensor_stream.py",
                "--dataset", "paderborn",
                "--model", "ewc",
                "--dry-run",
                "--n-samples", "10",
                "--n-tasks", "3",
                "--output", str(tmp_path / "stream_pad.json"),
            ],
            capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, f"Erreur stream Paderborn : {result.stderr}"

    @pytest.mark.skipif(
        not Path("configs/paderborn_feature_subset.yaml").exists(),
        reason="paderborn_feature_subset.yaml non généré (lancer S2311 d'abord)"
    )
    def test_paderborn_feature_subset_loaded(self):
        import yaml
        d = yaml.safe_load(Path("configs/paderborn_feature_subset.yaml").read_text())
        assert "selected_features" in d
        assert len(d["selected_features"]) == 5
