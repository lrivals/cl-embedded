"""Tests du streaming CMAPSS via sensor_stream.py."""

import pytest
from pathlib import Path
import subprocess
import sys


class TestCmapssStream:
    """Tests dry-run du streaming CMAPSS — ne requièrent pas de board."""

    def test_cmapss_stream_dryrun_runs(self, tmp_path):
        """sensor_stream.py --dataset cmapss --dry-run ne lève pas d'exception."""
        result = subprocess.run(
            [
                sys.executable, "scripts/sensor_stream.py",
                "--dataset", "cmapss",
                "--model", "ewc",
                "--dry-run",
                "--n-samples", "10",
                "--n-tasks", "2",
                "--output", str(tmp_path / "stream_test.json"),
            ],
            capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, f"Erreur stream CMAPSS : {result.stderr}"

    @pytest.mark.skipif(
        not Path("configs/cmapss_feature_subset.yaml").exists(),
        reason="cmapss_feature_subset.yaml non généré (lancer S2305 d'abord)"
    )
    def test_cmapss_feature_subset_loaded(self):
        """Le fichier cmapss_feature_subset.yaml est chargé sans erreur."""
        import yaml
        d = yaml.safe_load(Path("configs/cmapss_feature_subset.yaml").read_text())
        assert "selected_features" in d
        assert len(d["selected_features"]) == 5

    def test_cmapss_stream_hdc_dryrun(self, tmp_path):
        """sensor_stream.py --model hdc --dataset cmapss --dry-run (S2304 intégré)."""
        result = subprocess.run(
            [
                sys.executable, "scripts/sensor_stream.py",
                "--dataset", "cmapss",
                "--model", "hdc",
                "--dry-run",
                "--n-samples", "10",
                "--n-tasks", "2",
                "--output", str(tmp_path / "stream_hdc.json"),
            ],
            capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, f"Erreur stream HDC CMAPSS : {result.stderr}"
