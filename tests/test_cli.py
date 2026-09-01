"""Tests for the Phase 1 CLI entrypoint (``python -m src.main``)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.main import main

SHIPPED_CONFIG = "configs/preprocessing_search_space.yaml"


def test_show_preprocessing_prints_summary(capsys, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    rc = main(["show-preprocessing", "--config", SHIPPED_CONFIG])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Preprocessing search space" in out
    assert "imputation" in out


def test_show_preprocessing_missing_config_returns_1(capsys, tmp_path):
    rc = main(["show-preprocessing", "--config", str(tmp_path / "missing.yaml")])
    err = capsys.readouterr().err
    assert rc == 1
    assert "not found" in err


def test_no_subcommand_is_usage_error():
    with pytest.raises(SystemExit):
        main([])
