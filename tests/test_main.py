"""Tests for the module entry point."""

import runpy
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("run_name", ["__main__", "autofisher.__main__"])
def test_module_entry_point(
    monkeypatch: pytest.MonkeyPatch,
    run_name: str,
) -> None:
    """Verify entry and non-entry module guard behavior."""
    main = Mock()
    monkeypatch.setattr("autofisher.cli.app.main", main)
    runpy.run_module("autofisher.__main__", run_name=run_name)
    assert main.call_count == (1 if run_name == "__main__" else 0)
