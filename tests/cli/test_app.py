"""Tests for the Typer CLI application."""

from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from autofisher.cli import app as cli_module
from autofisher.cli.app import app


def test_cli_runs_without_subcommand() -> None:
    """Verify the root CLI help works without a subcommand."""
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "listen" not in result.stdout


def test_cli_callback_and_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify callback and main delegation."""
    listen = Mock()
    monkeypatch.setattr(cli_module, "listen", listen)
    cli_module.root()
    listen.assert_called_once_with()

    typer_app = Mock()
    monkeypatch.setattr(cli_module, "app", typer_app)
    cli_module.main()
    typer_app.assert_called_once_with()
