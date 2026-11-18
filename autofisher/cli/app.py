"""Typer application."""

import typer

from autofisher.cli.commands.listen import listen

app = typer.Typer(
    name="autofisher",
    help="Detect fishing splash sounds and recast automatically.",
    invoke_without_command=True,
)


@app.callback(invoke_without_command=True)
def root() -> None:
    """Start listening when the root command is invoked."""
    listen()


def main() -> None:
    """Run the AutoFisher command-line application."""
    app()
