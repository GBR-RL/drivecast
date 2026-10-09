"""Command line entry point."""

from __future__ import annotations

import typer

from drivecast import __version__

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Predicting hard-drive failures in the Backblaze fleet."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)
