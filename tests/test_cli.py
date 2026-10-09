from typer.testing import CliRunner

from drivecast import __version__
from drivecast.cli import app


def test_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == __version__
