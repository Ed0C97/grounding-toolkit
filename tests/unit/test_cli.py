"""CLI smoke tests: version / smoke / manifest subcommands."""

from __future__ import annotations

import io
import sys

import pytest

from grounding.cli.main import main


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["version"])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert out.startswith("2026.")


def test_cli_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["smoke"])
    err = capsys.readouterr().err.strip()
    assert rc == 0
    assert err == "OK"


def test_cli_manifest(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["manifest"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "grounding-toolkit" in out
    assert '"groups"' in out


def test_cli_unknown_subcommand_exits_with_error() -> None:
    with pytest.raises(SystemExit):
        main(["nonexistent-subcommand-xyz"])
