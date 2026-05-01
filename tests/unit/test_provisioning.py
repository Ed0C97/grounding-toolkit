"""Provisioning bridge tests: descriptor + runner contract."""

from __future__ import annotations

import json

import pytest

from grounding.provisioning import descriptor, runner


def test_descriptor_shape() -> None:
    desc = descriptor()
    assert desc["toolkit"] == "grounding-toolkit"
    assert desc["version"].startswith("2026.")
    assert "groups" in desc
    assert isinstance(desc["groups"], list)
    assert len(desc["groups"]) >= 1


def test_descriptor_groups_have_id_and_name() -> None:
    desc = descriptor()
    for group in desc["groups"]:
        assert "id" in group
        assert "name" in group
        assert "runner" in group or "path" in group


def test_runner_smoke_group() -> None:
    """The smoke group runner returns a junit-xml string."""
    junit_xml = runner("smoke")
    assert isinstance(junit_xml, str)
    assert "<testsuites" in junit_xml
    assert "grounding.provisioning" in junit_xml


def test_runner_unknown_group_raises() -> None:
    with pytest.raises(KeyError):
        runner("definitely-not-a-real-group-zzz")


def test_descriptor_is_json_serialisable() -> None:
    desc = descriptor()
    js = json.dumps(desc)
    assert "grounding-toolkit" in js
