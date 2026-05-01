"""Shared pytest fixtures for grounding-toolkit.

Plugs in the public test framework so unit tests can access stubs /
factories / fixtures without re-importing them.
"""

from __future__ import annotations

pytest_plugins = ["grounding.testing.fixtures"]
