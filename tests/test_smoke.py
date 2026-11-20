"""Smoke tests: confirm the package imports and the test harness runs."""

from __future__ import annotations

import offside


def test_package_has_version() -> None:
    assert isinstance(offside.__version__, str)
    assert offside.__version__
