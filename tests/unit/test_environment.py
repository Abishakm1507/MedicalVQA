"""Initial environment and test harness verification test."""

import sys


def test_python_version() -> None:
    """Asserts that the test runtime is Python 3.11."""
    assert sys.version_info.major == 3
    assert sys.version_info.minor == 11


def test_package_imports() -> None:
    """Verifies that all core packages import cleanly."""
    import api
    import core
    import datasets
    import evaluation
    import localization
    import verification
    import vlm

    assert core is not None
    assert api is not None
    assert datasets is not None
    assert localization is not None
    assert vlm is not None
    assert verification is not None
    assert evaluation is not None
