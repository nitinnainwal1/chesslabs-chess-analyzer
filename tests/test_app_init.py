"""Tests for Phase 1 - Project initialization & imports."""

import importlib


def test_core_dependencies():
    """Verify that all required Python libraries can be imported cleanly."""
    required_packages = [
        "chess",
        "streamlit",
        "pandas",
        "numpy",
        "matplotlib",
        "plotly",
        "sklearn",
        "joblib",
        "torch",
        "requests",
    ]
    for pkg in required_packages:
        mod = importlib.import_module(pkg)
        assert mod is not None, f"Failed to import {pkg}"


def test_internal_packages():
    """Verify that project packages can be imported."""
    for pkg in ["chess", "ml", "dl", "training", "tests"]:
        mod = importlib.import_module(pkg)
        assert mod is not None, f"Failed to import project package {pkg}"


def test_app_compilation():
    """Verify that app.py compiles without syntax or name errors at module level."""
    import py_compile
    res = py_compile.compile("app.py", doraise=True)
    assert res is not None

