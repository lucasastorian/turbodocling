"""Shared pytest configuration and fixtures."""

import sys
import os

# Ensure project roots are importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "shared"))


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: pure unit tests (no IO, no GPU)")
    config.addinivalue_line("markers", "contract: wire-format contract tests")
    config.addinivalue_line("markers", "integration: end-to-end pipeline tests")
    config.addinivalue_line("markers", "golden: golden file regression tests")
    config.addinivalue_line("markers", "perf: performance regression tests")
