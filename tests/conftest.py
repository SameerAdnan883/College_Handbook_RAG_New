"""
tests/conftest.py — Shared fixtures and path setup for pytest
"""
import sys
import os

# Ensure the project root is on the Python path for all tests
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
