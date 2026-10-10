"""Compatibility entry point for the root dashboard."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[1] / 'dashboard.py'), run_name='__main__')
