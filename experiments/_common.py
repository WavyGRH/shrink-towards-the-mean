"""Shared paths. Importing this puts src/ on the path, so the repo works without installation."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data" / "school.mat"
RESULTS = ROOT / "results"
RAW = RESULTS / "raw"
REFERENCE = RESULTS / "reference"
FIGURES = ROOT / "figures"
TABLES = ROOT / "paper" / "tables"


def fresh_or_reference(fresh, name):
    """Use the output of a fresh run if there is one, otherwise the stored results behind the paper."""
    return fresh if fresh.exists() else REFERENCE / name
