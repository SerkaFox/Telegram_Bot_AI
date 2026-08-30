import sys
from pathlib import Path

# Ensure the project root (with generator/ and fox_worker/) is importable when pytest runs.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
