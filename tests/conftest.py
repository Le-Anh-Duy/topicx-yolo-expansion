import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]   # `topicx` (harness) và `src.rav` (pipeline proposal, cùng đường import với P-026)
