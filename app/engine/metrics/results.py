"""
Append-only results log.

JSONL rather than JSON: appendable without reading
"""

import json
import os
import platform
import subprocess
from datetime import datetime, timezone

RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "results.jsonl")


def _git_state():
  try:
    commit = subprocess.check_output(
      ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
    ).decode().strip()
    dirty = bool(subprocess.check_output(
      ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
    ).decode().strip())
    return {"commit": commit, "dirty": dirty}
  except Exception:
    return {"commit": None, "dirty": None}


def log_result(label, metrics, config, reference=None, hardware=None, notes=None):
  import taichi as ti

  os.makedirs(RESULTS_DIR, exist_ok=True)

  record = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "label": label,
    "metrics": metrics,
    "config": config,
    "reference": reference or {},
    "provenance": {
      **_git_state(),
      "taichi": ti.__version__,
      "python": platform.python_version(),
      "platform": platform.platform(),
      "hardware": hardware,
    },
    "notes": notes,
  }

  with open(RESULTS_FILE, "a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\n")

  return record


def load_results(path=RESULTS_FILE):
  if not os.path.exists(path):
    return []
  with open(path, encoding="utf-8") as f:
    return [json.loads(line) for line in f if line.strip()]