import os
import sys

import pytest
import taichi as ti

# pytest puts tests/ on sys.path, not app/engine/, so `tracer` never resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session", autouse=True)
def ti_runtime():
  """Initialise Taichi exactly once for the whole session.

  ti.init() RESETS the runtime and invalidates every field allocated before
  it. If two test modules each call it, whichever ran first has its fields
  silently pulled out from under it. One session-scoped init is the only
  arrangement that survives multiple test files.

  NOTE: remove any ti.init() call from tests/test_brdf.py -- it will fight
  with this one.
  """
  ti.init(arch=ti.cuda)
  yield