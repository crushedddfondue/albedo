import os
import sys

import pytest
import numpy as np
import taichi as ti
from taichi.math import vec3

from tracer.camera import Camera

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

@ti.kernel
def _fill_ray_dirs(d: ti.template(), r: vec3, u: vec3, fw: vec3, fov: ti.f32, ar: ti.f32):  # type: ignore
  for px, py in d:
    d[px, py] = Camera.ray_direction_for_pixel(px, py, 0.5, 0.5, d.shape[0], d.shape[1], r, u, fw, fov, ar)


def ray_dirs(cam, w, h):
  """Per-pixel primary ray directions at pixel centre, (w, h, 3) float64.

  Calls the same ray_direction_for_pixel that gbuffer_kernel uses, so any
  test built on these directions also checks that the shared function is
  genuinely shared.
  """
  f = ti.Vector.field(3, ti.f32, shape=(w, h))
  right, up, forward = cam.basis_from_yaw_pitch()
  _fill_ray_dirs(f, right, up, forward, cam.fov, cam.aspect_ratio)
  return f.to_numpy().astype(np.float64)