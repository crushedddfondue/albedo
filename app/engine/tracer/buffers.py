from typing import Any

import taichi as ti
from taichi.math import vec3, vec2

width = 0
height = 0

albedo: Any = None
normal: Any = None
object_id: Any = None
hit_mask: Any = None
depth: Any = None
motion: Any = None


def init_aov_fields(w: int, h: int):
  global width, height
  global albedo, normal, object_id, hit_mask, depth, motion

  width = w
  height = h
  shape = (w, h)

  albedo = ti.Vector.field(3, ti.f32, shape)
  normal = ti.Vector.field(3, ti.f32, shape)
  object_id = ti.field(ti.i32, shape)
  hit_mask = ti.field(ti.i32, shape)
  depth = ti.field(ti.f32, shape)
  motion = ti.Vector.field(2, ti.f32, shape)


@ti.kernel
def _clear_kernel(albedo_f: ti.template(), normal_f: ti.template(), object_id_f: ti.template(), hit_mask_f: ti.template(), depth_f: ti.template(), motion_f: ti.template()):  # type: ignore
  # One kernel, one launch. Six separate .fill() calls from Python would be
  # six launches for the same work.
  for i, j in albedo_f:
    albedo_f[i, j] = vec3(0.0)
    normal_f[i, j] = vec3(0.0)
    object_id_f[i, j] = -1
    hit_mask_f[i, j] = 0
    # NOTE: 0.0 is a placeholder sentinel. Decide alongside the depth
    # representation -- zero is a plausible depth in some conventions, so
    # either use a large sentinel or make hit_mask the sole authority and
    # attach no meaning to background depth.
    depth_f[i, j] = 0.0
    motion_f[i, j] = vec2(0.0)


def clear_aovs():
  """Python-scope wrapper. Reads the module globals at call time, so a
  reallocation (window resize) is picked up automatically -- Taichi
  recompiles the kernel when the ti.template() arguments change identity.
  Call sites stay one word long."""
  _clear_kernel(albedo, normal, object_id, hit_mask, depth, motion)