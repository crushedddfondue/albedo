from typing import Any

import taichi as ti
from taichi.math import vec3, vec2

from tracer.constants import BACKGROUND_DEPTH

# Allocated by init_aov_fields(), which must run after ti.init().
#
# Consumers: use `from tracer import buffers` and reference buffers.albedo.
# A `from tracer.buffers import albedo` snapshots None at import time and
# never sees the reassignment below.
width = 0
height = 0

albedo: Any = None
normal: Any = None
object_id: Any = None
hit_mask: Any = None
depth: Any = None
motion: Any = None


def init_aov_fields(w: int, h: int):
  """Allocate the AOV set. Idempotent.

  ⚠ Re-allocation cannot be allowed silently. Any @ti.kernel reading these
  module globals is permanently bound to whatever they pointed at when that
  kernel first compiled. Rebinding the names afterwards is invisible to the
  kernel and visible to Python, and the two views then disagree forever.

  Allocate once, return early if already done, raise if someone asks for a
  different shape rather than silently handing back the old one.
  """
  global width, height
  global albedo, normal, object_id, hit_mask, depth, motion

  if albedo is not None:
    if (width, height) != (w, h):
      raise RuntimeError(
        f"AOV fields already allocated at {width}x{height}; cannot reallocate "
        f"to {w}x{h}. Kernels compiled against the old fields would keep "
        f"writing to them."
      )
    return

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
    # Huge sentinel, not zero. hit_mask is the authority, but bilinear
    # history fetches in 2.2 will straddle silhouette edges and pick up
    # background taps. Zero reads as "very close" and could wrongly pass a
    # depth rejection test; a huge value reads as "infinitely far" and fails
    # it. Fail-safe direction.
    depth_f[i, j] = BACKGROUND_DEPTH
    motion_f[i, j] = vec2(0.0)


def clear_aovs():
  """Python-scope wrapper. Resolves the module globals at call time and
  passes them as ti.template() arguments, so the kernel is never bound to a
  stale field. Call sites stay one word long."""
  _clear_kernel(albedo, normal, object_id, hit_mask, depth, motion)