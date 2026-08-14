from typing import Any

import taichi as ti
from taichi.math import vec2, vec3

from tracer.constants import BACKGROUND_DEPTH

width, height = 0, 0
frame_index = 0          # 0 means "no history yet"

colour: Any = None       
moments: Any = None      
depth: Any = None
normal: Any = None
object_id: Any = None
length: Any = None      


def init_history_fields(w: int, h: int):
  global width, height
  global colour, moments, depth, normal, object_id, length

  if colour is not None:
    if width != w or height != h:
      raise RuntimeError(
        f"History buffer shape mismatch. Already allocated {width}x{height}, "
        f"but requested {w}x{h}. Re-allocating would silently orphan "
        "compiled Taichi kernels."
      )
    return

  width = w
  height = h

  colour = ti.Vector.field(3, dtype=ti.f32, shape=(w, h))
  moments = ti.Vector.field(2, dtype=ti.f32, shape=(w, h))
  depth = ti.field(dtype=ti.f32, shape=(w, h))
  normal = ti.Vector.field(3, dtype=ti.f32, shape=(w, h))
  object_id = ti.field(dtype=ti.i32, shape=(w, h))
  length = ti.field(dtype=ti.i32, shape=(w, h))


@ti.kernel
def _store_kernel(normal_f: ti.template(), object_id_f: ti.template(), depth_f: ti.template(), denoised_f: ti.template(), moments_f: ti.template(), length_f: ti.template()):  # type: ignore
  for i, j in colour:
    normal[i, j] = normal_f[i, j]
    object_id[i, j] = object_id_f[i, j]
    depth[i, j] = depth_f[i, j]
    colour[i, j] = denoised_f[i, j]
    moments[i, j] = moments_f[i, j]
    length[i, j] = length_f[i, j]


def store(normal_f, object_id_f, depth_f, denoised_f, moments_f, length_f):
  """Copy this frame's state into history, at end of frame.

  ⚠ denoised_f is the FILTERED output, not the raw radiance. Storing the
  noisy input would mean the accumulation never converges -- you would be
  averaging noise into noise forever.

  No albedo: history holds demodulated colour, and you remultiply by the
  CURRENT frame's albedo after filtering.
  """
  global frame_index
  _store_kernel(normal_f, object_id_f, depth_f, denoised_f, moments_f, length_f)
  frame_index += 1


@ti.kernel
def _reset_kernel():
  for i, j in length:
    length[i, j] = 0
    depth[i, j] = BACKGROUND_DEPTH
    object_id[i, j] = -1
    normal[i, j] = vec3(0.0)
    moments[i, j] = vec2(0.0)
    colour[i, j] = vec3(0.0)


def reset():
  """Invalidate all history -- camera teleport, scene change, resolution change.

  Zeroing `length` is what actually matters: reprojection reads it to set
  alpha = max(1/n, alpha_min), so n = 0 makes the next frame take its sample
  outright rather than blending with a stale buffer. The sentinels are
  belt-and-braces for anything that skips the length check.
  """
  global frame_index
  frame_index = 0
  if length is not None:
    _reset_kernel()