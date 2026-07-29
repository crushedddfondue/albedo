from __future__ import annotations

import taichi as ti
import math

from typing import Any

@ti.func
def build_orthonormal_basis(normal: Any):
  sign = 1.0 if normal.z >= 0.0 else -1.0
  a = -1.0 / (sign + normal.z)
  b = normal.x * normal.y * a

  tangent = ti.math.vec3(1.0 + sign + normal.x * normal.x * a, sign * b, -sign * normal.x)
  bitangent = ti.math.vec3(b, sign + normal.y * normal.y * a, -normal.y)

  return tangent, bitangent, normal

@ti.func
def cosine_sample_hemisphere(normal):
  r1 = ti.random(ti.f32)
  r2 = ti.random(ti.f32)

  phi = 2.0 * math.pi*r1
  r = ti.math.sqrt(r2)

  local_x = math.cos(phi) * r
  local_y = math.sin(phi) * r
  local_z = math.sqrt(1.0 - r2)

  t, b, n = build_orthonormal_basis(normal)

  world_dir = local_x * t + local_y * b + local_z * n
  return ti.math.normalize(world_dir)