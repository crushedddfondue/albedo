import taichi as ti
from taichi.math import vec2, vec3



MIN_HISTORY_FOR_TEMPORAL = 4

SPATIAL_RADIUS = 3

DEPTH_TOLERANCE = 0.1
NORMAL_TOLERANCE = 0.9

@ti.func
def _luminance(c: vec3)-> ti.f32: # type: ignore
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

@ti.func
def _variance_from_moments(m: vec2)-> ti.f32: # type: ignore
  v = m[1] - m[0] * m[0]

  return ti.max(v, 0.0)

@ti.kernel
def estimate_variance(moments: ti.template(), length: ti.template(), colour: ti.template(), depth: ti.template(), normal: ti.template(), object_id: ti.template(), out_variance: ti.template(), width: ti.i32, height: ti.i32):  # type: ignore
  for i, j in out_variance:
    if object_id[i, j] == -1:
      out_variance[i, j] = 0.0
      continue

    n = length[i, j]

    if n >= MIN_HISTORY_FOR_TEMPORAL:
      out_variance[i, j] = _variance_from_moments(moments[i, j])
    else:
      z_p = depth[i, j]
      n_p = normal[i, j]
      o_p = object_id[i, j]

      sum_w = 0.0
      sum_l = 0.0
      sum_l2 = 0.0

      for dy in range(-SPATIAL_RADIUS, SPATIAL_RADIUS + 1):
        for dx in range(-SPATIAL_RADIUS, SPATIAL_RADIUS + 1):
          qi = i + dx
          qj = j + dy
          if qi < 0 or qi >= width or qj < 0 or qj >= height:
            continue
          if object_id[qi, qj] != o_p:
            continue
          if ti.abs(depth[qi, qj] - z_p) / (z_p + 1e-5) > DEPTH_TOLERANCE:
            continue
          if ti.math.dot(normal[qi, qj], n_p) < NORMAL_TOLERANCE:
            continue

          l = _luminance(colour[qi, qj])
          sum_w += 1.0
          sum_l += l
          sum_l2 += l * l

      var = 0.0
      if sum_w > 1.0:
        mean = sum_l / sum_w
        var = ti.max(sum_l2 / sum_w - mean * mean, 0.0)

      out_variance[i, j] = var * (4.0 / ti.max(ti.cast(n, ti.f32), 1.0))