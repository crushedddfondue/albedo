"""Temporal accumulation with history rejection.

Three of the rejection conditions are the ones from test_motion.py --
geometry present, lookup in bounds, same object. This adds what that test did
not need: BILINEAR fetch, a depth tolerance, and a normal test.
"""

import taichi as ti
from taichi.math import vec2, vec3


@ti.func
def _bilinear_weights(fx: ti.f32, fy: ti.f32):  # type: ignore
  """Four weights for fractional offsets in [0,1), summing to 1.

  Ordering must match the tap indexing below. Getting it wrong shifts the
  image by a pixel in a way that reads as a motion vector bug.
  """
  w00 = (1.0 - fx) * (1.0 - fy)
  w10 = fx * (1.0 - fy)
  w01 = (1.0 - fx) * fy
  w11 = fx * fy
  return w00, w10, w01, w11


@ti.func
def _tap_is_valid(tx: ti.i32, ty: ti.i32, curr_depth: ti.f32, curr_object: ti.i32, curr_normal: vec3, hist_depth: ti.template(), hist_object: ti.template(), hist_normal: ti.template(), width: ti.i32, height: ti.i32) -> ti.i32:  # type: ignore
  """Is this individual tap usable history?"""
  valid = 1

  if tx < 0 or tx >= width or ty < 0 or ty >= height:
    valid = 0
  else:
    if hist_object[tx, ty] != curr_object:
      valid = 0
    else:
      # RELATIVE depth tolerance. A 0.05 world-unit disagreement is a
      # different surface at depth 3 and the same surface at depth 300.
      # SVGF derives this from the depth gradient; 10% relative is the
      # simplified version.
      rel = ti.abs(hist_depth[tx, ty] - curr_depth) / (curr_depth + 1e-5)
      if rel > 0.1:
        valid = 0
      else:
        # Catches what object_id cannot: same object, different face --
        # the occluder's top versus its underside. 0.9 is about 25 degrees.
        if ti.math.dot(hist_normal[tx, ty], curr_normal) < 0.9:
          valid = 0

  return valid


@ti.kernel
def reproject_kernel(curr_colour: ti.template(), curr_depth: ti.template(), curr_normal: ti.template(), curr_object_id: ti.template(), motion: ti.template(), hist_colour: ti.template(), hist_moments: ti.template(), hist_depth: ti.template(), hist_normal: ti.template(), hist_object_id: ti.template(), hist_length: ti.template(), out_colour: ti.template(), out_moments: ti.template(), out_length: ti.template(), width: ti.i32, height: ti.i32, alpha_min: ti.f32):  # type: ignore
  """Fetch history at pixel + motion, reject what is invalid, blend.

  curr_colour is DEMODULATED radiance -- divided by albedo. Remultiply after
  filtering, using the current frame's albedo.
  """
  max_len = ti.cast(1.0 / alpha_min, ti.i32)

  for i, j in curr_colour:
    c_col = curr_colour[i, j]
    c_dep = curr_depth[i, j]
    c_nor = curr_normal[i, j]
    c_obj = curr_object_id[i, j]
    c_lum = 0.2126 * c_col[0] + 0.7152 * c_col[1] + 0.0722 * c_col[2]

    if c_obj == -1:
      # Background is the environment -- deterministic, no noise to average.
      out_colour[i, j] = c_col
      out_moments[i, j] = vec2(c_lum, c_lum * c_lum)
      out_length[i, j] = 1
      continue

    mv = motion[i, j]
    rx = ti.cast(i, ti.f32) + mv[0]
    ry = ti.cast(j, ti.f32) + mv[1]

    tx = ti.cast(ti.floor(rx), ti.i32)
    ty = ti.cast(ti.floor(ry), ti.i32)
    fx = rx - ti.cast(tx, ti.f32)
    fy = ry - ti.cast(ty, ti.f32)

    w00, w10, w01, w11 = _bilinear_weights(fx, fy)

    # Validity from the UNCLAMPED coordinates.
    v00 = _tap_is_valid(tx, ty, c_dep, c_obj, c_nor, hist_depth, hist_object_id, hist_normal, width, height)
    v10 = _tap_is_valid(tx + 1, ty, c_dep, c_obj, c_nor, hist_depth, hist_object_id, hist_normal, width, height)
    v01 = _tap_is_valid(tx, ty + 1, c_dep, c_obj, c_nor, hist_depth, hist_object_id, hist_normal, width, height)
    v11 = _tap_is_valid(tx + 1, ty + 1, c_dep, c_obj, c_nor, hist_depth, hist_object_id, hist_normal, width, height)

    # ⚠ Reads must be CLAMPED. A validity flag gates the contribution, not
    # the memory access -- and an out-of-bounds read returning NaN makes
    # 0 * NaN = NaN, poisoning the pixel despite the weight being zero.
    cx0 = ti.math.clamp(tx, 0, width - 1)
    cy0 = ti.math.clamp(ty, 0, height - 1)
    cx1 = ti.math.clamp(tx + 1, 0, width - 1)
    cy1 = ti.math.clamp(ty + 1, 0, height - 1)

    a00 = w00 * v00
    a10 = w10 * v10
    a01 = w01 * v01
    a11 = w11 * v11
    sum_w = a00 + a10 + a01 + a11

    if sum_w > 1e-4:
      inv_w = 1.0 / sum_w

      # Renormalising over the surviving taps is not optional -- skipping it
      # darkens every silhouette edge by exactly the rejected weight, which
      # is uniform, subtle, and easy to mistake for a filter artefact.
      h_col = (a00 * hist_colour[cx0, cy0] + a10 * hist_colour[cx1, cy0] +
               a01 * hist_colour[cx0, cy1] + a11 * hist_colour[cx1, cy1]) * inv_w

      h_mom = (a00 * hist_moments[cx0, cy0] + a10 * hist_moments[cx1, cy0] +
               a01 * hist_moments[cx0, cy1] + a11 * hist_moments[cx1, cy1]) * inv_w

      h_len = (a00 * ti.cast(hist_length[cx0, cy0], ti.f32) +
               a10 * ti.cast(hist_length[cx1, cy0], ti.f32) +
               a01 * ti.cast(hist_length[cx0, cy1], ti.f32) +
               a11 * ti.cast(hist_length[cx1, cy1], ti.f32)) * inv_w

      new_len = ti.min(ti.cast(ti.round(h_len), ti.i32) + 1, max_len)

      # alpha = max(1/n, alpha_min). The 1/n is a BIAS CORRECTION and is not
      # optional: with a fixed alpha the first frames are dominated by
      # whatever the buffer was initialised to. n = 1 gives alpha = 1, so the
      # first frame takes its sample outright.
      #
      # alpha_min caps how far back history reaches -- 0.05 is about a
      # 20-frame window. Larger means less noise and more ghosting. That one
      # number is the entire temporal quality tradeoff.
      alpha = ti.max(1.0 / ti.cast(new_len, ti.f32), alpha_min)

      out_colour[i, j] = ti.math.mix(h_col, c_col, alpha)
      out_moments[i, j] = ti.math.mix(h_mom, vec2(c_lum, c_lum * c_lum), alpha)
      out_length[i, j] = new_len
    else:
      out_colour[i, j] = c_col
      out_moments[i, j] = vec2(c_lum, c_lum * c_lum)
      out_length[i, j] = 1