import taichi as ti
from taichi.math import vec2, vec3

# B3 spline, [1,4,6,4,1]/16 outer product.
KERNEL = (1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0)

SIGMA_Z = 1.0      # depth tolerance, in units of the expected depth change
SIGMA_N = 128.0    # normal exponent -- large makes it nearly a step function
SIGMA_L = 4.0      # luminance tolerance, in standard deviations

ATROUS_LEVELS = 5
EPS = 1e-3

@ti.func
def _luminance(c: vec3) -> ti.f32:  # type: ignore
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

@ti.func
def _depth_gradient(depth: ti.template(), object_id: ti.template(), i: ti.i32, j: ti.i32, width: ti.i32, height: ti.i32) -> vec2:  # type: ignore
  o = object_id[i, j]
  z = depth[i, j]

  gx = 0.0
  if i + 1 < width and object_id[i + 1, j] == o:
    gx = depth[i + 1, j] - z
  elif i - 1 >= 0 and object_id[i - 1, j] == o:
    gx = z - depth[i - 1, j]

  gy = 0.0
  if j + 1 < height and object_id[i, j + 1] == o:
    gy = depth[i, j + 1] - z
  elif j - 1 >= 0 and object_id[i, j - 1] == o:
    gy = z - depth[i, j - 1]

  return vec2(gx, gy)


@ti.kernel
def _copy_colour(src: ti.template(), dst: ti.template()):  # type: ignore
  for i, j in src:
    dst[i, j] = src[i, j]


@ti.kernel
def prefilter_variance(variance_in: ti.template(), object_id: ti.template(), variance_out: ti.template(), width: ti.i32, height: ti.i32):  # type: ignore

  for i, j in variance_out:
    o = object_id[i, j]
    sum_v = 0.0
    sum_w = 0.0

    for dy in ti.static(range(-1, 2)):
      for dx in ti.static(range(-1, 2)):
        qi = i + dx
        qj = j + dy

        if qi >= 0 and qi < width and qj >= 0 and qj < height:
          if object_id[qi, qj] == o:
            w = 1.0
            if dx != 0:
              w *= 0.5
            if dy != 0:
              w *= 0.5
            sum_v += w * variance_in[qi, qj]
            sum_w += w

    variance_out[i, j] = sum_v / ti.max(sum_w, 1e-6)


@ti.kernel
def atrous_level(step: ti.i32, colour_in: ti.template(), variance_in: ti.template(), var_prefiltered: ti.template(), depth: ti.template(), normal: ti.template(), object_id: ti.template(), colour_out: ti.template(), variance_out: ti.template(), width: ti.i32, height: ti.i32):  # type: ignore
  """One a-trous iteration at the given stride."""
  for i, j in colour_out:
    o_p = object_id[i, j]

    # Legal here -- the pixel loop is a runtime loop, unlike the tap loops.
    if o_p == -1:
      colour_out[i, j] = colour_in[i, j]
      variance_out[i, j] = variance_in[i, j]
      continue

    c_p = colour_in[i, j]
    l_p = _luminance(c_p)
    z_p = depth[i, j]
    n_p = normal[i, j]

    l_scale = SIGMA_L * ti.sqrt(var_prefiltered[i, j]) + EPS

    grad = _depth_gradient(depth, object_id, i, j, width, height)

    sum_c = vec3(0.0)
    sum_v = 0.0
    sum_w = 0.0

    for dy in ti.static(range(-2, 3)):
      for dx in ti.static(range(-2, 3)):
        qi = i + dx * step
        qj = j + dy * step

        if qi >= 0 and qi < width and qj >= 0 and qj < height:
          if object_id[qi, qj] == o_p:
            h = KERNEL[dx + 2] * KERNEL[dy + 2]

            # DEPTH: tolerance scaled by the depth change EXPECTED over this
            # offset, from the gradient. A slanted floor changes depth
            # rapidly across the screen and is still one continuous surface;
            # a fixed tolerance would refuse to blend along it. This is why
            # depth is linear view-space z -- a plane must have constant
            # gradient.
            expected_dz = ti.abs(grad[0] * dx * step + grad[1] * dy * step)
            w_z = ti.exp(-ti.abs(z_p - depth[qi, qj]) / (SIGMA_Z * expected_dz + EPS))

            # NORMAL: large exponent makes this nearly a step function.
            w_n = ti.pow(ti.max(ti.math.dot(n_p, normal[qi, qj]), 0.0), SIGMA_N)

            # LUMINANCE: the adaptive term, and why variance is needed.
            l_q = _luminance(colour_in[qi, qj])
            w_l = ti.exp(-ti.abs(l_p - l_q) / l_scale)

            w = h * w_z * w_n * w_l

            sum_c += w * colour_in[qi, qj]
            sum_v += w * w * variance_in[qi, qj]
            sum_w += w

    if sum_w > 1e-6:
      colour_out[i, j] = sum_c / sum_w

      # ⚠ VARIANCE DOES NOT FILTER LIKE COLOUR. The effective weights are
      # a_k = w_k / sum_w, summing to 1, and for independent variables
      #
      #     Var(sum a_k x_k) = sum a_k^2 Var(x_k)
      #
      # so the numerator carries w^2 and the denominator is sum_w SQUARED.
      # Using w throughout makes the variance decay too slowly, the
      # luminance term stays loose for too many levels, and the filter
      # over-blurs -- a uniform, subtle loss of detail, very hard to
      # attribute afterwards.
      variance_out[i, j] = sum_v / (sum_w * sum_w)
    else:
      colour_out[i, j] = c_p
      variance_out[i, j] = variance_in[i, j]


def filter_image(colour_a, colour_b, variance_a, variance_b, var_prefiltered, history_out, depth, normal, object_id, width, height, levels=ATROUS_LEVELS):
  src_c, dst_c = colour_a, colour_b
  src_v, dst_v = variance_a, variance_b

  for level in range(levels):
    prefilter_variance(src_v, object_id, var_prefiltered, width, height)
    atrous_level(
      1 << level, src_c, src_v, var_prefiltered,
      depth, normal, object_id, dst_c, dst_v, width, height,
    )
    src_c, dst_c = dst_c, src_c
    src_v, dst_v = dst_v, src_v

    if level == 0:
      _copy_colour(src_c, history_out)

  return src_c, src_v