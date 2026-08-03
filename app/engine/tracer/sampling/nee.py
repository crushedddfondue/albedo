import taichi as ti
from taichi.math import vec3


@ti.func
def sample_emissive_triangle(v0, v1, v2, u1, u2):
  sqrt_u1 = ti.sqrt(u1)
  b0 = 1.0 - sqrt_u1
  b1 = u2 * sqrt_u1
  b2 = sqrt_u1 * (1.0 - u2)

  x_l = b0 * v0 + b1 * v1 + b2 * v2

  edge1 = v1 - v0
  edge2 = v2 - v0
  cross_prod = ti.math.cross(edge1, edge2)

  cross_len = cross_prod.norm()
  area = 0.5 * cross_len

  pdf_area = 0.0
  n_l = vec3(0.0)

  if area > 1e-8:
    pdf_area = 1.0 / area
    n_l = cross_prod / cross_len

  return x_l, n_l, pdf_area


@ti.func
def solid_angle_pdf(x: vec3, x_l: vec3, n_l: vec3, pdf_area: ti.f32, single_sided: ti.i32) -> ti.f32:  # type: ignore
  to_light = x_l - x
  dist_sq = ti.math.dot(to_light, to_light)

  p_light = 0.0
  if dist_sq > 1e-8 and pdf_area > 0.0:
    w_l = to_light / ti.sqrt(dist_sq)
    cos_theta_l = ti.math.dot(-w_l, n_l)
    if not single_sided:
      cos_theta_l = ti.abs(cos_theta_l)

    if cos_theta_l > 1e-6:
      p_light = pdf_area * dist_sq / cos_theta_l

  return p_light