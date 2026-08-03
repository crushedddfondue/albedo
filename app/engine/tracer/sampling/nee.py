import taichi as ti
from taichi.math import vec3

from tracer.bvh.traverse import compute_visibility


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


@ti.func
def compute_direct_lighting(x: vec3, n: vec3, albedo: vec3, v0: vec3, v1: vec3, v2: vec3, l_e: vec3, u1: ti.f32, u2: ti.f32, triangles: ti.template(), num_triangles: ti.i32, single_sided: ti.i32, brdf: ti.template()):  # type: ignore
  l_dir = vec3(0.0)
  w_l = vec3(0.0)
  p_light = 0.0

  x_l, n_l, pdf_area = sample_emissive_triangle(v0, v1, v2, u1, u2)

  if pdf_area > 0.0:
    to_light = x_l - x
    dist_to_light = to_light.norm()

    if dist_to_light > 1e-8:
      w_l = to_light / dist_to_light

      cos_theta = ti.math.dot(n, w_l)
      if cos_theta > 0.0:
        p_light = solid_angle_pdf(x, x_l, n_l, pdf_area, single_sided)

        if p_light > 0.0:
          V = compute_visibility(x, n, x_l, dist_to_light, triangles, num_triangles)

          if V > 0.0:
            f_r = brdf.evaluate(albedo)
            l_dir = (f_r * l_e * cos_theta * V) / p_light

  return l_dir, w_l, p_light


@ti.func
def weighted_nee_sample(x: vec3, n: vec3, albedo: vec3, v0: vec3, v1: vec3, v2: vec3, l_e: vec3, u1: ti.f32, u2: ti.f32, triangles: ti.template(), num_triangles: ti.i32, single_sided: ti.i32, brdf: ti.template()):  # type: ignore
  l_dir, w_l, p_light = compute_direct_lighting(x, n, albedo, v0, v1, v2, l_e, u1, u2, triangles, num_triangles, single_sided, brdf)

  final_l_dir = vec3(0.0)

  if p_light > 0.0:
    cos_theta = ti.math.dot(n, w_l)
    p_bsdf = brdf.pdf(cos_theta)

    p_light_sq = p_light * p_light
    p_bsdf_sq = p_bsdf * p_bsdf

    w_light = p_light_sq / (p_light_sq + p_bsdf_sq)
    final_l_dir = l_dir * w_light

  return final_l_dir