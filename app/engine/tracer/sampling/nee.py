import taichi as ti
from taichi.math import vec3

from tracer.bvh.traverse import compute_visibility
from tracer.geometry.primitives import Triangle


@ti.func
def sample_emissive_triangle(triangle: Triangle, u1: ti.f32, u2: ti.f32) -> vec3:  # type: ignore
  sqrt_u1 = ti.sqrt(u1)
  b0 = 1.0 - sqrt_u1
  b1 = u2 * sqrt_u1
  b2 = sqrt_u1 * (1.0 - u2)

  return b0 * triangle.v0 + b1 * triangle.v1 + b2 * triangle.v2


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
def compute_direct_lighting(x: vec3, n: vec3, albedo: vec3, triangle: Triangle, pdf_area: ti.f32, u1: ti.f32, u2: ti.f32, bvh_node_min: ti.template(), bvh_node_max: ti.template(), bvh_node_left: ti.template(), bvh_node_right: ti.template(), bvh_node_start: ti.template(), bvh_node_count: ti.template(), bvh_indices: ti.template(), triangles: ti.template(), single_sided: ti.i32, brdf: ti.template()):  # type: ignore
  l_dir = vec3(0.0)
  w_l = vec3(0.0)
  p_light = 0.0
  x_l = sample_emissive_triangle(triangle, u1, u2)
  if pdf_area > 0.0:
    to_light = x_l - x
    dist_to_light = to_light.norm()
    if dist_to_light > 1e-8:
      w_l = to_light / dist_to_light
      cos_theta = ti.math.dot(n, w_l)
      if cos_theta > 0.0:
        p_light = solid_angle_pdf(x, x_l, triangle.normal, pdf_area, single_sided)
        if p_light > 0.0:
          V = compute_visibility(
            x, n, x_l, dist_to_light,
            bvh_node_min, bvh_node_max, bvh_node_left, bvh_node_right,
            bvh_node_start, bvh_node_count, bvh_indices, triangles,
          )
          if V > 0.0:
            f_r = brdf.evaluate(albedo)
            l_dir = (f_r * triangle.emission * cos_theta * V) / p_light
  return l_dir, w_l, p_light


@ti.func
def weighted_nee_sample(x: vec3, n: vec3, albedo: vec3, triangle: Triangle, pdf_area: ti.f32, u1: ti.f32, u2: ti.f32, bvh_node_min: ti.template(), bvh_node_max: ti.template(), bvh_node_left: ti.template(), bvh_node_right: ti.template(), bvh_node_start: ti.template(), bvh_node_count: ti.template(), bvh_indices: ti.template(), triangles: ti.template(), single_sided: ti.i32, brdf: ti.template()):  # type: ignore
  l_dir, w_l, p_light = compute_direct_lighting(
    x, n, albedo, triangle, pdf_area, u1, u2,
    bvh_node_min, bvh_node_max, bvh_node_left, bvh_node_right,
    bvh_node_start, bvh_node_count, bvh_indices, triangles,
    single_sided, brdf,
  )
  final_l_dir = vec3(0.0)
  if p_light > 0.0:
    cos_theta = ti.math.dot(n, w_l)
    p_bsdf = brdf.pdf(cos_theta)
    p_light_sq = p_light * p_light
    p_bsdf_sq = p_bsdf * p_bsdf
    w_light = p_light_sq / (p_light_sq + p_bsdf_sq)
    final_l_dir = l_dir * w_light
  return final_l_dir


@ti.func
def pick_light(u3: ti.f32, num_lights: ti.i32) -> ti.i32:  # type: ignore
  light_idx = ti.cast(ti.floor(u3 * num_lights), ti.i32)
  light_idx = ti.min(light_idx, num_lights - 1)
  return light_idx