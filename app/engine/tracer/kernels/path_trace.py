import taichi as ti
from taichi.math import vec3, normalize

from tracer.geometry.primitives import Ray, Triangle, HitRecord, ray_triangle_intersect
from tracer.sampling.rng import rotate_to_world_space
from tracer.sampling.nee import weighted_nee_sample, solid_angle_pdf

RAY_OFFSET_EPSILON = 1e-4


@ti.func
def build_tangent_space(n: vec3):  # type: ignore
  up = vec3(1.0, 0.0, 0.0)
  if ti.abs(n.x) > 0.999:
    up = vec3(0.0, 1.0, 0.0)

  tangent = normalize(ti.math.cross(up, n))
  bitangent = ti.math.cross(n, tangent)

  return tangent, bitangent


@ti.func
def intersect_scene(ray: Ray, triangles: ti.template(), num_triangles: ti.i32) -> HitRecord:  # type: ignore
  closest = HitRecord(hit=0, t=0.0, position=vec3(0.0), normal=vec3(0.0), albedo=vec3(0.0), light_index=-1)
  closest_t = 1e30

  for i in range(num_triangles):
    candidate = ray_triangle_intersect(ray, triangles[i], RAY_OFFSET_EPSILON, closest_t)
    if candidate.hit == 1:
      closest = candidate
      closest_t = candidate.t

  return closest


@ti.func
def path_trace(ray: Ray, brdf: ti.template(), triangles: ti.template(), num_triangles: ti.i32, light_triangle: Triangle, light_area_pdf: ti.f32, single_sided: ti.i32, max_bounces: ti.i32):  # type: ignore
  radiance = vec3(0.0)
  throughput = vec3(1.0)

  prev_p_bsdf = 1.0
  prev_x = ray.origin

  current_ray = ray

  for bounce in range(max_bounces):
    hit = intersect_scene(current_ray, triangles, num_triangles)

    if hit.hit == 0:
      break

    if hit.light_index >= 0:
      if bounce == 0:
        radiance += throughput * hit.emission
      else:
        p_light_here = solid_angle_pdf(prev_x, hit.position, hit.normal, light_area_pdf, single_sided)

        w_brdf = 0.0
        denom = prev_p_bsdf * prev_p_bsdf + p_light_here * p_light_here
        if denom > 0.0:
          w_brdf = (prev_p_bsdf * prev_p_bsdf) / denom

        radiance += throughput * w_brdf * hit.emission

    u1, u2 = ti.random(ti.f32), ti.random(ti.f32)
    l_dir = weighted_nee_sample(hit.position, hit.normal, hit.albedo, light_triangle, u1, u2, triangles, num_triangles, single_sided, brdf)
    radiance += throughput * l_dir

    tangent, bitangent = build_tangent_space(hit.normal)
    bounce_dir, pdf_brdf = rotate_to_world_space(tangent, bitangent, hit.normal)
    cos_theta = ti.math.dot(hit.normal, bounce_dir)

    throughput *= brdf.sample_cosine_weighted_bounce(hit.albedo, cos_theta, vec3(1.0))

    # Russian roulette goes here once task 10 exists

    prev_x = hit.position
    prev_p_bsdf = pdf_brdf

    current_ray = Ray(origin=hit.position + hit.normal * RAY_OFFSET_EPSILON, direction=bounce_dir)

  return radiance