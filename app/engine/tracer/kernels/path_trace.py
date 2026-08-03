# path_trace.py
import taichi as ti
from taichi.math import vec3, normalize

from tracer.geometry.primitives import Ray, Triangle, HitRecord, ray_triangle_intersect
from tracer.sampling.rng import rotate_to_world_space
from tracer.sampling.nee import weighted_nee_sample, solid_angle_pdf, pick_light
from tracer.bvh.traverse import traverse_closest_hit

RAY_OFFSET_EPSILON = 1e-4
RR_MIN_BOUNCES = 3
RR_Q_MIN = 0.05
RR_Q_MAX = 0.95


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
def path_trace(ray: Ray, brdf: ti.template(), triangles: ti.template(), num_triangles: ti.i32, bvh_node_min: ti.template(), bvh_node_max: ti.template(), bvh_node_left: ti.template(), bvh_node_right: ti.template(), bvh_node_start: ti.template(), bvh_node_count: ti.template(), bvh_indices: ti.template(), light_triangle_index: ti.template(), light_pdf_area: ti.template(), num_lights: ti.i32, single_sided: ti.i32, max_bounces: ti.i32,):  # type: ignore
  radiance = vec3(0.0)
  throughput = vec3(1.0)

  prev_p_bsdf = 1.0
  prev_x = ray.origin

  current_ray = ray

  for bounce in range(max_bounces):
    hit = traverse_closest_hit(
      current_ray.origin, current_ray.direction,
      bvh_node_min, bvh_node_max, bvh_node_left, bvh_node_right,
      bvh_node_start, bvh_node_count, bvh_indices, triangles,
    )

    if hit.hit == 0:
      break

    if hit.light_index >= 0:
      if bounce == 0:
        radiance += throughput * hit.emission
      else:
        hit_pdf_area = light_pdf_area[hit.light_index] / num_lights
        p_light_here = solid_angle_pdf(prev_x, hit.position, hit.normal, hit_pdf_area, single_sided)

        w_brdf = 0.0
        denom = prev_p_bsdf * prev_p_bsdf + p_light_here * p_light_here
        if denom > 0.0:
          w_brdf = (prev_p_bsdf * prev_p_bsdf) / denom

        radiance += throughput * w_brdf * hit.emission

    if num_lights > 0:
      u1, u2, u3 = ti.random(ti.f32), ti.random(ti.f32), ti.random(ti.f32)
      light_idx = pick_light(u3, num_lights)
      light_tri = triangles[light_triangle_index[light_idx]]
      pdf_area = light_pdf_area[light_idx] / num_lights

      l_dir = weighted_nee_sample(
        hit.position, hit.normal, hit.albedo, light_tri, pdf_area, u1, u2,
        bvh_node_min, bvh_node_max, bvh_node_left, bvh_node_right,
        bvh_node_start, bvh_node_count, bvh_indices,
        triangles, single_sided, brdf,
      )
      radiance += throughput * l_dir

    tangent, bitangent = build_tangent_space(hit.normal)
    bounce_dir, pdf_brdf = rotate_to_world_space(tangent, bitangent, hit.normal)
    cos_theta = ti.math.dot(hit.normal, bounce_dir)

    throughput *= brdf.sample_cosine_weighted_bounce(hit.albedo, cos_theta, vec3(1.0))

    if bounce >= RR_MIN_BOUNCES:
      gamma = ti.random(ti.f32)
      q = ti.math.clamp(ti.max(throughput.x, ti.max(throughput.y, throughput.z)), RR_Q_MIN, RR_Q_MAX)

      if gamma > q:
        break

      throughput /= q

    prev_x = hit.position
    prev_p_bsdf = pdf_brdf

    current_ray = Ray(origin=hit.position + hit.normal * RAY_OFFSET_EPSILON, direction=bounce_dir)

  return radiance