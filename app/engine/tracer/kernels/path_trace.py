# path_trace.py
import taichi as ti
from taichi.math import vec3, normalize

from tracer.constants import RAY_ORIGIN_EPSILON, EMISSION_FACING_EPSILON
from tracer.geometry.primitives import Ray
from tracer.sampling.rng import rotate_to_world_space
from tracer.sampling.nee import weighted_nee_sample, solid_angle_pdf, pick_light
from tracer.bvh.traverse import traverse_closest_hit

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
def path_trace(ray: Ray, brdf: ti.template(), environment: ti.template(), triangles: ti.template(), num_triangles: ti.i32, bvh_node_min: ti.template(), bvh_node_max: ti.template(), bvh_node_left: ti.template(), bvh_node_right: ti.template(), bvh_node_start: ti.template(), bvh_node_count: ti.template(), bvh_indices: ti.template(), light_triangle_index: ti.template(), light_pdf_area: ti.template(), num_lights: ti.i32, single_sided: ti.i32, use_nee: ti.i32, max_bounces: ti.i32,):  # type: ignore
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
      # A ray that hits nothing is not carrying zero radiance -- it is
      # carrying whatever the environment emits in that direction. The
      # environment has direction but no position, so unlike an area light
      # there is no distance falloff and no area-to-solid-angle conversion.
      # With Environment(mode=ENV_BLACK) this reduces exactly to the previous
      # bare `break`.
      radiance += throughput * environment.sample(current_ray.direction)
      break

    # ---- Emission. Uses hit.normal, the TRUE geometric normal, deliberately:
    # the single-sided test below has to know which physical face of the
    # emitter is being seen, and the viewer-facing normal computed further
    # down would make that test pass unconditionally.
    if hit.light_index >= 0:
      # Single-sidedness has to be enforced here as well as inside
      # solid_angle_pdf. If only the pdf side honours it, the two sampling
      # strategies disagree about whether the light exists at all when seen
      # from behind: NEE reports p_light = 0, while a BSDF ray landing on the
      # back face gets w_brdf = p_b^2 / (p_b^2 + 0) = 1 and adds the full
      # emission. MIS cannot reconcile that -- the result is biased upward,
      # not merely noisy. Tested against current_ray, not ray: on bounce > 0
      # they differ, and it is the incoming direction at *this* hit that
      # decides which face is being seen.
      facing = ti.math.dot(-current_ray.direction, hit.normal)
      emitting = 1
      if single_sided != 0 and facing <= EMISSION_FACING_EPSILON:
        emitting = 0

      if emitting == 1:
        if bounce == 0:
          radiance += throughput * hit.emission
        else:
          # Declared before the branch, not inside it: Taichi requires the
          # name to already exist in the enclosing scope before a conditional
          # assigns to it. 1.0 is also the correct BSDF-only weight, not a
          # placeholder -- with NEE disabled there is no second strategy to
          # share this estimate with, so the BSDF path carries all of it.
          w_brdf = 1.0

          if use_nee != 0:
            hit_pdf_area = light_pdf_area[hit.light_index] / num_lights
            p_light_here = solid_angle_pdf(prev_x, hit.position, hit.normal, hit_pdf_area, single_sided)

            w_brdf = 0.0
            denom = prev_p_bsdf * prev_p_bsdf + p_light_here * p_light_here
            if denom > 0.0:
              w_brdf = (prev_p_bsdf * prev_p_bsdf) / denom

          radiance += throughput * w_brdf * hit.emission

    # ---- Two-sided shading normal. Everything below this line shades with
    # shading_normal, never hit.normal.
    #
    # The scene is bare quads, not closed meshes, so rays legitimately land on
    # back faces, and compute_triangle_normal stores exactly one geometric
    # normal per triangle. Without the flip, a point on the underside of the
    # occluder is shaded as though it faced the light: cos_theta = dot(n, w_l)
    # comes out positive, and compute_visibility offsets the shadow ray origin
    # by +n -- straight through the quad onto the lit side -- so the shadow
    # ray reports no occlusion. The underside renders as a blown-out slab and
    # then feeds that radiance back down into its own umbra on the next
    # bounce, washing out the shadow it is supposed to be casting. One flip
    # fixes both symptoms.
    shading_normal = hit.normal
    if ti.math.dot(-current_ray.direction, hit.normal) < 0.0:
      shading_normal = -hit.normal

    if use_nee != 0 and num_lights > 0:
      u1, u2, u3 = ti.random(ti.f32), ti.random(ti.f32), ti.random(ti.f32)
      light_idx = pick_light(u3, num_lights)
      light_tri = triangles[light_triangle_index[light_idx]]
      pdf_area = light_pdf_area[light_idx] / num_lights

      l_dir = weighted_nee_sample(
        hit.position, shading_normal, hit.albedo, light_tri, pdf_area, u1, u2,
        bvh_node_min, bvh_node_max, bvh_node_left, bvh_node_right,
        bvh_node_start, bvh_node_count, bvh_indices,
        triangles, single_sided, brdf,
      )
      radiance += throughput * l_dir

    tangent, bitangent = build_tangent_space(shading_normal)
    bounce_dir, pdf_brdf = rotate_to_world_space(tangent, bitangent, shading_normal)
    cos_theta = ti.math.dot(shading_normal, bounce_dir)

    # f_r * cos / pdf = (rho/pi) * cos / (cos/pi) = rho. The cosine and both
    # pi terms cancel exactly, which is why this is a multiply and not a
    # divide -- and why throughput is literally the product of albedos along
    # the path, and must stay inside [0,1]^3 before Russian roulette.
    throughput *= brdf.sample_cosine_weighted_bounce(hit.albedo, cos_theta, vec3(1.0))

    if bounce >= RR_MIN_BOUNCES:
      gamma = ti.random(ti.f32)
      q = ti.math.clamp(ti.max(throughput.x, ti.max(throughput.y, throughput.z)), RR_Q_MIN, RR_Q_MAX)

      if gamma > q:
        break

      throughput /= q

    prev_x = hit.position
    prev_p_bsdf = pdf_brdf

    current_ray = Ray(origin=hit.position + shading_normal * RAY_ORIGIN_EPSILON, direction=bounce_dir)

  return radiance