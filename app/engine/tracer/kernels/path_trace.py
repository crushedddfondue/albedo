from __future__ import annotations

import math
import taichi as ti
from taichi.math import vec2, vec3
from taichi import random

from engine.tracer.sampling.hemisphere import cosine_sample_hemisphere

EPSILON = 1e-4
MAX_BOUNCES = 4
RUSSIAN_ROULETTE_START = 2

@ti.func
def evaluate_lambertian_brdf(albedo):
  return albedo / math.pi

@ti.func
def sample_next_event_estimation(hit_p, hit_normal, hit_albedo, light_pos, light_colour, light_intensity, light_normal, light_area, scene):
  direct_radiance = vec3(0.0)

  u_rand = random(ti.f32) - 0.5
  v_rand = random(ti.f32) - 0.5
  sampled_light_p = light_pos + vec3(u_rand, 0.0, v_rand)

  light_dir = sampled_light_p - hit_p
  dist_sq = ti.math.dot(light_dir, light_dir)
  dist = ti.math.sqrt(dist_sq)
  light_dir = light_dir / dist

  cos_theta = ti.math.dot(hit_normal, light_dir)
  cos_theta_light = ti.math.dot(light_normal, -light_dir)

  if cos_theta > 0.0 and cos_theta_light > 0.0:
    shadow_ray_origin = hit_p + hit_normal * EPSILON
    shadow_hit = scene.intersect(shadow_ray_origin, light_dir, dist - EPSILON)

    if not shadow_hit.is_hit:
      brdf = evaluate_lambertian_brdf(hit_albedo)
      pdf = (1.0/light_area) * (dist_sq / cos_theta_light)
      emiteed_radiance = light_colour * light_intensity

      direct_radiance = (brdf * emiteed_radiance * cos_theta ) / pdf

  return direct_radiance

@ti.func
def path_trace_pixel(ray_origin, ray_dir, camera, scene, light_data):
  ray_o = ray_origin
  ray_d = ray_dir

  throughput = vec3(1.0)
  accumulated_radiance = vec3(0.0)

  gbuffer_normal = vec3(0.0)
  gbuffer_albedo = vec3(0.0)
  gbuffer_depth = 0.0
  first_hit_captured = False

  for bounce in range(MAX_BOUNCES):
    hit = scene.intersect(ray_o, ray_d, 1e-20)

    if not hit.is_hit:
      if bounce == 0:
        gbuffer_depth = 1e-20

      accumulated_radiance += throughput * vec3(0.05, 0.05, 0.08)
      break

    if not first_hit_captured:
      gbuffer_normal = hit.normal
      gbuffer_albedo = hit.albedo
      gbuffer_depth = hit.t
      first_hit_captured = True

    if hit.is_emissive:
      if bounce == 0:
        accumulated_radiance += hit.emissive_colour * hit.emissive_intensity

      break

    direct_light = sample_next_event_estimation(
      hit.p, hit.normal, hit.albedo,
      light_data.pos, light_data.colour, light_data.intensity,
      light_data.normal, light_data.area, scene
    )
    accumulated_radiance += throughput * direct_light

    next_d = cosine_sample_hemisphere(hit.normal)
    next_o = hit.p + hit.normal * EPSILON

    throughput *= hit.albedo

    ray_o = next_o
    ray_d = next_d

    if bounce >= RUSSIAN_ROULETTE_START:
      p_continue = ti.max(throughput.x, ti.max(throughput.y, throughput.z))
      p_continue = ti.min(p_continue, 0.95)

      if ti.random(ti.f32) > p_continue:
        break

      throughput /= p_continue

  return accumulated_radiance, gbuffer_normal, gbuffer_albedo, gbuffer_depth

@ti.func
def render_path_traced_frame(
  output_gbuffer: ti.template(), # type: ignore
  camera: ti.template(), # type: ignore
  scene: ti.template(),  # type: ignore
  light_data: ti.template() # type: ignore
):
  """
    Main Taichi GPU entry kernel to compute a 10-channel G-buffer frame at 1 SPP.
    
    Output G-Buffer layout (10 contiguous float channels):
      Channels 0..2 : Radiance (R, G, B)
      Channels 3..5 : Surface Normal (X, Y, Z)
      Channels 6..8 : Albedo / Base Color (R, G, B)
      Channel  9    : Linear Depth
  """

  for x, y in output_gbuffer:
    ray_o, ray_d = camera.generate_ray(x, y)

    radiance, normal, albedo, depth = path_trace_pixel(
      ray_o, ray_d, camera, scene, light_data
    )

    output_gbuffer[x, y, 0] = radiance.x
    output_gbuffer[x, y, 1] = radiance.y
    output_gbuffer[x, y, 2] = radiance.z

    output_gbuffer[x, y, 3] = normal.x
    output_gbuffer[x, y, 4] = normal.y
    output_gbuffer[x, y, 5] = normal.z

    output_gbuffer[x, y, 6] = albedo.x
    output_gbuffer[x, y, 7] = albedo.y
    output_gbuffer[x, y, 8] = albedo.z

    output_gbuffer[x, y, 9] = depth
