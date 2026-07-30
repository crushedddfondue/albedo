import taichi as ti
from taichi.math import vec3, normalize

from sampling.hemisphere import sample_hemisphere

MAX_BOUNCE = 10


@ti.func
def next_event_estimation(surface_pos, surface_normal, surface_albedo):
  light_pos = vec3(0.0, 4.0, 0.0)
  light_normal = vec3(0.0, -1.0, 0.0)
  light_emission = vec3(10.0)
  light_area = 2.0

  d = light_pos - surface_pos

  r_squared = ti.math.dot(d, d)

  shadow_ray_dir = normalize(d)

  cos_theta_x = ti.math.max(0, ti.math.dot(surface_normal, shadow_ray_dir))
  cos_theta_y = ti.math.max(0, ti.math.dot(light_normal, -shadow_ray_dir))

  direct_light = vec3(0.0)

  if cos_theta_x > 0.0 and cos_theta_y > 0.0:
    f_r = surface_albedo / ti.math.pi

    visibility = 1.0

    direct_light = visibility * f_r * light_emission * (cos_theta_x * cos_theta_y * light_area) / r_squared

  return direct_light


@ti.func
def path_trace(ray_o, ray_d):
  radiance = vec3(0.0)
  throughput = vec3(1.0)

  for bounce in range(MAX_BOUNCE):
    hit = True
    pos = ray_o + ray_d * 2.0
    normal = vec3(0.0, 1.0, 0.0)
    albedo = vec3(0.8, 0.3, 0.3)
    emission = vec3(0.0)

    if not hit:
      radiance += throughput * vec3(0.05)
      break

    if bounce == 0:
      radiance += throughput * emission

    direct_light = next_event_estimation(pos, normal, albedo)
    radiance += throughput * direct_light

    throughput = throughput * albedo

    ray_d = sample_hemisphere(normal)
    ray_o = pos + normal * 0.001

    continuation_prob = ti.math.max(albedo.x, ti.math.max(albedo.y, albedo.z))
    r = ti.random(ti.f32)

    if r > continuation_prob:
      break

    throughput /= continuation_prob

  return radiance

