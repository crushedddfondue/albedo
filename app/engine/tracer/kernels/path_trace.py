import taichi as ti
from taichi.math import vec3, normalize
from app.engine.tracer.sampling.hemisphere import sample_hemisphere

from app.engine.tracer.geometry.sphere import intersect_sphere
from app.engine.tracer.geometry.plane import plane_intersection

MAX_BOUNCE = 10

@ti.func
def next_event_estimation(surface_pos, surface_normal, surface_albedo):
  light_pos = vec3(0.0, 4.0, -3.0)
  light_normal = ti.math.normalize(vec3(0.0, -1.0, 1.0))
  light_emission = vec3(15.0)
  light_area = 2.0

  d = light_pos - surface_pos
  r_squared = ti.math.dot(d, d)
  shadow_ray_dir = ti.math.normalize(d)

  cos_theta_x = ti.math.max(0.0, ti.math.dot(surface_normal, shadow_ray_dir))
  cos_theta_y = ti.math.max(0.0, ti.math.dot(light_normal, -shadow_ray_dir))

  direct_light = vec3(0.0)

  if cos_theta_x > 0.0 and cos_theta_y > 0.0:
    f_r = surface_albedo / ti.math.pi
    visibility = 1.0
    
    # Lift the ray slightly off the surface to avoid self-intersection
    shadow_ray_o = surface_pos + surface_normal * 0.001
    light_distance = ti.math.sqrt(r_squared)
    
    # STRICTLY define the sphere properties
    sphere_center = vec3(0.0, 0.0, 0.0)
    sphere_radius = 1.0 
    
    # Explicitly check for shadows
    hit_sph, t_sph, shadow_norm = intersect_sphere(shadow_ray_o, shadow_ray_dir, sphere_center, sphere_radius)
    
    if hit_sph and t_sph < light_distance:
      visibility = 0.0

    direct_light = visibility * f_r * light_emission * (cos_theta_x * cos_theta_y * light_area) / r_squared

  return direct_light

@ti.func
def path_trace(ray_o, ray_d):
  radiance = vec3(0.0)
  throughput = vec3(1.0)

  for bounce in range(MAX_BOUNCE):
    hit = False
    t = 1e10  
    normal = vec3(0.0)
    albedo = vec3(0.0)
    emission = vec3(0.0)

    # 1. Sphere Intersection
    sphere_center = vec3(0.0, 0.0, 0.0)
    sphere_radius = 1.0
    hit_sph, t_sph, normal_sph = intersect_sphere(ray_o, ray_d, sphere_center, sphere_radius)

    if hit_sph and t_sph < t:
      hit = True
      t = t_sph
      normal = normal_sph
      albedo = vec3(0.8, 0.3, 0.3) 
      emission = vec3(0.0)

    # 2. Plane Intersection
    plane_pos = vec3(0.0, -1.0, 0.0) 
    plane_norm = vec3(0.0, 1.0, 0.0) 
    hit_pln, t_pln, norm_pln = plane_intersection(ray_o, ray_d, plane_pos, plane_norm)

    if hit_pln and t_pln < t:
      hit = True
      t = t_pln
      normal = norm_pln
      albedo = vec3(0.8, 0.8, 0.8) 
      emission = vec3(0.0)

    # 3. Sky Miss
    if not hit:
      radiance += throughput * vec3(0.05, 0.05, 0.08)
      break

    # 4. Surface Shading
    pos = ray_o + ray_d * t

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