import taichi as ti
from taichi.math import vec3, normalize
from app.engine.tracer.sampling.hemisphere import sample_hemisphere

# NEW: Import your intersection math
from app.engine.tracer.geometry.sphere import intersect_sphere

# Note: In a larger app, it's best to only call ti.init() once inside your main.py!
ti.init(arch=ti.cuda)

# Setting a limit to the maximum number of bounces (for light)
MAX_BOUNCE = 10

# Function to estimate the next actions of light after interaction with an object
@ti.func
def next_event_estimation(surface_pos, surface_normal, surface_albedo):
  # x = surface_pos; y = light_pos
  light_pos = vec3(0.0, 4.0, -3.0)
  light_normal = ti.math.normalize(vec3(0.0, -1.0, 1.0))
  # Denoted as L_e
  light_emission = vec3(15.0)
  # Denoted as A
  light_area = 2.0

  # d = y - x
  d = light_pos - surface_pos

  # r^2 = || d^2 ||
  r_squared = ti.math.dot(d, d)

  # w_l = d / || d ||
  shadow_ray_dir = normalize(d)

  """
  here cos(θ_x) = max(0, n_x * w_l)
  and cos(θ_y) = max(0, n_y * -w_l)
  """
  # FIX: Used 0.0 instead of 0 to avoid Taichi type casting errors
  cos_theta_x = ti.math.max(0.0, ti.math.dot(surface_normal, shadow_ray_dir))
  cos_theta_y = ti.math.max(0.0, ti.math.dot(light_normal, -shadow_ray_dir))

  # L_direct
  direct_light = vec3(0.0)

  # FIX: Only calculate if the surfaces are actually facing each other
  if cos_theta_x > 0.0 and cos_theta_y > 0.0:
    # f_r = ρ / π
    f_r = surface_albedo / ti.math.pi

    # denoted as V
    visibility = 1.0

    """
    Crux: Direct Light Estimator
    L_direct = V * f_r * L_e * (cos(θ_x) * cos(θ_y) * A) / r^2
    """
    direct_light = visibility * f_r * light_emission * (cos_theta_x * cos_theta_y * light_area) / r_squared

  return direct_light

# aggregates the noisy path tracer outputs into a converged target image X_1 for the AI model.
@ti.func
def path_trace(ray_o, ray_d):
  radiance = vec3(0.0)
  throughput = vec3(1.0)

  for bounce in range(MAX_BOUNCE):
    # --- REAL SCENE INTERSECTION ---
    sphere_center = vec3(0.0, 0.0, 0.0)
    sphere_radius = 1.0
    
    hit, t, normal = intersect_sphere(ray_o, ray_d, sphere_center, sphere_radius)

    # If ray escapes into the sky, add sky colour and terminate
    if not hit:
      radiance += throughput * vec3(0.05, 0.05, 0.08) # Slightly bluish dark sky
      break

    # If it hits, calculate the exact physical point in space
    pos = ray_o + ray_d * t
    albedo = vec3(0.8, 0.3, 0.3)
    emission = vec3(0.0)
    # -------------------------------

    # Add Emission only for the first bounce (Camera looking directly at the light)
    if bounce == 0:
      radiance += throughput * emission

    """
    3. Direct Lighting (L_direct) --> (Next-Event Estimation)
    Explicitly calculates shadow ray contribution
    """
    direct_light = next_event_estimation(pos, normal, albedo)
    radiance += throughput * direct_light

    """
    4. Indirect Lighting --> (Hemisphere Bounce)
    Update throughput using the albedo (cosine and PDF cancel out)
    """
    throughput = throughput * albedo

    # Generating the next ray
    ray_d = sample_hemisphere(normal)
    ray_o = pos + normal * 0.001

    # Russian Roulette (Path Termination)
    continuation_prob = ti.math.max(albedo.x, ti.math.max(albedo.y, albedo.z))
    r = ti.random(ti.f32)

    if r > continuation_prob:
      break

    throughput /= continuation_prob

  return radiance