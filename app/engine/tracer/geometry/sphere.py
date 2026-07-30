import taichi as ti
from taichi.math import vec3

@ti.func
def intersect_sphere(ray_o, ray_d, center, radius):
  hit = False
  t = 0.0
  normal = vec3(0.0)

  # Vector from sphere center to ray origin
  oc = ray_o - center
  
  # The quadratic coefficients
  h = ti.math.dot(ray_d, oc)
  c = ti.math.dot(oc, oc) - (radius * radius)

  discriminant = (h * h) - c

  # If discriminant is < 0, the ray completely missed
  if discriminant >= 0.0:
    sqrtd = ti.math.sqrt(discriminant)
    
    # Calculate the two possible hit distances
    t1 = -h - sqrtd
    t2 = -h + sqrtd

    # We want the closest hit (t1) that is strictly in front of the camera (> 0.001)
    t_nearest = t1
    if t_nearest < 0.001:
      t_nearest = t2 # If t1 is behind us/inside the sphere, try t2

    if t_nearest > 0.001:
      hit = True
      t = t_nearest
      
      # Now that we know t, we can finally calculate the hit position
      hit_pos = ray_o + t * ray_d
      
      # The normal is simply the hit position minus the center, divided by radius
      normal = (hit_pos - center) / radius

  return hit, t, normal