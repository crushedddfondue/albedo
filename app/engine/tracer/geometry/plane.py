import taichi as ti
from taichi.math import vec3

@ti.func
def plane_intersection(ray_o, ray_d, plane_pos, plane_normal):
  hit = False
  t = 0.0
  normal = vec3(0.0)

  denominator = ti.math.dot(ray_d, plane_normal)

  # Check absolute value to see if ray is perfectly parallel to the plane
  if ti.abs(denominator) > 1e-6:
    p0_minus_p = plane_pos - ray_o
    numerator = ti.math.dot(p0_minus_p, plane_normal)

    # Safe to divide because we are inside the if-block!
    t_hit = numerator / denominator

    if t_hit > 0.001:
      hit = True
      t = t_hit
      normal = plane_normal if denominator < 0 else -plane_normal

  return hit, t, normal