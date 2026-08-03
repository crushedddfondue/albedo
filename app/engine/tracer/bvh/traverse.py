import taichi as ti
from taichi.math import vec3, normalize

SHADOW_RAY_EPSILON = 1e-4

@ti.func
def ray_triangle_intersection(ray_o: vec3, ray_d: vec3, v0: vec3, v1: vec3, v2: vec3, t_min: ti.f32, t_max: ti.f32):  # type: ignore
  edge1 = v1-v0
  edge2 = v2-v0

  p_vec = ti.math.cross(ray_d, edge2)
  det = ti.math.dot(edge1, p_vec)

  hit = False
  t = 0.0

  if ti.abs(det) > 1e-8:
    inv_det = 1.0 / det
    t_vec = ray_o - v0

    u = ti.math.dot(t_vec, p_vec) * inv_det
    if u >= 0.0 and u <= 1.0:
      q_vec = ti.math.cross(t_vec, edge1)
      v = ti.math.dot(ray_d, q_vec) * inv_det
      if v >= 0.0 and v <= 1.0:
        candidate_t = ti.math.dot(edge2, q_vec) * inv_det
        if candidate_t > t_min and candidate_t < t_max:
          hit = True
          t = candidate_t

  return hit, t
@ti.func
def compute_visibility(x: vec3, n: vec3, x_l: vec3, dist_to_light: ti.f32, triangles: ti.template(), num_triangles: ti.i32)-> ti.f32:  # type: ignore
  ray_o = x + n * SHADOW_RAY_EPSILON
  ray_d = normalize(x_l-x)

  t_max = dist_to_light - SHADOW_RAY_EPSILON

  visibility = 1.0
  for i in range(num_triangles):
    v0 = triangles[i, 0]
    v1 = triangles[i, 1]
    v2 = triangles[i, 2]

    hit, _ = ray_triangle_intersection(ray_o, ray_d, v0, v1, v2, SHADOW_RAY_EPSILON, t_max)
    if hit:
      visibility = 0.0
      break


  return visibility
