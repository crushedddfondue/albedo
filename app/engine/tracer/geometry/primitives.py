import taichi as ti
from taichi.math import vec3

EPSILON = 1e-4


Ray = ti.types.struct(origin=vec3, direction=vec3)

Triangle = ti.types.struct(
  v0=vec3, v1=vec3, v2=vec3,
  normal=vec3, albedo=vec3, light_index=ti.i32,
  emission=vec3
)

HitRecord = ti.types.struct(
  hit=ti.i32, t=ti.f32,
  position=vec3, normal=vec3, albedo=vec3, light_index=ti.i32,
)


@ti.func
def compute_triangle_normal(triangle: Triangle) -> vec3:  # type: ignore
  edge1 = triangle.v1 - triangle.v0
  edge2 = triangle.v2 - triangle.v0
  return ti.math.normalize(ti.math.cross(edge1, edge2))


@ti.func
def ray_triangle_intersect(ray: Ray, triangle: Triangle, t_min: ti.f32, t_max: ti.f32) -> HitRecord:  # type: ignore
  edge1 = triangle.v1 - triangle.v0
  edge2 = triangle.v2 - triangle.v0

  p_vec = ti.math.cross(ray.direction, edge2)
  det = ti.math.dot(edge1, p_vec)

  record = HitRecord(hit=0, t=0.0, position=vec3(0.0), normal=vec3(0.0), albedo=vec3(0.0), light_index=-1)

  if ti.abs(det) > EPSILON:
    inv_det = 1.0 / det
    t_vec = ray.origin - triangle.v0

    u = ti.math.dot(t_vec, p_vec) * inv_det
    if u >= 0.0 and u <= 1.0:
      q_vec = ti.math.cross(t_vec, edge1)
      v = ti.math.dot(ray.direction, q_vec) * inv_det

      if v >= 0.0 and (u + v) <= 1.0:
        t = ti.math.dot(edge2, q_vec) * inv_det

        if t > t_min and t < t_max:
          record.hit = 1
          record.t = t
          record.position = ray.origin + t * ray.direction
          record.normal = triangle.normal
          record.albedo = triangle.albedo
          record.light_index = triangle.light_index

  return record