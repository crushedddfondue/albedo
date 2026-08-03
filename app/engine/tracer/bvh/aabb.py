import taichi as ti
from taichi.math import vec3

AABB = ti.types.struct(
  b_min=vec3,
  b_max=vec3
)


@ti.func
def aabb_union_point(box: AABB, p: vec3) -> AABB:  # type: ignore
  return AABB(
    b_min=ti.min(box.b_min, p),
    b_max=ti.max(box.b_max, p)
  )


@ti.func
def aabb_union_box(box1: AABB, box2: AABB) -> AABB:  # type: ignore
  return AABB(
    b_min=ti.min(box1.b_min, box2.b_min),
    b_max=ti.max(box1.b_max, box2.b_max)
  )


@ti.func
def ray_aabb_intersection(box: AABB, ray_o: vec3, inv_dir: vec3, t_min: ti.f32, t_max: ti.f32) -> bool:  # type: ignore
  t0 = (box.b_min - ray_o) * inv_dir
  t1 = (box.b_max - ray_o) * inv_dir

  t_near_vec = ti.min(t0, t1)
  t_far_vec = ti.max(t0, t1)

  t_near = ti.math.max(t_min, ti.math.max(t_near_vec.x, ti.math.max(t_near_vec.y, t_near_vec.z)))
  t_far = ti.math.min(t_max, ti.math.min(t_far_vec.x, ti.math.min(t_far_vec.y, t_far_vec.z)))

  return t_near <= t_far