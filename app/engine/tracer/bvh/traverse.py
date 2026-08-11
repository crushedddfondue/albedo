import taichi as ti
from taichi.math import vec3, normalize

from tracer.constants import RAY_ORIGIN_EPSILON, RAY_T_MIN, SHADOW_ENDPOINT_MARGIN
from tracer.geometry.primitives import Ray, HitRecord, ray_triangle_intersect
from tracer.bvh.aabb import AABB, ray_aabb_intersection


@ti.func
def compute_visibility(x: vec3, n: vec3, x_l: vec3, dist_to_light: ti.f32, bvh_node_min: ti.template(), bvh_node_max: ti.template(), bvh_node_left: ti.template(), bvh_node_right: ti.template(), bvh_node_start: ti.template(), bvh_node_count: ti.template(), bvh_indices: ti.template(), triangles: ti.template()) -> ti.f32:  # type: ignore
  ray_o = x + n * RAY_ORIGIN_EPSILON
  ray_d = normalize(x_l - x)

  # Relative margin. See constants.py -- the absolute form leaves a vanishing
  # gap for surface points directly beneath the light, where n . w_l -> 1.
  t_max = dist_to_light * (1.0 - SHADOW_ENDPOINT_MARGIN)

  occluded = traverse_any_hit(
    ray_o, ray_d, t_max,
    bvh_node_min, bvh_node_max, bvh_node_left, bvh_node_right,
    bvh_node_start, bvh_node_count, bvh_indices, triangles
  )

  visibility = 1.0
  if occluded == 1:
    visibility = 0.0

  return visibility


@ti.func
def traverse_closest_hit(ray_o: vec3, ray_d: vec3, bvh_node_min: ti.template(), bvh_node_max: ti.template(), bvh_node_left: ti.template(), bvh_node_right: ti.template(), bvh_node_start: ti.template(), bvh_node_count: ti.template(), bvh_indices: ti.template(), triangles: ti.template()) -> HitRecord:  # type: ignore
  inv_dir = 1.0 / ray_d
  t_min = RAY_T_MIN
  closest_t = 1e9

  ray = Ray(origin=ray_o, direction=ray_d)
  closest = HitRecord(hit=0, t=0.0, position=vec3(0.0), normal=vec3(0.0), albedo=vec3(0.0), light_index=-1, emission=vec3(0.0), object_id=-1)

  stack = ti.Vector.zero(ti.i32, 64)
  stack_ptr = 0
  stack[stack_ptr] = 0  # push root

  while stack_ptr >= 0:
    node_idx = stack[stack_ptr]
    stack_ptr -= 1

    box = AABB(b_min=bvh_node_min[node_idx], b_max=bvh_node_max[node_idx])

    if not ray_aabb_intersection(box, ray_o, inv_dir, t_min, closest_t):
      continue

    count = bvh_node_count[node_idx]
    if count > 0:
      start = bvh_node_start[node_idx]
      for i in range(start, start + count):
        tri_idx = bvh_indices[i]
        candidate = ray_triangle_intersect(ray, triangles[tri_idx], t_min, closest_t)
        if candidate.hit == 1:
          closest = candidate
          closest_t = candidate.t
    else:
      # FOLLOW-UP: children are pushed in fixed order, so descent is
      # independent of ray direction and closest_t pruning is far weaker than
      # it could be. Pushing the far child first (LIFO, so the near child pops
      # first) requires storing the split axis per node -- builder.py already
      # computes it and discards it. Not a refactor change; it alters
      # traversal order and therefore floating-point accumulation order.
      stack_ptr += 1
      stack[stack_ptr] = bvh_node_right[node_idx]

      stack_ptr += 1
      stack[stack_ptr] = bvh_node_left[node_idx]

  return closest


@ti.func
def traverse_any_hit(ray_o: vec3, ray_d: vec3, t_max: ti.f32, bvh_node_min: ti.template(), bvh_node_max: ti.template(), bvh_node_left: ti.template(), bvh_node_right: ti.template(), bvh_node_start: ti.template(), bvh_node_count: ti.template(), bvh_indices: ti.template(), triangles: ti.template()) -> ti.i32:  # type: ignore
  inv_dir = 1.0 / ray_d
  t_min = RAY_T_MIN
  ray = Ray(origin=ray_o, direction=ray_d)

  occluded = 0

  stack = ti.Vector.zero(ti.i32, 64)
  stack_ptr = 0
  stack[stack_ptr] = 0

  while stack_ptr >= 0 and occluded == 0:
    node_idx = stack[stack_ptr]
    stack_ptr -= 1

    box = AABB(b_min=bvh_node_min[node_idx], b_max=bvh_node_max[node_idx])
    if not ray_aabb_intersection(box, ray_o, inv_dir, t_min, t_max):
      continue

    count = bvh_node_count[node_idx]
    if count > 0:
      start = bvh_node_start[node_idx]
      for i in range(start, start + count):
        tri_idx = bvh_indices[i]
        candidate = ray_triangle_intersect(ray, triangles[tri_idx], t_min, t_max)
        if candidate.hit == 1:
          occluded = 1
          break
    else:
      # No ordering needed here: any-hit returns on the first hit and has no
      # notion of nearest, so descent order cannot affect the result.
      stack_ptr += 1
      stack[stack_ptr] = bvh_node_right[node_idx]
      stack_ptr += 1
      stack[stack_ptr] = bvh_node_left[node_idx]

  return occluded