import taichi as ti
from taichi.math import vec2, vec3

from tracer.camera import Camera
from tracer.constants import EMISSION_FACING_EPSILON

mat4 = ti.types.matrix(4, 4, ti.f32)

@ti.kernel
def motion_kernel(motion_f: ti.template(), depth_f: ti.template(), hit_mask_f: ti.template(), right: vec3, up: vec3, forward: vec3, position: vec3, fov: ti.f32, aspect_ratio: ti.f32, prev_view: mat4, prev_proj: mat4): # type: ignore
  width = motion_f.shape[0]
  height = motion_f.shape[1]

  jitter_x, jitter_y = 0.5, 0.5

  for px, py in motion_f:
    motion_f[px, py] = vec2(0.0)

    if hit_mask_f[px, py] == 1:
      d = Camera.ray_direction_for_pixel(
        px, py, jitter_x, jitter_y, width, height, right, up, forward, fov, aspect_ratio
      )

      t = depth_f[px, py] / ti.math.dot(d, forward)
      pos = position + t*d

      clip = prev_proj @ prev_view @ ti.Vector([pos.x, pos.y, pos.z, 1.0])

      if clip.w > EMISSION_FACING_EPSILON:  # w > 1e-6 check
        ndc_x = clip.x / clip.w
        ndc_y = clip.y / clip.w

        prev_x = (ndc_x + 1.0) * 0.5 * width
        prev_y = (ndc_y + 1.0) * 0.5 * height

        motion_f[px, py] = vec2(prev_x - (px + 0.5), prev_y - (py + 0.5))



      

