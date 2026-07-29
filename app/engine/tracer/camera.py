from __future__ import annotations

import taichi as ti
from taichi.math import vec3, normalize, cross

import math

@ti.dataclass
class CameraState:
  pos: vec3 # type: ignore
  forward: vec3 # type: ignore
  right: vec3 # type: ignore
  up: vec3  # type: ignore
  fov_scale: ti.f32 # type: ignore
  aspect_ratio: ti.f32# type: ignore

def create_camera(pos, yaw, pitch, fov_degrees, aspect_ratio):
  yaw_rad = math.radians(yaw)
  pitch_rad = math.radians(pitch)

  forward_x = math.cos(yaw_rad) * math.cos(pitch_rad)
  forward_y = math.sin(pitch_rad)
  forward_z = math.sin(yaw_rad) * math.cos(pitch_rad)

  forward = normalize(vec3(forward_x, forward_y, forward_z))
  world_up = vec3(0.0, 1.0, 0.0)

  right = normalize(vec3(cross(forward, world_up)))
  up = cross(right, forward)

  fov_scale = math.tan(math.radians(fov_degrees * 0.5))

  return CameraState(
    pos = vec3(pos),
    forward=forward,
    right=right,
    up=up,
    fov_scale=fov_scale,
    aspect_ratio=aspect_ratio
  )

@ti.func
def generate_ray(camera: CameraState, u: ti.f32, v: ti.f32):  # type: ignore
  ndc_x = (2.0 * u - 1.0) * camera.aspect_ratio * camera.aspect_ratio
  ndc_y = (2.0 * v - 1.0) * camera.fov_scale

  ray_dir = normalize(
    ndc_x * camera.right + ndc_y * camera.up + camera.forward
  )

  return camera.pos, ray_dir
