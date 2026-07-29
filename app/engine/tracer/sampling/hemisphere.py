import math
import torch
import taichi as ti
from taichi.math import vec2, vec3

ti.init(arch=ti.cuda)

@ti.func
def cosine_weighted_local_coordinates():
  r1 = ti.random(ti.f32)
  r2 = ti.random(ti.f32)

  phi = 2.0 * math.pi * r1

  x = ti.math.cos(phi) * ti.math.sqrt(r2)
  y = ti.math.sin(phi) * ti.math.sqrt(r2)
  z = ti.math.sqrt(1.0 - r2)

  return x, y, z

@ti.func
def tbn_orthonormal_basis(normal):
  s = 1.0 if normal.z >= 0 else -1.0

  a = -(1.0 / (s + normal.z))
  b = normal.x * normal.y * a

  tangent = vec3(1.0 + s * (normal.x * normal.x * a), s * b, -s * normal.x)
  bitangent = vec3(b, s + (normal.y * normal.y * a), -normal.y)

  return tangent, bitangent

@ti.func
def world_space_transformation(normal):
  x, y, z = cosine_weighted_local_coordinates()
  tangent, bitangent = tbn_orthonormal_basis(normal)

  d_world = x * tangent + y * bitangent + z * normal
  return ti.math.normalize(d_world)

@ti.func
def sample_hemisphere(normal):
  world_dir = world_space_transformation(normal)
  return world_dir