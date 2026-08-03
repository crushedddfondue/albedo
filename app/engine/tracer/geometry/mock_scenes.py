import taichi as ti
from taichi.math import vec3

from tracer.geometry import scene
from tracer.bvh.builder import build_bvh, upload_to_taichi

@ti.kernel
def _write_test_room():
  # Floor (2 triangles), albedo = 0.8 gray. Winding order swapped from the
  # original so cross(edge1, edge2) actually produces an upward normal --
  # normal is left at zero here regardless; scene.recompute_normals()
  # derives it right after this kernel runs, so it can never again
  # silently disagree with the real geometry.
  scene.triangles[0] = scene.Triangle(
    v0=vec3(-5.0, 0.0, -5.0), v1=vec3(5.0, 0.0, 5.0), v2=vec3(5.0, 0.0, -5.0),
    normal=vec3(0.0), albedo=vec3(0.8), emission=vec3(0.0), light_index=-1,
  )
  scene.triangles[1] = scene.Triangle(
    v0=vec3(-5.0, 0.0, -5.0), v1=vec3(-5.0, 0.0, 5.0), v2=vec3(5.0, 0.0, 5.0),
    normal=vec3(0.0), albedo=vec3(0.8), emission=vec3(0.0), light_index=-1,
  )

  # Light quad (2 triangles), hovering at y=4, emission = 10.0 white.
  # This winding already produced the correct downward normal, unchanged.
  scene.triangles[2] = scene.Triangle(
    v0=vec3(-1.0, 4.0, -1.0), v1=vec3(1.0, 4.0, -1.0), v2=vec3(1.0, 4.0, 1.0),
    normal=vec3(0.0), albedo=vec3(0.0), emission=vec3(10.0, 10.0, 10.0), light_index=-1,
  )
  scene.triangles[3] = scene.Triangle(
    v0=vec3(-1.0, 4.0, -1.0), v1=vec3(1.0, 4.0, 1.0), v2=vec3(-1.0, 4.0, 1.0),
    normal=vec3(0.0), albedo=vec3(0.0), emission=vec3(10.0, 10.0, 10.0), light_index=-1,
  )

  scene.num_triangles[None] = 4


def build_test_room():
  _write_test_room()
  scene.recompute_normals()
  scene.build_light_list()
  build_bvh()
  upload_to_taichi()