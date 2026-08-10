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
    normal=vec3(0.0), albedo=vec3(0.8), emission=vec3(0.0), light_index=-1, object_id=0
  )
  scene.triangles[1] = scene.Triangle(
    v0=vec3(-5.0, 0.0, -5.0), v1=vec3(-5.0, 0.0, 5.0), v2=vec3(5.0, 0.0, 5.0),
    normal=vec3(0.0), albedo=vec3(0.8), emission=vec3(0.0), light_index=-1, object_id=0
  )

  # Light quad (2 triangles), hovering at y=4, emission = 10.0 white.
  # This winding already produced the correct downward normal, unchanged.
  scene.triangles[2] = scene.Triangle(
    v0=vec3(-1.0, 4.0, -1.0), v1=vec3(1.0, 4.0, -1.0), v2=vec3(1.0, 4.0, 1.0),
    normal=vec3(0.0), albedo=vec3(0.0), emission=vec3(10.0, 10.0, 10.0), light_index=-1,
    object_id = 1
  )
  scene.triangles[3] = scene.Triangle(
    v0=vec3(-1.0, 4.0, -1.0), v1=vec3(1.0, 4.0, 1.0), v2=vec3(-1.0, 4.0, 1.0),
    normal=vec3(0.0), albedo=vec3(0.0), emission=vec3(10.0, 10.0, 10.0), light_index=-1,
    object_id=1
  )

  # Occluder (2 triangles), y=2, albedo 0.8, non-emissive. Same winding as
  # the floor, so the geometric normal points up. Takes the scene to 6
  # triangles against LEAF_SIZE = 4, which is what makes build() recurse and
  # produce an actual tree rather than a single leaf node.
  scene.triangles[4] = scene.Triangle(
    v0=vec3(-0.75, 2.0, -0.75), v1=vec3(0.75, 2.0, 0.75), v2=vec3(0.75, 2.0, -0.75),
    normal=vec3(0.0), albedo=vec3(0.8), emission=vec3(0.0), light_index=-1, object_id=2
  )
  scene.triangles[5] = scene.Triangle(
    v0=vec3(-0.75, 2.0, -0.75), v1=vec3(-0.75, 2.0, 0.75), v2=vec3(0.75, 2.0, 0.75),
    normal=vec3(0.0), albedo=vec3(0.8), emission=vec3(0.0), light_index=-1, object_id=2
  )

  scene.num_triangles[None] = 6


@ti.kernel
def _make_furnace():
  """Overwrite material properties for the white furnace test.

  Geometry is unchanged -- only albedo and emission move. Unit albedo makes
  the analytic answer exactly the environment radiance L, and removing all
  emission leaves num_lights = 0 so NEE is skipped and the environment is the
  only light source in the scene, which is the entire point of the test.

  Reads num_triangles rather than hardcoding 6 so it stays correct if the
  test room grows.
  """
  for i in range(scene.num_triangles[None]):  # type: ignore
    scene.triangles[i].albedo = vec3(1.0)  # type: ignore
    scene.triangles[i].emission = vec3(0.0)  # type: ignore
    scene.triangles[i].light_index = -1  # type: ignore


def build_test_room():
  _write_test_room()
  scene.recompute_normals()
  scene.build_light_list()
  build_bvh()
  upload_to_taichi()


def build_furnace_scene():
  _write_test_room()
  _make_furnace()
  scene.recompute_normals()
  scene.build_light_list()
  build_bvh()
  upload_to_taichi()