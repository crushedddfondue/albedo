import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from typing import Any

import taichi as ti
from taichi.math import vec3

from tracer import buffers
from tracer.camera import Camera
from tracer.geometry import scene
from tracer.geometry.mock_scenes import build_test_room
from tracer.bvh import builder
from tracer.kernels.gbuffer import gbuffer_kernel

WIDTH, HEIGHT = 1280, 720

MODE_ALBEDO = 1
MODE_NORMAL = 2
MODE_OBJECT_ID = 3
MODE_HIT_MASK = 4
MODE_DEPTH = 5

MODE_NAMES = {
  MODE_ALBEDO: "Albedo (1)",
  MODE_NORMAL: "Normal (2)",
  MODE_OBJECT_ID: "Object ID (3)",
  MODE_HIT_MASK: "Hit Mask (4)",
  MODE_DEPTH: "Depth (5)",
}

display: Any = None


@ti.kernel
def render_mode_albedo():
  # Light quad renders BLACK here and that is correct -- emissive surfaces
  # have albedo vec3(0.0), no diffuse response. Grey would mean emission is
  # leaking into the albedo channel.
  for i, j in display:
    if buffers.hit_mask[i, j] == 1:
      display[i, j] = ti.pow(buffers.albedo[i, j], 1.0 / 2.2)
    else:
      display[i, j] = vec3(0.0)


@ti.kernel
def render_mode_normal():
  # Floor should be uniform green (+Y). The occluder's UNDERSIDE must be the
  # exact inverse -- magenta. If it reads green like the floor, the
  # viewer-facing flip is not being applied and gbuffer.py stored the raw
  # geometric normal.
  for i, j in display:
    if buffers.hit_mask[i, j] == 1:
      display[i, j] = 0.5 * (buffers.normal[i, j] + 1.0)
    else:
      display[i, j] = vec3(0.0)


@ti.kernel
def render_mode_object_id():
  # THE important check: the two floor triangles must be exactly the same
  # colour. Any seam along the diagonal from (-5,0,-5) to (5,0,5) means the
  # ID is per-triangle rather than per-object, and SVGF will draw a false
  # disocclusion line corner to corner across the floor.
  for i, j in display:
    obj_id = buffers.object_id[i, j]
    if obj_id == -1:
      display[i, j] = vec3(0.0)
    elif obj_id == 0:
      display[i, j] = vec3(0.8, 0.2, 0.2)
    elif obj_id == 1:
      display[i, j] = vec3(0.2, 0.8, 0.2)
    elif obj_id == 2:
      display[i, j] = vec3(0.2, 0.2, 0.8)
    else:
      display[i, j] = vec3(0.8, 0.8, 0.2)


@ti.kernel
def render_mode_hit_mask():
  for i, j in display:
    if buffers.hit_mask[i, j] == 1:
      display[i, j] = vec3(1.0)
    else:
      display[i, j] = vec3(0.0)


@ti.kernel
def render_mode_depth():
  # Stub until the depth representation is decided. Writes both branches so
  # switching to this mode does not leave the previous mode's pixels behind.
  for i, j in display:
    display[i, j] = vec3(0.0)


RENDERERS = {
  MODE_ALBEDO: render_mode_albedo,
  MODE_NORMAL: render_mode_normal,
  MODE_OBJECT_ID: render_mode_object_id,
  MODE_HIT_MASK: render_mode_hit_mask,
  MODE_DEPTH: render_mode_depth,
}


def main():
  global display

  ti.init(arch=ti.cuda)

  scene.init_scene_fields()
  build_test_room()

  buffers.init_aov_fields(WIDTH, HEIGHT)
  buffers.clear_aovs()

  display = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

  print(f"triangles: {scene.num_triangles[None]}  bvh nodes: {len(builder.nodes)}")

  # Same camera as main.py, so you can flip between the two and compare.
  camera = Camera(
    position=vec3(0.0, 1.2, 4.0),
    yaw=0.0,
    pitch=math.radians(5),
    fov=math.radians(60.0),
    aspect_ratio=WIDTH / HEIGHT,
    near=0.1,
    far=1000.0,
  )
  right, up, forward = camera.basis_from_yaw_pitch()

  # Static camera, so once is enough. If camera controls are added,
  # clear_aovs() then gbuffer_kernel() on every move, in that order -- the
  # miss branch deliberately does not write albedo/normal, so stale values
  # would survive in background pixels.
  gbuffer_kernel(
    buffers.albedo, buffers.normal, buffers.object_id, buffers.hit_mask,
    right, up, forward, camera.position,
    camera.fov, camera.aspect_ratio,
    scene.triangles,
    builder.bvh_node_min, builder.bvh_node_max,
    builder.bvh_node_left, builder.bvh_node_right,
    builder.bvh_node_start, builder.bvh_node_count, builder.bvh_indices,
  )

  gui = ti.GUI("AOV Viewer", res=(WIDTH, HEIGHT))  # type: ignore
  current_mode = MODE_ALBEDO
  RENDERERS[current_mode]()

  while gui.running:
    for e in gui.get_events(ti.GUI.PRESS):
      if e.key == ti.GUI.ESCAPE:
        gui.running = False
      elif e.key in ('1', '2', '3', '4', '5'):
        requested = int(e.key)
        if requested != current_mode:
          current_mode = requested
          RENDERERS[current_mode]()

    gui.set_image(display)
    gui.text(MODE_NAMES[current_mode], pos=(0.02, 0.97), font_size=20, color=0xFFFFFF)
    gui.show()


if __name__ == "__main__":
  main()