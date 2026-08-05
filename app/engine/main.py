import math

import taichi as ti

from tracer.camera import Camera
from tracer.sampling.brdf import BRDF
from tracer.kernels.render import render_kernel
from tracer.geometry import scene
from tracer.geometry.mock_scenes import build_test_room
from tracer.bvh import builder

WIDTH = 1280
HEIGHT = 720

FOV = 60.0
NEAR = 0.1
FAR = 1000.0

MAX_BOUNCES = 1
SPP = 1

SINGLE_SIDED = 1


def main():
  ti.init(arch=ti.cuda)

  scene.init_scene_fields()
  build_test_room()

  camera = Camera(
    position=ti.math.vec3(0.0, 1.2, 4.0),
    yaw=0.0,
    pitch=0.0,
    fov=math.radians(FOV),
    aspect_ratio=WIDTH / HEIGHT,
    near=NEAR,
    far=FAR,
  )

  right, up, forward = camera.basis_from_yaw_pitch()

  brdf = BRDF()

  image = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

  render_kernel(
    image,
    right, up, forward, camera.position,
    camera.fov, camera.aspect_ratio,
    brdf,
    scene.triangles, scene.num_triangles[None],
    builder.bvh_node_min, builder.bvh_node_max, builder.bvh_node_left, builder.bvh_node_right,
    builder.bvh_node_start, builder.bvh_node_count, builder.bvh_indices,
    scene.light_triangle_index, scene.light_pdf_area, scene.num_lights[None],
    SINGLE_SIDED, MAX_BOUNCES, SPP,
  )

  gui = ti.GUI("Project Albedo", res=(WIDTH, HEIGHT)) # type: ignore

  while gui.running:
    gui.set_image(image)
    gui.show()


if __name__ == "__main__":
  main()