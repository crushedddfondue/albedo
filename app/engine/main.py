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

MAX_BOUNCES = 8
SPP = 1024
USE_NEE = 1

SINGLE_SIDED = 1


def main():
  ti.init(arch=ti.cuda)

  scene.init_scene_fields()
  build_test_room()

  camera = Camera(
    position=ti.math.vec3(0.0, 1.2, 4.0),
    yaw=0.0,
    pitch=math.radians(5),
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
    SINGLE_SIDED, USE_NEE, MAX_BOUNCES, SPP
  )

  import numpy as np

  a = np.load("nee.npy").astype(np.float64) ** 2.2
  b = np.load("bsdf.npy").astype(np.float64) ** 2.2

  # image is indexed [px, py, c]; after the ndc_y fix py = 0 is the bottom row,
  # so low py is floor. Stay well clear of the light quad -- it clamps at 1.0
  # and carries no information.
  floor_a = a[:, :250].mean()
  floor_b = b[:, :250].mean()

  print(f"nee  {floor_a:.6f}")
  print(f"bsdf {floor_b:.6f}")
  print(f"ratio {floor_a / floor_b:.4f}")

  print(f"floor  {a[:, :250].mean() / b[:, :250].mean():.4f}")
  print(f"umbra  {a[600:700, 100:170].mean() / b[600:700, 100:170].mean():.4f}")

  # import numpy as np
  # np.save("nee.npy" if USE_NEE else "bsdf.npy", image.to_numpy())

  gui = ti.GUI("Project Albedo", res=(WIDTH, HEIGHT)) # type: ignore

  while gui.running:
    gui.set_image(image)
    gui.show()


if __name__ == "__main__":
  main()