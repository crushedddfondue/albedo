import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import numpy as np
import taichi as ti
from taichi.math import vec3

from tracer.bvh import builder
from tracer.camera import Camera
from tracer.environment import Environment, ENV_BLACK
from tracer.geometry import scene
from tracer.geometry.mock_scenes import build_test_room
from tracer.io import frame_io
from tracer.kernels.render import accumulate_kernel
from tracer.sampling.brdf import BRDF

WIDTH, HEIGHT = 1280, 720
FOV, NEAR, FAR = 60.0, 0.1, 1000.0
PITCH = math.radians(5)

MAX_BOUNCES = 8
USE_NEE, SINGLE_SIDED = 1, 1

# Per half. Total is twice this. 256 * 32 = 8192 spp each, 16384 combined.
FRAMES_PER_HALF = 256
SPP_PER_FRAME = 32

OUT_PATH = "reference_static"


def make_camera(position, yaw):
  return Camera(
    position=position, yaw=yaw, pitch=PITCH,
    fov=math.radians(FOV), aspect_ratio=WIDTH / HEIGHT,
    near=NEAR, far=FAR,
  )


def _accumulate(target, frame_index, camera, brdf, environment):
  right, up, forward = camera.basis_from_yaw_pitch()
  accumulate_kernel(
    target, frame_index, right, up, forward, camera.position,
    camera.fov, camera.aspect_ratio,
    brdf, environment,
    scene.triangles, scene.num_triangles[None],
    builder.bvh_node_min, builder.bvh_node_max,
    builder.bvh_node_left, builder.bvh_node_right,
    builder.bvh_node_start, builder.bvh_node_count, builder.bvh_indices,
    scene.light_triangle_index, scene.light_pdf_area, scene.num_lights[None],
    SINGLE_SIDED, USE_NEE, MAX_BOUNCES, SPP_PER_FRAME,
  )


def main():
  ti.init(arch=ti.cuda)

  scene.init_scene_fields()
  build_test_room()

  brdf = BRDF()
  environment = Environment(mode=ENV_BLACK)

  # camera_path.static's pose -- the one SVGF is scored at.
  camera = make_camera(vec3(0.0, 1.2, 4.0), 0.0)

  half_a = ti.Vector.field(3, ti.f32, shape=(WIDTH, HEIGHT))
  half_b = ti.Vector.field(3, ti.f32, shape=(WIDTH, HEIGHT))
  half_a.fill(0.0)
  half_b.fill(0.0)

  # Two INDEPENDENT accumulations. Taichi's RNG stream continues across
  # kernel launches within a run, so alternating frames between the two
  # buffers gives each an independent set of samples with no seed juggling.
  for n in range(1, FRAMES_PER_HALF + 1):
    _accumulate(half_a, n, camera, brdf, environment)
    _accumulate(half_b, n, camera, brdf, environment)
    if n % 32 == 0:
      print(f"  {n}/{FRAMES_PER_HALF} frames per half "
            f"({n * SPP_PER_FRAME} spp each)")

  a = half_a.to_numpy().astype(np.float64)
  b = half_b.to_numpy().astype(np.float64)

  reference = 0.5 * (a + b)

  # SPLIT-HALF ERROR BAR, with no ground truth anywhere in sight.
  #
  # A and B are independent unbiased estimates of the same image, each with
  # variance V. So Var(A - B) = 2V, giving sigma_A = RMS(A - B) / sqrt(2).
  # The reference is their mean, with variance V/2, so
  #
  #     sigma_reference = RMS(A - B) / 2
  #
  # Worth remembering well beyond here: it is how you get an error estimate
  # for any Monte Carlo quantity that has no analytic answer, which
  # describes essentially all of Phase 2.
  diff = a - b
  sigma_ref = np.sqrt((diff ** 2).mean()) / 2.0
  mean_lum = reference.mean()

  print(f"\nreference: {2 * FRAMES_PER_HALF * SPP_PER_FRAME} spp total")
  print(f"  mean radiance     {mean_lum:.6f}")
  print(f"  own noise (sigma) {sigma_ref:.6f}")
  print(f"  relative          {sigma_ref / max(mean_lum, 1e-9):.2%}")
  print("\nFor this reference to adjudicate, its own noise must be well "
        "below the gap you expect between SVGF and the model. If it is not, "
        "raise FRAMES_PER_HALF -- error falls as 1/sqrt(N), so 4x the frames "
        "halves it.")

  path = frame_io.write_frame(
    OUT_PATH,
    radiance=reference,
    meta={
      "scene_id": "test_room",
      "kind": "reference",
      "camera": {
        "position": [0.0, 1.2, 4.0], "yaw": 0.0, "pitch": PITCH,
        "fov": math.radians(FOV), "aspect_ratio": WIDTH / HEIGHT,
      },
      "render_config": {
        "spp_total": 2 * FRAMES_PER_HALF * SPP_PER_FRAME,
        "max_bounces": MAX_BOUNCES, "use_nee": USE_NEE,
        "single_sided": SINGLE_SIDED,
      },
      "split_half_sigma": float(sigma_ref),
    },
  )
  print(f"\nsaved {path}")


if __name__ == "__main__":
  main()