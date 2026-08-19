import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import numpy as np
import taichi as ti
from taichi.math import vec3

from metrics.image_metrics import report
from tracer import buffers
from tracer.bvh import builder
from tracer.camera import Camera
from tracer.denoise import history
from tracer.denoise.atrous import filter_image, ATROUS_LEVELS
from tracer.denoise.demodulate import demodulate, remodulate
from tracer.denoise.moments import estimate_variance
from tracer.denoise.reproject import reproject_kernel
from tracer.environment import Environment, ENV_BLACK
from tracer.geometry import scene
from tracer.geometry.mock_scenes import build_test_room
from tracer.io import frame_io
from tracer.kernels.gbuffer import gbuffer_kernel
from tracer.kernels.render import accumulate_kernel
from tracer.sampling.brdf import BRDF

WIDTH, HEIGHT = 1280, 720
FOV, NEAR, FAR = 60.0, 0.1, 1000.0
PITCH = math.radians(5)

MAX_BOUNCES = 8
SPP = 2
USE_NEE, SINGLE_SIDED = 1, 1
ALPHA_MIN = 0.05

# Long enough for history to saturate at 1/ALPHA_MIN and settle.
WARMUP_FRAMES = 120

REFERENCE_PATH = "reference_static"


def main():
  ti.init(arch=ti.cuda)

  scene.init_scene_fields()
  build_test_room()
  buffers.init_aov_fields(WIDTH, HEIGHT)
  history.init_history_fields(WIDTH, HEIGHT)
  history.reset()

  brdf = BRDF()
  environment = Environment(mode=ENV_BLACK)

  camera = Camera(
    position=vec3(0.0, 1.2, 4.0), yaw=0.0, pitch=PITCH,
    fov=math.radians(FOV), aspect_ratio=WIDTH / HEIGHT, near=NEAR, far=FAR,
  )
  right, up, forward = camera.basis_from_yaw_pitch()

  shape = (WIDTH, HEIGHT)
  raw = ti.Vector.field(3, ti.f32, shape=shape)
  demod = ti.Vector.field(3, ti.f32, shape=shape)
  accum_col = ti.Vector.field(3, ti.f32, shape=shape)
  accum_mom = ti.Vector.field(2, ti.f32, shape=shape)
  accum_len = ti.field(ti.i32, shape=shape)
  colour_b = ti.Vector.field(3, ti.f32, shape=shape)
  variance_a = ti.field(ti.f32, shape=shape)
  variance_b = ti.field(ti.f32, shape=shape)
  var_prefiltered = ti.field(ti.f32, shape=shape)
  history_colour = ti.Vector.field(3, ti.f32, shape=shape)
  final = ti.Vector.field(3, ti.f32, shape=shape)

  # Static camera, so the G-buffer is identical every frame and motion is
  # zero throughout. Compute it once.
  buffers.clear_aovs()
  gbuffer_kernel(
    buffers.albedo, buffers.normal, buffers.object_id,
    buffers.hit_mask, buffers.depth,
    right, up, forward, camera.position, camera.fov, camera.aspect_ratio,
    scene.triangles,
    builder.bvh_node_min, builder.bvh_node_max,
    builder.bvh_node_left, builder.bvh_node_right,
    builder.bvh_node_start, builder.bvh_node_count, builder.bvh_indices,
  )
  buffers.motion.fill(0.0)

  for n in range(WARMUP_FRAMES):
    accumulate_kernel(
      raw, 1, right, up, forward, camera.position,
      camera.fov, camera.aspect_ratio, brdf, environment,
      scene.triangles, scene.num_triangles[None],
      builder.bvh_node_min, builder.bvh_node_max,
      builder.bvh_node_left, builder.bvh_node_right,
      builder.bvh_node_start, builder.bvh_node_count, builder.bvh_indices,
      scene.light_triangle_index, scene.light_pdf_area, scene.num_lights[None],
      SINGLE_SIDED, USE_NEE, MAX_BOUNCES, SPP,
    )
    demodulate(raw, buffers.albedo, buffers.hit_mask, demod)
    reproject_kernel(
      demod, buffers.depth, buffers.normal, buffers.object_id, buffers.motion,
      history.colour, history.moments, history.depth, history.normal,
      history.object_id, history.length,
      accum_col, accum_mom, accum_len, WIDTH, HEIGHT, ALPHA_MIN,
    )
    estimate_variance(
      accum_mom, accum_len, accum_col,
      buffers.depth, buffers.normal, buffers.object_id,
      variance_a, WIDTH, HEIGHT,
    )
    filtered, _ = filter_image(
      accum_col, colour_b, variance_a, variance_b, var_prefiltered,
      history_colour, buffers.depth, buffers.normal, buffers.object_id,
      WIDTH, HEIGHT, levels=ATROUS_LEVELS,
    )
    remodulate(filtered, buffers.albedo, buffers.hit_mask, final)
    history.store(
      buffers.normal, buffers.object_id, buffers.depth,
      history_colour, accum_mom, accum_len,
    )

  ref = frame_io.read_frame(REFERENCE_PATH)
  reference = ref["radiance"].astype(np.float64)
  sigma_ref = ref["meta"].get("split_half_sigma")

  denoised = final.to_numpy().astype(np.float64)
  one_spp = raw.to_numpy().astype(np.float64)
  geo = buffers.hit_mask.to_numpy() == 1

  print(f"\nSVGF at {SPP} spp/frame, {WARMUP_FRAMES} frames warmup")
  print(f"reference own sigma: {sigma_ref:.6f}\n")

  # ⚠ Both scores use the SAME metric code as 2.4 will. That is the only
  # thing making the eventual comparison meaningful.
  print("raw (no denoising):")
  for k, v in report(one_spp, reference, mask=geo).items():
    print(f"  {k:14s} {v:.6f}")

  print("\nSVGF:")
  svgf_scores = report(denoised, reference, mask=geo)
  for k, v in svgf_scores.items():
    print(f"  {k:14s} {v:.6f}")

  print(f"\n>>> THE BAR: relMSE {svgf_scores['relmse']:.6f}")
  print("2.4 has to beat this, measured exactly this way.")


if __name__ == "__main__":
  main()