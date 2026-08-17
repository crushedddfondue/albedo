import math

import numpy as np
import taichi as ti
from taichi.math import vec3

from tracer import buffers
from tracer.bvh import builder
from tracer.camera import Camera
from tracer.camera_path import scripted
from tracer.denoise import history
from tracer.denoise.demodulate import demodulate, remodulate
from tracer.denoise.reproject import reproject_kernel
from tracer.environment import Environment, ENV_BLACK
from tracer.geometry import scene
from tracer.geometry.mock_scenes import build_test_room
from tracer.kernels.gbuffer import gbuffer_kernel
from tracer.kernels.motion import motion_kernel
from tracer.kernels.render import (
  accumulate_kernel, resolve_kernel, TONEMAP_REINHARD_EXT,
)
from tracer.sampling.brdf import BRDF

WIDTH, HEIGHT = 1280, 720

FOV = 60.0
NEAR, FAR = 0.1, 1000.0
PITCH = math.radians(5)

MAX_BOUNCES = 8

# ⚠ SPP is now samples per frame with NO cross-frame accumulation. The
# denoiser's whole job is to make a noisy per-frame estimate usable, so
# feeding it a progressively-converged image would be testing nothing.
# accumulate_kernel is called with frame_index=1 every frame, which makes
# its running mean reduce to a plain assignment.
SPP = 2

USE_NEE = 1
SINGLE_SIDED = 1

# Caps how far back history reaches. 0.05 is roughly a 20-frame window.
# Larger means less noise and more ghosting -- this one number is the entire
# temporal quality tradeoff.
ALPHA_MIN = 0.05

# A camera jump larger than this invalidates all history.
CUT_DISTANCE = 0.5

EXPOSURE = 1.0
TONEMAP = TONEMAP_REINHARD_EXT
WHITE_POINT = 3.0


def make_camera(position, yaw):
  return Camera(
    position=position, yaw=yaw, pitch=PITCH,
    fov=math.radians(FOV), aspect_ratio=WIDTH / HEIGHT,
    near=NEAR, far=FAR,
  )


def main():
  ti.init(arch=ti.cuda)

  print("init ok")

  scene.init_scene_fields()
  build_test_room()
  print("scene ok")

  buffers.init_aov_fields(WIDTH, HEIGHT)
  history.init_history_fields(WIDTH, HEIGHT)
  history.reset()
  print("buffers ok")

  scene.init_scene_fields()
  build_test_room()
  buffers.init_aov_fields(WIDTH, HEIGHT)
  history.init_history_fields(WIDTH, HEIGHT)
  history.reset()

  print(f"triangles: {scene.num_triangles[None]}  bvh nodes: {len(builder.nodes)}  "
        f"lights: {scene.num_lights[None]}")

  brdf = BRDF()
  environment = Environment(mode=ENV_BLACK)

  shape = (WIDTH, HEIGHT)
  raw = ti.Vector.field(3, ti.f32, shape=shape)        # noisy radiance, linear
  demod = ti.Vector.field(3, ti.f32, shape=shape)      # radiance / albedo
  accum_col = ti.Vector.field(3, ti.f32, shape=shape)  # temporally accumulated
  accum_mom = ti.Vector.field(2, ti.f32, shape=shape)
  accum_len = ti.field(ti.i32, shape=shape)
  final = ti.Vector.field(3, ti.f32, shape=shape)      # remodulated
  display = ti.Vector.field(3, ti.f32, shape=shape)

  gui = ti.GUI("Project Albedo", res=(WIDTH, HEIGHT))  # type: ignore

  frame = 0
  prev = None   # (view, proj, position) of the previous frame

  while gui.running:
    position, yaw = scripted(frame)
    camera = make_camera(position, yaw)
    right, up, forward = camera.basis_from_yaw_pitch()

    # A teleport invalidates every pixel's history. Detected by distance
    # rather than by frame number so it also catches the discontinuity where
    # the scripted path switches phases.
    if prev is not None:
      if float(np.linalg.norm(np.asarray(position) - prev[2])) > CUT_DISTANCE:
        history.reset()
        prev = None

    # --- 1. G-buffer. clear_aovs FIRST: the miss branch deliberately leaves
    #        albedo, normal and depth alone, so stale background survives.
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

    # --- 2. Motion vectors: current geometry, PREVIOUS matrices.
    if prev is not None:
      motion_kernel(
        buffers.motion, buffers.depth, buffers.hit_mask,
        right, up, forward, camera.position, camera.fov, camera.aspect_ratio,
        prev[0], prev[1],
      )
    else:
      buffers.motion.fill(0.0)

    # --- 3. Noisy radiance. frame_index=1 makes the running mean a plain
    #        assignment -- one independent estimate per frame.
    accumulate_kernel(
      raw, 1, right, up, forward, camera.position,
      camera.fov, camera.aspect_ratio,
      brdf, environment,
      scene.triangles, scene.num_triangles[None],
      builder.bvh_node_min, builder.bvh_node_max,
      builder.bvh_node_left, builder.bvh_node_right,
      builder.bvh_node_start, builder.bvh_node_count, builder.bvh_indices,
      scene.light_triangle_index, scene.light_pdf_area, scene.num_lights[None],
      SINGLE_SIDED, USE_NEE, MAX_BOUNCES, SPP,
    )

    # --- 4. Demodulate, so the filter sees irradiance not texture.
    demodulate(raw, buffers.albedo, buffers.hit_mask, demod)

    # --- 5. Temporal accumulation.
    reproject_kernel(
      demod, buffers.depth, buffers.normal, buffers.object_id, buffers.motion,
      history.colour, history.moments, history.depth, history.normal,
      history.object_id, history.length,
      accum_col, accum_mom, accum_len,
      WIDTH, HEIGHT, ALPHA_MIN,
    )

    # --- 6. TODO: variance from accum_mom, then the a-trous spatial filter.
    #        Until those exist this is temporal-only denoising, which is
    #        still worth looking at -- it should visibly stabilise a static
    #        camera and ghost under motion.

    # --- 7. Remodulate BEFORE display, but store the DEMODULATED result.
    remodulate(accum_col, buffers.albedo, buffers.hit_mask, final)

    # --- 8. History takes the FILTERED, demodulated colour. Storing the raw
    #        input would average noise into noise and never converge.
    history.store(
      buffers.normal, buffers.object_id, buffers.depth,
      accum_col, accum_mom, accum_len,
    )

    prev = (
      np.asarray(camera.view_matrix(), dtype=np.float32),
      np.asarray(camera.projection_matrix(), dtype=np.float32),
      np.asarray(position, dtype=np.float64),
    )

    resolve_kernel(final, display, EXPOSURE, TONEMAP, WHITE_POINT)
    gui.set_image(display)
    gui.show()

    frame += 1
    if frame % 60 == 0:
      print(f"frame {frame}  mean history length "
            f"{accum_len.to_numpy().mean():.1f}")


if __name__ == "__main__":
  main()