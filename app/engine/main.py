import math

import numpy as np
import taichi as ti
from taichi.math import vec3

from tracer import buffers
from tracer.bvh import builder
from tracer.camera import Camera
from tracer.camera_path import scripted
from tracer.denoise import history
from tracer.denoise.atrous import filter_image, ATROUS_LEVELS
from tracer.denoise.demodulate import demodulate, remodulate
from tracer.denoise.moments import estimate_variance
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

# Samples per FRAME, with no cross-frame accumulation. The denoiser's job is
# to make a noisy per-frame estimate usable, so feeding it a progressively
# converged image would test nothing. accumulate_kernel is called with
# frame_index=1 each frame, which reduces its running mean to an assignment.
SPP = 2

USE_NEE = 1
SINGLE_SIDED = 1

# Caps how far back temporal history reaches. 0.05 is roughly a 20-frame
# window. Larger means less noise and more ghosting -- this one number is the
# entire temporal quality tradeoff.
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

  raw = ti.Vector.field(3, ti.f32, shape=shape)          # noisy radiance, linear
  demod = ti.Vector.field(3, ti.f32, shape=shape)        # radiance / albedo
  accum_col = ti.Vector.field(3, ti.f32, shape=shape)    # temporally accumulated
  accum_mom = ti.Vector.field(2, ti.f32, shape=shape)
  accum_len = ti.field(ti.i32, shape=shape)

  # A-trous ping-pong. accum_col is the first source, so only one extra
  # colour buffer is needed alongside it.
  colour_b = ti.Vector.field(3, ti.f32, shape=shape)
  variance_a = ti.field(ti.f32, shape=shape)
  variance_b = ti.field(ti.f32, shape=shape)
  var_prefiltered = ti.field(ti.f32, shape=shape)

  # ⚠ Dedicated buffer for the level-one output. It cannot be a reference
  # into the ping-pong pair: with five levels, the buffer holding level one
  # is written again at level three, so a reference would silently give
  # history the level-three image instead.
  history_colour = ti.Vector.field(3, ti.f32, shape=shape)

  final = ti.Vector.field(3, ti.f32, shape=shape)        # remodulated, for display
  display = ti.Vector.field(3, ti.f32, shape=shape)

  gui = ti.GUI("Project Albedo", res=(WIDTH, HEIGHT))  # type: ignore

  frame = 0
  prev = None   # (view, proj, position) of the previous frame

  while gui.running:
    position, yaw = scripted(frame)
    camera = make_camera(position, yaw)
    right, up, forward = camera.basis_from_yaw_pitch()

    # A teleport invalidates every pixel's history. Detected by distance
    # rather than frame number, so it also catches the discontinuity where
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

    # --- 3. Noisy radiance, one independent estimate per frame.
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

    # --- 4. Demodulate, so the filter sees irradiance rather than texture.
    demodulate(raw, buffers.albedo, buffers.hit_mask, demod)

    # --- 5. Temporal accumulation.
    reproject_kernel(
      demod, buffers.depth, buffers.normal, buffers.object_id, buffers.motion,
      history.colour, history.moments, history.depth, history.normal,
      history.object_id, history.length,
      accum_col, accum_mom, accum_len,
      WIDTH, HEIGHT, ALPHA_MIN,
    )

    # --- 6. Variance: temporal where history allows, spatial where it does not.
    estimate_variance(
      accum_mom, accum_len, accum_col,
      buffers.depth, buffers.normal, buffers.object_id,
      variance_a, WIDTH, HEIGHT,
    )

    var_snapshot = None
    if (frame + 1) % 60 == 0:
      var_snapshot = variance_a.to_numpy()

    # --- 7. Spatial filter. history_colour is filled in place with the
    #        level-one result; `filtered` is the level-five output.
    filtered, _ = filter_image(
      accum_col, colour_b, variance_a, variance_b, var_prefiltered,
      history_colour,
      buffers.depth, buffers.normal, buffers.object_id,
      WIDTH, HEIGHT, levels=ATROUS_LEVELS,
    )

    # --- 8. Remodulate for display only. History stays demodulated.
    remodulate(filtered, buffers.albedo, buffers.hit_mask, final)

    # --- 9. ⚠ History takes LEVEL ONE, from its own buffer. Feeding level
    #        five back compounds the blur every frame and a static camera
    #        slowly dissolves into mush.
    history.store(
      buffers.normal, buffers.object_id, buffers.depth,
      history_colour, accum_mom, accum_len,
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
      lengths = accum_len.to_numpy()
      geo = buffers.hit_mask.to_numpy() == 1
      n_geo = int(geo.sum())

      if n_geo == 0:
        print(f"after frame {frame - 1}  NO GEOMETRY IN FRAME")
      else:
        pre = var_snapshot[geo].mean()  # type: ignore
        post = variance_a.to_numpy()[geo].mean()
        print(f"after frame {frame - 1}  history {lengths[geo].mean():.1f}  "
              f"coverage {n_geo / (WIDTH * HEIGHT):.0%}  "
              f"variance pre {pre:.4f} -> post {post:.5f}")


if __name__ == "__main__":
  main()