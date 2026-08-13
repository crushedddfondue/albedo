import math

import numpy as np
import taichi as ti

from tracer.camera import Camera
from tracer.sampling.brdf import BRDF
from tracer.environment import Environment, ENV_BLACK, ENV_GRADIENT, ENV_CONSTANT
from tracer.kernels.render import (
  accumulate_kernel, resolve_kernel,
  TONEMAP_NONE, TONEMAP_REINHARD, TONEMAP_REINHARD_EXT,
)
from tracer.geometry import scene
from tracer.geometry.mock_scenes import build_test_room, build_furnace_scene
from tracer.bvh import builder

print(np.load("nee.npy")[:, :250].mean())

WIDTH = 1280
HEIGHT = 720

FOV = 60.0
NEAR = 0.1
FAR = 1000.0

MAX_BOUNCES = 8

# SPP is now samples per FRAME, not per image. Total samples = SPP * frames.
# Keep it small: the point of accumulation is many cheap launches rather than
# one expensive one, which is also what keeps each launch under the Windows
# TDR limit.
SPP = 4

USE_NEE = 1
SINGLE_SIDED = 1

# Display. Nothing here touches the linear accumulation buffer.
EXPOSURE = 1.0
TONEMAP = TONEMAP_REINHARD_EXT
WHITE_POINT = 3.0

# ---- Offline / validation mode ----------------------------------------------
# When OFFLINE is True, run a fixed number of frames, report, and stop
# accumulating. When False, accumulate indefinitely and watch it converge.
OFFLINE = True
OFFLINE_FRAMES = 256          # 256 * 4 = 1024 spp total

# White furnace test. Constant environment radiance, unit albedo, no emitters.
#     L_o = (rho/pi) * L * int_H cos(theta) dw = rho * L = L   for rho = 1
# The accumulation buffer is linear now, so this reads it directly -- no
# ** 2.2 round trip, and it stays valid even for values above 1.0.
FURNACE = False
FURNACE_L = 0.5


def main():
  ti.init(arch=ti.cuda)

  scene.init_scene_fields()

  if FURNACE:
    build_furnace_scene()
    environment = Environment(mode=ENV_CONSTANT, constant=ti.math.vec3(FURNACE_L))
  else:
    build_test_room()
    environment = Environment(mode=ENV_BLACK)

  # Cheap guard against rendering a scene that silently did not change --
  # e.g. occluder triangles written but num_triangles left at 4. Expect
  # 6 triangles / 3 nodes; 6 against LEAF_SIZE = 4 forces build() to recurse.
  print(f"triangles: {scene.num_triangles[None]}  bvh nodes: {len(builder.nodes)}  lights: {scene.num_lights[None]}")

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

  # accum holds linear HDR radiance and is the tracer's real output.
  # display is display-referred [0,1] and exists only for ti.GUI.
  accum = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
  display = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
  accum.fill(0.0)

  def trace(frame_index):
    accumulate_kernel(
      accum, frame_index,
      right, up, forward, camera.position,
      camera.fov, camera.aspect_ratio,
      brdf, environment,
      scene.triangles, scene.num_triangles[None],
      builder.bvh_node_min, builder.bvh_node_max, builder.bvh_node_left, builder.bvh_node_right,
      builder.bvh_node_start, builder.bvh_node_count, builder.bvh_indices,
      scene.light_triangle_index, scene.light_pdf_area, scene.num_lights[None],
      SINGLE_SIDED, USE_NEE, MAX_BOUNCES, SPP,
    )

  gui = ti.GUI("Project Albedo", res=(WIDTH, HEIGHT))  # type: ignore

  frame = 0

  if OFFLINE:
    for f in range(1, OFFLINE_FRAMES + 1):
      trace(f)
    frame = OFFLINE_FRAMES

    lin = accum.to_numpy().astype(np.float64)

    if FURNACE:
      err = np.abs(lin - FURNACE_L)
      print(f"furnace  bounces {MAX_BOUNCES}  spp {SPP * OFFLINE_FRAMES}  "
            f"min {lin.min():.6f}  max {lin.max():.6f}  mean {lin.mean():.6f}  target {FURNACE_L}")
      print(f"         max abs error {err.max():.3e}   mean abs error {err.mean():.3e}")
    else:
      # Linear now, so the comparison script no longer needs ** 2.2.
      np.save("nee.npy" if USE_NEE else "bsdf.npy", lin)
      print(f"saved {'nee' if USE_NEE else 'bsdf'}.npy  spp {SPP * OFFLINE_FRAMES}  "
            f"min {lin.min():.6f}  max {lin.max():.6f}  mean {lin.mean():.6f}")

  resolve_kernel(accum, display, EXPOSURE, TONEMAP, WHITE_POINT)

  while gui.running:
    if not OFFLINE:
      # Progressive: one launch per displayed frame, converging live.
      # Resetting on camera motion is frame = 0 plus accum.fill(0.0) --
      # that hookup belongs with Task 6, when input actually moves the camera.
      frame += 1
      trace(frame)
      resolve_kernel(accum, display, EXPOSURE, TONEMAP, WHITE_POINT)
      gui.set_image(display)
      gui.show(None)
      if frame % 32 == 0:
        print(f"frame {frame}  spp {frame * SPP}")
    else:
      gui.set_image(display)
      gui.show()


if __name__ == "__main__":
  main()