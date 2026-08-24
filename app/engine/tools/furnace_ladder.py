import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import taichi as ti
from taichi.math import vec3

ti.init(arch=ti.cuda)

from tracer.bvh import upload as bvh_upload
from tracer.camera import Camera
from tracer.environment import Environment, ENV_BLACK, ENV_CONSTANT
from tracer.geometry import scene
from tracer.geometry.scene_generator import SceneParams, SceneData, build_scene, upload_scene
from tracer.kernels.render import accumulate_kernel
from tracer.sampling.brdf import BRDF
from tracer.trajectory import TrajectoryParams, sample_trajectory

W, H = 128, 72
FOV = math.radians(60.0)
ASPECT = W / H

FURNACE_L = 0.5
LADDER = (4, 8, 16, 32, 64, 128)
SEEDS = (20260819, 20260823, 20260831)

FURNACE_SPP = 32
DATASET_SPP = 64

scene.init_scene_fields()
bvh_upload.init_bvh_fields()

accum = ti.Vector.field(3, ti.f32, shape=(W, H))
brdf = BRDF()

env_furnace = Environment(mode=ENV_CONSTANT, constant=vec3(FURNACE_L))
env_black = Environment(mode=ENV_BLACK)

def make_furnace(d: SceneData) -> SceneData:
  n = d.n_triangles
  return SceneData(
    v0=d.v0, v1=d.v1, v2=d.v2,
    albedo=np.ones((n, 3), dtype=np.float32),
    emission=np.zeros((n, 3), dtype=np.float32),
    object_id=d.object_id,
    light_index=np.full(n, -1, dtype=np.int32),
    light_triangle_index=np.zeros(0, dtype=np.int32),
    light_pdf_area=np.zeros(0, dtype=np.float32),
    room_size=d.room_size, occluders=d.occluders, scene_id=d.scene_id, seed=d.seed, meta=dict(d.meta),
  )

def camera_for(d: SceneData) -> Camera:
  poses, _ = sample_trajectory(d.seed * 1000, 8, d.bounds, TrajectoryParams(), obstacles=d.occluders)
  p = poses[0]
  return Camera(position=np.asarray(p.position, dtype=np.float32), yaw=p.yaw, pitch=p.pitch, fov=FOV, aspect_ratio=ASPECT, near=0.1, far=1000.0)

def render(camera: Camera, environment, bounces: int, spp: int, use_nee: int):
  right, up, forward = camera.basis_from_yaw_pitch()
  accum.fill(0.0)
  accumulate_kernel(
    accum, 1, right, up, forward, camera.position, camera.fov, camera.aspect_ratio,
    brdf, environment,
    scene.triangles, scene.num_triangles[None],
    *bvh_upload.kernel_args(),
    scene.light_triangle_index, scene.light_pdf_area, scene.num_lights[None],
    1, use_nee, bounces, spp,
  )
  return accum.to_numpy().astype(np.float64)


def section_a():
  print("=" * 78)
  print("A. WHITE FURNACE vs BOUNCE COUNT   (unit albedo, no emitters, env L =",
        f"{FURNACE_L})")
  print("=" * 78)
  print("open room  -> environment reachable, expect bias -> 0 as bounces rise")
  print("sealed room-> environment unreachable, TRUE answer is 0.0 at every bounce")
  print()

  for seed in SEEDS:
    d = build_scene(SceneParams(), seed)
    upload_scene(make_furnace(d))
    assert scene.num_lights[None] == 0
    camera = camera_for(d)

    sealed = d.meta["has_ceiling"]
    target = 0.0 if sealed else FURNACE_L

    print(f"seed {seed}  ceiling={sealed}  boxes={d.meta['n_boxes']}  "
          f"lights={d.meta['n_light_quads']}  room={np.round(d.room_size, 2).tolist()}")
    print(f"  expected limit: {target}")

    prev = None
    for b in LADDER:
      img = render(camera, env_furnace, b, FURNACE_SPP, use_nee=0)
      mean = img.mean()
      sem = img.std(ddof=1) / math.sqrt(img.size)
      bias = mean - target
      delta = "" if prev is None else f"  d={mean - prev:+.6f}"
      print(f"  bounces {b:4d}  mean {mean:.6f}  bias {bias:+.6f}  "
            f"sem {sem:.6f}  min {img.min():.4f}  max {img.max():.4f}{delta}")
      prev = mean
    print()

def section_b():
  print("=" * 78)
  print("B. DATASET SCENES vs BOUNCE COUNT   (real albedo + emitters, ENV_BLACK)")
  print("=" * 78)
  print("How much energy does the clean target miss at 8 bounces? Compared")
  print("against the highest rung, which stands in for the converged answer.")
  print()

  for seed in SEEDS:
    d = build_scene(SceneParams(), seed)
    upload_scene(d)
    camera = camera_for(d)

    print(f"seed {seed}  ceiling={d.meta['has_ceiling']}  boxes={d.meta['n_boxes']}  "
          f"lights={d.meta['n_light_quads']}")

    means = {}
    for b in LADDER:
      img = render(camera, env_black, b, DATASET_SPP, use_nee=1)
      means[b] = img.mean()
      print(f"  bounces {b:4d}  mean {means[b]:.6f}  max {img.max():.2f}")

    ref = means[LADDER[-1]]
    if ref > 0:
      print(f"  -> at 8 bounces the mean is {means[8] / ref - 1.0:+.2%} of the "
            f"{LADDER[-1]}-bounce value")
      print(f"  -> at 16 bounces, {means[16] / ref - 1.0:+.2%}")
    print()

  print("Compare those percentages to the clean target's split-half sigma")
  print("(manifest: clean_split_half_sigma). Truncation smaller than the")
  print("target's own noise is not worth paying for; larger is a BIAS, and")
  print("no amount of averaging removes it.")


if __name__ == "__main__":
  section_a()
  section_b()