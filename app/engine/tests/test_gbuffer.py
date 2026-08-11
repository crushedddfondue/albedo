from types import SimpleNamespace

import math
import numpy as np
import pytest
import taichi as ti
from taichi.math import vec3

from tracer import buffers
from tracer.camera import Camera
from tracer.geometry import scene
from tracer.geometry.mock_scenes import build_test_room, build_furnace_scene
from tracer.bvh import builder
from tracer.kernels.gbuffer import gbuffer_kernel
from tracer.kernels.render import accumulate_kernel
from tracer.sampling.brdf import BRDF
from tracer.environment import Environment, ENV_CONSTANT

W, H = 320, 180   # small: these tests are about correctness, not throughput


@ti.kernel
def _facing_kernel(facing: ti.template(), normal_f: ti.template(), hit_mask_f: ti.template(), right: vec3, up: vec3, forward: vec3, fov: ti.f32, aspect_ratio: ti.f32):  # type: ignore
  """dot(n, -ray_dir) per pixel.

  Deliberately calls the same ray_direction_for_pixel that gbuffer_kernel
  uses, so this doubles as a check that the shared function is actually
  shared -- if someone reintroduces a second copy of the mapping, the dot
  products drift and this fails.
  """
  width = normal_f.shape[0]
  height = normal_f.shape[1]
  for px, py in normal_f:
    facing[px, py] = 0.0
    if hit_mask_f[px, py] == 1:
      d = Camera.ray_direction_for_pixel(px, py, 0.5, 0.5, width, height, right, up, forward, fov, aspect_ratio)
      facing[px, py] = ti.math.dot(normal_f[px, py], -d)


@pytest.fixture(scope="module")
def frame():
  scene.init_scene_fields()
  build_test_room()
  buffers.init_aov_fields(W, H)

  camera = Camera(
    position=vec3(0.0, 1.2, 4.0), yaw=0.0, pitch=math.radians(5),
    fov=math.radians(60.0), aspect_ratio=W / H, near=0.1, far=1000.0,
  )
  right, up, forward = camera.basis_from_yaw_pitch()
  facing = ti.field(ti.f32, shape=(W, H))

  def render():
    buffers.clear_aovs()
    gbuffer_kernel(
      buffers.albedo, buffers.normal, buffers.object_id, buffers.hit_mask,
      right, up, forward, camera.position, camera.fov, camera.aspect_ratio,
      scene.triangles,
      builder.bvh_node_min, builder.bvh_node_max,
      builder.bvh_node_left, builder.bvh_node_right,
      builder.bvh_node_start, builder.bvh_node_count, builder.bvh_indices,
    )

  render()
  _facing_kernel(facing, buffers.normal, buffers.hit_mask,
                 right, up, forward, camera.fov, camera.aspect_ratio)

  return SimpleNamespace(
    albedo=buffers.albedo.to_numpy(),
    normal=buffers.normal.to_numpy(),
    object_id=buffers.object_id.to_numpy(),
    hit_mask=buffers.hit_mask.to_numpy(),
    facing=facing.to_numpy(),
    camera=camera, basis=(right, up, forward), render=render,
  )


def test_hit_mask_matches_object_id(frame):
  assert np.array_equal(frame.object_id == -1, frame.hit_mask == 0)


def test_normals_unit_length(frame):
  hit = frame.hit_mask == 1
  lengths = np.linalg.norm(frame.normal[hit], axis=-1)
  assert np.allclose(lengths, 1.0, atol=1e-5), f"worst deviation {np.abs(lengths - 1).max():.2e}"


def test_normals_are_viewer_facing(frame):
  """Every stored normal must face the camera.

  A back-facing stored normal is what collapses SVGF's edge weights and makes
  it refuse to blend across a surface that is in fact continuous.
  """
  hit = frame.hit_mask == 1
  assert (frame.facing[hit] > 0.0).all(), \
    f"{int((frame.facing[hit] <= 0).sum())} back-facing normals"


def test_albedo_in_range(frame):
  hit = frame.hit_mask == 1
  assert frame.albedo[hit].min() >= 0.0
  assert frame.albedo[hit].max() <= 1.0


def test_object_ids_are_known(frame):
  assert set(np.unique(frame.object_id)).issubset({-1, 0, 1, 2})


def test_background_channels_untouched(frame):
  """The miss branch deliberately does not write albedo or normal, so these
  must still hold whatever clear_aovs() set. Catches a stale buffer."""
  bg = frame.hit_mask == 0
  assert np.all(frame.albedo[bg] == 0.0)
  assert np.all(frame.normal[bg] == 0.0)


def test_downward_facing_surfaces_share_a_normal(frame):
  """Light quad (id 1) and occluder underside (id 2) reach (0,-1,0) by
  different routes -- the light's geometric normal already points down and
  does not flip, the occluder's points up and does. Same result, so the two
  code paths must agree bit for bit."""
  light = frame.normal[frame.object_id == 1]
  occl = frame.normal[frame.object_id == 2]
  assert len(light) and len(occl)
  assert np.array_equal(np.unique(light, axis=0), np.unique(occl, axis=0))


def test_deterministic(frame):
  """Zero ti.random calls anywhere in the G-buffer path, so re-rendering must
  be bit-identical. Anything else means a bug or an unzeroed buffer."""
  before = (frame.albedo.copy(), frame.normal.copy(),
            frame.object_id.copy(), frame.hit_mask.copy())
  frame.render()
  after = (buffers.albedo.to_numpy(), buffers.normal.to_numpy(),
           buffers.object_id.to_numpy(), buffers.hit_mask.to_numpy())
  for b, a in zip(before, after):
    assert np.array_equal(b, a)


def test_silhouette_matches_path_tracer():
  """Silhouette cross-check between the G-buffer and the path tracer.

  Furnace scene (no emitters) + constant environment L=1 + MAX_BOUNCES=1 +
  NEE off gives a binary image: a ray that misses picks up exactly 1.0 from
  the environment, a ray that hits runs one loop iteration with no emission
  and no NEE and contributes exactly 0.0.

  The two kernels sample DIFFERENT sub-pixel positions by design -- the
  G-buffer at pixel centre, accumulate_kernel with ti.random jitter -- so on
  a silhouette edge they can legitimately land on opposite sides of the
  boundary. The meaningful assertion is therefore that disagreement is
  confined to boundary pixels; any interior disagreement means the two
  kernels genuinely generate different rays.
  """
  build_furnace_scene()
  camera = Camera(
    position=vec3(0.0, 1.2, 4.0), yaw=0.0, pitch=math.radians(5),
    fov=math.radians(60.0), aspect_ratio=W / H, near=0.1, far=1000.0,
  )
  right, up, forward = camera.basis_from_yaw_pitch()

  buffers.clear_aovs()
  gbuffer_kernel(
    buffers.albedo, buffers.normal, buffers.object_id, buffers.hit_mask,
    right, up, forward, camera.position, camera.fov, camera.aspect_ratio,
    scene.triangles,
    builder.bvh_node_min, builder.bvh_node_max,
    builder.bvh_node_left, builder.bvh_node_right,
    builder.bvh_node_start, builder.bvh_node_count, builder.bvh_indices,
  )

  accum = ti.Vector.field(3, ti.f32, shape=(W, H))
  accum.fill(0.0)
  accumulate_kernel(
    accum, 1, right, up, forward, camera.position,
    camera.fov, camera.aspect_ratio,
    BRDF(), Environment(mode=ENV_CONSTANT, constant=vec3(1.0)),
    scene.triangles, scene.num_triangles[None],
    builder.bvh_node_min, builder.bvh_node_max,
    builder.bvh_node_left, builder.bvh_node_right,
    builder.bvh_node_start, builder.bvh_node_count, builder.bvh_indices,
    scene.light_triangle_index, scene.light_pdf_area, scene.num_lights[None],
    1, 0, 1, 1,   # single_sided, use_nee=0, max_bounces=1, spp=1
  )

  radiance = accum.to_numpy()[..., 0]
  hit_mask = buffers.hit_mask.to_numpy()

  disagree = (radiance == 1.0) != (hit_mask == 0)

  m = hit_mask == 1
  boundary = np.zeros_like(m)
  edge_x = m[:-1, :] != m[1:, :]
  edge_y = m[:, :-1] != m[:, 1:]
  boundary[:-1, :] |= edge_x
  boundary[1:, :] |= edge_x
  boundary[:, :-1] |= edge_y
  boundary[:, 1:] |= edge_y

  interior_disagree = int((disagree & ~boundary).sum())
  assert interior_disagree == 0, (
    f"{interior_disagree} interior pixels disagree "
    f"({int(disagree.sum())} total, rest are silhouette edges)"
  )

  build_test_room()   # restore, so later modules see the usual scene