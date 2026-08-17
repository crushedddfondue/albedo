import math

import numpy as np
import pytest
import taichi as ti
from taichi.math import vec3

from tracer import buffers
from tracer.bvh import builder
from tracer.camera import Camera
from tracer.geometry import scene
from tracer.geometry.mock_scenes import build_test_room
from tracer.kernels.gbuffer import gbuffer_kernel
from tracer.kernels.motion import motion_kernel

W, H = 320, 180


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

@ti.kernel
def _fill_ray_dirs(d: ti.template(), r: vec3, u: vec3, fw: vec3, fov: ti.f32, ar: ti.f32):  # type: ignore
  for px, py in d:
    d[px, py] = Camera.ray_direction_for_pixel(px, py, 0.5, 0.5, d.shape[0], d.shape[1], r, u, fw, fov, ar)


def _ray_dirs(cam):
  """Per-pixel primary ray directions at pixel centre, (W, H, 3) float64.

  Calls the same ray_direction_for_pixel that gbuffer_kernel uses, so
  anything built on these also checks that the shared function stayed shared.

  FOLLOW-UP: test_gbuffer.py has a near-identical kernel. Second copy --
  hoist both into a shared helper.
  """
  f = ti.Vector.field(3, ti.f32, shape=(W, H))
  right, up, forward = cam.basis_from_yaw_pitch()
  _fill_ray_dirs(f, right, up, forward, cam.fov, cam.aspect_ratio)
  return f.to_numpy().astype(np.float64)


def _make_camera(position, yaw):
  return Camera(
    position=position, yaw=yaw, pitch=math.radians(5),
    fov=math.radians(60.0), aspect_ratio=W / H, near=0.1, far=1000.0,
  )


def _render_gbuffer(cam):
  right, up, forward = cam.basis_from_yaw_pitch()
  buffers.clear_aovs()
  gbuffer_kernel(
    buffers.albedo, buffers.normal, buffers.object_id,
    buffers.hit_mask, buffers.depth,
    right, up, forward, cam.position, cam.fov, cam.aspect_ratio,
    scene.triangles,
    builder.bvh_node_min, builder.bvh_node_max,
    builder.bvh_node_left, builder.bvh_node_right,
    builder.bvh_node_start, builder.bvh_node_count, builder.bvh_indices,
  )
  return {
    "hit_mask": buffers.hit_mask.to_numpy(),
    "depth": buffers.depth.to_numpy().astype(np.float64),
    "object_id": buffers.object_id.to_numpy(),
    "normal": buffers.normal.to_numpy().astype(np.float64),
  }


def _run_motion(cam_current, cam_prev):
  """Current camera's geometry, previous camera's matrices.

  The asymmetry is the whole idea: the current camera defines WHICH world
  point each pixel sees, the previous camera answers WHERE it was.
  """
  right, up, forward = cam_current.basis_from_yaw_pitch()
  motion_kernel(
    buffers.motion, buffers.depth, buffers.hit_mask,
    right, up, forward, cam_current.position,
    cam_current.fov, cam_current.aspect_ratio,
    np.asarray(cam_prev.view_matrix(), dtype=np.float32),
    np.asarray(cam_prev.projection_matrix(), dtype=np.float32),
  )
  return buffers.motion.to_numpy().astype(np.float64)


def _world_positions(depth, dirs, cam):
  """depth + ray directions -> world positions, (W, H, 3).

  Same reconstruction as test_depth_reconstructs_known_geometry, which is
  what makes it trustworthy here.
  """
  _, _, forward = cam.basis_from_yaw_pitch()
  forward = np.asarray(forward, np.float64)
  origin = np.asarray(cam.position, np.float64)
  t = depth / (dirs @ forward)
  return origin + t[..., None] * dirs


# ---------------------------------------------------------------------------
# FIXTURE
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_frames():
  """Static scene, camera moves slightly between A and B.

  Movement is deliberately small -- a little translation plus two degrees of
  yaw. Move further and most of frame B is disoccluded, the test discards it,
  and you end up asserting on almost nothing.
  """
  scene.init_scene_fields()
  build_test_room()
  buffers.init_aov_fields(W, H)

  cam_a = _make_camera(vec3(0.0, 1.0, 3.0), yaw=0.0)
  cam_b = _make_camera(vec3(0.1, 1.0, 3.0), yaw=math.radians(-2.0))

  frame_a = _render_gbuffer(cam_a)

  # B rendered second, so buffers.* now holds B's geometry -- which is what
  # motion_kernel needs.
  frame_b = _render_gbuffer(cam_b)
  motion = _run_motion(cam_b, cam_a)

  return {
    "a": frame_a, "b": frame_b, "motion": motion,
    "cam_a": cam_a, "cam_b": cam_b,
    "dirs_a": _ray_dirs(cam_a), "dirs_b": _ray_dirs(cam_b),
  }


# ---------------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------------

def test_static_camera_gives_zero_motion():
  """Same camera twice. Motion must be zero to within float round-trip error.

  Isolates the reconstruct-and-project path from the two-camera logic.

  ⚠ NOT exactly zero, and it cannot be. gbuffer_kernel stores
  depth = t * cos(angle from view axis); motion_kernel recovers
  t' = depth / cos and rebuilds the position from it. A multiply followed by
  a divide by the same f32 does not round-trip -- about an ulp is lost, which
  at depth ~5 is ~5e-7 world units, which at ~31 px per world unit is ~1.6e-5
  pixels. Residuals are largest near the frame edges, where 1/cos is largest.

  Any real bug -- wrong matrix order, wrong ndc convention, a half-pixel
  offset -- lands at 0.1 px or more, three orders of magnitude above this.
  """
  scene.init_scene_fields()
  build_test_room()
  buffers.init_aov_fields(W, H)

  cam = _make_camera(vec3(0.0, 1.0, 3.0), yaw=0.0)
  _render_gbuffer(cam)
  motion = _run_motion(cam, cam)

  worst = float(np.abs(motion).max())
  assert worst < 1e-3, f"max |motion| = {worst:.3e} px, expected float noise (~1e-5)"


def test_motion_vectors_reproject_to_the_same_world_point(two_frames):
  """Follow a pixel's motion vector back into frame A and land on the same
  physical point in space.

  Type 3 oracle -- metamorphic. Two independent routes to one world point
  must agree.
  """
  a, b = two_frames["a"], two_frames["b"]
  motion = two_frames["motion"]
  cam_a, cam_b = two_frames["cam_a"], two_frames["cam_b"]
  dirs_a, dirs_b = two_frames["dirs_a"], two_frames["dirs_b"]

  # Pixel-centre grid, matching the convention motion_kernel was written to:
  #   prev_pixel = current_pixel + motion
  px = np.arange(W, dtype=np.float64)[:, None] + 0.5
  py = np.arange(H, dtype=np.float64)[None, :] + 0.5

  prev_x = px + motion[..., 0]
  prev_y = py + motion[..., 1]

  ix = np.floor(prev_x).astype(int)
  iy = np.floor(prev_y).astype(int)

  # in_bounds computed BEFORE clipping. Clipping folds off-frame lookups onto
  # the border, where they would silently compare against an unrelated surface.
  in_bounds = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
  ixc = np.clip(ix, 0, W - 1)
  iyc = np.clip(iy, 0, H - 1)

  # ⚠ These three conditions ARE history rejection -- SVGF's temporal
  # validity check, arriving a phase early.
  #   1. B has geometry here at all
  #   2. the lookup landed inside frame A
  #   3. A had geometry there, and it was the SAME object
  # Condition 3 is why object_id exists: matching depth alone cannot separate
  # "same surface" from "different surface at coincidentally similar range".
  valid = (
    (b["hit_mask"] == 1)
    & in_bounds
    & (a["hit_mask"][ixc, iyc] == 1)
    & (a["object_id"][ixc, iyc] == b["object_id"])
  )

  pos_b = _world_positions(b["depth"], dirs_b, cam_b)
  pos_a = _world_positions(a["depth"][ixc, iyc], dirs_a[ixc, iyc], cam_a)

  err = np.linalg.norm(pos_a - pos_b, axis=-1)

  # ERROR BUDGET, written out rather than guessed -- getting this wrong is
  # what made the first two versions of this assertion meaningless.
  #
  # Nearest-neighbour lookup snaps to a pixel centre up to ~0.71 px away.
  # That screen-space error becomes world-space error via two factors:
  #
  #   1. Pixel footprint perpendicular to the view: 2*tan(fov/2)*depth/H.
  #      Grows linearly with distance -- ~0.03 at depth 5, ~0.06 at depth 9.
  #   2. Surface slant: snapping moves along the SURFACE, so on tilted
  #      geometry one pixel covers footprint / |n·d| world units. The floor
  #      near the horizon runs |n·d| ~ 0.13, a 7.7x amplification.
  #
  # Omitting (2) makes the tolerance far too tight on grazing geometry and
  # far too loose head-on. Dividing by the full footprint makes the error
  # scale-free, so one threshold means the same thing everywhere.
  footprint = 2.0 * math.tan(cam_b.fov / 2.0) * b["depth"] / H
  slant = np.abs((b["normal"] * dirs_b).sum(axis=-1))
  footprint_along_surface = footprint / np.maximum(slant, 0.05)

  rel = err[valid] / footprint_along_surface[valid]

  survived = valid.sum() / max((b["hit_mask"] == 1).sum(), 1)

  # Without this, a bug that rejects every pixel would pass trivially.
  assert survived > 0.6, f"only {survived:.1%} of pixels survived rejection"

  assert rel.max() < 2.0, (
    f"worst reprojection error {rel.max():.2f} pixel-footprints "
    f"({err[valid].max():.4f} world units), {survived:.1%} survived"
  )