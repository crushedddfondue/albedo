import math

import numpy as np
import pytest
import taichi as ti
from taichi.math import vec3

from tracer import buffers
from tracer.camera import Camera
from tracer.geometry import scene
from tracer.geometry.mock_scenes import build_test_room
from tracer.bvh import builder
from tracer.kernels.gbuffer import gbuffer_kernel

W, H = 320, 180


@pytest.fixture(scope="module")
def cam():
  c = Camera(
    position=vec3(0.0, 1.2, 4.0), yaw=0.0, pitch=math.radians(5),
    fov=math.radians(60.0), aspect_ratio=W / H, near=0.1, far=1000.0,
  )
  right, up, forward = c.basis_from_yaw_pitch()
  return c, np.asarray(right, np.float64), np.asarray(up, np.float64), np.asarray(forward, np.float64)


@pytest.fixture(scope="module")
def gbuf(cam):
  """G-buffer plus per-pixel ray directions, for the round-trip test."""
  c, right, up, forward = cam
  scene.init_scene_fields()
  build_test_room()
  buffers.init_aov_fields(W, H)
  buffers.clear_aovs()

  gbuffer_kernel(
    buffers.albedo, buffers.normal, buffers.object_id,
    buffers.hit_mask, buffers.depth,
    right, up, forward, c.position, c.fov, c.aspect_ratio,
    scene.triangles,
    builder.bvh_node_min, builder.bvh_node_max,
    builder.bvh_node_left, builder.bvh_node_right,
    builder.bvh_node_start, builder.bvh_node_count, builder.bvh_indices,
  )

  dirs_field = ti.Vector.field(3, ti.f32, shape=(W, H))

  @ti.kernel
  def fill_dirs(d: ti.template(), r: vec3, u: vec3, f: vec3, fov: ti.f32, ar: ti.f32):  # type: ignore
    for px, py in d:
      d[px, py] = Camera.ray_direction_for_pixel(px, py, 0.5, 0.5, d.shape[0], d.shape[1], r, u, f, fov, ar)

  fill_dirs(dirs_field, right, up, forward, c.fov, c.aspect_ratio)

  return {
    "depth": buffers.depth.to_numpy(),
    "hit_mask": buffers.hit_mask.to_numpy(),
    "dirs": dirs_field.to_numpy().astype(np.float64),
  }


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def M(mat):
  """Taichi matrix or numpy -> float64 numpy (4,4)."""
  return np.asarray(mat, dtype=np.float64).reshape((4, 4))


def apply(mat, v3, w):
  """Apply a 4x4 to a 3-vector with the given homogeneous w. Returns len-4."""
  # Assuming standard right-multiplication convention: M @ v
  v4 = np.array([v3[0], v3[1], v3[2], w], dtype=np.float64)
  return M(mat) @ v4


# ---------------------------------------------------------------------------
# ROW 1
# ---------------------------------------------------------------------------

def test_view_maps_position_to_origin(cam):
  """V · (position, 1) -> (0,0,0)"""
  c, *_ = cam
  V = c.view_matrix()
  pos = np.asarray(c.position, dtype=np.float64)
  
  origin = apply(V, pos, 1.0)
  
  assert np.allclose(origin[:3], 0.0, atol=1e-5), \
    f"Camera position mapped to {origin[:3]}, expected (0,0,0)"


# ---------------------------------------------------------------------------
# ROW 2
# ---------------------------------------------------------------------------

def test_view_maps_basis_to_canonical_axes(cam):
  """forward -> (0,0,-1), right -> (1,0,0), up -> (0,1,0)"""
  c, right, up, forward = cam
  V = c.view_matrix()

  mapped_forward = apply(V, forward, 0.0)
  mapped_right = apply(V, right, 0.0)
  mapped_up = apply(V, up, 0.0)

  # Note: Ray generators often treat forward as looking down -Z in view space
  assert np.allclose(mapped_forward[:3], [0.0, 0.0, -1.0], atol=1e-5), f"Forward maps to {mapped_forward[:3]}"
  assert np.allclose(mapped_right[:3], [1.0, 0.0, 0.0], atol=1e-5), f"Right maps to {mapped_right[:3]}"
  assert np.allclose(mapped_up[:3], [0.0, 1.0, 0.0], atol=1e-5), f"Up maps to {mapped_up[:3]}"


# ---------------------------------------------------------------------------
# ROW 3
# ---------------------------------------------------------------------------

def test_view_rotation_is_orthonormal(cam):
  """Upper-left 3x3 of V satisfies R @ R.T == I"""
  c, *_ = cam
  V = M(c.view_matrix())
  R = V[:3, :3]
  
  identity = R @ R.T
  assert np.allclose(identity, np.eye(3), atol=1e-5), "View matrix rotation is not orthonormal"


# ---------------------------------------------------------------------------
# ROW 4
# ---------------------------------------------------------------------------

def test_projection_maps_near_and_far(cam):
  """Camera-space (0,0,-near) and (0,0,-far) hit the ndc_z bounds."""
  c, *_ = cam
  P = c.projection_matrix()

  near_clip = apply(P, [0.0, 0.0, -c.near], 1.0)
  far_clip = apply(P, [0.0, 0.0, -c.far], 1.0)

  near_ndc_z = near_clip[2] / near_clip[3]
  far_ndc_z = far_clip[2] / far_clip[3]

  # Far plane should always map to 1.0 using a slightly looser tolerance for precision
  assert np.isclose(far_ndc_z, 1.0, atol=1e-3), f"Far plane mapped to {far_ndc_z}, expected 1.0"
  
  # Near plane maps to -1.0 in standard OpenGL, or 0.0 in D3D/Vulkan. 
  is_gl = np.isclose(near_ndc_z, -1.0, atol=1e-5)
  is_d3d = np.isclose(near_ndc_z, 0.0, atol=1e-5)
  
  assert is_gl or is_d3d, f"Near plane mapped to {near_ndc_z}, expected -1.0 or 0.0"


# ---------------------------------------------------------------------------
# ROWS 5 & 6  -- these exist as a PAIR
# ---------------------------------------------------------------------------

def test_projection_right_edge_maps_to_ndc_x_one(cam):
  """Camera-space (d·tan(fov/2)·aspect, 0, -d) -> ndc_x = +1"""
  c, *_ = cam
  P = c.projection_matrix()
  
  for d in [1.0, 5.0, 100.0]:
    x = d * math.tan(c.fov / 2.0) * c.aspect_ratio
    clip = apply(P, [x, 0.0, -d], 1.0)
    ndc_x = clip[0] / clip[3]
    
    assert np.isclose(ndc_x, 1.0, atol=1e-5), f"Right edge failed at depth {d}, ndc_x={ndc_x}"


def test_projection_top_edge_maps_to_ndc_y_one(cam):
  """Camera-space (0, d·tan(fov/2), -d) -> ndc_y = +1"""
  c, *_ = cam
  P = c.projection_matrix()
  
  for d in [1.0, 5.0, 100.0]:
    y = d * math.tan(c.fov / 2.0)
    clip = apply(P, [0.0, y, -d], 1.0)
    ndc_y = clip[1] / clip[3]
    
    assert np.isclose(ndc_y, 1.0, atol=1e-5), f"Top edge failed at depth {d}, ndc_y={ndc_y}"


# ---------------------------------------------------------------------------
# ROW 7
# ---------------------------------------------------------------------------

def test_clip_w_is_positive_in_front(cam):
  """clip.w > 0 for every point in front of the camera."""
  c, *_ = cam
  P = c.projection_matrix()
  
  for d in [0.1, 1.0, 10.0, 100.0]:
    clip = apply(P, [0.0, 0.0, -d], 1.0)
    assert clip[3] > 0.0, f"clip.w is {clip[3]} at depth {d}"


# ---------------------------------------------------------------------------
# ROW 8 -- THE CONTRACT
# ---------------------------------------------------------------------------

def test_world_to_screen_round_trip(cam, gbuf):
  """G-buffer world positions project back onto their own pixel."""
  c, right, up, forward = cam
  
  V = M(c.view_matrix())
  P = M(c.projection_matrix())
  PV = P @ V  # Matrix multiplication ordering
  
  depth_map = gbuf["depth"]
  hit_mask = gbuf["hit_mask"]
  dirs = gbuf["dirs"]
  
  pos_cam = np.asarray(c.position, dtype=np.float64)

  for px in range(W):
    for py in range(H):
      if hit_mask[px, py]:
        d = depth_map[px, py]
        ray_dir = dirs[px, py]
        
        # Reconstruct hit point in world space
        # t = depth / dot(dir, forward)
        t = d / np.dot(ray_dir, forward)
        world_pos = pos_cam + t * ray_dir
        
        # Project back to clip space
        clip = apply(PV, world_pos, 1.0)
        assert clip[3] > 0.0, f"Round trip: clip.w is negative at pixel {px}, {py}"
        
        # Perspective divide to NDC
        ndc = clip[:3] / clip[3]
        
        # Map to Screen coordinates
        # lower-left origin assumed per hint 2
        screen_x = (ndc[0] + 1.0) * 0.5 * W
        screen_y = (ndc[1] + 1.0) * 0.5 * H
        
        # Validate against pixel centers
        assert np.isclose(screen_x, px + 0.5, atol=1e-2), \
          f"X mismatch at ({px}, {py}): projected to {screen_x:.2f}"
        assert np.isclose(screen_y, py + 0.5, atol=1e-2), \
          f"Y mismatch at ({px}, {py}): projected to {screen_y:.2f}"