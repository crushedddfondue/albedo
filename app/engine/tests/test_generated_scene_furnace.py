import math

import numpy as np
import pytest
import taichi as ti
from taichi.math import vec3

from tracer.bvh import upload as bvh_upload
from tracer.camera import Camera
from tracer.environment import Environment, ENV_CONSTANT
from tracer.geometry import scene
from tracer.geometry.scene_generator import SceneParams, SceneData, build_scene, upload_scene
from tracer.kernels.gbuffer import gbuffer_kernel
from tracer.kernels.render import accumulate_kernel
from tracer.sampling.brdf import BRDF
from tracer.trajectory import TrajectoryParams, sample_trajectory

W, H = 96, 54
FOV = math.radians(60.0)
ASPECT = W / H

FURNACE_L = 0.5
FURNACE_SPP = 24
FURNACE_LADDER = (16, 64, 128)

SEEDS = (20260819, 20260823, 20260831)

# The environment must be reachable for the identity to hold, so the ceiling
# coin flip is removed rather than left to chance. Walls, floor and boxes are
# still generated geometry -- this narrows the distribution, it does not
# replace it with a toy.
FURNACE_PARAMS = SceneParams(ceiling_probability=0.0)
SEALED_PARAMS = SceneParams(ceiling_probability=1.0)
STATE_PARAMS = SceneParams()

# Backstop only. The convergence and sign assertions below carry the real
# content; this is here so a gross regression cannot hide behind them.
# ⚠ Tighten to what tools/furnace_ladder.py actually reports at 128 bounces.
FURNACE_TOL_REL = 5e-3

# Derived from measurement, not chosen: seed 20260823 sealed returned a mean
# of 0.000025 against L = 0.5, i.e. 5.0e-5 relative, from rays slipping
# between coincident quad edges at TRIANGLE_DET_EPSILON. 1e-3 leaves 20x
# headroom while still catching a genuine hole in the geometry.
LEAK_TOL_REL = 2e-3

_fields = {}


def _alloc():
  """One allocation for the whole module. Same idempotency contract as
  buffers.init_aov_fields, for the same reason: kernels bind permanently."""
  if not _fields:
    _fields["accum"] = ti.Vector.field(3, ti.f32, shape=(W, H))
    _fields["albedo"] = ti.Vector.field(3, ti.f32, shape=(W, H))
    _fields["normal"] = ti.Vector.field(3, ti.f32, shape=(W, H))
    _fields["object_id"] = ti.field(ti.i32, shape=(W, H))
    _fields["hit_mask"] = ti.field(ti.i32, shape=(W, H))
    _fields["depth"] = ti.field(ti.f32, shape=(W, H))
  return _fields


def _camera_for(data: SceneData) -> Camera:
  """First pose of the scene's own trajectory -- interior by construction, so
  the measurement is taken from somewhere the camera can actually be, rather
  than a hardcoded point that might sit inside a box."""
  poses, _ = sample_trajectory(data.seed * 1000, 8, data.bounds, TrajectoryParams(), obstacles=data.occluders)
  p = poses[0]
  return Camera(position=np.asarray(p.position, dtype=np.float32),
                yaw=p.yaw, pitch=p.pitch, fov=FOV, aspect_ratio=ASPECT,
                near=0.1, far=1000.0)


def _make_furnace(data: SceneData) -> SceneData:
  """Unit albedo, no emission, no lights.

  Geometry untouched -- only materials move, exactly as
  mock_scenes._make_furnace does. Unit albedo makes the analytic answer
  exactly the environment radiance, and emptying the light list leaves the
  environment as the only possible source, which is the whole point.

  ⚠ Unit albedo also means throughput never decays, so Russian roulette never
  fires: q = min(1, max(beta)) stays at 1. Every path runs to max_bounces
  unless it escapes. That makes the furnace the strictest possible truncation
  test and the ONLY instrument that can see an RR bug.
  """
  n = data.n_triangles
  return SceneData(
    v0=data.v0, v1=data.v1, v2=data.v2,
    albedo=np.ones((n, 3), dtype=np.float32),
    emission=np.zeros((n, 3), dtype=np.float32),
    object_id=data.object_id,
    light_index=np.full(n, -1, dtype=np.int32),
    light_triangle_index=np.zeros(0, dtype=np.int32),
    light_pdf_area=np.zeros(0, dtype=np.float32),
    room_size=data.room_size, occluders=data.occluders, 
    scene_id=data.scene_id + "_furnace",
    seed=data.seed, meta=dict(data.meta),
  )


def _render_furnace(camera: Camera, bounces: int):
  f = _alloc()
  right, up, forward = camera.basis_from_yaw_pitch()
  f["accum"].fill(0.0)
  accumulate_kernel(
    f["accum"], 1, right, up, forward, camera.position, camera.fov, camera.aspect_ratio,
    BRDF(), Environment(mode=ENV_CONSTANT, constant=vec3(FURNACE_L)),
    scene.triangles, scene.num_triangles[None],
    *bvh_upload.kernel_args(),
    scene.light_triangle_index, scene.light_pdf_area, scene.num_lights[None],
    1, 0, bounces, FURNACE_SPP,        # single_sided, use_nee = 0
  )
  img = f["accum"].to_numpy().astype(np.float64)
  return img, img.mean(), img.std(ddof=1) / math.sqrt(img.size)


def _render_gbuffer(camera: Camera):
  f = _alloc()
  right, up, forward = camera.basis_from_yaw_pitch()
  gbuffer_kernel(
    f["albedo"], f["normal"], f["object_id"], f["hit_mask"], f["depth"],
    right, up, forward, camera.position, camera.fov, camera.aspect_ratio,
    scene.triangles,
    *bvh_upload.kernel_args(),
  )
  return {k: f[k].to_numpy() for k in ("albedo", "normal", "object_id", "hit_mask", "depth")}


def _prepare():
  scene.init_scene_fields()
  bvh_upload.init_bvh_fields()


# =============================================================================
# 1. White furnace, where the identity is actually defined
# =============================================================================

@pytest.mark.parametrize("seed", SEEDS)
def test_white_furnace_converges_on_open_scene(seed):
  """Open room, unit albedo, constant environment L. The estimate must
  CONVERGE to L as the bounce limit rises.

  Written as convergence rather than a single value because the single-value
  form encodes a truncation residual nobody has derived. Three claims, in
  order of how much they can catch:

    1. The bias is never significantly POSITIVE. Truncation can only discard
       energy. A furnace reading above L means energy is being created, which
       is a far more serious defect than any deficit.
    2. |bias| is non-increasing along the ladder. A deficit that does not
       shrink with more bounces is a leak, not truncation.
    3. The sequence has settled -- the top two rungs agree within their
       combined noise -- and the settled value is L.

  Only the last needs an absolute tolerance, and by then it is a backstop
  rather than the substance of the test.
  """
  _prepare()
  data = build_scene(FURNACE_PARAMS, seed)
  assert not data.meta["has_ceiling"], "furnace scene must be open to the environment"

  upload_scene(_make_furnace(data))
  assert scene.num_lights[None] == 0, "furnace must have no emitters"

  camera = _camera_for(data)
  rows = []
  for bounces in FURNACE_LADDER:
    img, mean, sem = _render_furnace(camera, bounces)
    assert np.isfinite(img).all(), f"{(~np.isfinite(img)).sum()} non-finite pixels"
    rows.append((bounces, mean, sem, img.min(), img.max()))
    print(f"\nseed {seed} bounces {bounces:4d}: mean {mean:.6f}  "
          f"bias {mean - FURNACE_L:+.6f}  sem {sem:.6f}  "
          f"min {img.min():.4f}  max {img.max():.4f}")

  # 1. no energy creation, at any rung
  for bounces, mean, sem, _, _ in rows:
    assert mean - FURNACE_L < 4.0 * sem, (
      f"furnace OVERSHOOTS at {bounces} bounces: {mean:.6f} > {FURNACE_L}. "
      f"Truncation cannot add energy; this is a transport bug."
    )

  # 2. the deficit shrinks with more bounces
  for (b0, m0, s0, _, _), (b1, m1, s1, _, _) in zip(rows, rows[1:]):
    d0, d1 = abs(m0 - FURNACE_L), abs(m1 - FURNACE_L)
    assert d1 <= d0 + 3.0 * (s0 + s1), (
      f"deficit grew from {d0:.6f} at {b0} bounces to {d1:.6f} at {b1}. "
      f"A deficit that does not shrink is a leak, not truncation."
    )

  # 3. converged, and converged to L
  (_, m_prev, s_prev, _, _), (b_top, m_top, s_top, _, _) = rows[-2], rows[-1]
  assert abs(m_top - m_prev) < 4.0 * (s_prev + s_top), (
    f"not converged at {b_top} bounces: {m_prev:.6f} -> {m_top:.6f}. "
    f"Extend FURNACE_LADDER before reading anything into the value."
  )

  bias = m_top - FURNACE_L
  tol = max(4.0 * s_top, FURNACE_TOL_REL * FURNACE_L)
  assert abs(bias) < tol, (
    f"converged furnace returned {m_top:.6f}, expected {FURNACE_L}. "
    f"bias {bias:+.6f} against tolerance {tol:.6f} (4*sem = {4 * s_top:.6f})"
  )

  # A single runaway pixel is invisible in the mean and fatal in a dataset.
  assert rows[-1][4] < 4.0 * FURNACE_L, f"firefly: max {rows[-1][4]:.3f}"


# =============================================================================
# 2. Sealed room must be dark -- an instrument test_room cannot provide
# =============================================================================

@pytest.mark.parametrize("seed", SEEDS)
def test_sealed_room_is_dark(seed):
  """A closed room with unit albedo and no emitters has no light source.

  No ray escapes, so the environment is never sampled and the true radiance
  is exactly 0. Anything above the leak tolerance means light is entering
  through the geometry -- a hole at a wall seam, a quad wound so a face is
  missing, or rays slipping between coincident edges.

  This is the check open geometry structurally cannot perform, and it is the
  reason the original failure on seed 20260823 was worth keeping rather than
  deleting.
  """
  _prepare()
  data = build_scene(SEALED_PARAMS, seed)
  assert data.meta["has_ceiling"], "sealed scene must have a ceiling"

  upload_scene(_make_furnace(data))
  assert scene.num_lights[None] == 0

  camera = _camera_for(data)
  img, mean, _ = _render_furnace(camera, FURNACE_LADDER[-1])

  leak_rel = mean / FURNACE_L
  print(f"\nseed {seed} sealed: mean {mean:.8f}  leak {leak_rel:.2e} of L  "
        f"max {img.max():.5f}")

  assert np.isfinite(img).all()
  assert leak_rel < LEAK_TOL_REL, (
    f"sealed room leaks {leak_rel:.2e} of L (mean {mean:.8f}). "
    f"With no emitters and no way out the answer must be 0."
  )


# =============================================================================
# 3. Scene upload leaves no residue
# =============================================================================

def test_scene_upload_is_stateless():
  """Upload A, then B, then A again. A's G-buffer must be bit-identical.

  This is the test the fixed-capacity BVH pool exists to pass. Refilling in
  place rather than reallocating is what stops kernels recompiling per scene,
  and the risk it introduces is stale entries surviving from the previous
  scene -- nodes past the new watermark, or triangles past num_triangles.

  Bit-identical, not approximately equal. gbuffer_kernel makes zero ti.random
  calls, so the primary-hit path is fully deterministic and there is no
  legitimate reason for a single bit to move.
  """
  _prepare()
  a = build_scene(STATE_PARAMS, SEEDS[0])
  b = build_scene(STATE_PARAMS, SEEDS[1])
  camera = _camera_for(a)

  upload_scene(a)
  first = _render_gbuffer(camera)

  upload_scene(b)
  _render_gbuffer(camera)          # pollute

  upload_scene(a)
  second = _render_gbuffer(camera)

  # hit_mask and object_id everywhere; the shading channels only where a
  # surface exists, because the miss branch deliberately leaves them
  # untouched and this test allocates no clear pass.
  assert np.array_equal(first["hit_mask"], second["hit_mask"]), "hit_mask changed"
  assert np.array_equal(first["object_id"], second["object_id"]), "object_id changed"

  hit = first["hit_mask"] == 1
  assert hit.any(), "camera sees no geometry; the test proves nothing"
  for name in ("albedo", "normal", "depth"):
    assert np.array_equal(first[name][hit], second[name][hit]), f"{name} changed"


# =============================================================================
# 4. What reached the GPU is what SceneData said
# =============================================================================

@pytest.mark.parametrize("seed", SEEDS)
def test_light_list_round_trips_to_gpu(seed):
  """upload_scene bypasses build_light_list and writes the list directly.
  That is a deliberate determinism decision, and it means nothing on the GPU
  side validates it -- so read it back and compare."""
  _prepare()
  data = build_scene(STATE_PARAMS, seed)
  upload_scene(data)

  assert scene.num_triangles[None] == data.n_triangles
  assert scene.num_lights[None] == data.n_lights

  n = data.n_lights
  assert np.array_equal(
    scene.light_triangle_index.to_numpy()[:n], data.light_triangle_index
  )
  assert np.allclose(scene.light_pdf_area.to_numpy()[:n], data.light_pdf_area, rtol=1e-6)

  # Normals are derived on the GPU by recompute_normals, so this also checks
  # the derivation ran at all -- an all-zero normal field means it did not.
  normals = scene.triangles.normal.to_numpy()[:data.n_triangles]
  lengths = np.linalg.norm(normals, axis=1)
  assert np.allclose(lengths, 1.0, atol=1e-5), f"lengths {lengths.min()}..{lengths.max()}"