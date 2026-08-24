"""The camera must never be inside solid geometry.

A camera inside an occluder sees that box's interior: a sealed volume with no
light in it. Every frame of the sequence renders black, at full clean-render
price, and NOTHING in the output looks wrong -- the AOVs are well formed, the
motion vectors are well formed, hit_mask is 1 almost everywhere. Only the
radiance is zero, and zero radiance is a perfectly legal value.

Found by the white furnace on seed 20260823, where all 32 poses sat inside
occluder 1 and every render -- furnace, sealed, emitter-lit -- came back at
~2e-5. This file is the tripwire that keeps it found.

Pure CPU. No Taichi, no GPU, sub-second. It has to be cheap enough to run
before every generation job.
"""

import numpy as np
import pytest

from tracer.geometry.scene_generator import SceneParams, build_scene
from tracer.trajectory import (
  TrajectoryParams, KINDS, sample_trajectory, invalid_poses, inside_index,
  as_obstacles,
)

PARAMS = SceneParams()
TRAJ = TrajectoryParams()

# Wide enough that a rare bad configuration shows up rather than hiding
# behind three hand-picked seeds -- which is exactly how the original bug
# survived: it was in the sample all along, on the seed nobody checked.
SWEEP = range(20260819, 20260819 + 200)

REGRESSION_SEED = 20260823   # all 32 poses inside occluder 1, before the fix


def _scene_and_poses(seed, n_frames=32, kind=None):
  data = build_scene(PARAMS, seed)
  poses, meta = sample_trajectory(
    data.seed * 1000, n_frames, data.bounds, TRAJ, kind=kind,
    obstacles=data.occluders,
  )
  return data, poses, meta


def test_regression_seed_20260823_is_no_longer_inside_a_box():
  """The seed that started this. Named explicitly so the failure is legible
  if it ever comes back, rather than surfacing as 'one of 200 seeds'."""
  data, poses, meta = _scene_and_poses(REGRESSION_SEED)
  assert data.occluders.shape[0] == 8
  bad = invalid_poses(poses, data.occluders, TRAJ.obstacle_margin)
  assert bad == [], (
    f"{len(bad)}/{len(poses)} poses inside geometry: {bad[:8]}. "
    f"This is the exact configuration that rendered 32 black frames."
  )
  assert meta["invalid_poses"] == 0
  assert meta["obstacles"] == 8


@pytest.mark.parametrize("seed", list(SWEEP)[:200])
def test_no_pose_is_inside_an_occluder(seed):
  data, poses, meta = _scene_and_poses(seed, n_frames=32)
  bad = invalid_poses(poses, data.occluders, TRAJ.obstacle_margin)
  assert bad == [], (
    f"seed {seed} ({meta['kind']}): {len(bad)}/32 poses inside geometry, "
    f"first at frame {bad[0] if bad else '-'}"
  )
  # The meta field is what generate_dataset reads to skip a scene, so it has
  # to agree with the direct check rather than merely coexist with it.
  assert meta["invalid_poses"] == 0, meta


@pytest.mark.parametrize("kind", KINDS)
def test_every_process_respects_obstacles(kind):
  """Forced per kind, because the mixture weights make `static` rare enough
  that a sweep can pass while one process is quietly broken."""
  failures = []
  for seed in list(SWEEP)[:60]:
    data, poses, meta = _scene_and_poses(seed, n_frames=24, kind=kind)
    bad = invalid_poses(poses, data.occluders, TRAJ.obstacle_margin)
    if bad:
      failures.append((seed, len(bad)))
  assert not failures, f"{kind}: {failures[:5]}"


def test_poses_stay_inside_the_room():
  """Pushing out of a box must not push through a wall. The two constraints
  are solved together, and it would be easy to satisfy one by breaking the
  other."""
  for seed in list(SWEEP)[:60]:
    data, poses, _ = _scene_and_poses(seed, n_frames=24)
    lo, hi = data.bounds
    pos = np.asarray([p.position for p in poses])
    assert np.all(pos[:, 0] >= lo[0] - 1e-9) and np.all(pos[:, 0] <= hi[0] + 1e-9), seed
    assert np.all(pos[:, 1] >= 0.0) and np.all(pos[:, 1] <= hi[1] + 1e-9), seed
    assert np.all(pos[:, 2] >= lo[2] - 1e-9) and np.all(pos[:, 2] <= hi[2] + 1e-9), seed


def test_margin_is_actually_enforced():
  """Not merely outside the box -- outside it by obstacle_margin.

  A camera flush against a face renders that face at the near plane and
  fills the frame with one flat surface, which is a legal frame and a
  worthless training sample.
  """
  for seed in list(SWEEP)[:60]:
    data, poses, _ = _scene_and_poses(seed, n_frames=24)
    obs = as_obstacles(data.occluders)
    if obs.shape[0] == 0:
      continue
    for i, p in enumerate(poses):
      # A slightly smaller margin must also find nothing, i.e. there is real
      # clearance rather than a hairline pass.
      assert inside_index(np.asarray(p.position), obs, TRAJ.obstacle_margin * 0.9) == -1, \
        f"seed {seed} frame {i} is within 0.9x the margin of a box"


def test_deterministic_in_seed():
  for seed in list(SWEEP)[:20]:
    data = build_scene(PARAMS, seed)
    a, ma = sample_trajectory(data.seed * 1000, 16, data.bounds, TRAJ,
                              obstacles=data.occluders)
    b, mb = sample_trajectory(data.seed * 1000, 16, data.bounds, TRAJ,
                              obstacles=data.occluders)
    assert ma == mb
    for x, y in zip(a, b):
      assert np.array_equal(x.position, y.position)
      assert x.yaw == y.yaw and x.pitch == y.pitch


def test_obstacles_change_the_trajectory():
  """Guards against the arguments being accepted and ignored -- a failure
  mode that would leave every test above passing vacuously."""
  differed = 0
  for seed in list(SWEEP)[:40]:
    data = build_scene(PARAMS, seed)
    with_obs, _ = sample_trajectory(data.seed * 1000, 16, data.bounds, TRAJ,
                                    obstacles=data.occluders)
    without, _ = sample_trajectory(data.seed * 1000, 16, data.bounds, TRAJ)
    if any(not np.allclose(a.position, b.position) for a, b in zip(with_obs, without)):
      differed += 1
  assert differed > 0, "obstacles had no effect on any trajectory"


def test_no_obstacles_is_still_valid():
  """An empty occluder set must not crash or degenerate -- mock_scenes and
  any hand-built scene will pass nothing at all."""
  data = build_scene(PARAMS, REGRESSION_SEED)
  poses, meta = sample_trajectory(data.seed * 1000, 8, data.bounds, TRAJ)
  assert len(poses) == 8
  assert meta["obstacles"] == 0
  assert meta["invalid_poses"] == 0