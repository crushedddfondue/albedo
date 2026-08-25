"""Shard format tests, v2.

The round trip is the whole contract: what comes out must be what went in,
within the precision the dtype actually offers and not one bit less.

v2 changed three things and each gets its own test:

  hit_mask is no longer stored. It is still ACCEPTED and checked against
  object_id -- the invariant is worth keeping, the bytes were not -- and
  reconstructed on read so no consumer sees a different channel set.

  object_id narrowed to int16, with an explicit overflow guard rather than a
  silent wrap. A wrapped object id is a silent relabelling of geometry, which
  is exactly the signal SVGF and the model use to reject history.

  noisy carries N independent realisations, stored contiguously and returned
  as (W, H, N, 3).
"""

import os

import numpy as np
import pytest

from data.shard import (
  ShardWriter, ShardReader, BASE_SPEC, channel_spec, frame_bytes, F16_MAX,
  OBJECT_ID_MAX,
)

W, H = 16, 9


def _frame(rng, n_real=1, scale=1.0):
  # object_id and hit_mask must agree: -1 exactly where the mask is 0. Faking
  # them independently would make the writer's cross-check unfalsifiable.
  oid = rng.integers(-1, 8, size=(W, H)).astype(np.int16)
  noisy = (rng.random((W, H, n_real, 3)) * scale).astype(np.float32)
  return {
    "noisy": noisy if n_real > 1 else noisy[:, :, 0, :],
    "clean": (rng.random((W, H, 3)) * scale).astype(np.float32),
    "albedo": rng.random((W, H, 3)).astype(np.float32),
    "normal": (rng.random((W, H, 3)) * 2 - 1).astype(np.float32),
    "depth": (rng.random((W, H)) * 100).astype(np.float32),
    "motion": (rng.random((W, H, 2)) * 20 - 10).astype(np.float32),
    "object_id": oid,
    "hit_mask": (oid != -1).astype(np.uint8),
  }


def _write(path, frames, n_real=1):
  with ShardWriter(path, W, H, noisy_realizations=n_real) as w:
    w.begin_sequence("scene_a", 1, 2, meta={"k": "v"})
    for f in frames:
      w.write_frame(**dict(f))
    w.end_sequence()


@pytest.mark.parametrize("n_real", [1, 2, 4])
def test_round_trip_exact_within_dtype(tmp_path, n_real):
  rng = np.random.default_rng(0)
  frames = [_frame(rng, n_real) for _ in range(4)]

  path = str(tmp_path / f"s{n_real}")
  _write(path, frames, n_real)

  r = ShardReader(path)
  assert len(r) == 4
  assert r.noisy_realizations == n_real

  for i, f in enumerate(frames):
    got = r.read_frame(i)
    for name, dt, _c in BASE_SPEC:
      # The only loss permitted is the dtype cast, so the comparison is
      # against the input cast to that dtype -- not a tolerance pulled out
      # of the air.
      assert np.array_equal(got[name], f[name].astype(dt)), name

    want = f["noisy"] if n_real > 1 else f["noisy"][:, :, None, :]
    assert got["noisy"].shape == (W, H, n_real, 3)
    assert np.array_equal(got["noisy"], want.astype(np.float16))


def test_realizations_are_independent(tmp_path):
  """Two realisations of one frame must not be the same array. Guards against
  the stack being built from a field that was reused before it was copied."""
  rng = np.random.default_rng(1)
  f = _frame(rng, n_real=2)
  path = str(tmp_path / "ind")
  _write(path, [f], n_real=2)
  got = ShardReader(path).read_frame(0)["noisy"]
  assert not np.array_equal(got[:, :, 0, :], got[:, :, 1, :])


def test_hit_mask_is_derived_not_stored(tmp_path):
  rng = np.random.default_rng(2)
  f = _frame(rng)
  path = str(tmp_path / "hm")
  _write(path, [f])

  # Not on disk: the stride is exactly the spec without it.
  assert os.path.getsize(path + ".bin") == frame_bytes(W, H, channel_spec(1))
  assert "hit_mask" not in [c["name"] for c in ShardReader(path).sidecar["channels"]]

  # But present on read, and identical to what was handed in.
  got = ShardReader(path).read_frame(0)
  assert np.array_equal(got["hit_mask"], f["hit_mask"])
  assert np.array_equal(got["object_id"] == -1, got["hit_mask"] == 0)


def test_inconsistent_hit_mask_raises(tmp_path):
  """The cross-check frame_io performs, kept alive now that only one of the
  two channels survives to disk."""
  rng = np.random.default_rng(3)
  f = _frame(rng)
  f["hit_mask"] = 1 - f["hit_mask"]
  with ShardWriter(str(tmp_path / "bad"), W, H) as w:
    w.begin_sequence("scene_a", 1, 2)
    with pytest.raises(ValueError, match="must exactly match"):
      w.write_frame(**f)
    w.write_frame(**_frame(rng))
    w.end_sequence()


def test_object_id_overflow_raises(tmp_path):
  """int16 is a deliberate narrowing, so it needs a loud edge. A wrapped id
  silently relabels geometry, which is the one signal history rejection has."""
  rng = np.random.default_rng(4)
  f = _frame(rng)
  f["object_id"] = f["object_id"].astype(np.int32)
  f["object_id"][0, 0] = OBJECT_ID_MAX + 1
  f["hit_mask"] = (f["object_id"] != -1).astype(np.uint8)
  with ShardWriter(str(tmp_path / "ovf"), W, H) as w:
    w.begin_sequence("scene_a", 1, 2)
    with pytest.raises(ValueError, match="exceeds int16"):
      w.write_frame(**f)
    w.write_frame(**_frame(rng))
    w.end_sequence()


def test_frame_stride_is_exact(tmp_path):
  rng = np.random.default_rng(5)
  path = str(tmp_path / "stride")
  _write(path, [_frame(rng, 2)], n_real=2)
  assert os.path.getsize(path + ".bin") == frame_bytes(W, H, channel_spec(2))


def test_nan_in_radiance_raises(tmp_path):
  rng = np.random.default_rng(6)
  f = _frame(rng)
  f["clean"][0, 0, 0] = np.nan
  with ShardWriter(str(tmp_path / "nan"), W, H) as w:
    w.begin_sequence("scene_a", 1, 2)
    with pytest.raises(ValueError, match="non-finite"):
      w.write_frame(**f)
    w.write_frame(**_frame(rng))
    w.end_sequence()


def test_f16_overflow_is_counted_not_silent(tmp_path):
  """A value past the f16 ceiling is clamped -- but the clamp is recorded.
  Silent clamping is how a dataset acquires a defect nobody can date."""
  rng = np.random.default_rng(7)
  f = _frame(rng)
  f["noisy"][0, 0, 0] = F16_MAX * 10

  path = str(tmp_path / "clamp")
  _write(path, [f])
  assert ShardReader(path).sidecar["clamped_f16"]["noisy"] == 1


def test_depth_survives_the_background_sentinel(tmp_path):
  """BACKGROUND_DEPTH = 1e30 is +inf in float16. This test is the reason
  depth is stored as float32 and it should fail loudly if that changes."""
  rng = np.random.default_rng(8)
  f = _frame(rng)
  f["depth"][:] = 1e30

  path = str(tmp_path / "depth")
  _write(path, [f])
  got = ShardReader(path).read_frame(0)
  assert np.isfinite(got["depth"]).all()
  assert got["depth"].max() == pytest.approx(1e30, rel=1e-6)


def test_sequence_boundaries(tmp_path):
  rng = np.random.default_rng(9)
  path = str(tmp_path / "seqs")
  with ShardWriter(path, W, H) as w:
    for name, n in (("a", 3), ("b", 2)):
      w.begin_sequence(f"scene_{name}", 1, 2)
      for _ in range(n):
        w.write_frame(**_frame(rng))
      w.end_sequence()

  r = ShardReader(path)
  assert [s.count for s in r.sequences] == [3, 2]
  assert r.sequence_for_frame(0).scene_id == "scene_a"
  assert r.sequence_for_frame(4).scene_id == "scene_b"


def test_v1_shard_is_rejected(tmp_path):
  """A format break must be loud. Silently reading a v1 shard with a v2
  layout would mis-slice every channel after `noisy`."""
  import json
  path = str(tmp_path / "old")
  _write(path, [_frame(np.random.default_rng(10))])
  side = json.load(open(path + ".json"))
  side["format_version"] = 1
  json.dump(side, open(path + ".json", "w"))
  with pytest.raises(ValueError, match="format v1"):
    ShardReader(path)