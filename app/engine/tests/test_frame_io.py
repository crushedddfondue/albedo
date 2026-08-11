import json

import numpy as np
import pytest

from tracer.io import frame_io

W, H = 16, 9


def _synthetic():
  rng = np.random.default_rng(0)
  hit_mask = (rng.random((W, H)) > 0.3).astype(np.uint8)
  object_id = np.where(hit_mask == 1, rng.integers(0, 3, (W, H)), -1).astype(np.int32)
  return {
    "radiance": rng.random((W, H, 3)).astype(np.float32) * 12.0,   # HDR, above 1
    "albedo": rng.random((W, H, 3)).astype(np.float32),
    "normal": rng.standard_normal((W, H, 3)).astype(np.float32),
    "object_id": object_id,
    "hit_mask": hit_mask,
  }


def test_round_trip_is_bit_identical(tmp_path):
  channels = _synthetic()
  path = frame_io.write_frame(str(tmp_path / "frame"), meta={"scene_id": "synthetic"}, **channels)
  out = frame_io.read_frame(path)
  for name, arr in channels.items():
    assert np.array_equal(arr, out[name]), f"{name} did not survive the round trip"


def test_metadata_survives(tmp_path):
  path = frame_io.write_frame(str(tmp_path / "f"), radiance=np.zeros((W, H, 3), np.float32),
                              meta={"scene_id": "abc", "render_config": {"spp": 7}})
  meta = frame_io.read_frame(path)["meta"]
  assert meta["scene_id"] == "abc"
  assert meta["render_config"]["spp"] == 7
  assert meta["format_version"] == frame_io.FORMAT_VERSION
  assert meta["conventions"]["layout"] == "WHC"
  assert meta["resolution"] == [W, H]


def test_extension_is_added_and_tolerated(tmp_path):
  """np.savez appends .npz when absent. Writing 'f' and then reading 'f'
  must still work, or every dump silently becomes unreadable."""
  a = np.zeros((W, H, 3), np.float32)
  frame_io.write_frame(str(tmp_path / "f"), radiance=a)
  assert frame_io.read_frame(str(tmp_path / "f"))["radiance"].shape == (W, H, 3)
  assert frame_io.read_frame(str(tmp_path / "f.npz"))["radiance"].shape == (W, H, 3)


def test_nan_in_radiance_raises(tmp_path):
  """NaN in a training set is poison. Catching it at write time beats
  discovering it as a loss going to nan three weeks in."""
  bad = np.zeros((W, H, 3), np.float32)
  bad[0, 0, 0] = np.nan
  with pytest.raises(ValueError, match="NaN or Inf"):
    frame_io.write_frame(str(tmp_path / "f"), radiance=bad)


def test_shape_mismatch_raises(tmp_path):
  with pytest.raises(ValueError, match="Shape mismatch"):
    frame_io.write_frame(
      str(tmp_path / "f"),
      radiance=np.zeros((W, H, 3), np.float32),
      albedo=np.zeros((W + 1, H, 3), np.float32),
    )


def test_mask_id_inconsistency_raises(tmp_path):
  hit_mask = np.ones((W, H), np.uint8)
  object_id = np.full((W, H), -1, np.int32)   # says background, mask says hit
  with pytest.raises(ValueError, match="object_id"):
    frame_io.write_frame(str(tmp_path / "f"), object_id=object_id, hit_mask=hit_mask)


def test_empty_frame_raises(tmp_path):
  with pytest.raises(ValueError, match="empty frame"):
    frame_io.write_frame(str(tmp_path / "f"))


def test_to_hwc(tmp_path):
  a = np.zeros((W, H, 3), np.float32)
  assert frame_io.to_hwc(a).shape == (H, W, 3)
  b = np.zeros((W, H), np.float32)
  assert frame_io.to_hwc(b).shape == (H, W)
  with pytest.raises(ValueError):
    frame_io.to_hwc(np.zeros((2, 2, 2, 2), np.float32))