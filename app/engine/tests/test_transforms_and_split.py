import numpy as np
import pytest

from data import transforms
from data.manifest import scene_split, effective_sample_size

H, W = 8, 12


def _basis():
  # A deliberately non-axis-aligned basis. An identity basis would make the
  # world/camera distinction invisible and the flip test would pass either way.
  yaw, pitch = 0.7, -0.2
  cp = np.cos(pitch)
  f = np.array([cp * np.sin(yaw), np.sin(pitch), -cp * np.cos(yaw)])
  f /= np.linalg.norm(f)
  r = np.cross(f, np.array([0.0, 1.0, 0.0])); r /= np.linalg.norm(r)
  u = np.cross(r, f); u /= np.linalg.norm(u)
  return r, u, f


def test_to_hwc_round_trip():
  a = np.arange(3 * 4 * 2).reshape(3, 4, 2)
  assert transforms.to_hwc(a).shape == (4, 3, 2)
  assert np.array_equal(transforms.to_hwc(transforms.to_hwc(a)), a)


def test_normal_space_round_trip():
  rng = np.random.default_rng(0)
  n = rng.normal(size=(H, W, 3)).astype(np.float32)
  n /= np.linalg.norm(n, axis=-1, keepdims=True)
  b = _basis()
  back = transforms.camera_to_world_normals(transforms.world_to_camera_normals(n, b), b)
  assert np.allclose(back, n, atol=1e-5)


def test_camera_space_preserves_length():
  rng = np.random.default_rng(1)
  n = rng.normal(size=(H, W, 3)).astype(np.float32)
  n /= np.linalg.norm(n, axis=-1, keepdims=True)
  c = transforms.world_to_camera_normals(n, _basis())
  assert np.allclose(np.linalg.norm(c, axis=-1), 1.0, atol=1e-5)


def test_flip_is_equivariant_for_a_view_dependent_quantity():
  """Mirror the frame, then shade -- versus shade, then mirror.

  The shading proxy is dot(n_cam, view) with view = camera-space -z, i.e. the
  facing ratio. Under a mirror of the IMAGE PLANE the facing ratio is
  invariant, so the two orders must agree exactly. In world space they do not:
  a world-space normal has no idea the image was mirrored.
  """
  rng = np.random.default_rng(2)
  n_world = rng.normal(size=(H, W, 3)).astype(np.float32)
  n_world /= np.linalg.norm(n_world, axis=-1, keepdims=True)
  basis = _basis()

  frame = transforms.to_model_space(
    {"normal": np.transpose(n_world, (1, 0, 2))}, basis
  )
  view = np.array([0.0, 0.0, -1.0], dtype=np.float32)

  shade_then_flip = (frame["normal"] @ view)[:, ::-1]
  flip_then_shade = transforms.flip_horizontal(frame)["normal"] @ view

  assert np.allclose(shade_then_flip, flip_then_shade, atol=1e-6)


def test_flip_negates_motion_x_only():
  rng = np.random.default_rng(3)
  m = rng.normal(size=(H, W, 2)).astype(np.float32)
  out = transforms.flip_horizontal({"motion": m})["motion"]
  assert np.allclose(out[..., 0], -m[:, ::-1, 0])
  assert np.allclose(out[..., 1], m[:, ::-1, 1])


def test_flip_is_an_involution():
  rng = np.random.default_rng(4)
  f = {
    "noisy": rng.random((H, W, 3)).astype(np.float32),
    "motion": rng.normal(size=(H, W, 2)).astype(np.float32),
    "normal": rng.normal(size=(H, W, 3)).astype(np.float32),
    "hit_mask": rng.integers(0, 2, (H, W)).astype(np.uint8),
  }
  twice = transforms.flip_horizontal(transforms.flip_horizontal(f))
  for k in f:
    assert np.array_equal(twice[k], f[k]), k


def test_crop_offset_is_shared():
  """Every channel of one frame must be cropped at the SAME offset, or the
  AOVs stop describing the pixels the radiance came from."""
  rng = np.random.default_rng(5)
  f = {"a": np.arange(H * W).reshape(H, W).astype(np.float32),
       "b": np.arange(H * W).reshape(H, W).astype(np.float32)}
  out, off = transforms.random_crop(f, (4, 4), rng)
  assert np.array_equal(out["a"], out["b"])
  out2, off2 = transforms.random_crop(f, (4, 4), rng, offset=off)
  assert off2 == off and np.array_equal(out2["a"], out["a"])


def test_split_is_deterministic_and_process_independent():
  ids = [f"scene_{i:08d}" for i in range(2000)]
  a = [scene_split(i, 0.1) for i in ids]
  b = [scene_split(i, 0.1) for i in ids]
  assert a == b
  # Not a claim about the exact count -- a claim that a hash of 2000 ids at
  # p=0.1 lands in a band no reasonable hash would miss.
  frac = a.count("val") / len(a)
  assert 0.07 < frac < 0.13, frac


def test_split_is_monotone_in_val_fraction():
  """Raising val_fraction may only ADD scenes to val. This is what lets a
  dataset be extended without invalidating an earlier training run."""
  ids = [f"scene_{i:08d}" for i in range(500)]
  small = {i for i in ids if scene_split(i, 0.1) == "val"}
  large = {i for i in ids if scene_split(i, 0.2) == "val"}
  assert small <= large


def test_effective_sample_size():
  assert effective_sample_size(100, 0.0) == pytest.approx(100.0)
  assert effective_sample_size(100, 0.9) == pytest.approx(100 * 0.1 / 1.9)
  with pytest.raises(ValueError):
    effective_sample_size(100, 1.0)


def test_every_split_path_uses_the_same_salt():
  """The defaults, not the behaviour.

  Two salt defaults ninety lines apart drifted to "albedo-p2.3" and
  "albedo-2.3". The budget print then reported 8 train / 0 val for the same
  eight scenes the manifest called 7/1, and the DataLoader used a different
  split again. Nothing raised, because each answer was internally consistent
  -- which is why this asserts on the signatures rather than on any value.
  """
  import inspect
  from data import manifest as M
  assert inspect.signature(M.scene_split).parameters["salt"].default == M.SPLIT_SALT
  for meth in (M.Manifest.split_summary, M.Manifest.sequences):
    d = inspect.signature(meth).parameters["salt"].default
    assert d in (None, M.SPLIT_SALT), (meth.__name__, d)