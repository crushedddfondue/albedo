from typing import Sequence, Dict

import numpy as np

SCALAR_CHANNELS = ("noisy", "clean", "albedo", "depth", "object_id", "hit_mask")

SIGNED_X_CHANNELS = ("normal", "motion")

def to_hwc(arr: np.ndarray) -> np.ndarray:
  if arr.ndim == 3:
    return np.transpose(arr, (1, 0, 2))
  if arr.ndim == 2:
    return np.transpose(arr, (1, 0))

  raise ValueError(f"expected 2D or 3D array, got {arr.ndim}D")

def world_to_camera_normals(normal_hwc: np.ndarray, basis: Sequence[np.ndarray]) -> np.ndarray:
  right, up, forward = (np.asarray(b, dtype=np.float32).reshape(3) for b in basis)
  m = np.stack([right, up, forward], axis = 1)
  return normal_hwc.astype(np.float32) @ m

def camera_to_world_normals(normal_hwc: np.ndarray, basis: Sequence[np.ndarray]) -> np.ndarray:
  right, up, forward = (np.asarray(b, dtype=np.float32).reshape(3) for b in basis)
  m = np.stack([right, up, forward], axis=1)
  return normal_hwc.astype(np.float32) @ m.T

def to_model_space(frame: Dict[str, np.ndarray], basis) -> Dict[str, np.ndarray]:
  out = {k: to_hwc(np.asarray(v)) for k, v in frame.items()}
  if "normal" in out:
    out["normal"] = world_to_camera_normals(out["normal"], basis)
  return out

def flip_horizontal(frame: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
  out = {}
  for name, arr in frame.items():
    a = np.ascontiguousarray(arr[:, ::-1])
    if name in SIGNED_X_CHANNELS:
      a = a.copy()
      a[..., 0] = -a[..., 0]
    out[name] = a
  return out

def random_crop(frame: Dict[str, np.ndarray], size, rng: np.random.Generator, offset=None):
  ch, cw = size
  ref = next(iter(frame.values()))
  h, w = ref.shape[:2]
  if ch > h or cw > w:
    raise ValueError(f"crop {size} does not fit in frame {(h, w)}")

  if offset is None:
    y = int(rng.integers(0, h - ch + 1))
    x = int(rng.integers(0, w - cw + 1))
  else:
    y, x = offset

  return {k: np.ascontiguousarray(v[y:y + ch, x:x + cw]) for k, v in frame.items()}, (y, x)

def stack_channels(frame: Dict[str, np.ndarray], order: Sequence[str]) -> np.ndarray:
  parts = []
  for name in order:
    a = np.asarray(frame[name])
    if a.ndim == 2:
      a = a[..., None]
    parts.append(a.astype(np.float32))
  return np.ascontiguousarray(np.concatenate(parts, axis=-1).transpose(2, 0, 1))