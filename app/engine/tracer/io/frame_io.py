import os
import json
import numpy as np

# =============================================================================
# Constants & Conventions
# =============================================================================

FORMAT_VERSION = 1

CONVENTIONS = {
  "ndc_y": "lower-left",
  "normal_space": "world",
  "normal_facing": "viewer-facing",
  "depth_rep": "TBD",   # Update once the depth representation is decided
  "layout": "WHC",      # Native Taichi layout (Width, Height, Channels)
}

_FLOAT_CHANNELS = ("radiance", "albedo", "normal", "depth", "motion")
_INT_CHANNELS = {"object_id": np.int32, "hit_mask": np.uint8}


# =============================================================================
# Helpers
# =============================================================================

def to_hwc(arr: np.ndarray) -> np.ndarray:
  """Taichi's native WHC (or WH) -> standard HWC (or HW).

  Applied explicitly at the PyTorch boundary, never on write. Storing the
  native layout means a round trip is provably lossless and a bug in this
  transpose cannot hide inside the write path.
  """
  if arr.ndim == 3:
    return np.transpose(arr, (1, 0, 2))
  if arr.ndim == 2:
    return np.transpose(arr, (1, 0))
  raise ValueError(f"Expected 2D or 3D array, got {arr.ndim}D")


def _vec_to_list(v):
  """Taichi vector or matrix (Python scope) -> nested Python lists."""
  if hasattr(v, "to_numpy"):
    return np.asarray(v.to_numpy()).tolist()
  return np.asarray(v).tolist()



def write_frame(path: str, *, radiance=None, albedo=None, normal=None, depth=None, motion=None, object_id=None, hit_mask=None, meta=None):
  """Write one frame plus metadata to a single uncompressed .npz."""

  supplied = {
    "radiance": radiance, "albedo": albedo, "normal": normal,
    "depth": depth, "motion": motion,
    "object_id": object_id, "hit_mask": hit_mask,
  }

  arrays = {}
  for name, arr in supplied.items():
    if arr is None:
      continue
    dtype = _INT_CHANNELS.get(name, np.float32)
    arrays[name] = np.asarray(arr, dtype=dtype)

  if not arrays:
    raise ValueError("Cannot write an empty frame. Provide at least one channel.")

  # --- Validation. Fail loudly: a corrupt frame silently written is worse
  # than a crash, because it surfaces weeks later as a loss going to nan.

  base_shape = None
  for name, arr in arrays.items():
    shape_wh = arr.shape[:2]
    if base_shape is None:
      base_shape = shape_wh
    elif shape_wh != base_shape:
      raise ValueError(
        f"Shape mismatch: expected {base_shape}, channel '{name}' has {shape_wh}"
      )

  if "radiance" in arrays and not np.isfinite(arrays["radiance"]).all():
    n_bad = int((~np.isfinite(arrays["radiance"])).sum())
    raise ValueError(f"NaN or Inf in radiance: {n_bad} non-finite values")

  if "object_id" in arrays and "hit_mask" in arrays:
    if not np.array_equal(arrays["object_id"] == -1, arrays["hit_mask"] == 0):
      raise ValueError("object_id == -1 must exactly match hit_mask == 0")

  # --- Metadata. Conventions written last so a caller cannot clobber them.
  final_meta = dict(meta) if meta else {}
  final_meta["format_version"] = FORMAT_VERSION
  final_meta["conventions"] = CONVENTIONS
  final_meta["resolution"] = list(base_shape) # type: ignore

  # np.savez appends .npz when absent, which would make read_frame(path) miss.
  if not path.endswith(".npz"):
    path = path + ".npz"

  np.savez(path, meta=json.dumps(final_meta), **arrays)
  return path


def read_frame(path: str) -> dict:
  """Read a frame. Returns channels plus a parsed 'meta' dict."""
  if not path.endswith(".npz"):
    path = path + ".npz"
  if not os.path.exists(path):
    raise FileNotFoundError(f"Frame file not found: {path}")

  result = {}
  with np.load(path) as data:
    for key in data.files:
      if key == "meta":
        # .item() unambiguously returns the Python str from a 0-d array.
        result["meta"] = json.loads(data["meta"].item())
      else:
        result[key] = data[key]

  return result


# =============================================================================
# Ergonomic Writer
# =============================================================================

def dump_current(path: str, *, camera, accum=None, render_config=None,
                 scene_id="test_room", meta_extra=None):
  """Pull the current AOVs from `buffers`, assemble metadata, write.

  accum and render_config are passed in explicitly rather than sniffed off a
  module. Sniffing defaults silently on a wrong guess, and metadata that is
  confidently wrong is worse than metadata that is absent -- a dataset
  labelled with the wrong spp is worse than one with no label at all.
  """
  from tracer import buffers

  kwargs = {}
  if accum is not None:
    kwargs["radiance"] = accum.to_numpy()

  for name in ("albedo", "normal", "object_id", "hit_mask", "depth", "motion"):
    field = getattr(buffers, name)
    if field is not None:
      kwargs[name] = field.to_numpy()

  right, up, forward = camera.basis_from_yaw_pitch()

  # view_matrix / projection_matrix have never been executed (Track A).
  # Record the failure rather than substituting an identity matrix, which
  # would look like a valid transform to anything reading this later.
  matrices = {}
  for name in ("view_matrix", "projection_matrix"):
    try:
      matrices[name] = _vec_to_list(getattr(camera, name)())
    except Exception as exc:
      matrices[name] = None
      matrices[f"{name}_error"] = repr(exc)

  meta = {
    "scene_id": scene_id,
    "camera": {
      "position": _vec_to_list(camera.position),
      "right": _vec_to_list(right),
      "up": _vec_to_list(up),
      "forward": _vec_to_list(forward),
      "yaw": float(camera.yaw),
      "pitch": float(camera.pitch),
      "fov": float(camera.fov),
      "aspect_ratio": float(camera.aspect_ratio),
      "near": float(camera.near),
      "far": float(camera.far),
    },
    "matrices": matrices,
    "render_config": dict(render_config) if render_config else {},
  }

  if meta_extra:
    meta.update(meta_extra)

  return write_frame(path, meta=meta, **kwargs)