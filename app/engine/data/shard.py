import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

FORMAT_VERSION = 1

CONVENTIONS = {
  "ndc_y": "lower-left",
  "normal_space": "world",
  "normal_facing": "viewer-facing",
  "depth_rep": "view_space_z_linear",
  "layout": "WHC",
  "radiance": "linear_hdr_unclamped",
}

CHANNEL_SPEC = (
  ("noisy",     np.dtype(np.float16), 3),
  ("clean",     np.dtype(np.float16), 3),
  ("albedo",    np.dtype(np.float16), 3),
  ("normal",    np.dtype(np.float16), 3),
  ("depth",     np.dtype(np.float32), 1),
  ("motion",    np.dtype(np.float32), 2),
  ("object_id", np.dtype(np.int32),   1),
  ("hit_mask",  np.dtype(np.uint8),   1),
)

F16_MAX = 65504.0


def channel_bytes(width: int, height: int, spec=CHANNEL_SPEC) -> Dict[str, int]:
  return {name: width * height * c * dt.itemsize for name, dt, c in spec}


def frame_bytes(width: int, height: int, spec=CHANNEL_SPEC) -> int:
  return sum(channel_bytes(width, height, spec).values())

@dataclass
class SequenceRecord:
  scene_id: str
  scene_seed: int
  trajectory_seed: int
  start: int
  count: int
  meta: dict

  def to_json(self):
    return {
      "scene_id": self.scene_id,
      "scene_seed": self.scene_seed,
      "trajectory_seed": self.trajectory_seed,
      "start": self.start,
      "count": self.count,
      "meta": self.meta,
    }
  
  @staticmethod
  def from_json(d):
    return SequenceRecord(
      scene_id=d["scene_id"], scene_seed=d["scene_seed"],
      trajectory_seed=d["trajectory_seed"], start=d["start"],
      count=d["count"], meta=d.get("meta", {}),
    )


class ShardWriter:
  def __init__(self, path: str, width: int, height: int, spec=CHANNEL_SPEC, render_config: Optional[dict] = None):
    self.path = path if not path.endswith(".bin") else path[:-4]
    self.width = width
    self.height = height
    self.spec = spec
    self.render_config = render_config or {}
    self.frame_bytes = frame_bytes(width, height, spec)

    self.sequences: List[SequenceRecord] = []
    self.n_frames = 0
    self._open_seq: Optional[dict] = None

    self._clamped = {name: 0 for name, _, _ in spec}
    self._nonfinite = {name: 0 for name, _, _ in spec}

    os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
    self._fh = open(self.path + ".bin", "wb")

  def begin_sequence(self, scene_id: str, scene_seed: int, trajectory_seed: int, meta: Optional[dict] = None):
    if self._open_seq is not None:
      raise RuntimeError("begin_sequence called with a sequence still open")
    self._open_seq = {
      "scene_id": scene_id, "scene_seed": int(scene_seed),
      "trajectory_seed": int(trajectory_seed), "start": self.n_frames,
      "meta": dict(meta or {}),
    }

  def end_sequence(self):
    if self._open_seq is None:
      raise RuntimeError("end_sequence called with no sequence open")
    count = self.n_frames - self._open_seq["start"]
    if count == 0:
      raise RuntimeError(f"sequence {self._open_seq['scene_id']} wrote no frames")
    self.sequences.append(SequenceRecord(count=count, **self._open_seq))
    self._open_seq = None

  def write_frame(self, **channels):
    if self._open_seq is None:
      raise RuntimeError("write_frame outside a sequence")

    missing = [n for n, _, _ in self.spec if n not in channels]
    if missing:
      raise ValueError(f"missing channels: {missing}")

    for name, dtype, c in self.spec:
      arr = np.asarray(channels[name])

      want = (self.width, self.height) if c == 1 else (self.width, self.height, c)
      if arr.ndim == 2 and c == 1:
        pass
      elif arr.shape[:2] != (self.width, self.height):
        raise ValueError(
          f"channel '{name}': expected leading shape "
          f"{(self.width, self.height)} (WHC), got {arr.shape}"
        )
      arr = arr.reshape(want)

      if dtype.kind == "f":
        bad = ~np.isfinite(arr)
        n_bad = int(bad.sum())
        if n_bad:
          # Fail loudly on the target and the input; a NaN in radiance is a
          # tracer bug and burying it in the dataset defers the diagnosis by
          # however long training takes.
          if name in ("noisy", "clean"):
            raise ValueError(f"{n_bad} non-finite values in '{name}'")
          self._nonfinite[name] += n_bad
          arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if dtype == np.float16:
          over = int((np.abs(arr) > F16_MAX).sum())
          if over:
            self._clamped[name] += over
            arr = np.clip(arr, -F16_MAX, F16_MAX)

      self._fh.write(np.ascontiguousarray(arr, dtype=dtype).tobytes())

    self.n_frames += 1

  def close(self):
    if self._open_seq is not None:
      raise RuntimeError("close() with a sequence still open")
    self._fh.close()

    sidecar = {
      "format_version": FORMAT_VERSION,
      "conventions": CONVENTIONS,
      "resolution": [self.width, self.height],
      "channels": [
        {"name": n, "dtype": dt.name, "channels": c, "bytes": self.width * self.height * c * dt.itemsize}
        for n, dt, c in self.spec
      ],
      "frame_bytes": self.frame_bytes,
      "n_frames": self.n_frames,
      "sequences": [s.to_json() for s in self.sequences],
      "render_config": self.render_config,
      "clamped_f16": {k: v for k, v in self._clamped.items() if v},
      "nonfinite_zeroed": {k: v for k, v in self._nonfinite.items() if v},
    }
    with open(self.path + ".json", "w") as f:
      json.dump(sidecar, f, indent=2)

    return self.path

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    if exc[0] is None:
      self.close()
    else:
      self._fh.close()


class ShardReader:
  def __init__(self, path: str):
    self.path = path[:-4] if path.endswith(".bin") else path
    with open(self.path + ".json") as f:
      self.sidecar = json.load(f)

    if self.sidecar["format_version"] != FORMAT_VERSION:
      raise ValueError(
        f"shard {self.path} is format v{self.sidecar['format_version']}, "
        f"reader is v{FORMAT_VERSION}"
      )

    self.width, self.height = self.sidecar["resolution"]
    self.frame_bytes = self.sidecar["frame_bytes"]
    self.n_frames = self.sidecar["n_frames"]
    self.sequences = [SequenceRecord.from_json(s) for s in self.sidecar["sequences"]]

    self._layout = []
    offset = 0
    for ch in self.sidecar["channels"]:
      dt = np.dtype(ch["dtype"])
      c = ch["channels"]
      self._layout.append((ch["name"], dt, c, offset))
      offset += self.width * self.height * c * dt.itemsize
    if offset != self.frame_bytes:
      raise ValueError(f"channel layout sums to {offset}, sidecar says {self.frame_bytes}")

    expect = self.frame_bytes * self.n_frames
    actual = os.path.getsize(self.path + ".bin")
    if actual != expect:
      raise ValueError(
        f"shard {self.path}.bin is {actual} bytes, sidecar implies {expect}. "
        f"Truncated write, or a spec change without a version bump."
      )

    self._mm = np.memmap(self.path + ".bin", dtype=np.uint8, mode="r", shape=(expect,))

  def __len__(self):
    return self.n_frames

  def read_frame(self, index: int, channels=None, copy: bool = False) -> Dict[str, np.ndarray]:
    if not 0 <= index < self.n_frames:
      raise IndexError(f"frame {index} out of range [0, {self.n_frames})")

    base = index * self.frame_bytes
    out = {}
    for name, dt, c, off in self._layout:
      if channels is not None and name not in channels:
        continue
      n = self.width * self.height * c
      raw = self._mm[base + off: base + off + n * dt.itemsize]
      arr = raw.view(dt).reshape((self.width, self.height) if c == 1 else (self.width, self.height, c))
      out[name] = np.array(arr) if copy else arr
    return out

  def sequence_for_frame(self, index: int) -> SequenceRecord:
    for s in self.sequences:
      if s.start <= index < s.start + s.count:
        return s
    raise IndexError(f"frame {index} belongs to no sequence")