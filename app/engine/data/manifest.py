import hashlib
import json
import os
from dataclasses import dataclass
from typing import List, Dict

MANIFEST_VERSION = 1
SPLIT_SALT = "albedo-p2.3"

def scene_split(scene_id: str, val_fraction: float, salt: str = "albedo-p2.3") -> str:
  if not 0.0 <= val_fraction < 1.0:
    raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
  h = hashlib.blake2b(f"{salt}:{scene_id}".encode("utf-8"), digest_size=8).digest()
  u = int.from_bytes(h, "big") / float(1 << 64)
  return "val" if u < val_fraction else "train"

@dataclass
class Manifest:
  root: str
  version: int
  config: dict
  shards: List[dict]

  def save(self, path: str | None = None) -> str:
    path = path or os.path.join(self.root, "dataset.json")
    with open(path, "w") as f:
      json.dump(
        {"manifest_version": self.version, "config": self.config, "shards": self.shards}, f, indent=2)
    return path

  @staticmethod
  def load(path: str) -> "Manifest":
    if os.path.isdir(path):
      path = os.path.join(path, "dataset.json")
    with open(path) as f:
      d = json.load(f)
    if d["manifest_version"] != MANIFEST_VERSION:
      raise ValueError(
        f"manifest is v{d['manifest_version']}, reader is v{MANIFEST_VERSION}"
      )
    return Manifest(
      root=os.path.dirname(os.path.abspath(path)),
      version=d["manifest_version"], config=d["config"], shards=d["shards"],
    )

  def scene_ids(self) -> List[str]:
    ids = []
    for shard in self.shards:
      for seq in shard["sequences"]:
        if seq["scene_id"] not in ids:
          ids.append(seq["scene_id"])
    return ids

  def sequences(self, split: str | None = None, val_fraction: float | None = None, salt: str | None = None):
    if val_fraction is None:
      val_fraction = self.config.get("val_fraction", 0.1)
    if salt is None:
      salt = self.config.get("split_salt", SPLIT_SALT)

    for shard in self.shards:
      path = os.path.join(self.root, shard["path"])
      for seq in shard["sequences"]:
        if split is None or scene_split(seq["scene_id"], val_fraction, salt) == split:  # type: ignore
          yield path, seq

  def split_summary(self, val_fraction: float | None = None, salt: str | None = None) -> Dict:
    if val_fraction is None:
      val_fraction = self.config.get("val_fraction", 0.1)
    if salt is None:
      salt = self.config.get("split_salt", SPLIT_SALT)

    out = {
      "train": {"scenes": set(), "sequences": 0, "frames": 0},
      "val": {"scenes": set(), "sequences": 0, "frames": 0},
    }
    for shard in self.shards:
      for seq in shard["sequences"]:
        s = scene_split(seq["scene_id"], val_fraction, salt)  # type: ignore
        out[s]["scenes"].add(seq["scene_id"])
        out[s]["sequences"] += 1
        out[s]["frames"] += seq["count"]

    overlap = out["train"]["scenes"] & out["val"]["scenes"]
    if overlap:
      raise AssertionError(
        f"{len(overlap)} scene(s) in BOTH splits: {sorted(overlap)[:5]}. "
        f"The split function is not a function of scene_id alone."
      )

    return {
      s: {"scenes": len(v["scenes"]), "sequences": v["sequences"], "frames": v["frames"]}
      for s, v in out.items()
    }


def effective_sample_size(n_frames: int, rho: float) -> float:
  if not -1.0 < rho < 1.0:
    raise ValueError(f"rho must be in (-1, 1), got {rho}")
  return n_frames * (1.0 - rho) / (1.0 + rho)