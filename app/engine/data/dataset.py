import os
from typing import Sequence, List, Dict

import numpy as np

try:
  import torch
  from torch.utils.data import Dataset
  _HAS_TORCH = True
except ImportError:                                   # pragma: no cover
  _HAS_TORCH = False
  Dataset = object

from data import transforms
from data.manifest import Manifest
from data.shard import ShardReader

DEFAULT_INPUT_CHANNELS = ("noisy", "albedo", "normal", "depth", "hit_mask")

class AlbedoSequenceDataset:
  def __init__(self, manifest: str, split: str = "train", seq_len: int = 8, window_stride: int = 1, augment: bool = True, crop: Sequence[int] | None = None, epoch_seed: int = 0, input_channels: Sequence[str] = DEFAULT_INPUT_CHANNELS, val_fraction: float | None = None, noisy_pair: bool = False):
    if not _HAS_TORCH:
      raise ImportError("torch is required for AlbedoSequenceDataset")

    self.manifest = Manifest.load(manifest) if isinstance(manifest, str) else manifest
    self.split = split
    self.seq_len = int(seq_len)
    self.augment = bool(augment)
    self.crop = tuple(crop) if crop else None 
    self.epoch_seed = int(epoch_seed)
    self.input_channels = tuple(input_channels)
    self.noisy_pair = bool(noisy_pair)

    self.index: List = []
    for shard_path, seq in self.manifest.sequences(split=split, val_fraction=val_fraction):
      n = seq["count"]
      if n < self.seq_len:
        continue
      for start in range(0, n - self.seq_len + 1, window_stride):
        self.index.append((shard_path, seq, seq["start"] + start))

    if not self.index:
      raise ValueError(
        f"split '{split}' produced no windows of length {seq_len}. "
        f"Either the split is empty or every sequence is shorter than the window."
      )

    self._readers: Dict[str, ShardReader] = {}

  def __len__(self):
    return len(self.index)

  def _reader(self, path: str) -> ShardReader:
    r = self._readers.get(path)
    if r is None:
      r = ShardReader(path)
      self._readers[path] = r
    return r

  def __getitem__(self, i: int):
    shard_path, seq, start = self.index[i]
    reader = self._reader(shard_path)
    rng = np.random.default_rng((self.epoch_seed, i))

    do_flip = self.augment and bool(rng.random() < 0.5)
    crop_offset = None

    n_real = reader.noisy_realizations
    if self.noisy_pair and n_real < 2:
      raise ValueError(
        f"noisy_pair needs >= 2 realisations, shard has {n_real}. "
        f"Regenerate with --noisy-realizations 2."
      )

    frames = []
    for t in range(self.seq_len):
      raw = reader.read_frame(start + t, copy=True)
      cam = seq["meta"]["frames"][(start - seq["start"]) + t]
      basis = (cam["right"], cam["up"], cam["forward"])

      order = rng.permutation(n_real)
      raw["noisy"] = raw["noisy"][:, :, order[0], :]
      if self.noisy_pair:
        raw["noisy2"] = reader.read_frame(start + t, copy=True)["noisy"][:, :, order[1], :]

      f = transforms.to_model_space(raw, basis)

      if self.crop is not None:
        f, crop_offset = transforms.random_crop(f, self.crop, rng, offset=crop_offset)
      if do_flip:
        f = transforms.flip_horizontal(f)

      frames.append(f)

    inp = np.stack([transforms.stack_channels(f, self.input_channels) for f in frames])
    target = np.stack([f["clean"].astype(np.float32).transpose(2, 0, 1) for f in frames])
    motion = np.stack([f["motion"].astype(np.float32).transpose(2, 0, 1) for f in frames])
    albedo = np.stack([f["albedo"].astype(np.float32).transpose(2, 0, 1) for f in frames])
    hit = np.stack([f["hit_mask"].astype(np.float32)[None] for f in frames])
    oid = np.stack([f["object_id"].astype(np.int32)[None] for f in frames])

    return {
      "input": torch.from_numpy(inp),          # (T, C, H, W)
      "target": torch.from_numpy(target),      # (T, 3, H, W) linear HDR
      "motion": torch.from_numpy(motion),      # (T, 2, H, W) pixels, x already mirrored
      "albedo": torch.from_numpy(albedo),      # (T, 3, H, W) for demodulation
      "hit_mask": torch.from_numpy(hit),       # (T, 1, H, W)
      "object_id": torch.from_numpy(oid),      # (T, 1, H, W)
      "scene_id": seq["scene_id"],
      "flipped": do_flip,
      "index": i,
    }

    
def make_loader(manifest, split="train", batch_size=4, seq_len=8, num_workers=4, epoch_seed=0, **kwargs):
  if not _HAS_TORCH:                                  # pragma: no cover
    raise ImportError("torch is required for make_loader")

  ds = AlbedoSequenceDataset(manifest, split=split, seq_len=seq_len,
                             augment=(split == "train"), epoch_seed=epoch_seed, **kwargs)
  return torch.utils.data.DataLoader(
    ds, batch_size=batch_size, shuffle=(split == "train"),  # type:ignore
    num_workers=num_workers, pin_memory=True, drop_last=(split == "train"),
    persistent_workers=num_workers > 0,
  )
