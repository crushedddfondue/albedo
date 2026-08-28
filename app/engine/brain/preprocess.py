from typing import Tuple

import torch
from torch import Tensor

MIN_ALBEDO = 1e-2
COMPRESS_MODES = ("log1p", "none")
DEFAULT_COMPRESS = "log1p"


def _check_mode(mode: str) -> str:
  if mode not in COMPRESS_MODES:
    raise ValueError(f"unknown compress mode {mode!r}, expected one of {COMPRESS_MODES}")
  return mode


def demodulate(radiance: Tensor, albedo: Tensor, hit_mask: Tensor, min_albedo: float = MIN_ALBEDO) -> Tensor:
  gate = _gate(albedo, hit_mask, min_albedo)
  safe = torch.clamp(albedo, min=min_albedo)
  return torch.where(gate, radiance / safe, radiance)


def remodulate(irradiance: Tensor, albedo: Tensor, hit_mask: Tensor, min_albedo: float = MIN_ALBEDO) -> Tensor:
  gate = _gate(albedo, hit_mask, min_albedo)
  safe = torch.clamp(albedo, min=min_albedo)
  return torch.where(gate, irradiance * safe, irradiance)


def _gate(albedo: Tensor, hit_mask: Tensor, min_albedo: float = MIN_ALBEDO) -> Tensor:
  if hit_mask.dim() == albedo.dim() - 1:
    hit_mask = hit_mask.unsqueeze(-3)
  return (hit_mask > 0) & (albedo >= min_albedo)

def compress(x: Tensor, mode: str = DEFAULT_COMPRESS) -> Tensor:
  if _check_mode(mode) == "none":
    return x

  return torch.log1p(torch.clamp(x, min=0.0))


def decompress(x: Tensor, mode: str = DEFAULT_COMPRESS) -> Tensor:
  if _check_mode(mode) == "none":
    return x
  return torch.expm1(x)

def prepare(batch: dict, compress_mode: str = DEFAULT_COMPRESS, min_albedo: float = MIN_ALBEDO) -> Tuple[Tensor, dict]:
  _check_mode(compress_mode)
  albedo, hit = batch["albedo"], batch["hit_mask"]

  noisy_irr = demodulate(batch["noisy"], albedo, hit, min_albedo)
  x = compress(noisy_irr, compress_mode)

  ctx = {
    "albedo": albedo,
    "hit_mask": hit,
    "compress_mode": compress_mode,
    "min_albedo": min_albedo,
  }
  if "clean" in batch:
    ctx["target"] = compress(demodulate(batch["clean"], albedo, hit, min_albedo), compress_mode)
  if "noisy2" in batch:
    ctx["target_noisy"] = compress(demodulate(batch["noisy2"], albedo, hit, min_albedo), compress_mode)
  return x, ctx

def to_radiance(prediction: Tensor, ctx: dict) -> Tensor:
  return remodulate(decompress(prediction, ctx["compress_mode"]), ctx["albedo"], ctx["hit_mask"], ctx["min_albedo"])