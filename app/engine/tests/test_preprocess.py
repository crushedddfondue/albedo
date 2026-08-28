"""The transforms must be exactly invertible, or the model learns the error.

demodulate/remodulate and compress/decompress are used in opposite directions
on either side of the network. Any asymmetry becomes a fixed spatial pattern
the model compensates for during training and that breaks the moment the gate
changes -- a failure that would surface as "the model got worse after an
unrelated edit".
"""

import numpy as np
import pytest
import torch

from brain.preprocess import (
  MIN_ALBEDO, demodulate, remodulate, compress, decompress, prepare, to_radiance,
)

H, W = 8, 12


def _batch(seed=0, albedo_floor=0.05):
  g = torch.Generator().manual_seed(seed)
  # Albedo floor 0.05 is the phase23 corpus's measured minimum. The
  # albedo_floor=0.0 case below is the distribution we do NOT have yet.
  albedo = albedo_floor + (0.9 - albedo_floor) * torch.rand(2, 3, H, W, generator=g)
  hit = (torch.rand(2, 1, H, W, generator=g) > 0.1).to(torch.uint8)
  radiance = 10.0 * torch.rand(2, 3, H, W, generator=g) ** 3
  return {"noisy": radiance, "clean": radiance * 0.99, "albedo": albedo, "hit_mask": hit}


def test_demodulate_round_trip_is_exact():
  b = _batch()
  back = remodulate(demodulate(b["noisy"], b["albedo"], b["hit_mask"]),
                    b["albedo"], b["hit_mask"])
  assert torch.allclose(back, b["noisy"], atol=1e-5, rtol=1e-5)


def test_round_trip_holds_below_the_gate():
  """Where albedo is too dark to divide by, both directions must pass through
  unchanged -- the identity has to hold on the excluded set too, not just the
  included one."""
  b = _batch(albedo_floor=0.0)
  assert (b["albedo"] < MIN_ALBEDO).any(), "test needs sub-gate albedo to be meaningful"
  back = remodulate(demodulate(b["noisy"], b["albedo"], b["hit_mask"]),
                    b["albedo"], b["hit_mask"])
  assert torch.allclose(back, b["noisy"], atol=1e-5, rtol=1e-5)


def test_gate_is_per_channel():
  """A saturated albedo must not be demodulated on its dark channels just
  because its bright channel carries the luminance. This is the hazard in
  tracer/denoise/demodulate.py, which gates on luminance and divides per
  channel."""
  albedo = torch.tensor([[0.002], [0.002], [0.30]]).reshape(1, 3, 1, 1).expand(1, 3, H, W)
  hit = torch.ones(1, 1, H, W, dtype=torch.uint8)
  radiance = torch.full((1, 3, H, W), 0.5)
  out = demodulate(radiance, albedo.contiguous(), hit)
  # Red and green are below the gate, so untouched. Blue is divided.
  assert torch.allclose(out[:, 0], radiance[:, 0])
  assert torch.allclose(out[:, 1], radiance[:, 1])
  assert torch.allclose(out[:, 2], radiance[:, 2] / 0.30, atol=1e-5)


def test_demodulate_never_amplifies_more_than_the_gate():
  b = _batch(albedo_floor=0.0)
  irr = demodulate(b["noisy"], b["albedo"], b["hit_mask"])
  amp = irr / torch.clamp(b["noisy"], min=1e-9)
  assert float(amp.max()) <= 1.0 / MIN_ALBEDO + 1e-3


def test_compress_round_trip():
  x = 10.0 * torch.rand(2, 3, H, W) ** 3
  assert torch.allclose(decompress(compress(x)), x, atol=1e-5, rtol=1e-5)
  assert torch.allclose(decompress(compress(x, "none"), "none"), x)


def test_compress_handles_tiny_negatives():
  """f16 storage plus a demodulation divide can produce a small negative.
  log1p of anything <= -1 is not finite, and one NaN propagates through an
  entire training step."""
  x = torch.full((1, 3, 4, 4), -1e-6)
  assert torch.isfinite(compress(x)).all()


def test_prepare_to_radiance_is_identity_on_the_input():
  b = _batch()
  x, ctx = prepare(b)
  assert torch.allclose(to_radiance(x, ctx), b["noisy"], atol=1e-4, rtol=1e-4)


def test_target_uses_the_same_albedo_as_the_input():
  """If the target were demodulated with different albedo the loss would be
  measuring the transform, not the prediction."""
  b = _batch()
  _, ctx = prepare(b)
  expected = compress(demodulate(b["clean"], b["albedo"], b["hit_mask"]))
  assert torch.allclose(ctx["target"], expected)


@pytest.mark.parametrize("mode", ["log1p", "none"])
def test_no_nans_on_corpus_like_input(mode):
  b = _batch(seed=3)
  x, ctx = prepare(b, compress_mode=mode)
  assert torch.isfinite(x).all()
  assert torch.isfinite(to_radiance(x, ctx)).all()