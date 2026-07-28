from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn

class RectifiedFlowUnet(nn.Module):
  def __init__(self)-> None:
    super(),__init__()  # type: ignore

    self.net = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(32, 3, kernel_size=3, padding=1)
    )

  def forward(self, x_noisy: Tensor)-> Tensor:
    velocity = self.net(x_noisy)

    x_clean = x_noisy - velocity

    return torch.clamp(x_clean, 0.0, 1.0)