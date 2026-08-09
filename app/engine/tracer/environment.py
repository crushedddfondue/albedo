import taichi as ti
from taichi.math import vec3

ENV_BLACK = 0

ENV_GRADIENT = 1

ENV_CONSTANT = 2

@ti.data_oriented
class Environment:
  def __init__(self, mode=ENV_GRADIENT, horizon = None, zenith = None, constant = None):
    self.mode = int(mode)
    self.horizon = vec3(1.0) if horizon is None else horizon
    self.zenith = vec3(0.5, 0.7, 0.1) if zenith is None else zenith
    self.constant = vec3(0.5, 0.5, 0.5) if constant is None else constant

  @ti.func
  def sample(self, d: vec3)-> vec3: # type: ignore
    result = vec3(0.0)

    if ti.static(self.mode == ENV_GRADIENT):
      t = 0.5 * (d.y + 1.0)
      result = (1.0 - t) * self.horizon + t * self.zenith

    if ti.static(self.mode == ENV_CONSTANT):
      result = self.constant

    return result

  