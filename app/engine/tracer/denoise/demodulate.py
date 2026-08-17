import taichi as ti
from taichi.math import vec3

DEMOD_EPS=1e-3

@ti.kernel
def demodulate(radiance: ti.template(), albedo: ti.template(), hit_mask: ti.template(), out: ti.template()): # type: ignore
  for i, j in radiance:
    if hit_mask[i, j] == 1:
      a = ti.max(albedo[i, j], vec3(DEMOD_EPS))
      out[i, j] = radiance[i, j] / a
    else:
      out[i, j] = radiance[i, j]

@ti.kernel
def remodulate(filtered: ti.template(), albedo: ti.template(), hit_mask: ti.template(), out: ti.template()):  # type: ignore
  for i, j in filtered:
    if hit_mask[i, j] == 1:
      a = ti.max(albedo[i, j], vec3(DEMOD_EPS))
      out[i, j] = filtered[i, j] * a
    else:
      out[i, j] = filtered[i, j]