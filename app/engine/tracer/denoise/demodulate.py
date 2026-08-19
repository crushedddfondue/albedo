import taichi as ti
from taichi.math import vec3

DEMOD_EPS=1e-3

DEMOD_MIN_ALBEDO = 1e-2

@ti.kernel
def demodulate(radiance: ti.template(), albedo: ti.template(), hit_mask: ti.template(), out: ti.template()):  # type: ignore
  for i, j in radiance:
    a = albedo[i, j]
    lum = 0.2126 * a[0] + 0.7152 * a[1] + 0.0722 * a[2]
    if hit_mask[i, j] == 1 and lum > DEMOD_MIN_ALBEDO:
      out[i, j] = radiance[i, j] / ti.max(a, vec3(DEMOD_EPS))
    else:
      out[i, j] = radiance[i, j]

@ti.kernel
def remodulate(filtered: ti.template(), albedo: ti.template(), hit_mask: ti.template(), out: ti.template()):  # type: ignore
  for i, j in filtered:
    a = albedo[i, j]
    lum = 0.2126 * a[0] + 0.7152 * a[1] + 0.0722 * a[2]
    if hit_mask[i, j] == 1 and lum > DEMOD_MIN_ALBEDO:
      out[i, j] = filtered[i, j] * ti.max(a, vec3(DEMOD_EPS))
    else:
      out[i, j] = filtered[i, j]