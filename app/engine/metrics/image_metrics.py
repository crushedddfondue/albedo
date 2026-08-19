import numpy as np

EPS = 1e-2

_LUMA = np.array([0.2126, 0.7152, 0.0722])

def luminance(img):
  return img @ _LUMA

def _reduce(per_pixel, mask = None):
  if mask is None:
    return float(per_pixel.mean())

  return float(per_pixel[mask].mean())

def relmse(test, ref, eps=EPS, mask = None):
  rel_mse = (test - ref)**2 / (ref + eps) ** 2
  if mask is not None:
    return _reduce(rel_mse, mask)

  return float(rel_mse.mean())

def smape(test, ref, eps=EPS, mask=None):
  denom = 0.5 * (np.abs(test) + np.abs(ref)) + eps
  s_ma_pe = np.abs(test - ref) / denom
  if mask is not None:
    return _reduce(s_ma_pe, mask)
  return s_ma_pe.mean()

def report(test, ref, mask = None):
  return {
    "relmse": relmse(test, ref, mask=mask),
    "smape": smape(test, ref, mask=mask),
    "relmse_luma": relmse(luminance(test), luminance(ref), mask=mask)
  }