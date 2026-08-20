from dataclasses import dataclass, asdict, field

import numpy as np

from tracer.geometry import scene

TRIS_PER_QUAD = 2
TRIS_PER_BOX = 12

@dataclass(frozen=True)
class SceneParams:
  room_size_min: tuple = (5.0, 3.0, 5.0)
  room_size_max: tuple = (12.0, 5.0, 12.0)
  ceiling_probability: float = 0.5

  n_boxes_min: int = 2
  n_boxes_max: int = 10
  box_size_min: tuple = (0.3, 0.3, 0.3)
  box_size_max: tuple = (2.0, 2.5, 2.0)

  albed_min: float = 0.05
  albed_max: float = 0.9
  saturation_max: float = 0.6

  n_lights_min: int = 1
  n_lights_max: int = 3
  light_size_min: float = 0.4
  light_size_max: float = 2.0
  light_intensity_min: float = 3.0
  light_intensity_max: float = 25.0
  light_height_margin: float = 0.15

  max_triangles: int = scene.MAX_TRIANGLES
  max_lights: int = scene.MAX_LIGHTS

  def to_json(self)-> dict:
    return asdict(self)


@dataclass
class SceneData:
  v0: np.ndarray
  v1: np.ndarray
  v2: np.ndarray
  albedo: np.ndarray
  emission: np.ndarray
  object_id: np.ndarray
  light_index: np.ndarray

  light_triangle_index: np.ndarray
  light_pdf_Area: np.ndarray

  room_size: np.ndarray
  scene_id: str = ""
  seed: int = 0
  meta: dict = field(default_factory=dict)

  @property
  def n_triangles(self) -> int:
    return int(self.v0.shape[0])

  @property
  def n_lights(self) -> int:
    return int(self.light_triangle_index.shape[0])

  @property
  def bounds(self):
    half = self.room_size * 0.5
    lo = np.array([-half[0], 0.0, -half[2]])
    hi = np.array([half[0], self.room_size[1], half[2]])
    return lo, hi

def _tri_normal(a, b, c):
  n = np.cross(b-a, c-a)
  ln = np.linalg.norm(n)

  return n / ln if n > 0.0 else n

def _quad(a, b, c, d, want_normal):
  a, b, c, d = (np.asarray(v, dtype=np.float64) for v in (a, b, c, d))

  if np.dot(_tri_normal(a, b, c), want_normal) < 0.0:
    a, b, c, d = a, d, c, b

  return [(a, b, c), (a, c, d)]

def _box_faces(lo, hi, inward=False):
  x0, y0, z0 = lo
  x1, y1, z1 = hi
  s = -1.0 if inward else 1.0

  faces = [
    ((x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (s, 0.0, 0.0)),
    ((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0), (-s, 0.0, 0.0)),
    ((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (0.0, s, 0.0)),
    ((x0, y0, z1), (x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (0.0, -s, 0.0)),
    ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), (0.0, 0.0, s)),
    ((x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (0.0, 0.0, -s)),
  ]

  tris = []
  for a, b, c, d, n in faces:
    tris.extend(_quad(a, b, c, d, np.asarray(n, dtype=np.float64)))
  return tris

def _random_albedo(rng, params):
  base = rng.uniform(params.albedo_min, params.albedo_max)
  tint = rng.uniform(-1.0, 1.0, size=3)
  tint = tint / max(np.abs(tint).max(), 1e-9) * rng.uniform(0.0, params.saturation_max)
  return np.clip(base * (1.0 + tint), params.albedo_min, params.albedo_max)

