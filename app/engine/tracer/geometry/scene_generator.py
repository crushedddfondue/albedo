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