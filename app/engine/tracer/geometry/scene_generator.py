import math
from dataclasses import dataclass, asdict, field
from typing import Optional

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
  box_yaw_max: float = math.pi

  albedo_min: float = 0.05
  albedo_max: float = 0.9
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
  light_pdf_area: np.ndarray

  room_size: np.ndarray
  occluders: np.ndarray = field(default_factory=lambda: np.zeros((0, 2, 3)))
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

  return n / ln if ln > 0.0 else n

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

def _rotate_y(tris, centre, yaw):
  c, s_ = math.cos(yaw), math.sin(yaw)

  def r(v):
    d = np.asarray(v, dtype=np.float64) - centre
    return centre + np.array([c * d[0] + s_ * d[2], d[1], -s_ * d[0] + c * d[2]])

  return [tuple(r(v) for v in tri) for tri in tris]

def _tris_aabb(tris):
  v = np.asarray([p for tri in tris for p in tri], dtype=np.float64)
  return v.min(axis=0), v.max(axis=0)

def _random_albedo(rng, params):
  base = rng.uniform(params.albedo_min, params.albedo_max)
  tint = rng.uniform(-1.0, 1.0, size=3)
  tint = tint / max(np.abs(tint).max(), 1e-9) * rng.uniform(0.0, params.saturation_max)
  return np.clip(base * (1.0 + tint), params.albedo_min, params.albedo_max)


def build_scene(params: SceneParams, seed: int, scene_id: Optional[str] = None) -> SceneData:
  rng = np.random.default_rng(seed)
  scene_id = scene_id if scene_id is not None else f"scene_{seed:08d}"

  room = np.array([
    rng.uniform(params.room_size_min[i], params.room_size_max[i]) for i in range(3)
  ])
  has_ceiling = bool(rng.random() < params.ceiling_probability)

  half_x, half_z = room[0] * 0.5, room[2] * 0.5
  room_lo = np.array([-half_x, 0.0, -half_z])
  room_hi = np.array([half_x, room[1], half_z])

  tri_v = []       
  tri_albedo = []
  tri_emission = []
  tri_object = []

  room_albedo = _random_albedo(rng, params)
  for k, tri in enumerate(_box_faces(room_lo, room_hi, inward=True)):
    is_ceiling = k in (4, 5)   # +Y face is faces[2] -> triangles 4 and 5
    if is_ceiling and not has_ceiling:
      continue
    tri_v.append(tri)
    tri_albedo.append(room_albedo)
    tri_emission.append(np.zeros(3))
    tri_object.append(0)

  n_boxes = int(rng.integers(params.n_boxes_min, params.n_boxes_max + 1))
  placed = []
  next_object_id = 1

  for _ in range(n_boxes):
    for _attempt in range(24):
      size = np.array([
        rng.uniform(params.box_size_min[i], params.box_size_max[i]) for i in range(3)
      ])
      size[1] = min(size[1], room[1] * 0.6)

      half_diag = 0.5 * math.hypot(size[0], size[2])
      if (room_hi[0] - room_lo[0] <= 2.0 * half_diag
          or room_hi[2] - room_lo[2] <= 2.0 * half_diag):
        continue

      cx = rng.uniform(room_lo[0] + size[0] * 0.5, room_hi[0] - size[0] * 0.5)
      cz = rng.uniform(room_lo[2] + size[2] * 0.5, room_hi[2] - size[2] * 0.5)

      lo = np.array([cx - size[0] * 0.5, 0.0, cz - size[2] * 0.5])
      hi = np.array([cx + size[0] * 0.5, size[1], cz + size[2] * 0.5])

      yaw = rng.uniform(0.0, params.box_yaw_max)
      box_tris = _rotate_y(_box_faces(lo, hi, inward=False), np.array([cx, 0.0, cz]), yaw)
      aabb_lo, aabb_hi = _tris_aabb(box_tris)

      if any(_overlaps_xz(aabb_lo, aabb_hi, p_lo, p_hi) for p_lo, p_hi in placed):
        continue

      placed.append((aabb_lo, aabb_hi))
      box_albedo = _random_albedo(rng, params)
      for tri in box_tris:
        tri_v.append(tri)
        tri_albedo.append(box_albedo)
        tri_emission.append(np.zeros(3))
        tri_object.append(next_object_id)
      next_object_id += 1
      break

  n_lights = int(rng.integers(params.n_lights_min, params.n_lights_max + 1))
  n_lights = min(n_lights, params.max_lights // TRIS_PER_QUAD)
  light_y = room[1] * (1.0 - params.light_height_margin)

  for _ in range(n_lights):
    sx = rng.uniform(params.light_size_min, params.light_size_max)
    sz = rng.uniform(params.light_size_min, params.light_size_max)
    cx = rng.uniform(room_lo[0] + sx, room_hi[0] - sx)
    cz = rng.uniform(room_lo[2] + sz, room_hi[2] - sz)

    intensity = rng.uniform(params.light_intensity_min, params.light_intensity_max)
    tint = 1.0 + rng.uniform(-0.15, 0.15, size=3)
    emission = np.clip(intensity * tint, 0.0, None)

    a = (cx - sx * 0.5, light_y, cz - sz * 0.5)
    b = (cx + sx * 0.5, light_y, cz - sz * 0.5)
    c = (cx + sx * 0.5, light_y, cz + sz * 0.5)
    d = (cx - sx * 0.5, light_y, cz + sz * 0.5)

    for tri in _quad(a, b, c, d, np.array([0.0, -1.0, 0.0])):
      tri_v.append(tri)
      tri_albedo.append(np.zeros(3))     # ⚠ zero albedo: this is what makes
      tri_emission.append(emission)      #   the demodulation threshold matter
      tri_object.append(next_object_id)
    next_object_id += 1

  n = len(tri_v)
  if n > params.max_triangles:
    raise ValueError(
      f"scene {scene_id} emitted {n} triangles, over the field capacity of "
      f"{params.max_triangles}. Lower n_boxes_max (each box is "
      f"{TRIS_PER_BOX} triangles) or raise scene.MAX_TRIANGLES."
    )

  v0 = np.ascontiguousarray([t[0] for t in tri_v], dtype=np.float32)
  v1 = np.ascontiguousarray([t[1] for t in tri_v], dtype=np.float32)
  v2 = np.ascontiguousarray([t[2] for t in tri_v], dtype=np.float32)
  albedo = np.ascontiguousarray(tri_albedo, dtype=np.float32)
  emission = np.ascontiguousarray(tri_emission, dtype=np.float32)
  object_id = np.ascontiguousarray(tri_object, dtype=np.int32)

  light_index = np.full(n, -1, dtype=np.int32)
  emissive = np.where(emission.max(axis=1) > 0.0)[0]

  if emissive.size > params.max_lights:
    raise ValueError(
      f"scene {scene_id} has {emissive.size} emissive triangles, over "
      f"MAX_LIGHTS = {params.max_lights}."
    )

  areas = 0.5 * np.linalg.norm(
    np.cross(v1[emissive] - v0[emissive], v2[emissive] - v0[emissive]), axis=1
  )
  light_pdf_area = np.where(areas > 1e-8, 1.0 / np.maximum(areas, 1e-12), 0.0)
  light_index[emissive] = np.arange(emissive.size, dtype=np.int32)

  occluders = (np.asarray([[lo, hi] for lo, hi in placed], dtype=np.float64) if placed else np.zeros((0, 2, 3), dtype=np.float64))

  return SceneData(
    v0=v0, v1=v1, v2=v2,
    albedo=albedo, emission=emission,
    object_id=object_id, light_index=light_index,
    light_triangle_index=emissive.astype(np.int32),
    light_pdf_area=light_pdf_area.astype(np.float32),
    room_size=room,
    occluders=occluders,
    scene_id=scene_id,
    seed=int(seed),
    meta={
      "n_triangles": int(n),
      "n_boxes": len(placed),
      "n_light_quads": int(emissive.size // TRIS_PER_QUAD),
      "has_ceiling": has_ceiling,
      "room_size": room.tolist(),
      "total_emission": float(emission.sum()),
    },
  )

def _overlaps_xz(a_lo, a_hi, b_lo, b_hi, margin=0.15):
  return (
    a_lo[0] - margin < b_hi[0] and a_hi[0] + margin > b_lo[0] and
    a_lo[2] - margin < b_hi[2] and a_hi[2] + margin > b_lo[2]
  )


def upload_scene(data: SceneData):
  from tracer.bvh import upload as bvh_upload

  cap = scene.MAX_TRIANGLES
  n = data.n_triangles

  def _pad(arr, width=None, dtype=np.float32, fill=0.0):
    shape = (cap,) if width is None else (cap, width)
    out = np.full(shape, fill, dtype=dtype)
    out[:n] = arr
    return out

  scene.triangles.v0.from_numpy(_pad(data.v0, 3))
  scene.triangles.v1.from_numpy(_pad(data.v1, 3))
  scene.triangles.v2.from_numpy(_pad(data.v2, 3))
  scene.triangles.albedo.from_numpy(_pad(data.albedo, 3))
  scene.triangles.emission.from_numpy(_pad(data.emission, 3))
  scene.triangles.normal.from_numpy(np.zeros((cap, 3), dtype=np.float32))
  scene.triangles.object_id.from_numpy(_pad(data.object_id, None, np.int32, -1))  # type: ignore
  scene.triangles.light_index.from_numpy(_pad(data.light_index, None, np.int32, -1))  # type: ignore

  scene.num_triangles[None] = n

  light_idx = np.full(scene.MAX_LIGHTS, -1, dtype=np.int32)
  light_pdf = np.zeros(scene.MAX_LIGHTS, dtype=np.float32)
  light_idx[: data.n_lights] = data.light_triangle_index
  light_pdf[: data.n_lights] = data.light_pdf_area

  scene.light_triangle_index.from_numpy(light_idx)
  scene.light_pdf_area.from_numpy(light_pdf)
  scene.num_lights[None] = data.n_lights

  scene.recompute_normals()

  bvh_upload.rebuild()
