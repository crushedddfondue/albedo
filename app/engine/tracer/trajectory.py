import math
from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence

import numpy as np

MAX_PITCH = math.radians(89.5)   # must match camera.MAX_PITCH

KINDS = ("static", "dolly", "pan", "orbit", "handheld")

_EXITS = ((0, False), (0, True), (1, False), (1, True), (2, False), (2, True))


@dataclass(frozen=True)
class Pose:
  position: np.ndarray   # (3,) f64, world space
  yaw: float
  pitch: float

  def to_json(self):
    return {
      "position": [float(v) for v in self.position],
      "yaw": float(self.yaw),
      "pitch": float(self.pitch),
    }


@dataclass(frozen=True)
class TrajectoryParams:
  weights: tuple = (0.08, 0.24, 0.24, 0.24, 0.20)   # aligned with KINDS

  speed_min: float = 0.005
  speed_max: float = 0.06

  yaw_rate_min: float = math.radians(0.15)
  yaw_rate_max: float = math.radians(1.6)

  wall_margin: float = 0.6
  height_min: float = 0.6
  height_margin_top: float = 0.5

  obstacle_margin: float = 0.30

  interior_attempts: int = 32

  max_attempts: int = 4

  handheld_damping: float = 0.90
  handheld_accel: float = 0.010
  handheld_look_jitter: float = math.radians(0.5)

  pitch_min: float = math.radians(-25.0)
  pitch_max: float = math.radians(15.0)

  def to_json(self) -> dict:
    return asdict(self)


def look_at(position, target):
  d = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
  n = np.linalg.norm(d)
  if n < 1e-9:
    return 0.0, 0.0
  d = d / n

  pitch = math.asin(float(np.clip(d[1], -1.0, 1.0)))
  yaw = math.atan2(float(d[0]), float(-d[2]))
  return yaw, float(np.clip(pitch, -MAX_PITCH, MAX_PITCH))


def as_obstacles(obstacles) -> np.ndarray:
  if obstacles is None:
    return np.zeros((0, 2, 3), dtype=np.float64)
  arr = np.asarray(obstacles, dtype=np.float64)
  if arr.size == 0:
    return np.zeros((0, 2, 3), dtype=np.float64)
  return arr.reshape(-1, 2, 3)


def inside_index(p, obstacles: np.ndarray, margin: float) -> int:
  for i in range(obstacles.shape[0]):
    lo, hi = obstacles[i, 0] - margin, obstacles[i, 1] + margin
    if bool(np.all(p >= lo)) and bool(np.all(p <= hi)):
      return i
  return -1


def _room_clamp(p, lo, hi, params):
  top = max(params.height_min, hi[1] - params.height_margin_top)
  return np.array([
    float(np.clip(p[0], lo[0] + params.wall_margin, hi[0] - params.wall_margin)),
    float(np.clip(p[1], params.height_min, top)),
    float(np.clip(p[2], lo[2] + params.wall_margin, hi[2] - params.wall_margin)),
  ])


def _push_out(p, box, lo, hi, params, obstacles):
  b_lo, b_hi = box[0] - params.obstacle_margin, box[1] + params.obstacle_margin

  best, best_d = None, float("inf")
  for axis, use_high in _EXITS:
    q = p.copy()
    q[axis] = b_hi[axis] if use_high else b_lo[axis]
    q = _room_clamp(q, lo, hi, params)
    if inside_index(q, obstacles, params.obstacle_margin) != -1:
      continue
    d = float(np.linalg.norm(q - p))
    if d < best_d:
      best, best_d = q, d
  return best


def _resolve(p, lo, hi, params, obstacles):
  q = _room_clamp(np.asarray(p, dtype=np.float64), lo, hi, params)
  if obstacles.shape[0] == 0:
    return q
  for _ in range(8):
    idx = inside_index(q, obstacles, params.obstacle_margin)
    if idx == -1:
      return q
    nxt = _push_out(q, obstacles[idx], lo, hi, params, obstacles)
    if nxt is None:
      return q                    # no legal exit; sample_trajectory catches it
    q = nxt
  return q


def _random_interior(rng, lo, hi, params, obstacles):
  top = max(params.height_min + 1e-3, hi[1] - params.height_margin_top)
  p = None
  for _ in range(params.interior_attempts):
    p = np.array([
      rng.uniform(lo[0] + params.wall_margin, hi[0] - params.wall_margin),
      rng.uniform(params.height_min, top),
      rng.uniform(lo[2] + params.wall_margin, hi[2] - params.wall_margin),
    ])
    if inside_index(p, obstacles, params.obstacle_margin) == -1:
      return _room_clamp(p, lo, hi, params)
  return _resolve(p, lo, hi, params, obstacles)


def _static(rng, n, lo, hi, params, obs):
  p = _random_interior(rng, lo, hi, params, obs)
  yaw, pitch = look_at(p, _random_interior(rng, lo, hi, params, obs))
  return [Pose(p.copy(), yaw, pitch) for _ in range(n)]


def _dolly(rng, n, lo, hi, params, obs):
  start = _random_interior(rng, lo, hi, params, obs)
  yaw, pitch = look_at(start, _random_interior(rng, lo, hi, params, obs))
  speed = rng.uniform(params.speed_min, params.speed_max)

  cp = math.cos(pitch)
  forward = np.array([cp * math.sin(yaw), math.sin(pitch), -cp * math.cos(yaw)])
  direction = forward * (1.0 if rng.random() < 0.7 else -1.0)

  return [Pose(_resolve(start + direction * speed * t, lo, hi, params, obs), yaw, pitch)
          for t in range(n)]


def _pan(rng, n, lo, hi, params, obs):
  p = _random_interior(rng, lo, hi, params, obs)
  yaw0, pitch = look_at(p, _random_interior(rng, lo, hi, params, obs))
  rate = rng.uniform(params.yaw_rate_min, params.yaw_rate_max) * (1.0 if rng.random() < 0.5 else -1.0)
  return [Pose(p.copy(), yaw0 + rate * t, pitch) for t in range(n)]


def _orbit(rng, n, lo, hi, params, obs):
  centre = np.array([
    rng.uniform(lo[0] + params.wall_margin, hi[0] - params.wall_margin),
    rng.uniform(params.height_min, max(params.height_min + 1e-3, hi[1] * 0.6)),
    rng.uniform(lo[2] + params.wall_margin, hi[2] - params.wall_margin),
  ])

  max_radius = min(
    hi[0] - params.wall_margin - centre[0], centre[0] - lo[0] - params.wall_margin,
    hi[2] - params.wall_margin - centre[2], centre[2] - lo[2] - params.wall_margin,
  )
  radius = float(np.clip(rng.uniform(1.0, 4.0), 0.5, max(0.5, max_radius)))

  rate = rng.uniform(params.yaw_rate_min, params.yaw_rate_max) * (1.0 if rng.random() < 0.5 else -1.0)
  theta0 = rng.uniform(0.0, 2.0 * math.pi)
  elevation = rng.uniform(0.2, 1.2)

  poses = []
  for t in range(n):
    theta = theta0 + rate * t
    p = _resolve(np.array([
      centre[0] + radius * math.sin(theta),
      centre[1] + elevation,
      centre[2] + radius * math.cos(theta),
    ]), lo, hi, params, obs)
    yaw, pitch = look_at(p, centre)
    poses.append(Pose(p, yaw, pitch))
  return poses


def _handheld(rng, n, lo, hi, params, obs):
  p = _random_interior(rng, lo, hi, params, obs)
  target = _random_interior(rng, lo, hi, params, obs)
  velocity = np.zeros(3)
  yaw, pitch = look_at(p, target)

  poses = []
  for _ in range(n):
    velocity = params.handheld_damping * velocity + params.handheld_accel * rng.normal(size=3)
    moved = _resolve(p + velocity, lo, hi, params, obs)

    velocity = moved - p
    p = moved

    yaw_t, pitch_t = look_at(p, target)
    yaw += (yaw_t - yaw) * 0.15 + rng.normal() * params.handheld_look_jitter
    pitch += (pitch_t - pitch) * 0.15 + rng.normal() * params.handheld_look_jitter
    pitch = float(np.clip(pitch, params.pitch_min, params.pitch_max))

    poses.append(Pose(p.copy(), float(yaw), pitch))
  return poses


_PROCESSES = {
  "static": _static, "dolly": _dolly, "pan": _pan,
  "orbit": _orbit, "handheld": _handheld,
}


def invalid_poses(poses: Sequence[Pose], obstacles, margin: float) -> List[int]:
  obs = as_obstacles(obstacles)
  if obs.shape[0] == 0:
    return []
  return [i for i, p in enumerate(poses)
          if inside_index(np.asarray(p.position, dtype=np.float64), obs, margin) != -1]


def sample_trajectory(seed: int, n_frames: int, bounds, params: Optional[TrajectoryParams] = None, kind: Optional[str] = None, obstacles=None):
  params = params or TrajectoryParams()
  lo, hi = bounds
  obs = as_obstacles(obstacles)

  poses, k, bad = [], kind, []
  for attempt in range(max(1, params.max_attempts)):
    rng = np.random.default_rng((seed, attempt))
    k = kind
    if k is None:
      w = np.asarray(params.weights, dtype=np.float64)
      k = str(rng.choice(KINDS, p=w / w.sum()))

    poses = _PROCESSES[k](rng, n_frames, lo, hi, params, obs)
    bad = invalid_poses(poses, obs, params.obstacle_margin)
    if not bad:
      return poses, {
        "kind": k, "seed": int(seed), "n_frames": int(n_frames),
        "obstacles": int(obs.shape[0]), "attempts": attempt + 1, "invalid_poses": 0,
      }

  return poses, {
    "kind": k, "seed": int(seed), "n_frames": int(n_frames),
    "obstacles": int(obs.shape[0]), "attempts": params.max_attempts,
    "invalid_poses": len(bad),
  }


def rotation_pixels_per_radian(width: int, fov: float, aspect_ratio: float) -> float:
  return width / (2.0 * aspect_ratio * math.tan(fov / 2.0))


def trajectory_diagnostics(poses: List[Pose], width: int, fov: float, aspect_ratio: float) -> dict:
  if len(poses) < 2:
    return {"frames": len(poses)}

  pos = np.asarray([p.position for p in poses])
  yaw = np.unwrap(np.asarray([p.yaw for p in poses]))
  pitch = np.asarray([p.pitch for p in poses])

  d_pos = np.linalg.norm(np.diff(pos, axis=0), axis=1)
  d_yaw = np.abs(np.diff(yaw))
  d_pitch = np.abs(np.diff(pitch))
  k = rotation_pixels_per_radian(width, fov, aspect_ratio)

  return {
    "frames": len(poses),
    "translation_per_frame_mean": float(d_pos.mean()),
    "translation_per_frame_max": float(d_pos.max()),
    "yaw_rate_mean_rad": float(d_yaw.mean()),
    "yaw_rate_max_rad": float(d_yaw.max()),
    "pitch_rate_max_rad": float(d_pitch.max()),
    # Rotation-only, image centre. Excludes all parallax.
    "rotation_px_per_frame_mean_centre": float(d_yaw.mean() * k),
    "rotation_px_per_frame_max_centre": float(d_yaw.max() * k),
  }