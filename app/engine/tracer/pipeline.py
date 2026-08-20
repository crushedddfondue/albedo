from dataclasses import dataclass, asdict
from typing import Any, Optional

import numpy as np
import taichi as ti

from tracer import buffers
from tracer.bvh import upload as bvh_upload
from tracer.camera import Camera
from tracer.denoise import history
from tracer.denoise.atrous import filter_image, ATROUS_LEVELS
from tracer.denoise.demodulate import demodulate, remodulate
from tracer.denoise.moments import estimate_variance
from tracer.denoise.reproject import reproject_kernel
from tracer.environment import Environment, ENV_BLACK
from tracer.geometry import scene
from tracer.kernels.gbuffer import gbuffer_kernel
from tracer.kernels.motion import motion_kernel
from tracer.kernels.render import accumulate_kernel
from tracer.sampling.brdf import BRDF


@dataclass(frozen=True)
class RenderConfig:
  width: int = 512
  height: int = 288
  fov_deg: float = 60.0
  near: float = 0.1
  far: float = 1000.0

  spp: int = 2                 
  max_bounces: int = 8
  use_nee: int = 1
  single_sided: int = 1

  clean_chunks: int = 16
  clean_spp_per_chunk: int = 32

  alpha_min: float = 0.05
  atrous_levels: int = ATROUS_LEVELS

  @property
  def aspect_ratio(self) -> float:
    return self.width / self.height

  @property
  def clean_spp(self) -> int:
    return self.clean_chunks * self.clean_spp_per_chunk

  def to_json(self) -> dict:
    d = asdict(self)
    d["clean_spp"] = self.clean_spp
    return d


_INSTANCE: Optional["Renderer"] = None


def get_renderer(config: RenderConfig) -> "Renderer":
  global _INSTANCE
  if _INSTANCE is None:
    _INSTANCE = Renderer(config)
  elif _INSTANCE.config != config:
    raise RuntimeError(
      f"Renderer already built for {_INSTANCE.config}; refusing to rebuild "
      f"for {config}. Restart the process to change render configuration."
    )
  return _INSTANCE


@ti.kernel
def _copy_vec3(src: ti.template(), dst: ti.template()):  # type: ignore
  for i, j in src:
    dst[i, j] = src[i, j]


class Renderer:
  def __init__(self, config: RenderConfig, environment: Optional[Environment] = None):
    self.config = config
    w, h = config.width, config.height
    shape = (w, h)

    scene.init_scene_fields()
    bvh_upload.init_bvh_fields()
    buffers.init_aov_fields(w, h)
    history.init_history_fields(w, h)
    history.reset()

    self.brdf = BRDF()
    self.environment = environment if environment is not None else Environment(mode=ENV_BLACK)

    self.raw = ti.Vector.field(3, ti.f32, shape=shape)
    self.clean = ti.Vector.field(3, ti.f32, shape=shape)
    self.demod = ti.Vector.field(3, ti.f32, shape=shape)

    self.accum_col = ti.Vector.field(3, ti.f32, shape=shape)
    self.accum_mom = ti.Vector.field(2, ti.f32, shape=shape)
    self.accum_len = ti.field(ti.i32, shape=shape)

    self.colour_b = ti.Vector.field(3, ti.f32, shape=shape)
    self.variance_a = ti.field(ti.f32, shape=shape)
    self.variance_b = ti.field(ti.f32, shape=shape)
    self.var_prefiltered = ti.field(ti.f32, shape=shape)

    self.history_colour = ti.Vector.field(3, ti.f32, shape=shape)
    self.final = ti.Vector.field(3, ti.f32, shape=shape)

    self._prev: Any = None      # (view, proj, position) of the previous frame


  def make_camera(self, pose) -> Camera:
    c = self.config
    return Camera(
      position=np.asarray(pose.position, dtype=np.float32),
      yaw=pose.yaw, pitch=pose.pitch,
      fov=np.radians(c.fov_deg), aspect_ratio=c.aspect_ratio,
      near=c.near, far=c.far,
    )

  def reset_sequence(self):
    history.reset()
    self._prev = None


  def render_aovs(self, camera: Camera):
    right, up, forward = camera.basis_from_yaw_pitch()

    buffers.clear_aovs()
    gbuffer_kernel(
      buffers.albedo, buffers.normal, buffers.object_id,
      buffers.hit_mask, buffers.depth,
      right, up, forward, camera.position, camera.fov, camera.aspect_ratio,
      scene.triangles,
      *bvh_upload.kernel_args(),
    )

    if self._prev is not None:
      motion_kernel(
        buffers.motion, buffers.depth, buffers.hit_mask,
        right, up, forward, camera.position, camera.fov, camera.aspect_ratio,
        self._prev[0], self._prev[1],
      )
    else:
      buffers.motion.fill(0.0)

    return right, up, forward

  def commit_frame(self, camera: Camera):
    self._prev = (
      np.asarray(camera.view_matrix(), dtype=np.float32),
      np.asarray(camera.projection_matrix(), dtype=np.float32),
      np.asarray(camera.position, dtype=np.float64),
    )


  def render_noisy(self, camera: Camera, spp: Optional[int] = None):
    c = self.config
    right, up, forward = camera.basis_from_yaw_pitch()
    accumulate_kernel(
      self.raw, 1, right, up, forward, camera.position,
      camera.fov, camera.aspect_ratio,
      self.brdf, self.environment,
      scene.triangles, scene.num_triangles[None],
      *bvh_upload.kernel_args(),
      scene.light_triangle_index, scene.light_pdf_area, scene.num_lights[None],
      c.single_sided, c.use_nee, c.max_bounces,
      c.spp if spp is None else spp,
    )
    return self.raw

  def render_clean(self, camera: Camera, chunks: Optional[int] = None, spp_per_chunk: Optional[int] = None):
    c = self.config
    chunks = c.clean_chunks if chunks is None else chunks
    spp_per_chunk = c.clean_spp_per_chunk if spp_per_chunk is None else spp_per_chunk

    right, up, forward = camera.basis_from_yaw_pitch()
    self.clean.fill(0.0)

    for k in range(1, chunks + 1):
      accumulate_kernel(
        self.clean, k, right, up, forward, camera.position,
        camera.fov, camera.aspect_ratio,
        self.brdf, self.environment,
        scene.triangles, scene.num_triangles[None],
        *bvh_upload.kernel_args(),
        scene.light_triangle_index, scene.light_pdf_area, scene.num_lights[None],
        c.single_sided, c.use_nee, c.max_bounces, spp_per_chunk,
      )
    return self.clean

  def clean_split_half(self, camera: Camera, chunks: Optional[int] = None, spp_per_chunk: Optional[int] = None) -> float:
    c = self.config
    chunks = c.clean_chunks if chunks is None else chunks
    spp_per_chunk = c.clean_spp_per_chunk if spp_per_chunk is None else spp_per_chunk
    half = max(1, chunks // 2)

    a = self.render_clean(camera, half, spp_per_chunk).to_numpy().astype(np.float64)
    b = self.render_clean(camera, half, spp_per_chunk).to_numpy().astype(np.float64)
    return float(np.sqrt(((a - b) ** 2).mean()) / 2.0)


  def denoise(self, radiance) -> Any:
    c = self.config
    w, h = c.width, c.height

    demodulate(radiance, buffers.albedo, buffers.hit_mask, self.demod)

    reproject_kernel(
      self.demod, buffers.depth, buffers.normal, buffers.object_id, buffers.motion,
      history.colour, history.moments, history.depth, history.normal,
      history.object_id, history.length,
      self.accum_col, self.accum_mom, self.accum_len,
      w, h, c.alpha_min,
    )

    estimate_variance(
      self.accum_mom, self.accum_len, self.accum_col,
      buffers.depth, buffers.normal, buffers.object_id,
      self.variance_a, w, h,
    )

    filtered, _ = filter_image(
      self.accum_col, self.colour_b, self.variance_a, self.variance_b,
      self.var_prefiltered, self.history_colour,
      buffers.depth, buffers.normal, buffers.object_id,
      w, h, levels=c.atrous_levels,
    )

    remodulate(filtered, buffers.albedo, buffers.hit_mask, self.final)

    history.store(
      buffers.normal, buffers.object_id, buffers.depth,
      self.history_colour, self.accum_mom, self.accum_len,
    )
    return self.final

  def render_and_denoise(self, camera: Camera):
    """Full frame, in the order main.py needs it."""
    self.render_aovs(camera)
    self.render_noisy(camera)
    out = self.denoise(self.raw)
    self.commit_frame(camera)
    return out