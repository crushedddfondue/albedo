import taichi as ti
from taichi.math import vec3

from app.engine.tracer.camera import PinholeCamera
from app.engine.tracer.kernels.path_trace import path_trace

ti.init(arch=ti.cuda)

@ti.kernel
def render_target(
  image: ti.template(),  # The pixel buffer # type: ignore
  camera: ti.template(), # Your PinholeCamera object  # type: ignore
  spp: ti.i32,           # Samples per pixel (integer)  # type: ignore
  width: ti.i32,         # Integer  # type: ignore
  height: ti.i32,        # Integer  # type: ignore
  fov: ti.f32            # Field of view (float)  # type: ignore
):

  for i, j in image:
    accumulated_colour = vec3(0.0)

    for _ in range(spp):
      jitter_x = ti.random(ti.f32)
      jitter_y = ti.random(ti.f32)

      x_ndc = 2.0 * (i + jitter_x) / float(width) - 1.0
      y_ndc = 2.0 * (j + jitter_y) / float(height) - 1.0

      aspect_ratio = float(width) / float(height)

      ray_d = camera.ray_generation(x_ndc, y_ndc, fov, aspect_ratio)
      ray_o = camera.position

      sample_colour = path_trace(ray_o, ray_d)

      accumulated_colour += sample_colour

    image[i, j] = accumulated_colour / float(spp)