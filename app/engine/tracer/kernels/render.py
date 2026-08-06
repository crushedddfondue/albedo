"""
app/engine/tracer/kernels/render.py

Split into two kernels, deliberately.

  accumulate_kernel  traces paths and accumulates LINEAR HDR radiance
  resolve_kernel     tone maps and gamma encodes for display

The previous single kernel applied gamma and clamped to [0,1] internally, so
the only thing leaving the tracer was display-referred, range-limited,
non-linear data. Three problems with that, all of which this split fixes:

  - Information was destroyed. The light quad's radiance of 2.85 clamped to
    1.0; anything brighter than white was unrecoverable.
  - Every numerical check -- furnace, form factor, A/B ratio -- is a statement
    about linear radiance, so each had to undo the gamma with ** 2.2. Lossy,
    and invalid on any saturated pixel.
  - The Phase 2 Rectified Flow denoiser needs linear HDR. Feeding it gamma
    encoded, clamped data means it has to learn to invert a non-invertible
    transform as part of denoising.

Accumulation also removes the Windows TDR ceiling. The driver resets the GPU
on any kernel running past ~2 s, which bounded how much work a single launch
could do; an unbounded number of bounded launches has no such limit.
"""

import taichi as ti
from taichi.math import vec3, normalize

from tracer.kernels.path_trace import path_trace
from tracer.geometry.primitives import Ray

TONEMAP_NONE = 0        # clamp only -- reproduces the old behaviour
TONEMAP_REINHARD = 1
TONEMAP_REINHARD_EXT = 2


@ti.kernel
def accumulate_kernel(accum: ti.template(), frame_index: ti.i32, right: vec3, up: vec3, forward: vec3, position: vec3, fov: ti.f32, aspect_ratio: ti.f32, brdf: ti.template(), environment: ti.template(), triangles: ti.template(), num_triangles: ti.i32, bvh_node_min: ti.template(), bvh_node_max: ti.template(), bvh_node_left: ti.template(), bvh_node_right: ti.template(), bvh_node_start: ti.template(), bvh_node_count: ti.template(), bvh_indices: ti.template(), light_triangle_index: ti.template(), light_pdf_area: ti.template(), num_lights: ti.i32, single_sided: ti.i32, use_nee: ti.i32, max_bounces: ti.i32, spp: ti.i32):  # type: ignore
  """frame_index is 1-based. accum must be zeroed before frame 1."""

  width = accum.shape[0]
  height = accum.shape[1]

  scale_phi = ti.tan(fov / 2)
  inv_n = 1.0 / ti.cast(frame_index, ti.f32)

  for px, py in accum:
    radiance = vec3(0.0)

    for _ in range(spp):
      jitter_x = ti.random(ti.f32)
      jitter_y = ti.random(ti.f32)

      ndc_x = (2.0 * ((ti.cast(px, ti.f32) + jitter_x) / width) - 1.0)

      # ti.GUI.set_image uses a lower-left origin, so py = 0 is the BOTTOM
      # row, not the top. The usual raster-convention mapping
      # (1 - 2*(py+j)/height) assumes upper-left and renders the frame
      # vertically flipped.
      ndc_y = (2.0 * ((ti.cast(py, ti.f32) + jitter_y) / height) - 1.0)

      world_dir = normalize(
        ndc_x * aspect_ratio * scale_phi * right +
        ndc_y * scale_phi * up +
        forward
      )

      ray = Ray(origin=position, direction=world_dir)

      radiance += path_trace(
        ray,
        brdf,
        environment,
        triangles,
        num_triangles,
        bvh_node_min,
        bvh_node_max,
        bvh_node_left,
        bvh_node_right,
        bvh_node_start,
        bvh_node_count,
        bvh_indices,
        light_triangle_index,
        light_pdf_area,
        num_lights,
        single_sided,
        use_nee,
        max_bounces
      )

    radiance /= ti.cast(spp, ti.f32)

    # Running mean, not sum-then-divide. Algebraically identical, but keeps
    # the accumulator at the scale of the signal rather than n times it --
    # f32 loses precision once the running sum grows large. Also gives a
    # displayable image after every frame instead of only at the end.
    #
    #   m_n = m_{n-1} + (x_n - m_{n-1}) / n
    #
    # At frame_index = 1 with accum zeroed, this reduces to accum = radiance.
    accum[px, py] += (radiance - accum[px, py]) * inv_n


@ti.kernel
def resolve_kernel(accum: ti.template(), display: ti.template(), exposure: ti.f32, tonemap: ti.i32, white_point: ti.f32):  # type: ignore
  """Linear HDR -> display referred. The only place gamma is applied."""

  for px, py in accum:
    L = ti.max(accum[px, py], vec3(0.0)) * exposure

    mapped = L
    if tonemap == TONEMAP_REINHARD:
      # L / (1 + L). Compresses rather than truncates: highlights roll off
      # smoothly instead of flattening into a plateau of pure white.
      mapped = L / (1.0 + L)
    elif tonemap == TONEMAP_REINHARD_EXT:
      # Extended form -- maps white_point exactly to 1.0 instead of only
      # approaching it asymptotically, so real whites stay white.
      w2 = white_point * white_point
      mapped = (L * (1.0 + L / w2)) / (1.0 + L)

    display[px, py] = ti.math.clamp(ti.pow(mapped, 1.0 / 2.2), 0.0, 1.0)