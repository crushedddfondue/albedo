import taichi as ti
from taichi.math import vec3, normalize

from tracer.kernels.path_trace import path_trace
from tracer.geometry.primitives import Ray

@ti.kernel
def render_kernel(image: ti.template(), right: vec3, up: vec3, forward: vec3, position: vec3, fov: ti.f32, aspect_ratio: ti.f32, brdf: ti.template(), triangles: ti.template(), num_triangles: ti.i32, bvh_node_min: ti.template(), bvh_node_max: ti.template(), bvh_node_left: ti.template(), bvh_node_right: ti.template(), bvh_node_start: ti.template(), bvh_node_count: ti.template(), bvh_indices: ti.template(), light_triangle_index: ti.template(), light_pdf_area: ti.template(), num_lights: ti.i32, single_sided: ti.i32, max_bounces: ti.i32, spp: ti.i32):  # type: ignore

  width = image.shape[0]
  height = image.shape[1]

  scale_phi = ti.tan(fov/2)

  for px, py in image:
    radiance = vec3(0.0)

    for _ in range(spp):
      jitter_x = 0.5 # ti.random(ti.f32)
      jitter_y = 0.5 # ti.random(ti.f32)

      ndc_x = (2.0 * ((ti.cast(px, ti.f32) + jitter_x) / width)- 1.0)

      ndc_y = (1.0 - 2.0 * ((ti.cast(py, ti.f32) + jitter_y) / height))

      world_dir = normalize(
        ndc_x * aspect_ratio * scale_phi * right +
        ndc_y * scale_phi * up +
        forward
      )

      ray = Ray(origin=position, direction=world_dir)

      radiance += path_trace(
        ray,
        brdf,
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
        max_bounces
      )

    radiance /= ti.cast(spp, ti.f32)

    radiance = ti.max(radiance, vec3(0.0))
    radiance = ti.pow(radiance, 1.0 / 2.2)
    radiance = ti.math.clamp(radiance, 0.0, 1.0)

    image[px, py] = radiance