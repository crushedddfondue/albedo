import taichi as ti

from tracer.geometry.primitives import Triangle, compute_triangle_normal

MAX_TRIANGLES = 1024
MAX_LIGHTS = 64

triangles = Triangle.field(shape=(MAX_TRIANGLES,))
num_triangles = ti.field(dtype=ti.i32, shape=())

light_triangle_index = ti.field(dtype=ti.i32, shape=(MAX_LIGHTS,))
light_pdf_area = ti.field(dtype=ti.f32, shape=(MAX_LIGHTS,))
num_lights = ti.field(dtype=ti.i32, shape=())


@ti.kernel
def recompute_normals():
  for i in range(num_triangles[None]):  # type: ignore
    triangles[i].normal = compute_triangle_normal(triangles[i])


@ti.kernel
def build_light_list():
  num_lights[None] = 0

  for i in range(num_triangles[None]):  # type: ignore
    emission = triangles[i].emission

    if emission.x > 0.0 or emission.y > 0.0 or emission.z > 0.0:
      idx = ti.atomic_add(num_lights[None], 1)

      if idx < MAX_LIGHTS:
        v0 = triangles[i].v0
        v1 = triangles[i].v1
        v2 = triangles[i].v2

        cross_prod = ti.math.cross(v1 - v0, v2 - v0)
        area = 0.5 * cross_prod.norm()

        light_triangle_index[idx] = i
        triangles[i].light_index = idx

        if area > 1e-8:
          light_pdf_area[idx] = 1.0 / area
        else:
          light_pdf_area[idx] = 0.0