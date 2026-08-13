import taichi as ti

from tracer.geometry.primitives import Triangle, compute_triangle_normal

MAX_TRIANGLES = 1024
MAX_LIGHTS = 64

triangles = None
num_triangles = None
light_triangle_index = None
light_pdf_area = None
num_lights = None


def init_scene_fields():
  """Call once, right after ti.init(). Idempotent.

  Field allocation can't live at module level -- import order isn't something
  to depend on -- which is the original reason this function exists.

  ⚠ The guard below matters just as much. recompute_normals, build_light_list
  and mock_scenes' kernels all read these module globals, and Taichi captures
  them at COMPILE time, meaning the first call. Re-allocating rebinds the
  Python names but leaves those kernels writing into the orphaned originals:
  _write_test_room() sets num_triangles = 6 on a field nobody reads while
  build_bvh() sees 0 on the new one. It surfaces as a zero-size array error
  deep inside numpy, nowhere near the cause.
  """
  global triangles, num_triangles, light_triangle_index, light_pdf_area, num_lights

  if triangles is not None:
    return

  triangles = Triangle.field(shape=(MAX_TRIANGLES,))
  num_triangles = ti.field(dtype=ti.i32, shape=())
  light_triangle_index = ti.field(dtype=ti.i32, shape=(MAX_LIGHTS,))
  light_pdf_area = ti.field(dtype=ti.f32, shape=(MAX_LIGHTS,))
  num_lights = ti.field(dtype=ti.i32, shape=())


@ti.kernel
def recompute_normals():
  for i in range(num_triangles[None]):  # type: ignore
    triangles[i].normal = compute_triangle_normal(triangles[i])  # type: ignore


@ti.kernel
def build_light_list():
  num_lights[None] = 0  # type: ignore

  # FOLLOW-UP: ti.atomic_add in a parallel loop means light ordering is
  # nondeterministic between runs. Harmless for correctness -- the pmf is
  # uniform either way -- but it breaks the "same scene gives bit-identical
  # output" property that has caught two bugs so far. Worth a serial pass
  # or a sort during the next cleanup.
  for i in range(num_triangles[None]):  # type: ignore
    emission = triangles[i].emission  # type: ignore

    if emission.x > 0.0 or emission.y > 0.0 or emission.z > 0.0:
      idx = ti.atomic_add(num_lights[None], 1)  # type: ignore

      if idx < MAX_LIGHTS:
        v0 = triangles[i].v0  # type: ignore
        v1 = triangles[i].v1  # type: ignore
        v2 = triangles[i].v2  # type: ignore

        cross_prod = ti.math.cross(v1 - v0, v2 - v0)
        area = 0.5 * cross_prod.norm()

        light_triangle_index[idx] = i  # type: ignore
        triangles[i].light_index = idx  # type: ignore

        if area > 1e-8:
          light_pdf_area[idx] = 1.0 / area  # type: ignore
        else:
          light_pdf_area[idx] = 0.0  # type: ignore