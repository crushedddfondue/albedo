import taichi as ti

ti.init(arch=ti.cuda)

@ti.kernel
def render_1spp_kernel(cam_x: ti.f32, cam_y: ti.f32, cam_z: ti.f32, vram_buffer: ti.types.ndarray(dtype=ti.math.vec3))-> None:  # type: ignore
  width, height = vram_buffer.shape[0], vram_buffer.shape[1]

  for i, j in ti.ndrange(width, height):
    # TODO: Implement BVH Traversal

    noise = ti.random(ti.f32) * 0.5
    r = (i / width) * noise + (cam_x * 0.1)
    g = (i / height) * noise + (cam_y * 0.1)
    b = noise + (cam_z * 0.1)

    vram_buffer[i, j] = ti.math.vec3(r, g, b)