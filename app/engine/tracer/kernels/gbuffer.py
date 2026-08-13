import taichi as ti
from taichi.math import vec3

from tracer.camera import Camera
from tracer.bvh.traverse import traverse_closest_hit


@ti.kernel
def gbuffer_kernel(albedo_f: ti.template(), normal_f: ti.template(), object_id_f: ti.template(), hit_mask_f: ti.template(), depth_f: ti.template(), right: vec3, up: vec3, forward: vec3, position: vec3, fov: ti.f32, aspect_ratio: ti.f32, triangles: ti.template(), bvh_node_min: ti.template(), bvh_node_max: ti.template(), bvh_node_left: ti.template(), bvh_node_right: ti.template(), bvh_node_start: ti.template(), bvh_node_count: ti.template(), bvh_indices: ti.template()):  # type: ignore
  """Primary hit only. No bounce loop, no accumulation, no randomness.

  These channels are exact on the first sample, so this kernel overwrites in
  full every frame rather than accumulating -- running a noise-free signal
  through the running mean would blur exactly the edges SVGF depends on.

  Zero ti.random calls anywhere in this path, which is what makes the
  run-twice-and-compare test meaningful.
  """
  width = albedo_f.shape[0]
  height = albedo_f.shape[1]

  for px, py in albedo_f:
    world_dir = Camera.ray_direction_for_pixel(
      px, py, 0.5, 0.5, width, height,
      right, up, forward, fov, aspect_ratio,
    )

    hit = traverse_closest_hit(
      position, world_dir,
      bvh_node_min, bvh_node_max, bvh_node_left, bvh_node_right,
      bvh_node_start, bvh_node_count, bvh_indices, triangles,
    )

    if hit.hit == 0:
      # Background has no surface. Deliberately leave albedo, normal and
      # depth at whatever clear_aovs() set -- writing plausible zeros invites
      # downstream code to read them as meaningful. hit_mask is the authority,
      # and the depth sentinel is huge so anything that forgets to check the
      # mask rejects history rather than blending it.
      hit_mask_f[px, py] = 0
      object_id_f[px, py] = -1
    else:
      # Store the VIEWER-FACING normal, matching what path_trace shades with.
      # Storing the raw geometric normal makes the occluder's underside
      # report a normal pointing away from the camera; SVGF's edge weights
      # then collapse and the filter refuses to blend across a surface that
      # is in fact continuous.
      shading_normal = hit.normal
      if ti.math.dot(-world_dir, hit.normal) < 0.0:
        shading_normal = -hit.normal

      albedo_f[px, py] = hit.albedo
      normal_f[px, py] = shading_normal
      object_id_f[px, py] = hit.object_id
      hit_mask_f[px, py] = 1

      # View-space z, not ray distance. world_dir is normalised and forward
      # is unit, so the dot is exactly cos(angle from view axis) -- one
      # multiply converts t to z. Ray distance would give a flat wall a
      # depth gradient across the screen, and SVGF's edge-stopping compares
      # depth against its local gradient to decide whether two pixels share
      # a surface. A plane would look curved and the filter would refuse to
      # blend across it.
      depth_f[px, py] = hit.t * ti.math.dot(world_dir, forward)