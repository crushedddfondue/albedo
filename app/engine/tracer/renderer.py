import taichi as ti
import math

ti.init(arch=ti.cuda)

vec3 = type(ti.math.vec3)
vec2 = type(ti.math.vec3)
@ti.dataclass
class Ray:
  ro: ti.types.vector(3, ti.f32)
  rd: ti.types.vector(3, ti.f32)

@ti.dataclass
class HitRecord:
  t: ti.f32
  normal: ti.types.vector(3, ti.f32)
  albedo: ti.types.vector(3, ti.f32)
  hit: ti.i32

@ti.dataclass
class Triangle:
  v0: ti.types.vector(3, ti.f32)
  v1: ti.types.vector(3, ti.f32)
  v2: ti.types.vector(3, ti.f32)
  normal: ti.types.vector(3, ti.f32)
  albedo: ti.types.vector(3, ti.f32)

@ti.dataclass
class BVHNode:
  aabb_min: ti.types.vector(3, ti.f32)
  aabb_max: ti.types.vector(3, ti.f32)
  left_child: ti.i32
  right_child: ti.i32
  first_tri: ti.i32
  num_tris: ti.i32

MAX_TRIS = 12
MAX_NODES = 3
triangles = Triangle.field(shape=MAX_TRIS)
bvh_nodes = BVHNode.field(shape=MAX_NODES)

@ti.func
def intersect_triangle(ray: Ray, tri: Triangle) -> HitRecord:
  hit_rec = HitRecord(t=1e20, normal=ti.math.vec3(0), albedo=ti.math.vec3(0), hit=0)
  edge1 = tri.v1 - tri.v0
  edge2 = tri.v2 - tri.v0
  h = ti.math.cross(ray.rd, edge2)
  a = ti.math.dot(edge1, h)
  
  if abs(a) > 1e-6:
    f = 1.0 / a
    s = ray.ro - tri.v0
    u = f * ti.math.dot(s, h)
    if 0.0 <= u <= 1.0:
      q = ti.math.cross(s, edge1)
      v = f * ti.math.dot(ray.rd, q)
      if 0.0 <= v and u + v <= 1.0:
        t = f * ti.math.dot(edge2, q)
        if t > 1e-4:
          hit_rec.t = t
          hit_rec.normal = tri.normal
          hit_rec.albedo = tri.albedo
          hit_rec.hit = 1
  return hit_rec

@ti.func
def intersect_aabb(ray: Ray, aabb_min: ti.math.vec3, aabb_max: ti.math.vec3) -> ti.f32:
  tmin = -1e20
  tmax = 1e20
  
  for i in ti.static(range(3)):
    invD = 1.0 / ray.rd[i]
    t0 = (aabb_min[i] - ray.ro[i]) * invD
    t1 = (aabb_max[i] - ray.ro[i]) * invD
    
    if invD < 0.0:
      t0, t1 = t1, t0
        
    tmin = ti.max(tmin, t0)
    tmax = ti.min(tmax, t1)
      
  hit_t = -1.0
  if tmax >= tmin and tmax > 0.0:
    hit_t = tmin if tmin > 0.0 else tmax
  return hit_t

@ti.func
def traverse_bvh(ray: Ray) -> HitRecord:
  closest_hit = HitRecord(t=1e20, normal=ti.math.vec3(0), albedo=ti.math.vec3(0), hit=0)
  
  # REDUCED STACK SIZE: 16 instead of 64.
  # Fits directly into CUDA registers.
  stack = ti.types.vector(16, ti.i32)(0)
  
  stack_ptr = 0
  stack[stack_ptr] = 0
  
  while stack_ptr >= 0:
    node_idx = stack[stack_ptr]
    stack_ptr -= 1
    
    node = bvh_nodes[node_idx]
    
    if intersect_aabb(ray, node.aabb_min, node.aabb_max) != -1.0:
      if node.num_tris > 0:
        for i in range(node.first_tri, node.first_tri + node.num_tris):
          hit = intersect_triangle(ray, triangles[i])
          if hit.hit == 1 and hit.t < closest_hit.t:
            closest_hit = hit
      else:
        stack_ptr += 1
        stack[stack_ptr] = node.left_child
        stack_ptr += 1
        stack[stack_ptr] = node.right_child
            
  return closest_hit

@ti.kernel
def render_1spp_kernel(
  cam_x: ti.f32, cam_y: ti.f32, cam_z: ti.f32, 
  vram_buffer: ti.types.ndarray(dtype=ti.f32, ndim=3)
):
  width = vram_buffer.shape[0]
  height = vram_buffer.shape[1]
  
  camera_origin = ti.math.vec3(cam_x, cam_y, cam_z + 3.0)
  
  for i, j in ti.ndrange(width, height):
    u = (i / width) * 2.0 - 1.0
    v = (j / height) * 2.0 - 1.0
    
    ray_dir = ti.math.normalize(ti.math.vec3(u, v, -1.0))
    ray = Ray(ro=camera_origin, rd=ray_dir)
    
    hit = traverse_bvh(ray)
    
    if hit.hit == 1:
      light_dir = ti.math.normalize(ti.math.vec3(1.0, 1.0, 1.0))
      diffuse = ti.max(0.0, ti.math.dot(hit.normal, light_dir))
      noise = (ti.random(ti.f32) - 0.5) * 0.5 
      radiance = hit.albedo * diffuse + ti.math.vec3(noise)
      
      # [0:3] Radiance
      vram_buffer[i, j, 0] = ti.math.clamp(radiance[0], 0.0, 1.0)
      vram_buffer[i, j, 1] = ti.math.clamp(radiance[1], 0.0, 1.0)
      vram_buffer[i, j, 2] = ti.math.clamp(radiance[2], 0.0, 1.0)
      
      # [3:6] Normals (remapped from [-1, 1] to [0, 1])
      vram_buffer[i, j, 3] = hit.normal[0] * 0.5 + 0.5
      vram_buffer[i, j, 4] = hit.normal[1] * 0.5 + 0.5
      vram_buffer[i, j, 5] = hit.normal[2] * 0.5 + 0.5
      
      # [6:9] Albedo
      vram_buffer[i, j, 6] = hit.albedo[0]
      vram_buffer[i, j, 7] = hit.albedo[1]
      vram_buffer[i, j, 8] = hit.albedo[2]
      
      # [9] Depth
      vram_buffer[i, j, 9] = ti.math.clamp(hit.t / 10.0, 0.0, 1.0)
      
    else:
      # Background Skybox
      vram_buffer[i, j, 0] = 0.05
      vram_buffer[i, j, 1] = 0.05
      vram_buffer[i, j, 2] = 0.1
      for c in ti.static(range(3, 10)):
        vram_buffer[i, j, c] = 0.0

def setup_mock_scene():
  v = [
    ti.math.vec3(-1, -1, -1), ti.math.vec3( 1, -1, -1), ti.math.vec3( 1,  1, -1), ti.math.vec3(-1,  1, -1),
    ti.math.vec3(-1, -1,  1), ti.math.vec3( 1, -1,  1), ti.math.vec3( 1,  1,  1), ti.math.vec3(-1,  1,  1)
  ]
  indices = [
    0, 1, 2, 0, 2, 3, 
    1, 5, 6, 1, 6, 2, 
    5, 4, 7, 5, 7, 6, 
    4, 0, 3, 4, 3, 7, 
    3, 2, 6, 3, 6, 7, 
    4, 5, 1, 4, 1, 0  
  ]
  
  for i in range(12):
    idx0, idx1, idx2 = indices[i*3], indices[i*3+1], indices[i*3+2]
    v0, v1, v2 = v[idx0], v[idx1], v[idx2]
    
    e1x, e1y, e1z = v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]
    e2x, e2y, e2z = v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]
    nx = e1y*e2z - e1z*e2y
    ny = e1z*e2x - e1x*e2z
    nz = e1x*e2y - e1y*e2x
    length = math.sqrt(nx*nx + ny*ny + nz*nz)
    
    triangles[i] = Triangle(
      v0=v0, v1=v1, v2=v2, 
      normal=ti.math.vec3(nx/length, ny/length, nz/length) if length > 0 else ti.math.vec3(0,0,1),
      albedo=ti.math.vec3(0.8, 0.2, 0.2)
    )
      
  bvh_nodes[0] = BVHNode(
    aabb_min=ti.math.vec3(-1.1, -1.1, -1.1), 
    aabb_max=ti.math.vec3(1.1, 1.1, 1.1),
    left_child=-1, right_child=-1,
    first_tri=0, num_tris=12
  )

setup_mock_scene()