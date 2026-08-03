import numpy as np
import taichi as ti

from tracer.geometry.scene import triangles, num_triangles

LEAF_SIZE = 4

nodes = []
bvh_triangle_indices = []
root_index = -1

centroids = None
tri_min = None
tri_max = None

bvh_node_min = None
bvh_node_max = None
bvh_node_left = None
bvh_node_right = None
bvh_node_start = None
bvh_node_count = None
bvh_indices_field = None


def build(tri_idx_list):
  node = {"min": None, "max": None, "left": -1, "right": -1, "start": -1, "count": 0}

  node["min"] = tri_min[tri_idx_list].min(axis=0) # type: ignore
  node["max"] = tri_max[tri_idx_list].max(axis=0) # type: ignore

  this_node_index = len(nodes)
  nodes.append(node)

  if len(tri_idx_list) <= LEAF_SIZE:
    node["start"] = len(bvh_triangle_indices)
    node["count"] = len(tri_idx_list)
    bvh_triangle_indices.extend(tri_idx_list)
    return this_node_index

  node_centroids = centroids[tri_idx_list]  # type: ignore
  centroid_min = node_centroids.min(axis=0)
  centroid_max = node_centroids.max(axis=0)
  axis = np.argmax(centroid_max - centroid_min)

  tri_idx_list = sorted(tri_idx_list, key=lambda i: centroids[i][axis]) # type: ignore
  mid = len(tri_idx_list) // 2

  left_index = build(tri_idx_list[:mid])
  right_index = build(tri_idx_list[mid:])

  node["left"] = left_index
  node["right"] = right_index
  node["count"] = 0

  return this_node_index


def build_bvh():
  """Call once the scene is actually populated -- e.g. right after
  mock_scenes.build_test_room(). Everything reading triangle data has to
  live in here, not at module level, for the same reason ti.init() couldn't
  live at the top of camera.py: import order isn't something to depend on."""
  global centroids, tri_min, tri_max, root_index

  n = num_triangles[None]

  v0 = triangles.v0.to_numpy()[:n]
  v1 = triangles.v1.to_numpy()[:n]
  v2 = triangles.v2.to_numpy()[:n]

  centroids = (v0 + v1 + v2) / 3.0
  tri_min = np.minimum(np.minimum(v0, v1), v2)
  tri_max = np.maximum(np.maximum(v0, v1), v2)

  nodes.clear()
  bvh_triangle_indices.clear()

  root_index = build(list(range(n)))  # type: ignore
  return root_index


def upload_to_taichi():
  global bvh_node_min, bvh_node_max, bvh_node_left, bvh_node_right, bvh_node_start, bvh_node_count, bvh_indices_field

  n_nodes = len(nodes)
  n_indices = len(bvh_triangle_indices)

  bvh_node_min = ti.Vector.field(3, dtype=ti.f32, shape=(n_nodes,))
  bvh_node_max = ti.Vector.field(3, dtype=ti.f32, shape=(n_nodes,))
  bvh_node_left = ti.field(dtype=ti.i32, shape=(n_nodes,))
  bvh_node_right = ti.field(dtype=ti.i32, shape=(n_nodes,))
  bvh_node_start = ti.field(dtype=ti.i32, shape=(n_nodes,))
  bvh_node_count = ti.field(dtype=ti.i32, shape=(n_nodes,))
  bvh_indices_field = ti.field(dtype=ti.i32, shape=(n_indices,))

  bvh_node_min.from_numpy(np.array([n["min"] for n in nodes], dtype=np.float32))
  bvh_node_max.from_numpy(np.array([n["max"] for n in nodes], dtype=np.float32))
  bvh_node_left.from_numpy(np.array([n["left"] for n in nodes], dtype=np.int32))
  bvh_node_right.from_numpy(np.array([n["right"] for n in nodes], dtype=np.int32))
  bvh_node_start.from_numpy(np.array([n["start"] for n in nodes], dtype=np.int32))
  bvh_node_count.from_numpy(np.array([n["count"] for n in nodes], dtype=np.int32))
  bvh_indices_field.from_numpy(np.array(bvh_triangle_indices, dtype=np.int32))