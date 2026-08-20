from typing import Any

import numpy as np
import taichi as ti

from tracer.bvh import builder
from tracer.geometry import scene

capacity_nodes = 0
capacity_indices = 0

bvh_node_min: Any = None
bvh_node_max: Any = None
bvh_node_left: Any = None
bvh_node_right: Any = None
bvh_node_start: Any = None
bvh_node_count: Any = None
bvh_indices: Any = None

_EMPTY_MIN = 1e30
_EMPTY_MAX = -1e30


def init_bvh_fields(max_nodes: int = None, max_indices: int = None):  # type: ignore
  global capacity_nodes, capacity_indices
  global bvh_node_min, bvh_node_max, bvh_node_left, bvh_node_right
  global bvh_node_start, bvh_node_count, bvh_indices

  if max_nodes is None:
    max_nodes = 2 * scene.MAX_TRIANGLES
  if max_indices is None:
    max_indices = scene.MAX_TRIANGLES

  if bvh_node_min is not None:
    if capacity_nodes < max_nodes or capacity_indices < max_indices:
      raise RuntimeError(
        f"BVH fields already allocated for {capacity_nodes} nodes / "
        f"{capacity_indices} indices; cannot grow to {max_nodes} / "
        f"{max_indices}. Kernels compiled against the old fields would keep "
        f"writing to them."
      )
    return

  capacity_nodes = int(max_nodes)
  capacity_indices = int(max_indices)

  bvh_node_min = ti.Vector.field(3, ti.f32, shape=(capacity_nodes,))
  bvh_node_max = ti.Vector.field(3, ti.f32, shape=(capacity_nodes,))
  bvh_node_left = ti.field(ti.i32, shape=(capacity_nodes,))
  bvh_node_right = ti.field(ti.i32, shape=(capacity_nodes,))
  bvh_node_start = ti.field(ti.i32, shape=(capacity_nodes,))
  bvh_node_count = ti.field(ti.i32, shape=(capacity_nodes,))
  bvh_indices = ti.field(ti.i32, shape=(capacity_indices,))


def rebuild():
  if bvh_node_min is None:
    raise RuntimeError("init_bvh_fields() must be called before rebuild()")

  root = builder.build_bvh()
  if root != 0:
    raise RuntimeError(
      f"BVH root index is {root}, expected 0. Traversal starts at node 0 "
      f"unconditionally."
    )

  n_nodes = len(builder.nodes)
  n_indices = len(builder.bvh_triangle_indices)

  if n_nodes > capacity_nodes or n_indices > capacity_indices:
    raise RuntimeError(
      f"BVH overflow: {n_nodes} nodes / {n_indices} indices against a "
      f"capacity of {capacity_nodes} / {capacity_indices}."
    )

  node_min = np.full((capacity_nodes, 3), _EMPTY_MIN, dtype=np.float32)
  node_max = np.full((capacity_nodes, 3), _EMPTY_MAX, dtype=np.float32)
  node_left = np.full(capacity_nodes, -1, dtype=np.int32)
  node_right = np.full(capacity_nodes, -1, dtype=np.int32)
  node_start = np.zeros(capacity_nodes, dtype=np.int32)
  node_count = np.zeros(capacity_nodes, dtype=np.int32)
  indices = np.zeros(capacity_indices, dtype=np.int32)

  node_min[:n_nodes] = np.asarray([n["min"] for n in builder.nodes], dtype=np.float32)
  node_max[:n_nodes] = np.asarray([n["max"] for n in builder.nodes], dtype=np.float32)
  node_left[:n_nodes] = np.asarray([n["left"] for n in builder.nodes], dtype=np.int32)
  node_right[:n_nodes] = np.asarray([n["right"] for n in builder.nodes], dtype=np.int32)
  node_start[:n_nodes] = np.asarray([n["start"] for n in builder.nodes], dtype=np.int32)
  node_count[:n_nodes] = np.asarray([n["count"] for n in builder.nodes], dtype=np.int32)
  indices[:n_indices] = np.asarray(builder.bvh_triangle_indices, dtype=np.int32)

  bvh_node_min.from_numpy(node_min)
  bvh_node_max.from_numpy(node_max)
  bvh_node_left.from_numpy(node_left)
  bvh_node_right.from_numpy(node_right)
  bvh_node_start.from_numpy(node_start)
  bvh_node_count.from_numpy(node_count)
  bvh_indices.from_numpy(indices)

  return n_nodes, n_indices


def kernel_args():
  return (
    bvh_node_min, bvh_node_max,
    bvh_node_left, bvh_node_right,
    bvh_node_start, bvh_node_count, bvh_indices,
  )