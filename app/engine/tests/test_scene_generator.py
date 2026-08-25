"""Scene generator tests. No GPU, no Taichi -- everything here runs on
SceneData, which is why build_scene and upload_scene are separate functions.

The winding tests are the ones that matter. SINGLE_SIDED = 1 means an emitter
wound the wrong way lights nothing and the frame renders black, and a black
frame looks exactly like a scene with no lights in it.
"""

import numpy as np
import pytest

from tracer.geometry.scene_generator import (
  SceneParams, build_scene, _tri_normal, TRIS_PER_QUAD,
)


def _normals(d):
  return np.stack([_tri_normal(d.v0[i].astype(np.float64),
                               d.v1[i].astype(np.float64),
                               d.v2[i].astype(np.float64))
                   for i in range(d.n_triangles)])


def test_deterministic_in_seed():
  a = build_scene(SceneParams(), 12345)
  b = build_scene(SceneParams(), 12345)
  for name in ("v0", "v1", "v2", "albedo", "emission", "object_id", "light_index"):
    assert np.array_equal(getattr(a, name), getattr(b, name)), name


def test_different_seeds_differ():
  a = build_scene(SceneParams(), 1)
  b = build_scene(SceneParams(), 2)
  assert a.n_triangles != b.n_triangles or not np.array_equal(a.v0, b.v0)


def test_emitters_face_down():
  """Every light triangle's geometric normal must have a negative y.

  Not "roughly downward" -- exactly negative, because the emission test in
  path_trace is a sign test on dot(n, w) and there is no tolerance band in
  which a sideways emitter half works.
  """
  for seed in range(20):
    d = build_scene(SceneParams(), seed)
    n = _normals(d)
    for t in d.light_triangle_index:
      assert n[t][1] < -0.99, f"seed {seed} emitter {t} normal {n[t]}"


def test_room_normals_face_inward():
  for seed in range(10):
    d = build_scene(SceneParams(), seed)
    n = _normals(d)
    room = np.where(d.object_id == 0)[0]
    centroid = (d.v0[room] + d.v1[room] + d.v2[room]) / 3.0
    interior = np.array([0.0, d.room_size[1] * 0.5, 0.0])
    to_interior = interior - centroid
    assert np.all((n[room] * to_interior).sum(axis=1) > 0), f"seed {seed}"


def test_box_normals_face_outward():
  for seed in range(10):
    d = build_scene(SceneParams(), seed)
    n = _normals(d)
    for oid in np.unique(d.object_id):
      if oid == 0:
        continue
      tris = np.where(d.object_id == oid)[0]
      if d.emission[tris].max() > 0:
        continue
      verts = np.concatenate([d.v0[tris], d.v1[tris], d.v2[tris]])
      centre = verts.mean(axis=0)
      centroid = (d.v0[tris] + d.v1[tris] + d.v2[tris]) / 3.0
      assert np.all((n[tris] * (centroid - centre)).sum(axis=1) > 0), f"seed {seed} obj {oid}"


def test_light_list_matches_emissive_triangles():
  """The Python light list must reproduce build_light_list's semantics
  exactly: same membership, pdf = 1/area, light_index round-trips."""
  for seed in range(20):
    d = build_scene(SceneParams(), seed)
    emissive = np.where(d.emission.max(axis=1) > 0.0)[0]
    assert np.array_equal(np.sort(d.light_triangle_index), np.sort(emissive))
    assert d.n_lights % TRIS_PER_QUAD == 0

    for li, ti_ in enumerate(d.light_triangle_index):
      assert d.light_index[ti_] == li
      e1 = d.v1[ti_].astype(np.float64) - d.v0[ti_]
      e2 = d.v2[ti_].astype(np.float64) - d.v0[ti_]
      area = 0.5 * np.linalg.norm(np.cross(e1, e2))
      assert d.light_pdf_area[li] == pytest.approx(1.0 / area, rel=1e-5)

    assert np.all(d.light_index[d.emission.max(axis=1) == 0.0] == -1)


def test_triangle_budget_respected():
  p = SceneParams()
  for seed in range(64):
    d = build_scene(p, seed)
    assert d.n_triangles <= p.max_triangles
    assert d.n_lights <= p.max_lights


def test_no_degenerate_triangles():
  for seed in range(20):
    d = build_scene(SceneParams(), seed)
    e1 = d.v1.astype(np.float64) - d.v0
    e2 = d.v2.astype(np.float64) - d.v0
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    assert area.min() > 1e-6, f"seed {seed} min area {area.min()}"


def test_geometry_inside_room():
  for seed in range(20):
    d = build_scene(SceneParams(), seed)
    lo, hi = d.bounds
    verts = np.concatenate([d.v0, d.v1, d.v2]).astype(np.float64)
    assert np.all(verts[:, 0] >= lo[0] - 1e-4) and np.all(verts[:, 0] <= hi[0] + 1e-4)
    assert np.all(verts[:, 1] >= -1e-4) and np.all(verts[:, 1] <= hi[1] + 1e-4)
    assert np.all(verts[:, 2] >= lo[2] - 1e-4) and np.all(verts[:, 2] <= hi[2] + 1e-4)


def test_box_normals_are_not_all_axis_aligned():
  """The normal channel must carry a continuum, not six values.

  Axis-aligned boxes give normals that are exactly +-1 on one axis. They are
  exactly representable in f16, so the first pilot reported "|normal| max
  deviation from 1: 0.00e+00" -- which looked like a clean round trip and was
  actually the tell that the entire corpus held six distinct normals.

  A model trained on that has never seen an oblique surface, and terrain or
  interpolated vertex normals produce nothing else. This is the tripwire.
  """
  oblique = total = 0
  for seed in range(20260819, 20260819 + 40):
    d = build_scene(SceneParams(), seed)
    n = _normals(d)
    boxes = np.where((d.object_id > 0) & (d.emission.max(axis=1) == 0.0))[0]
    if boxes.size == 0:
      continue
    axis_aligned = np.abs(n[boxes]).max(axis=1) > 0.999
    oblique += int((~axis_aligned).sum())
    total += boxes.size

  assert total > 0, "no occluder triangles in 40 scenes"
  frac = oblique / total
  # Vertical faces rotate; the top and bottom faces stay +-Y whatever the
  # yaw, so a fraction near 1.0 is not expected and would mean something odd.
  assert 0.4 < frac < 0.9, f"{frac:.2%} of box normals are oblique"


def test_box_yaw_can_be_disabled():
  """box_yaw_max = 0 must reproduce the axis-aligned distribution exactly,
  so the change is reversible and the old behaviour stays reachable."""
  d = build_scene(SceneParams(box_yaw_max=0.0), 20260819)
  n = _normals(d)
  boxes = np.where((d.object_id > 0) & (d.emission.max(axis=1) == 0.0))[0]
  assert np.all(np.abs(n[boxes]).max(axis=1) > 0.999)