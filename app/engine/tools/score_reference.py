"""Score SVGF on the reference set, on the SAME frames the model will use.

SVGF's published 0.00140 came from score_denoiser.py: ONE hardcoded scene,
a STATIC camera, 120 warmup frames, 1280x720, ENV_BLACK. Every one of those
differs from the validation corpus, and the static camera in particular is
SVGF's best case -- zero motion vectors, no disocclusion, history saturated.
Comparing a model's val relMSE against that number would be comparing
populations, not methods.

This re-measures the bar on the reference set's own frames.

    python tools/score_reference.py --dataset datasets/phase23 --reference datasets/ref23

⚠ No path tracing happens here. Every noisy frame and every AOV is already
in the shards, so the SVGF chain is replayed off disk. That is why it takes
seconds, and it is also what makes the comparison exact: the model will
consume the identical bytes.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data.manifest import Manifest
from data.shard import ShardReader
from metrics.image_metrics import relmse, EPS as RELMSE_EPS

REF_CHANNELS = 4


def parse_args(argv=None):
  p = argparse.ArgumentParser(description="Score a denoiser against the reference set.")
  p.add_argument("--dataset", required=True)
  p.add_argument("--reference", required=True, help="directory containing reference.json")
  p.add_argument("--method", default="svgf", help="label recorded in the results file")
  p.add_argument("--out", default=None, help="results json; defaults to <reference>/score_<method>.json")
  p.add_argument("--no-warmup", action="store_true",
                 help="score the window cold instead of replaying from frame 0. "
                      "Reports what a denoiser does with no history, which is a "
                      "different question and not the bar")
  return p.parse_args(argv)


def load_reference(root):
  with open(os.path.join(root, "reference.json")) as f:
    index = json.load(f)
  w, h = index["width"], index["height"]
  stride = w * h * REF_CHANNELS
  mm = np.memmap(os.path.join(root, "reference.bin"), dtype=np.float32, mode="r",
                 shape=(len(index["entries"]), stride))

  def get(i):
    flat = mm[i]
    ref = flat[: w * h * 3].reshape(w, h, 3)
    sigma = flat[w * h * 3:].reshape(w, h)
    return np.asarray(ref), np.asarray(sigma)

  return index, get


def sequence_map(manifest):
  """(shard_path, frame) -> (seq, shard_path). Built once; the reference index
  stores the join key but not the sequence bounds, and the warmup replay needs
  to know where the sequence starts."""
  out = {}
  for shard in manifest.shards:
    for seq in shard["sequences"]:
      for t in range(seq["count"]):
        out[(shard["path"], seq["start"] + t)] = seq
  return out


def main(argv=None):
  args = parse_args(argv)

  manifest = Manifest.load(args.dataset)
  index, ref_at = load_reference(args.reference)
  render = manifest.config["render"]
  if (index["width"], index["height"]) != (render["width"], render["height"]):
    raise SystemExit("error: reference and dataset disagree on resolution")

  seq_of = sequence_map(manifest)
  readers = {}

  # Group the scored frames by sequence so each sequence is replayed once.
  work = {}
  for i, e in enumerate(index["entries"]):
    key = (e["shard"], seq_of[(e["shard"], e["frame"])]["start"])
    work.setdefault(key, []).append((i, e))
  for v in work.values():
    v.sort(key=lambda x: x[1]["frame"])

  import taichi as ti
  from tracer import buffers
  from tracer.pipeline import RenderConfig, get_renderer
  from dataclasses import fields as dataclass_fields

  ti.init(arch=ti.cuda)
  known = {f.name for f in dataclass_fields(RenderConfig)}
  cfg = RenderConfig(**{k: v for k, v in render.items() if k in known})
  renderer = get_renderer(cfg)
  raw = ti.Vector.field(3, ti.f32, shape=(cfg.width, cfg.height))

  def load_frame(reader, idx):
    """Push one stored frame into the AOV buffers the SVGF chain reads.

    ⚠ Realisation 0, always. Which noisy realisation is scored must not
    depend on anything, or the bar moves between runs of the same code.
    """
    f = reader.read_frame(idx)
    noisy = f["noisy"]
    noisy = noisy[:, :, 0, :] if noisy.ndim == 4 else noisy
    raw.from_numpy(np.ascontiguousarray(noisy, dtype=np.float32))
    buffers.albedo.from_numpy(np.ascontiguousarray(f["albedo"], dtype=np.float32))
    buffers.normal.from_numpy(np.ascontiguousarray(f["normal"], dtype=np.float32))
    buffers.depth.from_numpy(np.ascontiguousarray(f["depth"], dtype=np.float32))
    buffers.motion.from_numpy(np.ascontiguousarray(f["motion"], dtype=np.float32))
    buffers.object_id.from_numpy(np.ascontiguousarray(f["object_id"], dtype=np.int32))
    buffers.hit_mask.from_numpy(np.ascontiguousarray(f["hit_mask"], dtype=np.int32))
    return f

  rows = []
  print("=" * 70)
  print(f"SCORING {args.method} on {len(index['entries'])} reference frames")
  print("=" * 70)

  for (shard_path, seq_start), items in sorted(work.items()):
    if shard_path not in readers:
      readers[shard_path] = ShardReader(os.path.join(manifest.root, shard_path))
    reader = readers[shard_path]
    seq = seq_of[(shard_path, items[0][1]["frame"])]

    # ⚠ Replay from the sequence's first frame. SVGF is a temporal filter;
    # scoring a mid-sequence window with empty history measures a cold start,
    # not the filter. The published 0.00140 used 120 warmup frames, so
    # anything less here would tilt the comparison the other way.
    first_scored = items[0][1]["frame"]
    start = seq_start if not args.no_warmup else first_scored
    scored = {e["frame"]: i for i, e in items}

    renderer.reset_sequence()
    for idx in range(start, items[-1][1]["frame"] + 1):
      f = load_frame(reader, idx)
      out = renderer.denoise(raw).to_numpy().astype(np.float64)

      if idx not in scored:
        continue
      ref, sigma = ref_at(scored[idx])
      hit = f["hit_mask"] == 1
      rows.append({
        "scene_id": seq["scene_id"],
        "trajectory_seed": seq["trajectory_seed"],
        "shard": shard_path,
        "frame": idx,
        "frame_in_sequence": idx - seq_start,
        "relmse": float(relmse(out, ref)),
        "relmse_hit": float(relmse(out, ref, mask=hit)),
        "floor": float((sigma[..., None] ** 2 / (ref + RELMSE_EPS) ** 2).mean()),
      })

    print(f"{seq['scene_id']} traj {seq['trajectory_seed']}: "
          f"{len(items)} frames, warmup {first_scored - start}")

  vals = np.asarray([r["relmse"] for r in rows])
  hits = np.asarray([r["relmse_hit"] for r in rows])
  floors = np.asarray([r["floor"] for r in rows])

  by_scene = {}
  for r in rows:
    by_scene.setdefault(r["scene_id"], []).append(r["relmse"])
  scene_means = np.asarray([np.mean(v) for v in by_scene.values()])

  out_path = args.out or os.path.join(args.reference, f"score_{args.method}.json")
  with open(out_path, "w") as f:
    json.dump({
      "method": args.method,
      "dataset": os.path.abspath(args.dataset),
      "reference": os.path.abspath(args.reference),
      "relmse_eps": RELMSE_EPS,
      "warmup": not args.no_warmup,
      "frames": rows,
    }, f, indent=1)

  print()
  print("=" * 70)
  print(f"relMSE all pixels: {vals.mean():.6f}")
  print(f"relMSE on geometry: {hits.mean():.6f}")
  print(f"reference floor: {floors.mean():.6f} "
        f"({100 * floors.mean() / vals.mean():.1f}% of the score)")
  print()
  # ⚠ Per-SCENE, because frames inside one window are near-duplicates. The
  # spread here is the number that decides how large an improvement is
  # defensible -- see effective_sample_size.
  print(f"per scene: {len(scene_means)} scenes, "
        f"min {scene_means.min():.6f} max {scene_means.max():.6f}")
  print(f"  median {np.median(scene_means):.6f}  "
        f"p25 {np.percentile(scene_means, 25):.6f}  "
        f"p75 {np.percentile(scene_means, 75):.6f}")
  print()
  print(f"score_denoiser.py reported 0.00140 on ONE static-camera scene at")
  print(f"1280x720 with 120 warmup frames. This is the matched number; the")
  print(f"two are not interchangeable and only this one can judge the model.")
  print(f"results: {out_path}")
  print("=" * 70)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())