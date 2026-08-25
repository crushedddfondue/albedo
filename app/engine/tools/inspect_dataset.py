import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data.manifest import Manifest
from data.shard import ShardReader


def pct(a, ps=(1, 5, 25, 50, 75, 95, 99)):
  q = np.percentile(a, ps)
  return "  ".join(f"p{p}={v:.3f}" for p, v in zip(ps, q))


def main(argv=None):
  ap = argparse.ArgumentParser(description="Describe a rendered dataset.")
  ap.add_argument("root", help="dataset directory containing dataset.json")
  ap.add_argument("--max-frames", type=int, default=400, help="cap on frames read, sampled evenly across sequences")
  args = ap.parse_args(argv)

  m = Manifest.load(args.root)
  cfg = m.config
  print("=" * 74)
  print(f"DATASET  {os.path.abspath(args.root)}")
  print("=" * 74)
  print(f"generator: {cfg.get('generator')}   seed: {cfg.get('seed')}   "
        f"resumed: {cfg.get('resumed')}")
  print(f"scenes: {cfg.get('scenes')}   frames/seq: {cfg.get('frames_per_sequence')}   "
        f"noisy realizations: {cfg.get('noisy_realizations')}")
  render = cfg.get("render", {})
  print(f"render: {render.get('width')}x{render.get('height')}  spp {render.get('spp')}  "
        f"clean {render.get('clean_spp')}  bounces {render.get('max_bounces')}")
  sigma = cfg.get("clean_split_half_sigma")
  print(f"clean target own sigma: {sigma if sigma is None else f'{sigma:.6f}'}")
  print()

  for split, v in m.split_summary().items():
    print(f"{split}: {v['scenes']} scenes, {v['sequences']} sequences, {v['frames']} frames")
  print()

  clamped, nonfinite, total_frames = {}, {}, 0
  readers = []
  for shard in m.shards:
    r = ShardReader(os.path.join(m.root, shard["path"]))
    readers.append(r)
    total_frames += len(r)
    for k, n in r.sidecar.get("clamped_f16", {}).items():
      clamped[k] = clamped.get(k, 0) + n
    for k, n in r.sidecar.get("nonfinite_zeroed", {}).items():
      nonfinite[k] = nonfinite.get(k, 0) + n
  print(f"shards: {len(readers)}   frames on disk: {total_frames}")
  print(f"f16 clamped: {clamped or 'none'}")
  print(f"non-finite zeroed: {nonfinite or 'none'}")
  print()

  picks = []
  for r in readers:
    for s in r.sequences:
      for t in range(s.count):
        picks.append((r, s, t))
  if len(picks) > args.max_frames:
    step = len(picks) / args.max_frames
    picks = [picks[int(i * step)] for i in range(args.max_frames)]

  cov, first_motion, later_motion, out_of_frame = [], [], [], []
  noisy_mean, clean_mean, realization_std = [], [], []
  depth_lo, depth_hi, normal_len = [], [], []
  oid_max, per_scene = 0, {}

  for r, s, t in picks:
    f = r.read_frame(s.start + t)
    hit = f["hit_mask"] == 1
    c = float(hit.mean())
    cov.append(c)
    per_scene.setdefault(s.scene_id, []).append(c)

    mag = np.linalg.norm(f["motion"].astype(np.float64), axis=-1)
    (first_motion if t == 0 else later_motion).append(float(mag.max()))

    W, H = r.width, r.height
    xs = np.arange(W)[:, None] + 0.5 + f["motion"][..., 0].astype(np.float64)
    ys = np.arange(H)[None, :] + 0.5 + f["motion"][..., 1].astype(np.float64)
    outside = (xs < 0) | (xs >= W) | (ys < 0) | (ys >= H)
    out_of_frame.append(float(outside[hit].mean()) if hit.any() else 0.0)

    if hit.any():
      noisy_mean.append(float(f["noisy"][hit].mean()))
      clean_mean.append(float(f["clean"][hit].mean()))

      if r.noisy_realizations > 1:
        d = f["noisy"][hit].astype(np.float64)
        realization_std.append(float(np.sqrt(((d[:, 0] - d[:, 1]) ** 2).mean() / 2.0)))
      d = f["depth"][hit]
      depth_lo.append(float(d.min())); depth_hi.append(float(d.max()))
      n = f["normal"][hit].astype(np.float64)
      normal_len.append(float(np.abs(np.linalg.norm(n, axis=1) - 1.0).max()))
      oid_max = max(oid_max, int(f["object_id"][hit].max()))

    bg = ~hit
    if bg.any():
      assert np.isfinite(f["depth"][bg]).all(), "background depth not finite"

  cov = np.asarray(cov)
  print("=" * 74)
  print(f"GEOMETRY COVERAGE   ({len(picks)} frames sampled)")
  print("=" * 74)
  print(f"mean {cov.mean():.3f}   min {cov.min():.3f}   max {cov.max():.3f}")
  print(f"  {pct(cov)}")
  low = [(k, float(np.mean(v))) for k, v in per_scene.items() if np.mean(v) < 0.5]
  print(f"scenes with mean coverage < 0.50: {len(low)}/{len(per_scene)}"
        + (f"  worst {sorted(low, key=lambda x: x[1])[:3]}" if low else ""))
  print("(test_room was 0.32. Higher is better: background pixels carry no")
  print("signal, and a frame that is mostly background spends clean-render")
  print("budget on nothing.)")
  print()

  print("=" * 74)
  print("MOTION")
  print("=" * 74)
  print(f"frame 0 max magnitude: {max(first_motion) if first_motion else 0:.6f}  "
        f"(MUST be 0.0 -- no previous frame exists)")
  lm = np.asarray(later_motion)
  print(f"later frames max px:   mean {lm.mean():.2f}   {pct(lm, (50, 90, 99))}")
  oof = np.asarray(out_of_frame)
  print(f"reprojects outside frame: mean {oof.mean():.2%}   {pct(oof, (50, 90, 99))}")
  print()

  print("=" * 74)
  print("RADIANCE AND AOVs")
  print("=" * 74)
  nm, cm = np.asarray(noisy_mean), np.asarray(clean_mean)
  rel = np.abs(nm - cm) / np.maximum(cm, 1e-9)
  print(f"noisy vs clean mean: median |rel diff| {np.median(rel):.3%}  "
        f"p95 {np.percentile(rel, 95):.3%}")
  print("(both are unbiased estimates of the same integrand; a large or")
  print("one-sided gap means the two render paths disagree about the scene.)")
  if realization_std:
    rs = np.asarray(realization_std)
    print(f"input noise sigma at {render.get('spp')} spp: {rs.mean():.4f}   "
          f"(split-half, same estimator as the clean target)")
    cs = cfg.get("clean_split_half_sigma")
    if cs:
      print(f"ratio to clean target sigma: {rs.mean() / cs:.1f}x   "
            f"(sqrt(clean_spp/spp) predicts "
            f"{(render.get('clean_spp', 1) / max(render.get('spp', 1), 1)) ** 0.5:.1f}x)")
      
  print(f"depth on geometry: {min(depth_lo):.3f} .. {max(depth_hi):.3f}")
  print(f"|normal| max deviation from 1: {max(normal_len):.2e}  (f16 storage)")
  print(f"object_id max: {oid_max}   (int16 ceiling 32767)")
  print("=" * 74)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())