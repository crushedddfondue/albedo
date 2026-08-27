"""Render the held-out evaluation reference.

The dataset's `clean` channel carries sigma ~0.0265, which puts a relMSE
floor near 0.0029 -- roughly TWICE SVGF's measured 0.00140. That target can
train a model under an unbiased loss. It cannot referee one. This renders a
separate, much quieter reference on the validation split so a claim about
0.00140 can actually be defended.

    python tools/make_reference.py --dataset datasets/phase23 --out datasets/ref23late
    python tools/make_reference.py --dataset datasets/phase23 --out datasets/ref23late --dry-run

The poses are read back OUT OF THE MANIFEST rather than re-derived from
sample_trajectory. That is deliberate: a reference is only a reference if it
lands on the same pixels as the frame it judges, and re-deriving would make
that correspondence depend on the trajectory sampler never changing again.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import fields as dataclass_fields

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data.manifest import Manifest
from metrics.image_metrics import EPS as RELMSE_EPS

FORMAT_VERSION = 1

# The bar. Hardcoded on purpose: this tool exists to say whether a reference
# can resolve a difference against THIS number, and a flag would invite
# quietly moving it.
SVGF_RELMSE = 0.00140

# reference RGB + per-pixel sigma, both f32.
#
# ⚠ f32, not f16. The whole point of this file is a quiet signal; f16 has
# ~3 decimal digits, and sigma here is ~1e-3 of the radiance it sits next to.
# Storing the reference in f16 would quantise away the precision it was
# rendered to obtain.
REF_CHANNELS = 4


def parse_args(argv=None):
  p = argparse.ArgumentParser(description="Render the held-out evaluation reference.")
  p.add_argument("--dataset", required=True, help="corpus directory containing dataset.json")
  p.add_argument("--out", required=True, help="output directory for the reference set")
  p.add_argument("--split", default="val", choices=("val", "train"),
                 help="which split to reference. val unless you know why not")
  p.add_argument("--frames-per-sequence", type=int, default=8,
                 help="frames sampled per sequence. Frames within one "
                      "trajectory are near-duplicates, so this buys far less "
                      "than the scene count does -- see effective_sample_size")
  p.add_argument("--frame-selection", default="window", choices=("window", "spread"),
                 help="window: one contiguous run per sequence, so temporal "
                      "metrics are computable. spread: evenly spaced, which "
                      "covers more of the trajectory but leaves no two "
                      "referenced frames adjacent")
  p.add_argument("--window-position", default="late", choices=("late", "early", "hashed"),
                 help="where in each sequence the contiguous window sits. "
                      "late gives maximum accumulated history, the steady "
                      "state a real-time denoiser spends its life in")
  p.add_argument("--spp", type=int, default=16384,
                 help="total samples per reference frame, split across two halves")
  p.add_argument("--chunk-spp", type=int, default=32,
                 help="samples per kernel launch; the Windows TDR sets the ceiling")
  p.add_argument("--max-scenes", type=int, default=0,
                 help="cap on scenes, 0 for all. For a timed pilot")
  p.add_argument("--resume", action="store_true")
  p.add_argument("--force", action="store_true", help="overwrite an existing reference set")
  p.add_argument("--dry-run", action="store_true")
  return p.parse_args(argv)


def validate_args(args):
  """Fail before the CUDA context, not after an hour."""
  if args.spp < 2:
    raise SystemExit(f"error: --spp must be >= 2, got {args.spp}")
  if args.chunk_spp < 1:
    raise SystemExit(f"error: --chunk-spp must be >= 1, got {args.chunk_spp}")
  if args.frames_per_sequence < 1:
    raise SystemExit("error: --frames-per-sequence must be >= 1")

  half = args.spp // 2
  if half % args.chunk_spp != 0:
    raise SystemExit(
      f"error: --spp/2 ({half}) is not a multiple of --chunk-spp "
      f"({args.chunk_spp}); the two halves would not carry equal weight and "
      f"RMS(A-B)/2 would not estimate the mean's sigma"
    )


def window_start(position, trajectory_seed, count, n):
  """Where the contiguous window begins inside a sequence.

  ⚠ NOT `trajectory_seed % (count - n + 1)`. Seeds are scene_seed * 1000,
  count - n + 1 was 25, and 1000 is a multiple of 25 -- so that expression
  returned 0 for every sequence in the corpus and silently placed every
  window at frame 0. Modulo of a structured value is not a hash. An entire
  reference set was rendered on cold-start frames before anything noticed,
  and the only tell was the scorer printing `warmup 0` twenty-four times.
  """
  span = max(count - n + 1, 1)
  if position == "late":
    # SVGF's history saturates at 1/alpha_min = 20 frames. A window at the
    # end of a 32-frame sequence is fully warmed; one at the start measures
    # cold start, which is a legitimate question and a different one.
    return max(0, count - n)
  if position == "early":
    return 0
  h = hashlib.blake2b(str(trajectory_seed).encode("utf-8"), digest_size=8).digest()
  return int.from_bytes(h, "big") % span


def select_frames(manifest, split, per_sequence, max_scenes,
                  selection="window", position="late"):
  """(shard basename, seq, frame_indices) for every sequence in the split.

  Sequences are grouped by scene by the caller, because uploading a scene is
  the expensive part of switching and doing it per frame would dominate a run
  whose per-frame cost is otherwise ~17 s.

  ⚠ The shard is stored as a BASENAME. manifest.sequences() yields a
  root-joined absolute path, and an absolute path written into a data file is
  a join key that breaks the moment the dataset moves or is read elsewhere.
  """
  picked, scenes = [], []
  for shard_path, seq in manifest.sequences(split=split):
    if seq["scene_id"] not in scenes:
      if max_scenes and len(scenes) >= max_scenes:
        continue
      scenes.append(seq["scene_id"])

    count = seq["count"]
    n = min(per_sequence, count)

    if selection == "window":
      # ⚠ Contiguous, and that is the whole reason this branch exists.
      # Temporal stability is measured between ADJACENT frames after
      # reprojection. A set of evenly spaced references contains no adjacent
      # pair, so it cannot support that metric at all -- and the gap is
      # invisible until you try to compute it, by which point the render is
      # spent.
      start = window_start(position, seq["trajectory_seed"], count, n)
      idx = list(range(start, start + n))
    else:
      step = count / n
      idx = sorted({int(i * step) for i in range(n)})

    picked.append((os.path.basename(shard_path), seq, idx))

  picked.sort(key=lambda x: (x[1]["scene_id"], x[1]["trajectory_seed"]))
  return picked, scenes


class ReferenceWriter:
  """Fixed-stride f32, append-only, with the index rewritten per frame.

  ⚠ The index is rewritten after EVERY frame, not at the end. At ~17 s per
  frame a run is an hour long, and an index written only on exit turns any
  interruption into a total loss. Rewriting a few-KB json per frame costs
  nothing next to the render it follows.
  """

  def __init__(self, root, width, height, config, resume=False):
    self.root = root
    self.width, self.height = width, height
    self.config = config
    self.stride = width * height * REF_CHANNELS * 4
    self.index_path = os.path.join(root, "reference.json")
    self.bin_path = os.path.join(root, "reference.bin")
    self.entries = []

    os.makedirs(root, exist_ok=True)
    if resume and os.path.exists(self.index_path):
      with open(self.index_path) as f:
        d = json.load(f)
      if d["format_version"] != FORMAT_VERSION:
        raise SystemExit(f"error: reference is v{d['format_version']}, "
                         f"writer is v{FORMAT_VERSION}")
      self.entries = d["entries"]
      # Truncate any partially written frame. A short final frame reads as
      # garbage rather than failing, which is the worst possible outcome.
      want = len(self.entries) * self.stride
      have = os.path.getsize(self.bin_path) if os.path.exists(self.bin_path) else 0
      if have < want:
        raise SystemExit(f"error: {self.bin_path} holds {have} bytes but the "
                         f"index claims {want}. Refusing to guess; rerun "
                         f"without --resume")
      self._fh = open(self.bin_path, "r+b")
      self._fh.truncate(want)
      self._fh.seek(want)
    else:
      self._fh = open(self.bin_path, "wb")

  def done_keys(self):
    return {(e["shard"], e["frame"]) for e in self.entries}

  def write(self, shard, frame, scene_id, trajectory_seed, frame_in_sequence,
            reference, sigma):
    assert reference.shape == (self.width, self.height, 3), reference.shape
    assert sigma.shape == (self.width, self.height), sigma.shape
    if not np.isfinite(reference).all():
      raise ValueError(f"non-finite reference at {shard}:{frame}")

    self._fh.write(np.ascontiguousarray(reference, dtype=np.float32).tobytes())
    self._fh.write(np.ascontiguousarray(sigma, dtype=np.float32).tobytes())
    self.entries.append({
      "shard": shard,
      "frame": int(frame),
      "scene_id": scene_id,
      "trajectory_seed": int(trajectory_seed),
      "frame_in_sequence": int(frame_in_sequence),
      "mean_sigma": float(sigma.mean()),
    })
    self._fh.flush()
    self._save_index()

  def _save_index(self):
    tmp = self.index_path + ".tmp"
    with open(tmp, "w") as f:
      json.dump({
        "format_version": FORMAT_VERSION,
        "width": self.width, "height": self.height,
        "channels": ["reference_r", "reference_g", "reference_b", "sigma"],
        "config": self.config,
        "entries": self.entries,
      }, f, indent=1)
    os.replace(tmp, self.index_path)   # atomic; a killed run never sees a half-written index

  def close(self):
    self._fh.close()
    self._save_index()


def print_budget(args, picked, scenes, render):
  n_frames = sum(len(idx) for _, _, idx in picked)
  w, h = render["width"], render["height"]
  per_frame = w * h * REF_CHANNELS * 4
  total = per_frame * n_frames
  scale = args.spp / max(render.get("clean_spp", 512), 1)

  windows = sorted({idx[0] for _, _, idx in picked})

  print("=" * 70)
  print("REFERENCE BUDGET")
  print("=" * 70)
  print(f"source: {args.dataset}")
  print(f"split: {args.split}   scenes: {len(scenes)}   sequences: {len(picked)}")
  print(f"frames: {n_frames} ({args.frames_per_sequence} per sequence)")
  print(f"selection: {args.frame_selection}/{args.window_position}")
  # ⚠ Printed because the last run placed every window at frame 0 by
  # accident and nothing said so until the scorer reported `warmup 0`.
  print(f"window starts: {windows if len(windows) <= 6 else f'{len(windows)} distinct'}")
  print(f"resolution: {w}x{h}   bounces: {render.get('max_bounces')}")
  print()
  print(f"reference spp: {args.spp} (2 halves x {args.spp // 2})")
  print(f"launches per frame: {args.spp // args.chunk_spp} x {args.chunk_spp} spp")
  print(f"sample cost vs the dataset's clean target: {scale:.0f}x")
  print()
  print(f"per frame: {per_frame / 1024:.1f} KiB")
  print(f"on disk: {total / 1e9:.2f} GB")
  print()
  print(f"PREDICTION -- the previous run settled near 17 s/frame, so expect")
  print(f"about {17.0 * n_frames / 60.0:.0f} min. Record the real rate against that.")
  print("=" * 70)
  return n_frames


def relmse_floor(reference, sigma):
  """What a PERFECT denoiser would score against this reference.

  A perfect predictor emits the true mean mu. Its residual against the noisy
  reference is exactly the reference's own error, whose per-pixel variance is
  sigma^2. So the floor is

      mean( sigma^2 / (reference + eps)^2 )

  using the same eps as metrics/image_metrics.py, because a floor computed
  with a different epsilon is not a floor for the metric you will report.

  ⚠ This is the number the file exists to produce. If it is not comfortably
  below SVGF_RELMSE, the reference cannot adjudicate and --spp must rise.
  """
  denom = (reference.astype(np.float64) + RELMSE_EPS) ** 2
  return float((sigma.astype(np.float64)[..., None] ** 2 / denom).mean())


def main(argv=None):
  argv = list(sys.argv[1:] if argv is None else argv)
  args = parse_args(argv)
  validate_args(args)

  manifest = Manifest.load(args.dataset)
  render = manifest.config["render"]
  picked, scenes = select_frames(manifest, args.split, args.frames_per_sequence,
                                 args.max_scenes, args.frame_selection,
                                 args.window_position)
  if not picked:
    raise SystemExit(f"error: no sequences in split '{args.split}'")

  n_frames = print_budget(args, picked, scenes, render)

  if args.dry_run:
    print()
    print("dry run: nothing rendered.")
    return 0

  if os.path.exists(os.path.join(args.out, "reference.json")) and not (args.resume or args.force):
    print()
    print(f"refusing to write into {args.out}: reference.json already exists.")
    print("Use --resume to continue it, --force to overwrite, or a fresh --out.")
    return 2

  import taichi as ti
  from tracer.geometry.scene_generator import SceneParams, build_scene, upload_scene
  from tracer.pipeline import RenderConfig, get_renderer
  from tracer.trajectory import Pose

  ti.init(arch=ti.cuda)

  # ⚠ Reconstructed from the manifest, not from CLI defaults. width, height,
  # fov, bounces, NEE and single-sidedness must match the frames being
  # referenced EXACTLY or the reference describes a different image. Only the
  # chunking is ours to choose.
  known = {f.name for f in dataclass_fields(RenderConfig)}
  cfg = RenderConfig(**{k: v for k, v in render.items() if k in known})
  half_chunks = (args.spp // 2) // args.chunk_spp
  renderer = get_renderer(cfg)
  scene_params = SceneParams()

  writer = ReferenceWriter(args.out, cfg.width, cfg.height,
                           config={
                             "source_dataset": os.path.abspath(args.dataset),
                             "split": args.split,
                             "spp": args.spp,
                             "chunk_spp": args.chunk_spp,
                             "frames_per_sequence": args.frames_per_sequence,
                             "frame_selection": args.frame_selection,
                             "window_position": args.window_position,
                             "relmse_eps": RELMSE_EPS,
                             "render": cfg.to_json(),
                             "argv": argv,
                           },
                           resume=args.resume)
  done = writer.done_keys()
  if args.resume:
    print()
    print(f"resume: {len(done)} frames already on disk")

  floors, sigmas = [], []
  current_scene = None
  seen, written = 0, 0
  t0 = time.perf_counter()
  t_first = None

  try:
    for shard_path, seq, frame_idx in picked:
      pending = [t for t in frame_idx if (shard_path, seq["start"] + t) not in done]
      seen += len(frame_idx) - len(pending)
      if not pending:
        continue

      if seq["scene_id"] != current_scene:
        data = build_scene(scene_params, seq["scene_seed"], scene_id=seq["scene_id"])
        upload_scene(data)
        current_scene = seq["scene_id"]

      poses = seq["meta"]["frames"]
      for t in pending:
        f = poses[t]
        cam = renderer.make_camera(
          Pose(position=np.asarray(f["position"], dtype=np.float64),
               yaw=float(f["yaw"]), pitch=float(f["pitch"]))
        )

        # Two independent halves. The mean IS the reference and RMS(A-B)/2 IS
        # its sigma, so the error bar costs nothing beyond the render itself.
        a = renderer.render_clean(cam, half_chunks, args.chunk_spp).to_numpy().astype(np.float64)
        b = renderer.render_clean(cam, half_chunks, args.chunk_spp).to_numpy().astype(np.float64)
        reference = 0.5 * (a + b)
        sigma = np.sqrt(((a - b) ** 2).mean(axis=-1)) / 2.0

        floor = relmse_floor(reference, sigma)
        floors.append(floor)
        sigmas.append(float(sigma.mean()))

        writer.write(shard_path, seq["start"] + t, seq["scene_id"],
                     seq["trajectory_seed"], t,
                     reference.astype(np.float32), sigma.astype(np.float32))
        seen += 1
        written += 1

        if t_first is None:
          t_first = time.perf_counter()
          print(f"[{seen}/{n_frames}] {seq['scene_id']} f{t} "
                f"{t_first - t0:.1f}s including JIT, floor {floor:.6f}")
          continue
        elapsed = time.perf_counter() - t_first
        rate = max(written - 1, 1) / max(elapsed, 1e-9)
        print(f"[{seen}/{n_frames}] {seq['scene_id']} f{t} "
              f"{elapsed / max(written - 1, 1):.1f}s/frame, "
              f"ETA {(n_frames - seen) / rate / 60.0:.1f} min, floor {floor:.6f}")

  except BaseException:
    print()
    print("aborting: the index is current through the last completed frame")
    raise

  finally:
    writer.close()
    print()
    print("=" * 70)
    print(f"reference: {writer.index_path}")
    print(f"frames written this run: {written}")
    if floors:
      mean_floor = float(np.mean(floors))
      print(f"mean sigma: {np.mean(sigmas):.6f}")
      print(f"relMSE FLOOR: {mean_floor:.6f}")
      print(f"SVGF (static-camera, 720p): {SVGF_RELMSE:.5f}   "
            f"ratio: {SVGF_RELMSE / mean_floor:.1f}x above the floor")
      if mean_floor > SVGF_RELMSE * 0.2:
        print("!! The floor is within 5x of the bar. This reference cannot")
        print("!! resolve a modest improvement. Raise --spp by 4x to halve it,")
        print("!! and re-read topic 41(a) before spending the GPU time.")
    print("=" * 70)

  return 0


if __name__ == "__main__":
  raise SystemExit(main())