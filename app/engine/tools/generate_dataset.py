"""
  Headless dataset generation. No GUI, no window, no blocking.

  python tools/generate_dataset.py --out datasets/albedo_v1 \
      --scenes 64 --frames 32 --width 512 --height 288 --dry-run
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data.manifest import Manifest, MANIFEST_VERSION, scene_split
from data.shard import ShardWriter, frame_bytes, channel_bytes


def parse_args(argv=None):
  p = argparse.ArgumentParser(description="Render a training dataset for Phase 2.4.")
  p.add_argument("--out", required=True, help="output directory")
  p.add_argument("--scenes", type=int, default=64)
  p.add_argument("--frames", type=int, default=32, help="frames per trajectory")
  p.add_argument("--trajectories-per-scene", type=int, default=1)
  p.add_argument("--width", type=int, default=512)
  p.add_argument("--height", type=int, default=288)
  p.add_argument("--spp", type=int, default=2, help="samples per noisy frame")
  p.add_argument("--noisy-realizations", type=int, default=2,
    help="independent noisy renders per frame. Costs ~1.2%% of render "
    "time each and is the only augmentation with no domain gap; "
    ">=2 also enables the Noise2Noise ablation. Costs bytes.")
  p.add_argument("--clean-chunks", type=int, default=16)
  p.add_argument("--clean-spp-per-chunk", type=int, default=32)
  p.add_argument("--max-bounces", type=int, default=8)
  p.add_argument("--seed", type=int, default=20260819)
  p.add_argument("--sequences-per-shard", type=int, default=8)
  p.add_argument("--val-fraction", type=float, default=0.12)
  p.add_argument("--calibrate-frames", type=int, default=8,
                 help="frames used for the split-half noise estimate; 0 to skip")
  p.add_argument("--resume", action="store_true",
                 help="skip sequences already recorded in an existing manifest")
  p.add_argument("--dry-run", action="store_true",
                 help="print the budget and the split, render nothing")
  p.add_argument("--batch-size", type=int, default=4, help="for the bandwidth report only")
  p.add_argument("--seq-len", type=int, default=8, help="for the bandwidth report only")
  p.add_argument("--target-step-ms", type=float, default=250.0,
                 help="for the bandwidth report only")
  return p.parse_args(argv)


def validate_args(args):
  """Fail before the CUDA context, not after three hours of rendering."""
  positive = {
    "scenes": args.scenes,
    "trajectories_per_scene": args.trajectories_per_scene,
    "width": args.width,
    "height": args.height,
    "spp": args.spp,
    "clean_chunks": args.clean_chunks,
    "clean_spp_per_chunk": args.clean_spp_per_chunk,
    "max_bounces": args.max_bounces,
    "sequences_per_shard": args.sequences_per_shard,
    "batch_size": args.batch_size,
    "seq_len": args.seq_len,
  }
  for name, value in positive.items():
    if value < 1:
      raise SystemExit(f"error: --{name.replace('_', '-')} must be >= 1, got {value}")

  if args.target_step_ms <= 0:
    raise SystemExit(f"error: --target-step-ms must be > 0, got {args.target_step_ms}")
  if args.calibrate_frames < 0:
    raise SystemExit(f"error: --calibrate-frames must be >= 0, got {args.calibrate_frames}")
  if not 0.0 <= args.val_fraction < 1.0:
    raise SystemExit(f"error: --val-fraction must be in [0, 1), got {args.val_fraction}")

  if args.frames < 2:
    raise SystemExit(f"error: --frames must be >= 2 for motion vectors, got {args.frames}")

  if args.frames < args.seq_len:
    print(f"warning: --frames {args.frames} < --seq-len {args.seq_len}, so no training "
          f"window of that length will fit in any sequence")


def scene_id_for(seed: int, index: int) -> str:
  return f"scene_{seed + index:08d}"


def print_budget(args) -> int:
  n_real = args.noisy_realizations
  per_frame = frame_bytes(args.width, args.height, noisy_realizations=n_real)
  per_channel = channel_bytes(args.width, args.height, noisy_realizations=n_real)
  n_seq = args.scenes * args.trajectories_per_scene
  total_frames = n_seq * args.frames
  total = per_frame * total_frames
  spp_total = args.clean_chunks * args.clean_spp_per_chunk

  print("=" * 70)
  print("DATASET BUDGET")
  print("=" * 70)
  print(f"resolution: {args.width}x{args.height}")
  print(f"sequences: {n_seq} ({args.scenes} scenes x {args.trajectories_per_scene} trajectories)")
  print(f"frames: {total_frames} ({args.frames} per sequence)")
  print(f"noisy realizations: {n_real} per frame")
  print()

  for name, b in per_channel.items():
    print(f"{name}: {b / 1024:.1f} KiB/frame ({b / per_frame:.1%})")
  print(f"total: {per_frame / 1024:.1f} KiB/frame")
  print()

  print(f"on disk: {total / 1e9:.2f} GB")
  print(f"shards: {int(np.ceil(n_seq / args.sequences_per_shard))} files")
  print()

  step_bytes = per_frame * args.batch_size * args.seq_len
  bw = step_bytes / (args.target_step_ms / 1000.0)
  print(f"one training step: batch {args.batch_size} x seq {args.seq_len} = {step_bytes / 1e6:.1f} MB")
  print(f"required read rate: {bw / 1e6:.0f} MB/s at {args.target_step_ms:.0f} ms/step")
  print(f"(cold cache, no reuse. With {total / 1e9:.1f} GB total and enough RAM to")
  print("cache it, the steady-state rate is ~0.)")
  print()

  print(f"clean target: {spp_total} spp ({args.clean_chunks} x {args.clean_spp_per_chunk})")
  samples = total_frames * (spp_total + args.spp) * args.width * args.height
  print(f"primary samples: {samples / 1e9:.1f} G")
  print()

  train = sum(1 for s in range(args.scenes)
            if scene_split(scene_id_for(args.seed, s), args.val_fraction) == "train")
  print(f"split: {train} train / {args.scenes - train} val SCENES "
        f"(val_fraction={args.val_fraction})")
  if args.scenes - train == 0:
    print("!! ZERO validation scenes. Raise --scenes or --val-fraction.")
  print("=" * 70)
  return total


def load_resume_state(out_dir: str):
  path = os.path.join(out_dir, "dataset.json")
  if not os.path.exists(path):
    return [], set(), -1

  manifest = Manifest.load(path)
  done = set()
  last_index = -1
  for shard in manifest.shards:
    for seq in shard["sequences"]:
      done.add((seq["scene_id"], seq["trajectory_seed"]))
    stem = os.path.splitext(os.path.basename(shard["path"]))[0]
    try:
      last_index = max(last_index, int(stem.rsplit("_", 1)[-1]))
    except ValueError:
      pass
  return list(manifest.shards), done, last_index


def _aov_frame(buffers):
  return {
    "albedo": buffers.albedo.to_numpy(),
    "normal": buffers.normal.to_numpy(),
    "depth": buffers.depth.to_numpy(),
    "motion": buffers.motion.to_numpy(),
    "object_id": buffers.object_id.to_numpy().astype(np.int16),
    "hit_mask": buffers.hit_mask.to_numpy().astype(np.uint8),
  }


def calibrate(renderer, args, build_scene, upload_scene, scene_params, sample_trajectory, traj_params):
  if args.calibrate_frames <= 0:
    return []

  data = build_scene(scene_params, args.seed, scene_id=scene_id_for(args.seed, 0))
  upload_scene(data)
  poses, _ = sample_trajectory(args.seed * 1000, args.frames, data.bounds, traj_params, obstacles=data.occluders)

  return [renderer.clean_split_half(renderer.make_camera(p)) for p in poses[:args.calibrate_frames]]


def save_manifest(args, argv, shards, calib, cfg, scene_params, traj_params, resumed):
  manifest = Manifest(
    root=os.path.abspath(args.out),
    version=MANIFEST_VERSION,
    config={
      "generator": "phase_2.3",
      "argv": list(argv),
      "seed": args.seed,
      "scenes": args.scenes,
      "trajectories_per_scene": args.trajectories_per_scene,
      "frames_per_sequence": args.frames,
      "sequences_per_shard": args.sequences_per_shard,
      "noisy_realizations": args.noisy_realizations,
      "calibrate_frames": args.calibrate_frames,
      "val_fraction": args.val_fraction,
      "resumed": resumed,
      "render": cfg.to_json(),
      "scene_params": scene_params.to_json(),
      "trajectory_params": traj_params.to_json(),
      "clean_split_half_sigma": float(np.mean(calib)) if calib else None,
      "clean_split_half_samples": len(calib),
    },
    shards=shards,
  )
  return manifest, manifest.save()


def main(argv=None):
  argv = list(sys.argv[1:] if argv is None else argv)
  args = parse_args(argv)
  validate_args(args)
  print_budget(args)

  if args.dry_run:
    print()
    print("dry run: nothing rendered.")
    return 0

  import taichi as ti
  from tracer import buffers
  from tracer.geometry.scene_generator import SceneParams, build_scene, upload_scene
  from tracer.pipeline import RenderConfig, get_renderer
  from tracer.trajectory import (
    TrajectoryParams, sample_trajectory, trajectory_diagnostics,
  )

  os.makedirs(args.out, exist_ok=True)
  shards, done, shard_index = load_resume_state(args.out) if args.resume else ([], set(), -1)
  if args.resume:
    print()
    print(f"resume: {len(done)} sequences on disk, continuing at shard {shard_index + 1}")

  ti.init(arch=ti.cuda)

  cfg = RenderConfig(
    width=args.width, height=args.height, spp=args.spp,
    max_bounces=args.max_bounces,
    clean_chunks=args.clean_chunks, clean_spp_per_chunk=args.clean_spp_per_chunk,
  )
  renderer = get_renderer(cfg)
  scene_params = SceneParams()
  traj_params = TrajectoryParams()

  writer = None
  writer_seq_count = 0

  calib = []
  n_seq = args.scenes * args.trajectories_per_scene
  seq_seen = 0
  seq_written = 0
  t0 = time.perf_counter()
  t_first = None

  def flush_shard():
    nonlocal writer, writer_seq_count
    if writer is None:
      return
    path = writer.close()
    shards.append({
      "path": os.path.basename(path),
      "sequences": [s.to_json() for s in writer.sequences],
    })
    writer = None
    writer_seq_count = 0

  def open_shard():
    nonlocal writer, shard_index
    flush_shard()
    shard_index += 1
    writer = ShardWriter(
    os.path.join(args.out, f"shard_{shard_index:05d}"),
    args.width, args.height,
    noisy_realizations=args.noisy_realizations,
    render_config=cfg.to_json(),
  )

  try:
    for s in range(args.scenes):
      scene_seed = args.seed + s
      scene_id = scene_id_for(args.seed, s)

      traj_seeds = [scene_seed * 1000 + k for k in range(args.trajectories_per_scene)]
      pending = [t for t in traj_seeds if (scene_id, t) not in done]
      seq_seen += len(traj_seeds) - len(pending)
      if not pending:
        continue

      data = build_scene(scene_params, scene_seed, scene_id=scene_id)
      upload_scene(data)

      for traj_seed in pending:
        poses, traj_meta = sample_trajectory(
          traj_seed, args.frames, data.bounds, traj_params,
          obstacles=data.occluders,
        )

        if traj_meta["invalid_poses"]:
          print(f"skip {scene_id} traj {traj_seed}: "
                f"{traj_meta['invalid_poses']} poses inside geometry")
          seq_seen += 1
          continue

        cameras = [renderer.make_camera(p) for p in poses]
        frame_meta = []
        for pose, cam in zip(poses, cameras):
          right, up, forward = cam.basis_from_yaw_pitch()
          frame_meta.append({
            "position": [float(v) for v in pose.position],
            "yaw": float(pose.yaw), "pitch": float(pose.pitch),
            "right": [float(v) for v in right],
            "up": [float(v) for v in up],
            "forward": [float(v) for v in forward],
          })

        if writer is None or writer_seq_count >= args.sequences_per_shard:
          open_shard()

        writer.begin_sequence(  # type: ignore
          scene_id=scene_id, scene_seed=scene_seed, trajectory_seed=traj_seed,
          meta={
            "scene": data.meta,
            "trajectory": traj_meta,
            "diagnostics": trajectory_diagnostics(
              poses, args.width, np.radians(cfg.fov_deg), cfg.aspect_ratio
            ),
            "frames": frame_meta,
          },
        )

        renderer.reset_sequence()

        for cam in cameras:
          renderer.render_aovs(cam)
          noisy = np.stack(
            [renderer.render_noisy(cam).to_numpy()
             for _ in range(args.noisy_realizations)],
            axis=2,
          )
          clean = renderer.render_clean(cam).to_numpy()
          writer.write_frame(noisy=noisy, clean=clean, **_aov_frame(buffers)) # type: ignore
          renderer.commit_frame(cam)

        writer.end_sequence() # type: ignore
        writer_seq_count += 1
        seq_seen += 1
        seq_written += 1

        if t_first is None:
          t_first = time.perf_counter()
          print(f"[{seq_seen}/{n_seq}] {scene_id} {traj_meta['kind']} "
                f"{t_first - t0:.1f}s including JIT")
          continue

        elapsed = time.perf_counter() - t_first
        rate = max(seq_written - 1, 1) / max(elapsed, 1e-9)
        print(f"[{seq_seen}/{n_seq}] {scene_id} {traj_meta['kind']} "
              f"{elapsed:.1f}s, ETA {(n_seq - seq_seen) / rate / 60.0:.1f} min")

    calib = calibrate(renderer, args, build_scene, upload_scene, scene_params,
                      sample_trajectory, traj_params)

  except BaseException:
    print()
    print("aborting: discarding the in-progress sequence, keeping completed ones")
    if writer is not None:
      writer.abort_sequence()
    raise

  finally:
    flush_shard()
    manifest, path = save_manifest(
      args, argv, shards, calib, cfg, scene_params, traj_params, resumed=args.resume
    )
    print()
    print("=" * 70)
    print(f"manifest: {path}")
    print(f"sequences written this run: {seq_written}")
    try:
      for split, v in manifest.split_summary().items():
        print(f"{split}: {v['scenes']} scenes, {v['sequences']} sequences, "
              f"{v['frames']} frames")
    except AssertionError as exc:
      print(f"!! SPLIT IS BROKEN: {exc}")
    if calib:
      print(f"clean target own sigma: {float(np.mean(calib)):.6f} "
            f"(split-half, {len(calib)} frame(s))")
      print("SVGF's measured relMSE is 0.00140. For the target to adjudicate a model")
      print("against that, this sigma must be small next to the DIFFERENCE you expect,")
      print("not next to the metric itself.")
    print("=" * 70)

  return 0


if __name__ == "__main__":
  raise SystemExit(main())