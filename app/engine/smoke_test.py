import os, numpy as np
from data.shard import ShardReader
from data.manifest import Manifest

m = Manifest.load("datasets/smoke")
r = ShardReader(os.path.join(m.root, m.shards[0]["path"]))

for i in range(len(r)):
  f = r.read_frame(i)
  hit = f["hit_mask"] == 1
  n = f["normal"][hit].astype(np.float64)
  ln = np.linalg.norm(n, axis=1)
  mag = np.linalg.norm(f["motion"].astype(np.float64), axis=-1)
  print(f"frame {i}")
  print(f"coverage {hit.mean():.3f} ids {sorted(set(f['object_id'][hit].tolist()))[:8]}")
  print(f"bg ids all -1: {(f['object_id'][~hit] == -1).all()}")
  print(f"depth geo {f['depth'][hit].min():.3f}..{f['depth'][hit].max():.3f}  bg {np.unique(f['depth'][~hit])[:2]}")
  print(f"|normal| min {ln.min():.4f} mean {ln.mean():.4f}")
  print(f"albedo {f['albedo'].min():.3f}..{f['albedo'].max():.3f}")
  print(f"noisy mean {f['noisy'][hit].mean():.4f} clean mean {f['clean'][hit].mean():.4f}")
  print(f"noisy max {f['noisy'].max():.1f} clean max {f['clean'].max():.1f}")
  print(f"motion max {mag.max():.4f} nonzero {(mag > 0).mean():.3f}")