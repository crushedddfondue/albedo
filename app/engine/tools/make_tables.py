import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import OrderedDict

from metrics.results import load_results

METRICS = ["relmse", "smape", "relmse_luma"]


def latest_per_label(records):
  """Most recent entry per label -- the log is append-only, so re-runs
  accumulate and only the newest is current."""
  out = OrderedDict()
  for r in sorted(records, key=lambda r: r["timestamp"]):
    out[r["label"]] = r
  return out


def markdown(records):
  rows = latest_per_label(records)
  lines = ["| Method | relMSE | SMAPE | relMSE (luma) | spp | Notes |",
           "|---|---|---|---|---|---|"]
  for label, r in rows.items():
    m, c = r["metrics"], r["config"]
    note = "—"
    if c.get("denoiser") == "svgf":
      note = f"{c['warmup_frames']} frame warmup, {c['atrous_levels']} à-trous levels"
    lines.append(
      f"| {label} | {m['relmse']:.5f} | {m['smape']:.5f} | "
      f"{m['relmse_luma']:.5f} | {c['spp_per_frame']} | {note} |"
    )

  any_r = next(iter(rows.values()))
  ref = any_r["reference"]
  lines.append("")
  lines.append(
    f"Scored against a {ref['spp_total']} spp reference "
    f"(split-half σ = {ref['split_half_sigma']:.5f}), "
    f"{any_r['config']['resolution'][0]}×{any_r['config']['resolution'][1]}, "
    f"scene `{any_r['config']['scene']}`, "
    f"commit `{any_r['provenance']['commit'][:8]}`."
  )

  dirty = [l for l, r in rows.items() if r["provenance"].get("dirty")]
  if dirty:
    lines.append("")
    lines.append(f"Produced from a dirty working tree: {', '.join(dirty)}. "
                 "Not reproducible from the recorded commit.")
  return "\n".join(lines)


def latex(records):
  rows = latest_per_label(records)
  lines = [r"\begin{tabular}{lrrr}", r"\toprule",
           r"Method & relMSE & SMAPE & relMSE$_\text{luma}$ \\", r"\midrule"]
  for label, r in rows.items():
    m = r["metrics"]
    lines.append(f"{label} & {m['relmse']:.5f} & {m['smape']:.5f} & "
                 f"{m['relmse_luma']:.5f} \\\\")
  lines += [r"\bottomrule", r"\end{tabular}"]
  return "\n".join(lines)


if __name__ == "__main__":
  records = load_results()
  if not records:
    print("No results logged yet.")
  else:
    print(markdown(records))
    print("\n\n" + latex(records))