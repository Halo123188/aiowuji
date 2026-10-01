#!/usr/bin/env python3
"""Plot how the joints tracked their references in a sysid_ref_collect.py run.

Writes one PNG per gain setting into the run's folder: a grid of fingers
(rows) by joints S1-S4 (columns), each panel showing the reference sent
(q_cmd) and the measured angle over the trajectory, with the segment
boundaries marked and the RMS and largest tracking error in the title. The
errors cover the sine and multisine segments: at a step the reference jumps
before any joint can follow, which would swamp them.

    python scripts/plot_sysid.py [RUN_DIR]     (default: the newest logs/sysid-*)

Needs matplotlib.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SURFACE, TEXT, TEXT_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MEASURED, TARGET = "#2a78d6", "#52514e"


def plot_gain(run: Path, manifest: dict, trials: list[dict], kp: float, kd: float) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fingers = [t["finger"] for t in trials]
    fig, axes = plt.subplots(len(trials), 4, figsize=(20, 2.6 * len(trials)), sharex=True,
                             layout="constrained", facecolor=SURFACE, squeeze=False)
    pre = manifest["pre_s"]
    for row, trial in enumerate(trials):
        data = np.load(run / trial["file"])
        t = (data["tick_ns"] - data["tick_ns"][0]) / 1e9 - pre      # 0 = the reference starts
        smooth = trial["segments"][[s["kind"] for s in trial["segments"]].index("sines")]["start"]
        scored = (t >= smooth) & (t <= trial["duration"])            # sines and multisine
        for col, joint in enumerate(trial["joints"]):
            ax = axes[row, col]
            target, measured = np.degrees(data["q_cmd"][:, joint]), np.degrees(data["q"][:, joint])
            error = (target - measured)[scored]
            ax.set_facecolor(SURFACE)
            ax.plot(t, measured, color=MEASURED, lw=1.4, label="measured")
            ax.plot(t, target, color=TARGET, lw=0.9, label="reference", zorder=3)
            for segment in trial["segments"][1:]:
                ax.axvline(segment["start"], color=AXIS, lw=0.8, zorder=0)
            ax.set_title(f"{trial['joint_names'][col]}   rms {np.sqrt(np.mean(error ** 2)):.1f}°   "
                         f"max {np.abs(error).max():.1f}°", loc="left", fontsize=10, color=TEXT)
            ax.grid(color=GRID, lw=0.6)
            ax.tick_params(colors=MUTED, labelcolor=TEXT_2, labelsize=8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(AXIS)
        axes[row, 0].set_ylabel(f"{fingers[row]}\ndeg", color=TEXT_2)
    # Name each segment once, centred over its span on every top-row panel.
    spans = {}
    for segment in trials[0]["segments"]:
        name = "steps" if segment["kind"] == "step" else segment["kind"]
        start, end = spans.get(name, (segment["start"], 0.0))
        spans[name] = (start, segment["start"] + segment["duration"])
    for ax in axes[0]:
        for name, (start, end) in spans.items():
            ax.annotate(name, xy=(0.5 * (start + end), 1.0), xycoords=("data", "axes fraction"),
                        xytext=(0, 14), textcoords="offset points", ha="center", fontsize=8, color=MUTED)
    for ax in axes[-1]:
        ax.set_xlabel("time since the reference started (s)", color=TEXT_2)
    fig.suptitle(f"Tracking at kp {kp:g} A/rad, kd {kd:g} A·s/rad   ({run.name}, {manifest['hand_sn']}, "
                 f"firmware {manifest.get('firmware')}; errors over the sine and multisine segments)",
                 x=0.005, ha="left", fontsize=13, color=TEXT)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper right", ncol=2,
               frameon=False, labelcolor=TEXT)
    path = run / f"tracking-kp{kp:g}-kd{kd:g}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", nargs="?", type=Path, help="a logs/sysid-* folder (default: the newest)")
    args = parser.parse_args()
    run = args.run or sorted((ROOT / "logs").glob("sysid-*"))[-1]
    manifest = json.loads((run / "manifest.json").read_text())
    settings = sorted({(t["kp"], t["kd"]) for t in manifest["trials"]})
    for kp, kd in settings:
        trials = [t for t in manifest["trials"] if (t["kp"], t["kd"]) == (kp, kd)]
        print(f"Saved {plot_gain(run, manifest, trials, kp, kd)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
