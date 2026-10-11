"""Per-joint comparison of a sim rollout against the training set's own episodes.

Six panels, one per joint, LeRobot normalized units vs time in seconds:

  * grey band  -- the min..max envelope of all 50 training episodes, resampled
                  onto a common time axis, with the median as a dashed line.
                  This is "what the policy has ever done", not a single demo.
  * blue       -- training episode 0, the reference demo.
  * red        -- the sim rollout's achieved state.
  * orange     -- what the policy ASKED for (`policy_target_lerobot`), where the
                  run recorded it, so a follower-emulation filter is visible as
                  the gap between orange and red.

Reading the panels against each other is what separates "the policy did nothing"
from "the policy did the right motion at the wrong speed", which is the whole
question for a relative-action checkpoint driven by a sim servo that is faster
than the real one.

    conda_envs/simfoundry/bin/python -I tools/plot_bluecube_vs_training.py \
        --run results/so101/vla/bluecube/y1_alpha
"""
import argparse
import glob
import os

import numpy as np

SNAP = ("/scratch/pandya.kei/hf_cache/hub/datasets--pdhinaka--"
        "so101_blue_cube_in_gray_box_pi05_cleaned/snapshots/*/")
J = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def training(n=200):
    import pyarrow.parquet as pq
    base = glob.glob(SNAP)[0]
    t = pq.read_table(base + "data/chunk-000/file-000.parquet")
    S = np.stack(t["observation.state"].to_numpy(zero_copy_only=False)).astype(float)
    ep = t["episode_index"].to_numpy()
    # Resample every episode onto the same 0..1 phase so the envelope is
    # meaningful despite the 212..414-frame spread in episode length.
    out = []
    for e in np.unique(ep):
        s = S[ep == e]
        x = np.linspace(0, 1, len(s))
        out.append(np.stack([np.interp(np.linspace(0, 1, n), x, s[:, i]) for i in range(6)], 1))
    return np.asarray(out), S[ep == 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    E = np.load(os.path.join(a.run, "episode.npz"), allow_pickle=True)
    S = E["state_lerobot"]
    tgt = E["policy_target_lerobot"] if "policy_target_lerobot" in E.files else None
    env, ep0 = training()
    lo, med, hi = env.min(0), np.median(env, 0), env.max(0)

    ts = np.arange(len(S)) / a.fps
    t0 = np.arange(len(ep0)) / a.fps
    te = np.linspace(0, len(ep0) / a.fps, env.shape[1])

    fig, ax = plt.subplots(2, 3, figsize=(16, 7.5), sharex=True)
    for i, j in enumerate(J):
        p = ax[i // 3, i % 3]
        p.fill_between(te, lo[:, i], hi[:, i], color="0.80", lw=0,
                       label="training, all 50 eps (min..max)")
        p.plot(te, med[:, i], "--", color="0.45", lw=1.0, label="training median")
        p.plot(t0, ep0[:, i], color="#1f77b4", lw=1.4, label="training episode 0")
        if tgt is not None:
            p.plot(ts, tgt[:, i], color="#ff7f0e", lw=1.0, alpha=0.8,
                   label="sim: policy target")
        p.plot(ts, S[:, i], color="#d62728", lw=1.6, label="sim: achieved")
        p.set_title(j, fontsize=11)
        p.grid(alpha=0.25)
        if i >= 3:
            p.set_xlabel("time (s)")
        if i % 3 == 0:
            p.set_ylabel("LeRobot normalized")
    ax[0, 0].legend(fontsize=7.5, loc="best")
    fig.suptitle(f"pi0.5 on the SO-101, blue cube in gray box — "
                 f"{os.path.basename(a.run.rstrip('/'))} vs the training set", fontsize=12)
    fig.tight_layout()
    out = a.out or os.path.join(a.run, "vs_training.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
