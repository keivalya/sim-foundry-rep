"""Plot what the pi0.5 closed loop actually did: commanded vs achieved state,
per joint, in LeRobot normalized units, with the checkpoint's own training
quantiles drawn on top.

The point of the [q01, q99] band is to show WHERE the sim arm sits relative to
the states this checkpoint ever saw. Anything outside the band was clamped
before it reached the model (see vla/pi05_server.py), i.e. the policy was told
the arm was somewhere it is not.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--ckpt", default="/scratch/pandya.kei/simfoundry-repro/vla/ckpt_so101_pi05_lerobot061")
ap.add_argument("--out", default="")
a = ap.parse_args()

d = np.load(os.path.join(a.run, "episode.npz"), allow_pickle=True)
names = [str(x) for x in d["lerobot_names"]]
st, ac = d["state_lerobot"], d["action_lerobot"]
plans = d["chunk_at"]

# The quantiles live in a .safetensors only the `openpi` env can read, and
# matplotlib is only in `simfoundry`, so they were dumped once to JSON.
q01 = q99 = None
try:
    qj = json.load(open("/scratch/pandya.kei/simfoundry-repro/vla/so101_state_quantiles.json"))
    q01 = np.asarray(qj["state"]["q01"]); q99 = np.asarray(qj["state"]["q99"])
except Exception as e:
    print(f"(no quantile band: {e})")

fig, axes = plt.subplots(3, 2, figsize=(13, 8.5), sharex=True)
for i, (ax, n) in enumerate(zip(axes.ravel(), names)):
    if q01 is not None:
        ax.axhspan(q01[i], q99[i], color="0.88", zorder=0,
                   label="checkpoint train range [q01,q99]")
    ax.plot(ac[:, i], lw=1.0, color="tab:red", label="pi0.5 action (commanded)")
    ax.plot(st[:, i], lw=1.0, color="tab:blue", label="achieved state")
    for p in plans:
        ax.axvline(p, color="0.6", lw=0.4, zorder=0)
    ax.set_title(n, fontsize=10)
    ax.grid(alpha=0.25, lw=0.4)
axes[0, 0].legend(fontsize=7, loc="upper left")
for ax in axes[-1]:
    ax.set_xlabel("control step (30 Hz); grey verticals = re-plan")
fig.suptitle(f"pi0.5 closed loop on SO-101 -- {os.path.basename(a.run)}\n"
             f"LeRobot normalized units; grey band = the states the checkpoint was trained on",
             fontsize=11)
fig.tight_layout()
out = a.out or os.path.join(a.run, "episode_traces.png")
fig.savefig(out, dpi=130)
print("wrote", out)
