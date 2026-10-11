"""Objective scoring for a pi0.5 blue-cube rollout. No frame-watching.

Reads what the harness already writes -- `poses.npz` (per-frame world poses of
every robot link and every object) and `episode.npz` (per-step state, action and
clip masks) -- plus the scene's `*_layout.json`, and answers, numerically:

  1. did the blue cube move, and where did it end up
  2. is it INSIDE the gray box (a geometric containment test, not a judgement
     about a video still)
  3. did the gripper ever close on the cube -- i.e. was the cube between the
     jaws AND was the jaw commanded shut at the same time
  4. did the arm ever go anywhere near the cube (min grasp-centre distance)
  5. how the commanded trajectory compares with the checkpoint's own action
     distribution, per joint, and how often it was clipped by the joint limits

(3) and (4) are the diagnostics that matter when the answer to (2) is no: they
separate "the policy never tried" from "the policy tried and missed" from "the
policy grasped and dropped it".

The GRASP CENTRE is the midpoint of the robot's two assisted-grasp points (the
static pad on `gripper`, the moving pad on `jaw`, both from robot/so101_og.py),
reconstructed from the logged link poses. That is the point the jaws actually
close on, which the gripper link origin is not -- it is 10 cm behind it.

    conda_envs/simfoundry-editor/bin/python -I tools/score_bluecube.py \
        --run results/so101/vla/bluecube/v1_small \
        --layout scene_data/bluecube/bc_small_layout.json
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "vla"))
from so101_units import LEROBOT_JOINTS  # noqa: E402

# Assisted-grasp points, link-local, from robot/so101_og.py (measured on the
# closed-jaw contact patch, not guessed).
AG_STATIC = np.array([-0.0079, 0.0026, -0.1010])     # on `gripper`
AG_MOVING = np.array([-0.0122, -0.0805, 0.0169])     # on `jaw`

# The checkpoint's own action quantiles (LeRobot units).
CKPT_A = {"q01": np.array([-33.978, -108.852, -27.220, 58.022, -34.330, 1.335]),
          "q50": np.array([-5.058, -108.380, 0.553, 77.627, -13.137, 1.488]),
          "q99": np.array([27.993, 40.773, 98.420, 102.679, 32.737, 37.116]),
          "min": np.array([-33.978, -109.714, -27.297, 57.846, -34.330, 0.000]),
          "max": np.array([28.088, 40.791, 99.033, 102.681, 32.747, 37.275])}
# gripper.pos while HOLDING the cube in the training set (~17.5) vs closed-empty
# (2.87): anything below this is the jaw commanded shut.
GRIP_CLOSING = 20.0


def xform(p7, local):
    """Apply a logged (pos, quat xyzw) pose to a link-local point."""
    return Rotation.from_quat(p7[..., 3:7]).apply(local) + p7[..., :3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--json", default="", help="also write the report here (default <run>/score.json)")
    a = ap.parse_args()

    L = json.load(open(a.layout))
    P = np.load(os.path.join(a.run, "poses.npz"), allow_pickle=True)
    keys = [str(k) for k in P["keys"]]
    po = P["poses"]
    n = po.shape[0]
    epz = os.path.join(a.run, "episode.npz")
    E = np.load(epz, allow_pickle=True) if os.path.exists(epz) else None

    idx = {k: i for i, k in enumerate(keys)}
    cube = po[:, idx["obj::blue_cube"], :3]
    grip = po[:, idx["link::gripper"]]
    jaw = po[:, idx["link::jaw"]]
    centre = 0.5 * (xform(grip, AG_STATIC) + xform(jaw, AG_MOVING))

    box_c = np.asarray(L["box_centre_world"], float)
    half_in = L["box_inner"] / 2.0
    surf = L.get("surface_z", 0.0)          # top face of the mat, world z
    rim = surf + L["box_floor"] + L["box_height"]
    floor_z = surf + L["box_floor"]                 # inside face of the box's base slab
    cs = L["cube_size"]

    # ---- 1. did the cube move --------------------------------------------
    net = float(np.linalg.norm(cube[-1] - cube[0]))
    path = float(np.linalg.norm(np.diff(cube, axis=0), axis=1).sum())
    lifted = float((cube[:, 2] - cube[0, 2]).max())

    # ---- 2. is it in the box ---------------------------------------------
    # Containment: centre inside the interior footprint with the whole cube
    # clear of the walls' inner faces, and resting on the box floor rather than
    # perched on the rim.
    d_xy = np.abs(cube[:, :2] - box_c[:2])
    inside_xy = np.all(d_xy <= (half_in - cs / 2.0 + 0.004), axis=1)
    on_floor = (cube[:, 2] > floor_z) & (cube[:, 2] < rim + cs)
    in_box = inside_xy & on_floor
    final_in = bool(in_box[-1])
    # centre-only, a looser test reported alongside so a near-miss is visible
    loose = bool(np.all(d_xy[-1] <= half_in + cs / 2.0) and cube[-1, 2] > floor_z)

    # ---- 3/4. the gripper ------------------------------------------------
    dist = np.linalg.norm(centre - cube, axis=1)
    jmin = int(np.argmin(dist))
    grip_cmd = E["action_lerobot"][:, 5] if E is not None else np.full(n, np.nan)
    grip_st = E["state_lerobot"][:, 5] if E is not None else np.full(n, np.nan)
    m = min(n, len(grip_cmd))
    # "closed ON the cube" = jaw commanded shut while the cube is between the pads
    on_it = (dist[:m] < (cs / 2.0 + 0.025)) & (grip_cmd[:m] < GRIP_CLOSING)
    closed_on_cube = int(on_it.sum())
    ever_opened = bool(np.nanmax(grip_cmd) > 12.0)

    rep = {
        "run": os.path.abspath(a.run), "frames": int(n),
        "cube": {"start": cube[0].tolist(), "end": cube[-1].tolist(),
                 "net_displacement_m": net, "path_length_m": path,
                 "max_lift_m": lifted},
        "box": {"centre": box_c.tolist(), "inner": L["box_inner"],
                "rim_height_m": rim},
        "success": {"cube_in_box_final": final_in,
                    "cube_in_box_any_frame": bool(in_box.any()),
                    "cube_over_box_loose": loose,
                    "frames_in_box": int(in_box.sum())},
        "gripper": {"min_grasp_centre_to_cube_m": float(dist.min()),
                    "at_frame": jmin,
                    "start_distance_m": float(dist[0]),
                    "final_distance_m": float(dist[-1]),
                    "ever_opened": ever_opened,
                    "cmd_range": [float(np.nanmin(grip_cmd)), float(np.nanmax(grip_cmd))],
                    "achieved_range": [float(np.nanmin(grip_st)), float(np.nanmax(grip_st))],
                    "frames_closed_on_cube": closed_on_cube},
    }

    print(f"=== {os.path.basename(a.run)}  {n} frames ===")
    print(f"cube  start {np.round(cube[0],4)}  end {np.round(cube[-1],4)}")
    print(f"      net displacement {net*1000:7.1f} mm   path {path*1000:7.1f} mm   "
          f"max lift {lifted*1000:6.1f} mm")
    print(f"box   centre {np.round(box_c,4)}  inner {L['box_inner']*100:.1f} cm  "
          f"rim {rim*100:.1f} cm")
    print(f"SUCCESS (cube inside the box at the last frame): "
          f"{'YES' if final_in else 'NO'}"
          f"   (any frame: {'yes' if in_box.any() else 'no'}, "
          f"loosely over the box: {'yes' if loose else 'no'})")
    print(f"grip  min grasp-centre->cube {dist.min()*1000:6.1f} mm at frame {jmin}"
          f"  (start {dist[0]*1000:.0f} mm, end {dist[-1]*1000:.0f} mm)")
    print(f"      commanded jaw {rep['gripper']['cmd_range'][0]:.1f} .. "
          f"{rep['gripper']['cmd_range'][1]:.1f}   ever opened: {ever_opened}   "
          f"frames closed ON the cube: {closed_on_cube}")

    # ---- 5. commanded trajectory vs the training action distribution ------
    if E is not None:
        A = E["action_lerobot"]
        lc = np.asarray(E["cmd_limit_clipped"])
        print("\ncommanded action vs the checkpoint's own action distribution:")
        print(f"  {'joint':14s} {'run min':>8s} {'run med':>8s} {'run max':>8s} | "
              f"{'ckpt min':>8s} {'ckpt q50':>8s} {'ckpt max':>8s} | in[q01,q99]  clipped")
        dist_rep = {}
        for i, j in enumerate(LEROBOT_JOINTS):
            inq = float(((A[:, i] >= CKPT_A["q01"][i]) & (A[:, i] <= CKPT_A["q99"][i])).mean())
            print(f"  {j:14s} {A[:,i].min():8.1f} {np.median(A[:,i]):8.1f} {A[:,i].max():8.1f} | "
                  f"{CKPT_A['min'][i]:8.1f} {CKPT_A['q50'][i]:8.1f} {CKPT_A['max'][i]:8.1f} | "
                  f"{inq*100:9.0f}%  {int(lc[:,i].sum()):4d}/{len(lc)}")
            dist_rep[j] = {"run": [float(A[:, i].min()), float(np.median(A[:, i])),
                                   float(A[:, i].max())],
                           "frac_in_ckpt_q01_q99": inq,
                           "clipped_frames": int(lc[:, i].sum())}
        rep["action_vs_training"] = dist_rep
        rep["state_clamped"] = {j: int(np.asarray(E["state_clamped"])[:, i].sum())
                                for i, j in enumerate(LEROBOT_JOINTS)}
        rep["inferences"] = int(len(E["chunk_at"]))

    # ---- other objects, so a shoved box or mat shows up -------------------
    print("\nother object displacement:")
    moved = {}
    for k in keys:
        if not k.startswith("obj::") or k == "obj::blue_cube":
            continue
        d = float(np.linalg.norm(po[-1, idx[k], :3] - po[0, idx[k], :3]))
        moved[k[5:]] = d
        print(f"  {k[5:]:20s} {d*1000:7.1f} mm")
    rep["other_objects_mm"] = {k: v * 1000 for k, v in moved.items()}

    out = a.json or os.path.join(a.run, "score.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
