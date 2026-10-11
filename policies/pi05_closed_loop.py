"""Closed-loop pi0.5 driving the SO-101 inside the reconstructed OmniGibson scene.

This is the SIM half (conda env `simfoundry`). The POLICY half is
vla/pi05_server.py (conda env `openpi`); the two cannot share an env (numpy
1.26 vs 2.2, conflicting transformers pins) and talk over a raw loopback socket.
jobs/run_pi05_closed_loop.sbatch starts both in one Slurm job.

READ THIS BEFORE JUDGING THE OUTPUT
-----------------------------------
The checkpoint (pdhinaka/pi05_so101_blue_cube_gray_box_cleaned_20k) is a
TASK-SPECIFIC fine-tune on "put the blue cube in the gray box". This scene has
an orange block, a red boat toy and a white paper cup, and no blue cube and no
gray box anywhere in it. There is no reason to expect task success, and success
is not what this script is for. It is for the PLUMBING: that live Isaac camera
images and live joint state reach a pi0.5 forward pass, that its action chunk
comes back in units the simulator can execute, and that the whole loop runs for
hundreds of steps without diverging. Whatever the arm does is reported as-is.

THE LOOP
--------
    obs -> wrist_cam + top_cam (224x224) and joint state (LeRobot units)
        -> socket -> pi0.5 -> chunk of 50 absolute joint targets
        -> execute the first --exec-steps of it -> re-plan

--exec-steps is a real choice, so it is a flag:
  * 50 (full chunk, pure open loop) -- the policy acts blind for 50/30 s = 1.7 s
    and never sees the consequence of its own motion inside a chunk.
  * 1 (re-plan every control step) -- one 4.1 B-param forward pass with 10
    denoising steps per 1/30 s step. Measured ~0.5 s on an L40S, i.e. ~15x
    slower than real time, and on a 1 h job it caps the episode at a few hundred
    steps.
  * 10 (DEFAULT) -- 1/3 s of open-loop execution, re-planning at ~3 Hz. This is
    what openpi's own serving loop does: consume a PREFIX of the chunk, not all
    of it. Inference is amortised over 10 cheap sim steps.

CONTROL AND UNITS
-----------------
arm_0 must be a JointController in position mode (use_delta_commands=False), so
the command vector IS the absolute joint angles in radians. Generate the scene
with `tools/swap_robot_so101.py --controller joint`. Feeding pi0.5's absolute
targets to the IK controller the scripted pick-and-place uses would read them as
Cartesian pose DELTAS and fling the arm. vla/so101_units.py does the
radians <-> LeRobot normalized conversion and documents its one assumption.

OUTPUTS
-------
  <out>/poses.npz          keys / poses, the format tools/render_episode.py and
                           tools/render_episode_multiview.py consume
  <out>/wrist_cam/*.png    the LIVE Isaac frames the policy actually saw
  <out>/top_cam/*.png
  <out>/episode.npz        state, action, commanded radians, clamp masks, timings
  <out>/meta.json          cameras, intrinsics, settings, summary
  <out>/OK                 written only on a clean finish

Always print the traceback BEFORE og.shutdown(): shutdown calls app.close(),
which terminates the process, so a bare try/finally exits 0 with no output.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch as th
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/scratch/pandya.kei/simfoundry-repro/vla")
from so101_units import SIM_JOINTS, LEROBOT_JOINTS, SO101Units  # noqa: E402

from simfoundry import CFG_DIR, import_og_dependencies  # noqa: E402
from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths  # noqa: E402

import_og_dependencies()
import omnigibson as og  # noqa: E402

T0 = time.time()
LINKS = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw"]
WRIST_CAM = "robot0:gripper:Camera:0"
HDR = struct.Struct(">Q")


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def _np(v):
    return np.asarray(v.cpu() if isinstance(v, th.Tensor) else v, dtype=float)


def _u8(f):
    f = _np(f)
    if f.ndim == 3 and f.shape[-1] == 4:
        f = f[:, :, :3]
    if f.dtype != np.uint8:
        f = f * 255.0 if f.max(initial=0) <= 1.0 else f
    return np.clip(f, 0, 255).astype(np.uint8)


def look_at_quat(cam, target, up=(0.0, 0.0, 1.0)):
    """USD/Isaac cameras look down -Z with +Y up; return xyzw."""
    f = np.asarray(target, float) - np.asarray(cam, float)
    f /= np.linalg.norm(f)
    z = -f
    x = np.cross(np.asarray(up, float), z)
    n = np.linalg.norm(x)
    x = np.array([1.0, 0.0, 0.0]) if n < 1e-6 else x / n
    y = np.cross(z, x)
    return [float(v) for v in Rotation.from_matrix(np.stack([x, y, z], axis=1)).as_quat()]


# ------------------------------------------------------------------ client --
class PolicyClient:
    """Raw TCP, length-prefixed .npz. NOT http: the cluster's squid proxy
    intercepts 127.0.0.1 and would answer with a 403 that looks like an app
    auth error, and numpy 2.2 pickles cannot be read by numpy 1.26 anyway."""

    def __init__(self, host, port, timeout=600.0, retries=240, log=log):
        self.s = None
        for i in range(retries):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(timeout)
                s.connect((host, port))
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.s = s
                log(f"policy server connected at {host}:{port} (attempt {i+1})")
                return
            except OSError:
                time.sleep(2.0)
        raise RuntimeError(f"no policy server on {host}:{port} after {retries} tries")

    def _send(self, **arrays):
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        b = buf.getvalue()
        self.s.sendall(HDR.pack(len(b)) + b)

    def _recv(self):
        def ex(n):
            out, got = [], 0
            while got < n:
                c = self.s.recv(min(1 << 20, n - got))
                if not c:
                    raise ConnectionError("policy server closed the connection")
                out.append(c)
                got += len(c)
            return b"".join(out)
        (n,) = HDR.unpack(ex(HDR.size))
        return np.load(io.BytesIO(ex(n)), allow_pickle=False)

    def infer(self, state, images, task):
        self._send(state=np.asarray(state, np.float64),
                   task=np.asarray(task), **images)
        return self._recv()

    def close(self):
        try:
            self._send(stop=np.asarray(1))
        except Exception:
            pass
        try:
            self.s.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-json", required=True,
                    help="MUST be a --controller joint scene (see tools/swap_robot_so101.py)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PI05_PORT", 48051)))
    ap.add_argument("--steps", type=int, default=300, help="control steps to execute")
    ap.add_argument("--exec-steps", type=int, default=10,
                    help="how many steps of each 50-step chunk to execute before "
                         "re-planning. 50 = pure open loop, 1 = re-plan every step.")
    ap.add_argument("--task", default="put the blue cube in the gray box",
                    help="the language prompt. The checkpoint was fine-tuned on "
                         "exactly this sentence; a different one is out of "
                         "distribution for it in a second, separate way.")
    ap.add_argument("--warmup", type=int, default=20,
                    help="steps holding the rest pose before the first inference, "
                         "so physics and the renderer have settled")
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=480)
    ap.add_argument("--policy-res", type=int, default=224,
                    help="images are resized to this before being sent; the model "
                         "resizes to 224 internally anyway, so this just keeps the "
                         "payload small")
    ap.add_argument("--action-frequency", type=int, default=30,
                    help="30 Hz: the training set is 18736 frames over 50 episodes "
                         "= 375 frames/episode, which at a typical ~13 s SO-101 "
                         "demo is ~30 fps, LeRobot's record default. Playing the "
                         "chunk at the wrong rate would stretch or compress every "
                         "motion the policy has learned.")
    ap.add_argument("--rendering-frequency", type=int, default=30)
    ap.add_argument("--physics-frequency", type=int, default=120)
    ap.add_argument("--top-az", type=float, default=180.0,
                    help="top camera azimuth in deg about the work area; 180 puts "
                         "it behind the robot looking out over its shoulder")
    ap.add_argument("--top-el", type=float, default=62.0)
    ap.add_argument("--top-dist", type=float, default=0.0,
                    help="0 = auto from the scene extent")
    ap.add_argument("--top-eye", default="",
                    help="x,y,z world position for top_cam, OVERRIDING --top-az/el/dist. "
                         "The auto framing is a bounding box over every categorised "
                         "object, which is useless once the scene contains a table "
                         "slab; and matching the training rig's viewpoint is worth "
                         "doing by hand (viewpoint mismatch is a standard reason a "
                         "VLA does nothing).")
    ap.add_argument("--top-target", default="",
                    help="x,y,z world point top_cam looks at (default: the auto target)")
    ap.add_argument("--top-focal", type=float, default=17.0,
                    help="top_cam focal length in mm over a 20.955 mm aperture: "
                         "17 = 63 deg horizontal, 12 = 81 deg, 24 = 47 deg")
    ap.add_argument("--frame-objects", default="",
                    help="comma-separated categories to use for the AUTO top_cam "
                         "framing; empty = all of them")
    ap.add_argument("--max-step-rad", type=float, default=0.0,
                    help="optional per-step per-joint rate limit in rad. 0 = OFF, "
                         "which is the default on purpose: limiting it would "
                         "flatter the policy by hiding how large its commanded "
                         "jumps are. The max observed jump is reported either way.")
    ap.add_argument("--action-offset", default="",
                    help="FOLLOWER-ERROR EMULATION, 6 comma-separated LeRobot "
                         "units added to every commanded action, or 'dataset' "
                         "for the measured values.\n"
                         "WHY THIS EXISTS. The checkpoint is a RELATIVE-action "
                         "policy (use_relative_actions=true): what it predicts is "
                         "action - state, and the postprocessor adds the state "
                         "back. On the real SO-101 the follower never reaches its "
                         "commanded position -- over the whole training set "
                         "mean(action - state) is -2.37 on shoulder_lift (-3.96 "
                         "while the shoulder is near its bottom stop) and -2.36 on "
                         "the gripper, and that offset does NOT disappear at any "
                         "lag up to 7 frames, so it is steady-state servo error "
                         "under gravity and squeeze force, not latency. In sim the "
                         "JointController reaches its target within one 1/30 s "
                         "step (measured: commanded and achieved differ by ~0 in "
                         "results/so101/vla/framed/episode.npz), so the policy's "
                         "systematic delta INTEGRATES instead of being absorbed, "
                         "and the joint walks to its stop. Adding mean(state - "
                         "action) back makes the sim arm settle where the real "
                         "follower settled for the same command. It is a model of "
                         "the real servo, not a nudge toward success, and it is "
                         "off by default so the uncorrected behaviour is the "
                         "default thing reported.")
    ap.add_argument("--follower-alpha", default="",
                    help="FOLLOWER SERVO EMULATION. 6 comma-separated per-joint "
                         "gains in (0,1], or 'dataset' for the measured ones.\n"
                         "The sim's JointController reaches its target inside ONE "
                         "1/30 s step; the real SO-101 does not. Fitting the "
                         "first-order model s[t+1] = s[t] + alpha*(a[t]-s[t]) to "
                         "all 18736 frames of the training set gives alpha = 0.27 / "
                         "0.20 / 0.24 / 0.20 / 0.27 / 0.07 (tau = 120-170 ms for "
                         "the arm, 470 ms for the jaw) with a per-step residual of "
                         "0.08-0.59 normalized units -- and during motion the "
                         "commanded position only matches the achieved one 3-4 "
                         "FRAMES later. So the real arm covers ~20-27% of each "
                         "commanded step per frame and the sim arm covers 100%: "
                         "replayed in sim, the policy's chunk moves the arm "
                         "through space ~4x faster than it did in training, the "
                         "re-planning loop sees a state far further along than it "
                         "should, and the reach sails past the object (measured: "
                         "the tip reaches the cube's x,y at step ~48 and then "
                         "overshoots to 8 cm beyond it). Filtering the commanded "
                         "joint target with the measured alpha makes the sim "
                         "servo behave like the one that recorded the data. Off by "
                         "default; the unfiltered behaviour is the default result.")
    ap.add_argument("--save-every", type=int, default=1, help="save camera PNGs every N steps")
    ap.add_argument("--flush-every", type=int, default=50,
                    help="rewrite poses.npz / episode.npz every N steps. The "
                         "`sharing` partition's 1 h cap is hard and Isaac's load "
                         "eats 5-15 min of it, so a run that is killed mid-rollout "
                         "must still leave a usable pose log behind. 0 = only at "
                         "the end.")
    a = ap.parse_args()

    # mean(observation.state - action) per joint over all 18736 frames of
    # pdhinaka/so101_blue_cube_in_gray_box_pi05_cleaned.
    DATASET_FOLLOWER_OFFSET = np.array([-0.082, 2.372, 0.697, -0.657, 0.028, 2.361])
    if a.action_offset == "dataset":
        act_off = DATASET_FOLLOWER_OFFSET.copy()
    elif a.action_offset:
        act_off = np.array([float(v) for v in a.action_offset.split(",")], float)
        assert act_off.shape == (6,)
    else:
        act_off = np.zeros(6)

    # Fitted on all 18736 frames of
    # pdhinaka/so101_blue_cube_in_gray_box_pi05_cleaned (see --follower-alpha).
    DATASET_FOLLOWER_ALPHA = np.array([0.272, 0.198, 0.235, 0.199, 0.273, 0.071])
    if a.follower_alpha == "dataset":
        alpha = DATASET_FOLLOWER_ALPHA.copy()
    elif a.follower_alpha:
        alpha = np.array([float(v) for v in a.follower_alpha.split(",")], float)
        assert alpha.shape == (6,) and np.all((alpha > 0) & (alpha <= 1.0))
    else:
        alpha = np.ones(6)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "wrist_cam").mkdir(exist_ok=True)
    (out / "top_cam").mkdir(exist_ok=True)

    log(f"python {sys.version.split()[0]}  out={out}")
    log(f"SO101_USD_PATH={os.environ.get('SO101_USD_PATH', '(class default)')}")
    log(f"scene={a.scene_json}")

    scene_file = load_json_with_absolute_usd_paths(a.scene_json)
    full = json.loads(Path(a.scene_json).read_text())
    rargs = scene_file["objects_info"]["init_info"]["robot0"]["args"]
    cc = rargs.get("controller_config", {}).get("arm_0", {})
    assert cc.get("name") == "JointController" and cc.get("use_delta_commands") is False, (
        "arm_0 must be a position-mode JointController with use_delta_commands=False. "
        "Regenerate the scene with: tools/swap_robot_so101.py --controller joint. "
        f"Got: {cc}")
    rargs["obs_modalities"] = ["rgb", "proprio"]   # NEVER a seg_* modality: see wrist_cam_probe.py

    # --- scene-frame overhead camera (`top_cam`, the checkpoint's 2nd camera) --
    base = np.asarray(full["state"]["registry"]["object_registry"]["robot0"]["root_link"]["pos"], float)
    cats = {k: v.get("args", {}).get("category") for k, v in full["objects_info"]["init_info"].items()}
    want = [c.strip() for c in a.frame_objects.split(",") if c.strip()]
    objp = [np.asarray(v["root_link"]["pos"], float)
            for k, v in full["state"]["registry"]["object_registry"].items()
            if cats.get(k) and k != "robot0" and (not want or cats.get(k) in want)]
    content = np.asarray(objp + [base])
    tgt = (content.min(0) + content.max(0)) / 2.0
    tgt[2] = max(float(tgt[2]), 0.04)
    diag = float(np.linalg.norm(content.max(0) - content.min(0)))
    dist = a.top_dist or max(0.45, 1.7 * diag)
    az, el = np.radians(a.top_az), np.radians(a.top_el)
    eye = tgt + dist * np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
    if a.top_target:
        tgt = np.array([float(v) for v in a.top_target.split(",")], float)
    if a.top_eye:
        eye = np.array([float(v) for v in a.top_eye.split(",")], float)
    log(f"top_cam: eye={np.round(eye,3)} -> target={np.round(tgt,3)}  "
        f"dist={float(np.linalg.norm(eye-tgt)):.3f} m  f={a.top_focal} mm"
        + ("  (explicit)" if a.top_eye else f"  az={a.top_az} el={a.top_el} deg"))

    ext = [{
        "sensor_type": "VisionSensor", "name": "top_cam",
        "relative_prim_path": "/top_cam", "modalities": ["rgb"],
        "sensor_kwargs": {"image_height": a.cam_height, "image_width": a.cam_width,
                          "focal_length": float(a.top_focal), "horizontal_aperture": 20.955,
                          "clipping_range": [0.01, 1000000.0]},
        "position": [float(v) for v in eye],
        "orientation": look_at_quat(eye, tgt),
        "pose_frame": "scene",
    }]

    cfg = {"env": {"external_sensors": ext,
                   "action_frequency": a.action_frequency,
                   "rendering_frequency": a.rendering_frequency,
                   "physics_frequency": a.physics_frequency},
           "scene": {"type": "Scene", "scene_file": scene_file,
                     "use_floor_plane": True, "floor_plane_visible": True,
                     "use_skybox": False, "include_robots": True}}

    os.chdir(CFG_DIR)
    extra = os.environ.get("KIT_EXTRA_ARGS", "").strip()
    if extra:
        import shlex
        sys.argv.extend(shlex.split(extra))

    env = cli = None
    try:
        log("constructing og.Environment ...")
        env = og.Environment(configs=cfg)
        log("built; reset ...")
        env.reset()
        log("reset done")

        robot = env.robots[0]
        jnames = [str(j) for j in robot.joints]
        units, jidx = SO101Units.from_sim_limits(
            jnames, _np(robot.joint_lower_limits), _np(robot.joint_upper_limits), log=log)
        log(f"robot joints {jnames}; policy order {SIM_JOINTS} -> sim idx {jidx}")

        cams = dict(robot.sensors)
        cams.update(env.external_sensors or {})
        log(f"cameras: {list(cams)}")
        if WRIST_CAM not in cams:
            # The canonical key is robot0:gripper:Camera:0, but it is derived from
            # the prim name, so fall back to any robot sensor on the gripper link.
            cand = [n for n in robot.sensors if "gripper" in n or "wrist" in n]
            assert cand, f"wrist camera missing; have {list(cams)}"
            log(f"wrist camera key is {cand[0]}, not the expected {WRIST_CAM}")
            globals()["WRIST_CAM"] = cand[0]
        assert "top_cam" in cams, f"top camera missing; have {list(cams)}"
        caminfo = {}
        for n in (WRIST_CAM, "top_cam"):
            s = cams[n]
            w, h, f, ha = s.image_width, s.image_height, s.focal_length, s.horizontal_aperture
            va = ha * h / w
            caminfo[n] = {"prim_path": str(s.prim_path), "width": int(w), "height": int(h),
                          "focal_length_mm": float(f), "horizontal_aperture_mm": float(ha),
                          "K": [w * f / ha, h * f / va, w / 2.0, h / 2.0],
                          "fov_deg": [2 * np.degrees(np.arctan(ha / (2 * f))),
                                      2 * np.degrees(np.arctan(va / (2 * f)))]}
            log(f"  {n}: {caminfo[n]}")

        objs = {}
        for o in env.scene.objects:
            c = getattr(o, "category", None)
            if c:
                objs[c] = o
        POSE_OBJS = sorted(objs)
        POSE_LINKS = [l for l in LINKS if l in robot.links]
        log(f"pose log: {len(POSE_LINKS)} links + {len(POSE_OBJS)} objects {POSE_OBJS}")

        adim = robot.action_space.shape[0]
        n_arm = len(jidx) - 1
        log(f"action dim {adim} = {n_arm} arm joints (rad) + 1 gripper cmd [-1,1]")

        # ---- helpers ------------------------------------------------------
        def grab(name):
            o = cams[name].get_obs()
            o = o[0] if isinstance(o, tuple) else o
            return _u8(o["rgb"])

        def small(img):
            return np.asarray(Image.fromarray(img).resize(
                (a.policy_res, a.policy_res), Image.BILINEAR))

        def snap(frame, rec_links, rec_objs):
            r = []
            for l in POSE_LINKS:
                p, q = robot.links[l].get_position_orientation()
                r.append(np.concatenate([_np(p), _np(q)]))
            for c in POSE_OBJS:
                p, q = objs[c].get_position_orientation()
                r.append(np.concatenate([_np(p), _np(q)]))
            return np.concatenate(r).reshape(len(POSE_LINKS) + len(POSE_OBJS), 7)

        pose_keys = [f"link::{l}" for l in POSE_LINKS] + [f"obj::{c}" for c in POSE_OBJS]
        # render_episode.py sorts its keys; match that so the npz is interchangeable.
        order = np.argsort(pose_keys)
        pose_keys_sorted = [pose_keys[i] for i in order]

        # ---- warmup: hold the rest pose -----------------------------------
        rest_rad = _np(robot.get_joint_positions())[jidx]
        hold = th.zeros(adim)
        hold[:n_arm] = th.tensor(rest_rad[:n_arm], dtype=th.float32)
        hold[-1] = units.gripper_norm_to_cmd(units.rad_to_norm(rest_rad)[5])
        for _ in range(a.warmup):
            env.step(hold)
        log(f"action offset (follower-error emulation): {np.round(act_off, 3)}")
        log(f"follower alpha (servo emulation, 1 = snap): {np.round(alpha, 3)}")
        # Seed the filter at the arm's actual pose, so step 0 is not a jump.
        exec_norm = units.rad_to_norm(rest_rad).copy()
        log(f"warmup {a.warmup} steps done; rest (rad) {np.round(rest_rad,4)}")
        log(f"rest in LeRobot units {np.round(units.rad_to_norm(rest_rad),2)}")

        cli = PolicyClient(a.host, a.port)

        # ---- the loop -------------------------------------------------------
        poses, states, acts, cmds, clamps, limclip, targs = [], [], [], [], [], [], []
        infer_s, step_s, chunk_at, bins_log = [], [], [], []
        chunk = None
        k = 0
        prev_cmd = rest_rad.copy()
        max_jump = 0.0
        t_loop = time.time()

        def dump():
            """Write the pose log and the episode record from whatever has been
            collected so far. Called periodically, not just at the end: the
            `sharing` partition's 1 h cap is hard, so a rollout that gets cut
            off must still leave a renderable pose log and the live frames
            (which are already on disk per step) behind. Written to a .tmp and
            renamed, so a kill mid-write cannot leave a truncated npz."""
            if not poses:
                return
            for name, arrays in (
                ("poses.npz", dict(keys=np.array(pose_keys_sorted),
                                   poses=np.asarray(poses))),
                ("episode.npz", dict(
                    joint_names=np.array(SIM_JOINTS),
                    lerobot_names=np.array(LEROBOT_JOINTS),
                    state_lerobot=np.asarray(states), action_lerobot=np.asarray(acts),
                    policy_target_lerobot=np.asarray(targs),
                    cmd_rad=np.asarray(cmds), chunk_at=np.asarray(chunk_at),
                    state_clamped=np.asarray(clamps),
                    cmd_limit_clipped=np.asarray(limclip),
                    infer_s=np.asarray(infer_s), step_s=np.asarray(step_s),
                    joint_lo=units.lo, joint_hi=units.hi))):
                tmp = out / (name + ".tmp.npz")
                np.savez_compressed(tmp, **arrays)
                os.replace(tmp, out / name)

        for step in range(a.steps):
            # (re)plan when the executed prefix of the current chunk runs out
            if chunk is None or k >= min(a.exec_steps, len(chunk)):
                imgs = {"wrist_cam": small(grab(WRIST_CAM)), "top_cam": small(grab("top_cam"))}
                q = _np(robot.get_joint_positions())[jidx]
                st = units.rad_to_norm(q)
                t0 = time.time()
                rep = cli.infer(st, imgs, a.task)
                chunk = np.asarray(rep["chunk"], float)
                infer_s.append(float(rep["infer_s"]))
                clamps.append(np.asarray(rep["clamped"], bool))
                bins_log.append(np.asarray(rep["bins"], int))
                chunk_at.append(step)
                k = 0
                if len(chunk_at) <= 3 or len(chunk_at) % 10 == 0:
                    log(f"  plan #{len(chunk_at)} @step {step}: {time.time()-t0:.2f}s "
                        f"(model {float(rep['infer_s']):.2f}s)  chunk{chunk.shape}  "
                        f"a0={np.array2string(chunk[0], precision=1)}")

            a_tgt = chunk[k] + act_off
            # First-order follower: the executed target chases the policy's
            # target instead of snapping to it. alpha = 1 (the default) is the
            # old behaviour exactly.
            a_norm = exec_norm + alpha * (a_tgt - exec_norm)
            exec_norm = a_norm.copy()
            q_cmd = units.norm_to_rad(a_norm)
            q_lim = units.clip_rad(q_cmd)
            limclip.append(q_cmd != q_lim)
            if a.max_step_rad > 0:
                q_lim = np.clip(q_lim, prev_cmd - a.max_step_rad, prev_cmd + a.max_step_rad)
            max_jump = max(max_jump, float(np.abs(q_lim[:n_arm] - prev_cmd[:n_arm]).max()))
            prev_cmd = q_lim.copy()

            act = th.zeros(adim)
            act[:n_arm] = th.tensor(q_lim[:n_arm], dtype=th.float32)
            act[-1] = units.gripper_norm_to_cmd(a_norm[5])

            ts = time.time()
            env.step(act)
            step_s.append(time.time() - ts)
            k += 1

            poses.append(snap(step, POSE_LINKS, POSE_OBJS)[order])
            states.append(units.rad_to_norm(_np(robot.get_joint_positions())[jidx]))
            acts.append(a_norm)
            targs.append(a_tgt)
            cmds.append(q_lim)
            if step % a.save_every == 0:
                Image.fromarray(grab(WRIST_CAM)).save(out / "wrist_cam" / f"rgb_{step:04d}.png")
                Image.fromarray(grab("top_cam")).save(out / "top_cam" / f"rgb_{step:04d}.png")
            if a.flush_every and step and step % a.flush_every == 0:
                dump()
            if step % 25 == 0:
                log(f"  step {step}/{a.steps}  sim {np.mean(step_s[-25:]):.3f}s/step")

        wall = time.time() - t_loop
        cli.close()
        cli = None
        dump()
        poses = np.asarray(poses)

        cl = np.asarray(clamps).sum(0)
        lc = np.asarray(limclip).sum(0)
        states_a = np.asarray(states)
        summary = {
            "steps": int(a.steps), "plans": len(chunk_at), "exec_steps": int(a.exec_steps),
            "task": a.task, "wall_s": wall,
            "sim_step_s_mean": float(np.mean(step_s)),
            "infer_s_mean": float(np.mean(infer_s)), "infer_s_p95": float(np.percentile(infer_s, 95)),
            "state_clamped_to_ckpt_quantiles": {j: int(c) for j, c in zip(LEROBOT_JOINTS, cl)},
            "cmd_clipped_to_joint_limits": {j: int(c) for j, c in zip(LEROBOT_JOINTS, lc)},
            "max_commanded_jump_rad": max_jump,
            "action_offset": act_off.tolist(),
            "follower_alpha": alpha.tolist(),
            "state_range_lerobot": {j: [float(states_a[:, i].min()), float(states_a[:, i].max())]
                                    for i, j in enumerate(LEROBOT_JOINTS)},
            "object_displacement_m": {},
        }
        for i, kname in enumerate(pose_keys_sorted):
            if kname.startswith("obj::"):
                d = float(np.linalg.norm(poses[-1, i, :3] - poses[0, i, :3]))
                summary["object_displacement_m"][kname[5:]] = d
        (out / "meta.json").write_text(json.dumps(
            {"args": vars(a), "cameras": caminfo, "pose_keys": pose_keys_sorted,
             "summary": summary}, indent=2))

        log("")
        log("=== SUMMARY ===")
        log(f"  {a.steps} control steps, {len(chunk_at)} inferences "
            f"(exec_steps={a.exec_steps}), {wall:.1f}s wall")
        log(f"  sim {np.mean(step_s):.3f}s/step   policy {np.mean(infer_s):.2f}s/call "
            f"(p95 {np.percentile(infer_s,95):.2f}s)")
        log(f"  state clamped into the checkpoint's [q01,q99] on: " +
            ", ".join(f"{j}={int(c)}/{len(clamps)}" for j, c in zip(LEROBOT_JOINTS, cl)))
        log(f"  commanded angle clipped to joint limits on: " +
            ", ".join(f"{j}={int(c)}/{a.steps}" for j, c in zip(LEROBOT_JOINTS, lc)))
        log(f"  largest commanded single-step joint jump: {max_jump:.4f} rad")
        for j, (mn, mx) in summary["state_range_lerobot"].items():
            log(f"    state {j:14s} {mn:8.2f} .. {mx:8.2f}")
        for k2, v in summary["object_displacement_m"].items():
            log(f"    object {k2:20s} moved {v*1000:7.1f} mm over the episode")
        log(f"  wrote {out}/poses.npz  ({poses.shape[0]} frames x {poses.shape[1]} prims)")
        (out / "OK").write_text(f"{a.steps}\n")
    except BaseException:
        import traceback
        traceback.print_exc()
        try:            # salvage whatever the rollout got to
            dump()
            log("flushed partial poses.npz / episode.npz")
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        (out / "FAILED").write_text(traceback.format_exc())
    finally:
        if cli is not None:
            cli.close()
        log("og.shutdown()")
        sys.stdout.flush()
        og.shutdown()


if __name__ == "__main__":
    main()
