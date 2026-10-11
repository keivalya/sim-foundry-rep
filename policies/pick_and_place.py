"""Scripted pick-and-place in a SimFoundry-reconstructed OmniGibson scene.

The Franka's arm_0 is an InverseKinematicsController in `pose_delta_ori` mode,
so it consumes DELTA end-effector poses, not absolute targets. We therefore run
closed-loop: read the current EEF position each step and command a clipped
delta toward the waypoint. grasping_mode is "assisted", so closing the gripper
near an object attaches it without needing exact finger contact.

Scene/video plumbing mirrors C_application/stages/0_smoke_random_actions.py.
"""
from __future__ import annotations
import argparse, os, json
from pathlib import Path

import imageio, numpy as np, torch as th
from PIL import Image

from scipy.spatial.transform import Rotation
from simfoundry import CFG_DIR, REPO_DIR, configure_omnigibson_data_path, import_og_dependencies
from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths
import_og_dependencies()
import omnigibson as og
from omnigibson.utils.config_utils import parse_config


# ---------- video helpers (same contract as the smoke test) ----------
def _u8(f):
    f = f.cpu().numpy() if isinstance(f, th.Tensor) else np.asarray(f)
    if f.ndim == 2: f = np.repeat(f[:, :, None], 3, axis=2)
    if f.shape[-1] == 4: f = f[:, :, :3]
    if f.dtype != np.uint8:
        f = (f * 255).astype(np.uint8) if f.max(initial=0) <= 1.0 else f.astype(np.uint8)
    return f

def _resize_h(f, h):
    if f.shape[0] == h: return f
    w = max(1, round(f.shape[1] * h / f.shape[0]))
    return np.asarray(Image.fromarray(f).resize((w, h), Image.Resampling.BILINEAR))

def collect_frame(obs, robot):
    frames = []
    for cam in obs.get(robot.name, {}).values():
        if isinstance(cam, dict) and "rgb" in cam: frames.append(cam["rgb"][:, :, :3])
    for cam in obs.get("external", {}).values():
        if isinstance(cam, dict) and "rgb" in cam: frames.append(cam["rgb"][:, :, :3])
    if not frames: return None
    fs = [_u8(f) for f in frames]
    h = max(f.shape[0] for f in fs)
    return np.concatenate([_resize_h(f, h) for f in fs], axis=1).astype(np.uint8)


def eef_pos(robot):
    """OmniGibson has moved this accessor around between versions."""
    for fn in ("get_eef_position",):
        if hasattr(robot, fn):
            try:
                v = getattr(robot, fn)()
            except TypeError:
                v = getattr(robot, fn)(robot.default_arm)
            return np.asarray(v.cpu() if isinstance(v, th.Tensor) else v, dtype=float)
    link = robot.eef_links[robot.default_arm]
    v = link.get_position_orientation()[0]
    return np.asarray(v.cpu() if isinstance(v, th.Tensor) else v, dtype=float)


def finger_tip_pos(robot):
    """Mean of the finger link origins. get_eef_position() returns the WRIST
    FLANGE, which on a Franka sits ~10 cm above the fingertips -- targeting the
    flange at the object put the fingers below the table, so assisted grasping
    had nothing between them. Control the fingertips instead."""
    links = robot.finger_links[robot.default_arm]
    ps = []
    for l in links:
        v = l.get_position_orientation()[0]
        ps.append(np.asarray(v.cpu() if isinstance(v, th.Tensor) else v, dtype=float))
    return np.mean(ps, axis=0)


def ag_point_pos(robot):
    """World position of the midpoint of OmniGibson's assisted-grasp points.

    THIS is the frame to control. franka.py defines the Panda hand's AG points
    at [0, 0.001, 0.045] in each finger link's frame -- 4.5 cm beyond the link
    ORIGINS that finger_tip_pos() returns. Driving the link origins to an object
    at z=0.018 therefore put the actual pads at z=-0.027, i.e. under the floor:
    the arm stalled against the ground plane (descend never converged, 4.1 cm
    short after 90 steps) and swept every object ~5.8 cm aside instead of
    grasping it. Targeting the AG midpoint is what assisted grasping actually
    tests for contact between.
    """
    pts = []
    for gp in list(robot._ag_start_points) + list(robot._ag_end_points):
        link = robot.links[gp.link_name]
        pq = link.get_position_orientation()
        lp = np.asarray(pq[0].cpu() if isinstance(pq[0], th.Tensor) else pq[0], dtype=float)
        lq = np.asarray(pq[1].cpu() if isinstance(pq[1], th.Tensor) else pq[1], dtype=float)
        off = gp.position
        off = np.asarray(off.cpu() if isinstance(off, th.Tensor) else off, dtype=float)
        pts.append(lp + Rotation.from_quat(lq).as_matrix() @ off)
    return np.mean(pts, axis=0)


def eef_quat(robot):
    """EEF orientation as xyzw."""
    link = robot.eef_links[robot.default_arm]
    q = link.get_position_orientation()[1]
    return np.asarray(q.cpu() if isinstance(q, th.Tensor) else q, dtype=float)


def _np(v):
    """Tensor/array -> float64 ndarray."""
    return np.asarray(v.cpu() if isinstance(v, th.Tensor) else v, dtype=float)


def _lpos(robot, name):
    v = robot.links[name].get_position_orientation()[0]
    return np.asarray(v.cpu() if isinstance(v, th.Tensor) else v, dtype=float)


def grasped_obj(robot):
    """Name of the assisted-grasp object in hand, or None."""
    try:
        o = robot._ag_obj_in_hand.get(robot.default_arm)
        return None if o is None else getattr(o, "name", str(o))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-name", default="fruits_demo")
    ap.add_argument("--scene-json", default=None,
                    help="override s14_og/reconstructed_og_scene.json (e.g. a repositioned-robot variant)")
    ap.add_argument("--settled-json", default=None)
    ap.add_argument("--pick", default=None,
                    help="comma-separated categories to attempt, in order. Default: auto-select by gripper fit.")
    ap.add_argument("--max-tasks", type=int, default=3)
    ap.add_argument("--place-dy", type=float, default=0.24,
                    help="place location offset in +y from the pick location")
    ap.add_argument("--video-name", default="pick_and_place")
    ap.add_argument("--reach", type=float, default=0.855,
                    help="nominal arm reach in m (Franka 0.855, SO-101 ~0.32)")
    ap.add_argument("--max-grasp-width", type=float, default=None,
                    help="max graspable object width in m. Required for a REVOLUTE "
                         "gripper like the SO-101's, where the joint limits are "
                         "radians and say nothing about the opening in metres.")
    ap.add_argument("--place-azimuth", type=float, default=0.0,
                    help="if non-zero, place by rotating the pick point about the "
                         "base by this many degrees, keeping its radius. The +y "
                         "offset mode can put the place spot outside the workspace "
                         "(the SO-101 was sent to 0.42 m with a 0.32 m reach, and "
                         "its +/-y fallback was out of reach too); rotating at a "
                         "constant radius is reachable by construction.")
    ap.add_argument("--grip-release", type=float, default=1.0,
                    help="gripper command used to RELEASE. Must be full travel: the "
                         "SO-101's partial 0.364 open command held the assisted grasp, "
                         "so the block was still in hand during the next task.")
    ap.add_argument("--approach", type=float, default=0.12,
                    help="height above the object to approach from, m")
    ap.add_argument("--lift", type=float, default=0.22,
                    help="height above the object to lift to, m. Franka-scaled by "
                         "default; a 0.32 m-reach SO-101 cannot reach 0.22 m up at "
                         "arm's length and the lift waypoint just stalls.")
    ap.add_argument("--poses-only", action="store_true",
                    help="record link/object poses and render NOTHING in Isaac. Its "
                         "RTX renderer both corrupts this scene's gripper and crashes "
                         "in libomni.hydra.rtx.plugin.so (job 10873560, SIGSEGV); "
                         "render_episode.py rebuilds the video offline from the log.")
    ap.add_argument("--cam-scene-frame", action="store_true",
                    help="place the workspace camera as a standalone SCENE-frame prim "
                         "instead of parenting it under the robot's base link. A "
                         "camera parented inside the articulation's prim hierarchy is "
                         "the only structural difference between the (correct) static "
                         "inspection renders and the (broken) episode video.")
    ap.add_argument("--cam-width", type=int, default=1280)
    ap.add_argument("--cam-height", type=int, default=720,
                    help="at 1280x720 with the scene framed ~0.9 m wide the SO-101 is "
                         "only ~250 px tall, and its thin brackets and hole grids "
                         "alias into looking like missing geometry.")
    ap.add_argument("--cam-dist-scale", type=float, default=1.25,
                    help="camera distance as a multiple of the content diagonal")
    ap.add_argument("--grip-open", type=float, default=1.0,
                    help="gripper command used for 'open'. 1.0 is full travel, which "
                         "is right for a parallel jaw but WRONG for the SO-101: its "
                         "jaw is a lever, and past ~30 deg the closing direction tilts "
                         "out of the plane perpendicular to the approach axis.")
    ap.add_argument("--cam-parent-link", default="panda_link0",
                    help="link the workspace camera is parented to (Franka "
                         "panda_link0, SO-101 base)")
    ap.add_argument("--robot-height", type=float, default=0.75,
                    help="approx standing height in m, used only for camera framing "
                         "(Franka ~0.75, SO-101 ~0.35)")
    ap.add_argument("--down-axis", default="+z", choices=["+z", "-z"],
                    help="which EEF-local axis should point at the table. Franka's "
                         "approach axis is +Z; the SO-101's pads sit at gripper-local "
                         "z=-0.10, so its approach axis is -Z.")
    ap.add_argument("--root-dir", default=str(Path(REPO_DIR) / "Data"))
    ap.add_argument("--external-sensors-config", default="sim_franka_teleop")
    ap.add_argument("--max-steps-per-waypoint", type=int, default=90)
    ap.add_argument("--pos-tol", type=float, default=0.02)
    ap.add_argument("--max-delta", type=float, default=0.03)
    ap.add_argument("--max-drot", type=float, default=0.15)   # rad per step
    ap.add_argument("--video-fps", type=int, default=15)
    a = ap.parse_args()

    root = Path(a.root_dir); scene_dir = root / a.scene_name
    scene_json_path = Path(a.scene_json) if a.scene_json else scene_dir / "s14_og" / "reconstructed_og_scene.json"
    settled_path = Path(a.settled_json) if a.settled_json else scene_dir / "s14_og" / "settled_poses.json"
    assert scene_json_path.exists(), f"missing {scene_json_path}"
    out_dir = scene_dir / "pick_place"; out_dir.mkdir(parents=True, exist_ok=True)

    settled = json.loads(settled_path.read_text())
    full = json.loads(scene_json_path.read_text())
    cat = {k: v.get("args", {}).get("category") for k, v in full["objects_info"]["init_info"].items()}
    by_cat = {}
    for k, v in settled.items():
        c = cat.get(k)
        if c: by_cat[c] = (k, np.asarray(v[0], dtype=float))

    # Franka Panda nominal reach is ~0.855 m from the base. In this scene the
    # base sits at x=-1.0 while most objects are near the origin, so only two
    # are actually reachable (measured: teal_square_plate 0.721 m, yellow_banana
    # 0.787 m; apple 1.016, orange 1.025, orange_plate 1.194 are NOT).
    REACH = a.reach
    base = np.asarray(settled.get("robot", [[-1.0, 0.0, 0.0]])[0], dtype=float)
    reach_of = {c: float(np.linalg.norm(p - base)) for c, (_, p) in by_cat.items()}
    for c, d in sorted(reach_of.items(), key=lambda kv: kv[1]):
        print(f"  reach {c:22s} {d:.3f} m  {'ok' if d < REACH else 'OUT OF REACH'}", flush=True)

    reachable = [c for c, d in reach_of.items() if d < REACH]
    print(f"  {len(reachable)}/{len(reach_of)} categories within reach", flush=True)
    assert reachable, "nothing within reach -- run reposition_robot.py first"
    # Which of those the GRIPPER can actually take is decided after the sim is
    # up, from each object's measured AABB vs the real finger travel.
    want = [c.strip() for c in a.pick.split(",")] if a.pick else None

    # Camera framing. The default sim_franka_teleop rig is set up for the DROID
    # cell and puts our objects at the frame edge against the skybox. Framing on
    # the PICK points alone is also wrong: it aims too low and too far +x, which
    # clipped the top of the arm and pushed the action off-centre. Frame on
    # everything the episode actually touches -- base, pick points, and the
    # place spots the task loop will generate.
    content = [base] + [by_cat[c][1] for c in reachable]
    for c in reachable:
        # The task loop places at +y, but flips to -y when +y is out of reach.
        # Framing only on +y put the centre on the wrong side when a flip
        # happened (the SO-101 sent the boat to y=-0.376), so include both.
        for sgn in (1.0, -1.0):
            content.append(by_cat[c][1] + np.array([0.0, sgn * (a.place_dy + 0.20), 0.0]))
    content = np.asarray(content, dtype=float)
    clo, chi = content.min(0), content.max(0)
    ws = (clo + chi) / 2
    ws[2] = 0.18                              # eye level for a ~0.7 m arm
    spread = np.array([chi[0]-clo[0], chi[1]-clo[1], a.robot_height])
    cam_dist = a.cam_dist_scale * float(np.linalg.norm(spread))

    # The camera prim path has to name the REAL robot class. It was hardcoded to
    # /controllable__frankapanda__robot0/panda_link0/..., so under the SO-101 the
    # parent prim did not exist and the camera was never placed relative to the
    # base -- which is why the first SO-101 video framed the arm off at the edge
    # with the cup filling the middle.
    _rcls = full["objects_info"]["init_info"]["robot0"]["class_name"].lower()
    print(f"  camera parent: /controllable__{_rcls}__robot0/{a.cam_parent_link}", flush=True)

    os.chdir(CFG_DIR)

    def look_at_quat(cam, target, up=(0.0, 0.0, 1.0)):
        """USD/Isaac cameras look down -Z with +Y up; return xyzw quaternion."""
        from scipy.spatial.transform import Rotation
        f = np.asarray(target, float) - np.asarray(cam, float)
        f /= np.linalg.norm(f)
        z = -f
        x = np.cross(np.asarray(up, float), z); n = np.linalg.norm(x)
        if n < 1e-6:
            x = np.array([1.0, 0.0, 0.0]); n = 1.0
        x /= n
        y = np.cross(z, x)
        return Rotation.from_matrix(np.stack([x, y, z], axis=1)).as_quat()

    # positions are relative to panda_link0 (the robot base)
    view_dir = np.array([0.75, -0.95, 0.62]); view_dir /= np.linalg.norm(view_dir)
    cam_rel = (ws + view_dir * cam_dist) - base
    tgt_rel = ws - base
    q = look_at_quat(cam_rel, tgt_rel)
    print(f"  framing centre (world) {np.round(ws,3)} dist={cam_dist:.2f} m", flush=True)
    print(f"  cam_rel {np.round(cam_rel,3)} quat {np.round(q,3)}", flush=True)
    _cam_path = ("/workspace_cam" if a.cam_scene_frame
                 else f"/controllable__{_rcls}__robot0/{a.cam_parent_link}/workspace_cam")
    _cam_frame = "scene" if a.cam_scene_frame else "parent"
    _cam_pos = (base + cam_rel) if a.cam_scene_frame else cam_rel
    _cam_tgt = ws if a.cam_scene_frame else tgt_rel
    _q = look_at_quat(_cam_pos, _cam_tgt)
    print(f"  camera prim {_cam_path} frame={_cam_frame}", flush=True)
    ext = [{
        "sensor_type": "VisionSensor",
        "name": "workspace_cam",
        "relative_prim_path": _cam_path,
        "modalities": ["rgb"],
        "sensor_kwargs": {"image_height": a.cam_height, "image_width": a.cam_width,
                          "clipping_range": [0.05, 1000000.0]},
        "position": [float(v) for v in _cam_pos],
        "orientation": _q,
        "pose_frame": _cam_frame,
    }]
    scene_file = load_json_with_absolute_usd_paths(str(scene_json_path))
    if a.poses_only:
        ext = []
        try:
            ri = scene_file["objects_info"]["init_info"]["robot0"]["args"]
            ri["obs_modalities"] = ["proprio"]
            print("  poses-only: external sensors off, robot obs -> proprio", flush=True)
        except Exception as e:
            print(f"  (could not strip robot obs modalities: {e})", flush=True)

    og_cfg = {
        "env": {"external_sensors": ext, "action_frequency": 15,
                "rendering_frequency": 15, "physics_frequency": 120},
        "scene": {"type": "Scene", "scene_file": scene_file,
                  "use_floor_plane": True, "floor_plane_visible": True,
                  "use_skybox": False, "include_robots": True},
    }
    env = og.Environment(configs=og_cfg)
    obs, _ = env.reset()
    robot = env.robots[0]
    adim = robot.action_space.shape[0]
    # vector from fingertips to flange; add it to a desired FINGERTIP target to
    # get the EEF target the IK controller should be driven to.
    print(f"robot={robot.name} action_dim={adim}", flush=True)
    print(f"  eef(flange)={eef_pos(robot)}  finger_links={finger_tip_pos(robot)}", flush=True)
    print(f"  AG midpoint (controlled)={ag_point_pos(robot)}", flush=True)
    print(f"  (flange->fingertip vector is pose-dependent; controlling fingertips directly)", flush=True)

    # Real finger travel, read from the articulation rather than assumed: the
    # default Franka hand is panda_finger_joint1/2, 0-0.04 m each, so the jaw
    # opens to ~8 cm. Anything wider cannot be grasped between the fingers no
    # matter how good the approach is.
    if a.max_grasp_width is not None:
        gripper_max = a.max_grasp_width
        print(f"  gripper max opening = {gripper_max*100:.1f} cm (given)", flush=True)
    else:
        try:
            jn = robot.finger_joint_names[robot.default_arm]
            gripper_max = float(sum(float(robot.joints[j].upper_limit) for j in jn))
        except Exception as e:
            gripper_max = 0.08
            print(f"  (could not read finger limits: {e}; assuming {gripper_max} m)", flush=True)
        print(f"  gripper max opening = {gripper_max*100:.1f} cm "
              f"(from prismatic finger joint limits)", flush=True)

    objs_by_cat = {}
    for o in env.scene.objects:
        c = getattr(o, "category", None)
        if c:
            objs_by_cat[c] = o

    FIT = 0.90 * gripper_max
    print(f"  {'category':22s} {'w_cm':>6s} {'reach_m':>8s}  verdict", flush=True)
    fits = []
    for c in sorted(reachable, key=lambda c: reach_of[c]):
        o = objs_by_cat.get(c)
        if o is None:
            print(f"  {c:22s} {'?':>6s} {reach_of[c]:8.3f}  not in sim", flush=True)
            continue
        ext = o.aabb_extent
        ext = np.asarray(ext.cpu() if isinstance(ext, th.Tensor) else ext, dtype=float)
        wmin = float(min(ext[0], ext[1]))   # narrowest horizontal span = best jaw axis
        v = "graspable" if wmin < FIT else f"TOO WIDE (>{FIT*100:.0f} cm jaw)"
        print(f"  {c:22s} {wmin*100:6.1f} {reach_of[c]:8.3f}  {v}", flush=True)
        if wmin < FIT:
            fits.append(c)

    if want:
        TASKS = [c for c in want if c in objs_by_cat]
        missing = [c for c in want if c not in objs_by_cat]
        if missing:
            print(f"  requested but not in scene: {missing}", flush=True)
    else:
        TASKS = fits[: a.max_tasks]
    print(f"  TASKS: {TASKS}", flush=True)
    assert TASKS, "no graspable object within reach"

    # Link-integrity watch. A joint fixes each child link's origin at a constant
    # offset from its parent, so any drift means the articulation came apart.
    # Added because the SO-101's jaw lever READS as detached in renders -- it is
    # a 9 cm arm on a 3.7 cm offset, so at a wide angle it swings clear of the
    # gripper body and looks like a separate flying piece.
    CHAIN = [("base", "shoulder"), ("shoulder", "upper_arm"), ("upper_arm", "lower_arm"),
             ("lower_arm", "wrist"), ("wrist", "gripper"), ("gripper", "jaw")]
    CHAIN = [(a_, b_) for a_, b_ in CHAIN if a_ in robot.links and b_ in robot.links]

    def link_offsets():
        out = {}
        for a_, b_ in CHAIN:
            pa = _lpos(robot, a_); pb = _lpos(robot, b_)
            out[f"{a_}->{b_}"] = float(np.linalg.norm(pb - pa))
        return out

    _ref = link_offsets()
    _drift = {k: 0.0 for k in _ref}
    print("  link rest offsets (m): " + ", ".join(f"{k}={v:.4f}" for k, v in _ref.items()),
          flush=True)

    frames = []
    POSE_LINKS = [l for l in ("base", "shoulder", "upper_arm", "lower_arm",
                              "wrist", "gripper", "jaw", "panda_link0")
                  if l in robot.links]
    POSE_OBJS = sorted(objs_by_cat)
    pose_log = []

    def snap():
        if not a.poses_only:
            f = collect_frame(obs, robot)
            if f is not None: frames.append(f)
        for k, v in link_offsets().items():
            _drift[k] = max(_drift[k], abs(v - _ref[k]))
        # Record world poses so the episode can be re-rendered OFFLINE from the
        # USD meshes. Isaac's own capture of this scene corrupts the gripper
        # (the jaw renders detached) even though every in-sim measurement -- link
        # poses, visual-mesh offsets, joint drift -- comes back at 0.00 mm.
        rec = {}
        for l in POSE_LINKS:
            pq = robot.links[l].get_position_orientation()
            rec[f"link::{l}"] = np.concatenate([_np(pq[0]), _np(pq[1])])
        for c in POSE_OBJS:
            pq = objs_by_cat[c].get_position_orientation()
            rec[f"obj::{c}"] = np.concatenate([_np(pq[0]), _np(pq[1])])
        pose_log.append(rec)
    snap()

    def goto(target_xyz, grip, label, use_fingers=True):
        """Closed-loop on the ASSISTED-GRASP MIDPOINT (pose-independent). IK
        takes a delta EEF pose; for small translations the hand follows the
        flange 1:1, so feeding the error straight in converges with no offset
        term. See ag_point_pos() for why this frame and not the finger links."""
        nonlocal obs
        goal = np.asarray(target_xyz, dtype=float)
        best, stall = np.inf, 0
        for i in range(a.max_steps_per_waypoint):
            cur = ag_point_pos(robot) if use_fingers else eef_pos(robot)
            err = goal - cur
            dist = float(np.linalg.norm(err))
            if dist < a.pos_tol:
                break
            # Stall guard: when the arm is jammed (typically the pads pressing
            # into the ground plane) it will otherwise burn every step shoving
            # the target around, which is exactly how the first run failed.
            if dist < best - 1e-3:
                best, stall = dist, 0
            else:
                stall += 1
                if stall >= 12:
                    print(f"  {label:26s} STALLED at dist={dist:.3f} "
                          f"after {i+1} steps", flush=True)
                    return
            step = np.clip(err, -a.max_delta, a.max_delta)
            act = np.zeros(adim, dtype=np.float32)
            act[0:3] = step              # delta position
            # Orientation: sending zeros froze the gripper in its HOME pose,
            # which points UP (fingertips sat above the flange) -- so the fingers
            # never faced the table and no grasp was possible. Drive the approach
            # axis to point down (+Z_gripper -> -Z_world) with a clipped
            # axis-angle delta, which is exactly what this IK mode consumes.
            d_rot = approach_delta(robot)
            n = np.linalg.norm(d_rot)
            if n > a.max_drot:
                d_rot = d_rot / n * a.max_drot
            act[3:6] = d_rot
            act[-1] = grip               # gripper: 1 open, 0 closed
            obs, _, term, trunc, _ = env.step(th.tensor(act))
            snap()
            if term or trunc:
                obs, _ = env.reset()
        print(f"  {label:26s} dist={dist:.3f} after {i+1} steps", flush=True)

    def hold(grip, n, label):
        nonlocal obs
        for _ in range(n):
            act = np.zeros(adim, dtype=np.float32); act[-1] = grip
            obs, _, term, trunc, _ = env.step(th.tensor(act)); snap()
        print(f"  {label:26s} in_hand={grasped_obj(robot)}", flush=True)

    # Franka gripper approaches along its local +Z; rotating pi about X maps that
    # to world -Z, i.e. pointing at the table.
    # Map the EEF-local approach axis onto world -Z. For +Z (Franka) that is a
    # pi rotation about X; for -Z (SO-101) the axis already points down in the
    # link's own frame, so no rotation is needed.
    # Orientation control: align ONLY the approach axis with world -Z, leaving
    # yaw free. Commanding a full target rotation also pins the gripper's yaw,
    # which a 5-DOF arm cannot honour together with a position -- once the base
    # yaw was corrected the required yaw became unreachable and every waypoint
    # stalled ~0.3 m short with the arm fighting its own IK. The minimal
    # rotation taking the current approach direction to vertical is always
    # achievable and costs no DOF.
    A_LOCAL = np.array([0.0, 0.0, 1.0]) if a.down_axis == "+z" else np.array([0.0, 0.0, -1.0])
    A_WORLD = np.array([0.0, 0.0, -1.0])
    GRIP_OPEN = a.grip_open
    print(f"  grip open={GRIP_OPEN} release={a.grip_release}", flush=True)
    print(f"  approach axis {a.down_axis} (local {A_LOCAL}) -> align to world -Z, yaw free", flush=True)

    def approach_delta(rb):
        """Shortest rotation bringing the EEF approach axis to world -Z."""
        cur = Rotation.from_quat(eef_quat(rb)).apply(A_LOCAL)
        v = np.cross(cur, A_WORLD)
        sn = float(np.linalg.norm(v)); cs = float(np.dot(cur, A_WORLD))
        if sn < 1e-9:
            return np.zeros(3) if cs > 0 else np.array([np.pi, 0.0, 0.0])
        return v / sn * float(np.arctan2(sn, cs))
    APPROACH, LIFT = a.approach, a.lift
    print(f"  approach={APPROACH} m  lift={LIFT} m  "
          f"grip open={a.grip_open} release={a.grip_release}", flush=True)
    try:
        results = []
        for ti, obj_cat in enumerate(TASKS):
            op = np.asarray(by_cat[obj_cat][1], dtype=float)
            # Place to a clear spot offset in +y, fanned out per task so later
            # objects are not dropped on top of earlier ones.
            if a.place_azimuth:
                v = op[:2] - base[:2]
                r = float(np.linalg.norm(v))
                th0 = float(np.arctan2(v[1], v[0]))
                th1 = th0 + np.radians(a.place_azimuth) * (1 if ti % 2 == 0 else -1)
                pp = np.array([base[0] + r * np.cos(th1),
                               base[1] + r * np.sin(th1), op[2]])
            else:
                pp = op + np.array([0.0, a.place_dy + 0.10 * ti, 0.0])
                if float(np.linalg.norm(pp - base)) >= REACH:
                    pp = op + np.array([0.0, -(a.place_dy + 0.10 * ti), 0.0])
            pd = float(np.linalg.norm(pp - base))
            print(f"TASK {ti+1}/{len(TASKS)}: {obj_cat} {np.round(op,3)} -> "
                  f"{np.round(pp,3)} (place reach {pd:.3f} m)", flush=True)
            # Grasp height: the AABB mid-height of the object itself, kept a
            # few mm clear of the ground plane. op[2] is the settled root-link
            # origin, which is not the centre of the visual bounds.
            o = objs_by_cat[obj_cat]
            try:
                ac = o.aabb_center
                ac = np.asarray(ac.cpu() if isinstance(ac, th.Tensor) else ac, dtype=float)
                gz = max(float(ac[2]), 0.006)
            except Exception:
                gz = max(float(op[2]), 0.006)
            print(f"  grasp height {gz:.3f} m (AG midpoint target)", flush=True)
            try:
                goto([op[0], op[1], op[2] + APPROACH], GRIP_OPEN, "approach above object")
                goto([op[0], op[1], gz], GRIP_OPEN, "descend (fingers astride object)")
                hold(0.0, 30, "close gripper")
                goto([op[0], op[1], op[2] + LIFT], 0.0, "lift")
                lift_held = grasped_obj(robot)
                goto([pp[0], pp[1], pp[2] + LIFT], 0.0, "traverse to place spot")
                goto([pp[0], pp[1], pp[2] + APPROACH], 0.0, "descend over place spot")
                hold(a.grip_release, 20, "open gripper (release)")
                goto([pp[0], pp[1], pp[2] + LIFT], a.grip_release, "retreat")
                fp = objs_by_cat[obj_cat].get_position_orientation()[0]
                fp = np.asarray(fp.cpu() if isinstance(fp, th.Tensor) else fp, dtype=float)
                moved = float(np.linalg.norm(fp[:2] - op[:2]))
                ok = lift_held is not None and moved > 0.05
                results.append((obj_cat, lift_held, moved, ok))
                print(f"  RESULT {obj_cat}: held_during_lift={lift_held} "
                      f"moved={moved:.3f} m -> {'SUCCESS' if ok else 'FAILED'}", flush=True)
            except Exception as e:
                results.append((obj_cat, None, 0.0, False))
                print(f"  RESULT {obj_cat}: ABORTED ({type(e).__name__}: {e})", flush=True)

        print("", flush=True)
        print("=== LINK INTEGRITY (max |offset - rest|) ===", flush=True)
        for k, v in _drift.items():
            print(f"  {k:22s} {v*1000:7.2f} mm" + ("   <<< DETACHED" if v > 0.005 else ""),
                  flush=True)
        print("", flush=True)
        print("=== SUMMARY ===", flush=True)
        for c, h, mv, ok in results:
            print(f"  {c:22s} grasped={str(h):10s} displaced={mv:.3f} m  "
                  f"{'SUCCESS' if ok else 'FAILED'}", flush=True)
        print("", flush=True)

        if pose_log:
            keys = sorted(pose_log[0])
            np.savez_compressed(
                out_dir / f"{a.video_name}_poses.npz",
                keys=np.array(keys),
                poses=np.stack([[r[k] for k in keys] for r in pose_log]))
            print(f"  wrote {a.video_name}_poses.npz  "
                  f"({len(pose_log)} frames x {len(keys)} prims)", flush=True)

        if frames:
            vp = out_dir / f"{a.video_name}.mp4"
            imageio.mimwrite(vp, frames, fps=a.video_fps)
            print(f"SAVED {vp}  ({len(frames)} frames)", flush=True)
            imageio.imwrite(out_dir / f"{a.video_name}_last.png", frames[-1])
        else:
            print("SAVED (poses only; no Isaac frames captured)", flush=True)
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
