"""Load the SO-101 alone in OmniGibson and measure what we had to guess.

Specifically:
  * does the class load at all, and in what DOF order
  * the real jaw opening in metres -- the Jaw joint is REVOLUTE, so unlike the
    Franka's prismatic fingers you cannot read the opening off the joint limits;
    it has to be measured by driving the joint and watching the geometry
  * where the assisted-grasp points actually land, so the guessed offsets in
    so101.py can be corrected from measurement instead of USD algebra

Renders a few frames so the pose is visually checkable too.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path

import imageio, numpy as np, torch as th
from simfoundry import CFG_DIR, REPO_DIR, import_og_dependencies
import_og_dependencies()
import omnigibson as og
from omnigibson.robots.manipulation_robot import GraspingPoint
from scipy.spatial.transform import Rotation

OUT = Path("/scratch/pandya.kei/simfoundry-repro/results/so101/probe")
OUT.mkdir(parents=True, exist_ok=True)


def _np(v):
    return np.asarray(v.cpu() if isinstance(v, th.Tensor) else v, dtype=float)


def ag_points(robot):
    """World positions of the AG start/end points."""
    out = {}
    for tag, pts in (("start", robot.assisted_grasp_start_points[robot.default_arm]),
                     ("end", robot.assisted_grasp_end_points[robot.default_arm])):
        ps = []
        for gp in pts:
            link = robot.links[gp.link_name]
            pq = link.get_position_orientation()
            p, q = _np(pq[0]), _np(pq[1])
            ps.append(p + Rotation.from_quat(q).as_matrix() @ _np(gp.position))
        out[tag] = np.array(ps)
    return out


def look_at_quat(cam, target, up=(0.0, 0.0, 1.0)):
    """USD/Isaac cameras look down -Z with +Y up; return a UNIT xyzw quaternion.
    Hardcoding one by hand gave [0.36,0.19,0.42,0.80], norm 1.009, and
    XformPrim.set_position_orientation asserts on non-unit quaternions."""
    f = np.asarray(target, float) - np.asarray(cam, float)
    f /= np.linalg.norm(f)
    z = -f
    x = np.cross(np.asarray(up, float), z)
    n = np.linalg.norm(x)
    x = np.array([1.0, 0.0, 0.0]) if n < 1e-6 else x / n
    y = np.cross(z, x)
    q = Rotation.from_matrix(np.stack([x, y, z], axis=1)).as_quat()
    return [float(v) for v in q / np.linalg.norm(q)]


def main():
    os.chdir(CFG_DIR)
    CAM_POS = [0.42, -0.38, 0.30]
    CAM_TGT = [0.18, 0.0, 0.12]        # roughly the SO-101's workspace centre
    cam = [{
        "sensor_type": "VisionSensor", "name": "probe_cam",
        "relative_prim_path": "/controllable__so101__robot0/base/probe_cam",
        "modalities": ["rgb"],
        "sensor_kwargs": {"image_height": 720, "image_width": 1280,
                          "clipping_range": [0.01, 1000000.0]},
        "position": CAM_POS,
        "orientation": look_at_quat(CAM_POS, CAM_TGT),
        "pose_frame": "parent",
    }]
    cfg = {
        "env": {"external_sensors": cam, "action_frequency": 20,
                "rendering_frequency": 20, "physics_frequency": 120},
        "scene": {"type": "Scene", "use_floor_plane": True,
                  "floor_plane_visible": True, "use_skybox": False},
        "robots": [{"type": "SO101", "name": "robot0",
                    "obs_modalities": ["proprio"],
                    "grasping_mode": "assisted",
                    "action_normalize": False,
                    "position": [0.0, 0.0, 0.0],
                    "controller_config": {
                        "arm_0": {"name": "InverseKinematicsController",
                                  "mode": "pose_delta_ori",
                                  "command_input_limits": None,
                                  "smoothing_filter_size": 5},
                        "gripper_0": {"name": "MultiFingerGripperController",
                                      "mode": "smooth",
                                      "command_input_limits": [0.0, 1.0]}}}],
    }
    # og.shutdown() -> app.close() TERMINATES THE PROCESS (omnigibson/__init__.py
    # :144), so a bare try/finally swallows the real exception and the job exits
    # 0 with no output -- which is exactly how job 10843029 reported success
    # while doing nothing. Print the traceback BEFORE shutting down.
    try:
        _run(og.Environment(configs=cfg))
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        Path("/scratch/pandya.kei/simfoundry-repro/results/so101/probe/FAILED").write_text(
            traceback.format_exc())
    finally:
        og.shutdown()


def _run(env):
    obs, _ = env.reset()
    robot = env.robots[0]
    frames = []

    print(f"\nrobot={robot.name}  action_dim={robot.action_space.shape[0]}", flush=True)
    print(f"DOF order ({len(robot.joints)}):", flush=True)
    for i, (jn, j) in enumerate(robot.joints.items()):
        print(f"  {i}: {jn:14s} lower={float(j.lower_limit):+.4f} "
              f"upper={float(j.upper_limit):+.4f}", flush=True)
    print(f"\narm joints : {robot.arm_joint_names[robot.default_arm]}", flush=True)
    print(f"finger jts : {robot.finger_joint_names[robot.default_arm]}", flush=True)
    print(f"eef link   : {robot.eef_link_names[robot.default_arm]}", flush=True)
    print(f"eef pos    : {np.round(_np(robot.get_eef_position()), 4)}", flush=True)

    jaw = robot.joints["Jaw"]
    lo, hi = float(jaw.lower_limit), float(jaw.upper_limit)
    print(f"\nJaw joint range: [{lo:+.4f}, {hi:+.4f}] rad "
          f"= [{np.degrees(lo):+.1f}, {np.degrees(hi):+.1f}] deg", flush=True)

    adim = robot.action_space.shape[0]
    print(f"\n{'grip_cmd':>9s} {'jaw_q':>9s} {'AG separation (m)':>18s}", flush=True)
    results = []
    for cmd in (0.0, 0.25, 0.5, 0.75, 1.0):
        for _ in range(40):
            act = np.zeros(adim, dtype=np.float32)
            act[-1] = cmd
            obs, _, _, _, _ = env.step(th.tensor(act))
        q = float(_np(robot.get_joint_positions()[list(robot.joints).index("Jaw")]))
        pts = ag_points(robot)
        sep = float(np.linalg.norm(pts["start"][0] - pts["end"][0]))
        results.append((cmd, q, sep))
        print(f"{cmd:9.2f} {q:+9.4f} {sep:18.4f}", flush=True)
        f = obs.get("external", {}).get("probe_cam", {}).get("rgb")
        if f is not None:
            f = _np(f)[:, :, :3]
            frames.append((f * 255).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8))

    seps = [r[2] for r in results]
    print(f"\nMEASURED JAW OPENING: closed={min(seps):.4f} m  open={max(seps):.4f} m"
          f"  travel={max(seps)-min(seps):.4f} m", flush=True)
    pts = ag_points(robot)
    print(f"AG start (world): {np.round(pts['start'], 4)}", flush=True)
    print(f"AG end   (world): {np.round(pts['end'], 4)}", flush=True)

    json.dump({"opening_closed_m": min(seps), "opening_open_m": max(seps),
               "samples": results,
               "dof_order": list(robot.joints)},
              open(OUT / "so101_probe.json", "w"), indent=2)
    for i, f in enumerate(frames):
        imageio.imwrite(OUT / f"probe_{i}.png", f)
    print(f"\nwrote {OUT}/so101_probe.json and {len(frames)} frames", flush=True)


if __name__ == "__main__":
    main()
