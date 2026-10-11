"""Wrist-camera + live-render reliability probe for the SO-101 scene.

Two questions, one job:

1. Does the gripper/wrist camera work? It is a UsdGeom.Camera authored at
   /so101_new_calib/gripper/gripper_cam by tools/add_wrist_cam.py, which
   OmniGibson's BaseRobot._load_sensors turns into a robot-owned VisionSensor.
   (`--external-wrist` instead declares it as an env `external_sensors` entry
   under the robot's prim path, which needs no asset change -- the fallback.)

2. Can Isaac render RELIABLY, frame after frame, for closed-loop use? Not
   "does one still look right" -- that question has been answered wrong three
   times from single frames. So this records, for EVERY probed frame and EVERY
   camera: rgb, linear depth, instance-id segmentation (which labels each pixel
   with the prim path of the visual mesh that drew it), the camera's world pose
   and intrinsics, and every link's world pose. tools/check_render_fidelity.py
   then re-projects the USD visual meshes through the logged camera and checks,
   per link, that the pixels Isaac actually drew are where that link really is.
   A jaw drawn as a detached lever is a ~10 cm, i.e. ~100 px, disagreement.

Environment matters more than anything in this script -- see jobs/run_wrist_cam.sbatch:
  OMNIGIBSON_HEADLESS=1        gm.HEADLESS defaults to FALSE, so every previous
                               job launched Kit with a GUI viewport on a node
                               with no display, and job 10873560's SIGSEGV
                               happened inside SimulationApp._wait_for_viewport
                               (py-spy dump), i.e. before the scene even loaded.
  OMNIGIBSON_APPDATA_PATH      defaults into the repo on networked /scratch,
                               shared by every concurrent Isaac job; macros.py:131
                               says outright to put it on a local path on SLURM.

Always print the traceback BEFORE og.shutdown(): shutdown calls app.close(),
which terminates the process, so a bare try/finally exits 0 with no output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch as th
from PIL import Image
from scipy.spatial.transform import Rotation

from simfoundry import CFG_DIR, REPO_DIR, import_og_dependencies
from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths

import_og_dependencies()
import omnigibson as og  # noqa: E402

T0 = time.time()
LINKS = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw"]


def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def _np(v):
    return np.asarray(v.cpu() if isinstance(v, th.Tensor) else v, dtype=float)


def _u8(f):
    f = _np(f)
    if f.ndim == 3 and f.shape[-1] == 4:
        f = f[:, :, :3]
    if f.dtype != np.uint8:
        f = (f * 255.0) if f.max(initial=0) <= 1.0 else f
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=125)
    ap.add_argument("--probe-stride", type=int, default=3,
                    help="save depth + instance-id segmentation every N frames "
                         "(rgb is saved every frame)")
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=480)
    ap.add_argument("--rendering-frequency", type=int, default=15)
    ap.add_argument("--action-frequency", type=int, default=15)
    ap.add_argument("--physics-frequency", type=int, default=120)
    ap.add_argument("--modalities", default="rgb,depth_linear",
                    help="camera modalities. NOT seg_instance_id by default: any "
                         "seg_* modality drags in seg_semantic, whose remapper "
                         "calls semantic_class_name_to_id() -> "
                         "get_all_system_categories(), which os.listdir()s "
                         "deps/BEHAVIOR-1K/datasets/behavior-1k-assets/systems. "
                         "That dataset is not installed here, so every get_obs() "
                         "dies with FileNotFoundError (job 10882646). "
                         "depth_linear alone is enough for the per-link check -- "
                         "see tools/check_render_fidelity.py.")
    ap.add_argument("--no-external", action="store_true",
                    help="disable BOTH external cameras, leaving only the wrist "
                         "camera -- isolates whether extra render products are "
                         "what destabilises the renderer")
    ap.add_argument("--external-wrist", action="store_true",
                    help="declare the wrist camera as an env external_sensor at "
                         "the robot's prim path instead of using the Camera prim "
                         "inside the USD (works with the plain so101.usd)")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    modalities = [m for m in a.modalities.split(",") if m]

    log(f"python {sys.version.split()[0]}  out={out}")
    log(f"SO101_USD_PATH={os.environ.get('SO101_USD_PATH', '(class default)')}")
    log(f"OMNIGIBSON_HEADLESS={os.environ.get('OMNIGIBSON_HEADLESS')} "
        f"OMNIGIBSON_APPDATA_PATH={os.environ.get('OMNIGIBSON_APPDATA_PATH')}")

    scene_file = load_json_with_absolute_usd_paths(a.scene_json)
    full = json.loads(Path(a.scene_json).read_text())
    rcls = full["objects_info"]["init_info"]["robot0"]["class_name"].lower()
    base = np.asarray(
        full["state"]["registry"]["object_registry"]["robot0"]["root_link"]["pos"], float)

    rargs = scene_file["objects_info"]["init_info"]["robot0"]["args"]
    rargs["obs_modalities"] = modalities + ["proprio"]
    # Joint-space control: deterministic, reproducible, and it exercises every
    # joint over its real range. The saved config is an IK controller whose
    # Franka-sized smoothing filters are a separate trap (see STATUS.md).
    rargs["controller_config"] = {
        "arm_0": {"name": "JointController", "motor_type": "position",
                  "command_input_limits": None, "command_output_limits": None,
                  "use_delta_commands": False},
        "gripper_0": {"name": "MultiFingerGripperController", "mode": "smooth",
                      "command_input_limits": [-1.0, 1.0],
                      "command_output_limits": None},
    }

    tgt = base + np.array([0.08, 0.0, 0.14])
    ext = []
    if not a.no_external:
        # (1) scene-frame camera and (2) a camera parented INSIDE the robot's
        # articulation. pick_and_place.py's standing hypothesis was that the
        # parented camera is what differs between the good static renders and
        # the bad episode video, so run both at once and compare them directly.
        for name, path, frame in (
            ("ext_scene", "/ext_scene", "scene"),
            ("ext_parented", f"/controllable__{rcls}__robot0/base/ext_parented", "parent"),
        ):
            eye = tgt + np.array([0.38, -0.40, 0.26])
            pos = eye if frame == "scene" else (eye - base)
            look = tgt if frame == "scene" else (tgt - base)
            ext.append({
                "sensor_type": "VisionSensor", "name": name,
                "relative_prim_path": path, "modalities": modalities,
                "sensor_kwargs": {"image_height": a.cam_height,
                                  "image_width": a.cam_width,
                                  "focal_length": 17.0,
                                  "clipping_range": [0.01, 1000000.0]},
                "position": [float(v) for v in pos],
                "orientation": look_at_quat(pos, look),
                "pose_frame": frame,
            })
    if a.external_wrist:
        ext.append({
            "sensor_type": "VisionSensor", "name": "wrist_ext",
            "relative_prim_path": f"/controllable__{rcls}__robot0/gripper/wrist_ext",
            "modalities": modalities,
            "sensor_kwargs": {"image_height": a.cam_height, "image_width": a.cam_width,
                              "focal_length": 13.5, "horizontal_aperture": 20.955,
                              "clipping_range": [0.01, 1000000.0]},
            "position": [-0.005, 0.06, -0.062],
            "orientation": [-0.3826834, 0.0, 0.0, 0.9238795],   # xyzw, Rx(-45 deg)
            "pose_frame": "parent",
        })

    cfg = {
        "env": {"external_sensors": ext,
                "action_frequency": a.action_frequency,
                "rendering_frequency": a.rendering_frequency,
                "physics_frequency": a.physics_frequency},
        "scene": {"type": "Scene", "scene_file": scene_file,
                  "use_floor_plane": True, "floor_plane_visible": True,
                  "use_skybox": False, "include_robots": True},
    }

    os.chdir(CFG_DIR)

    # Kit reads its "--/setting=value" overrides off sys.argv (which is how
    # OmniGibson injects --portable-root, simulator.py:224). Exposing a hook
    # lets a follow-up job try renderer settings without editing this file,
    # e.g. KIT_EXTRA_ARGS='--/rtx/rendermode=PathTracing'.
    extra = os.environ.get("KIT_EXTRA_ARGS", "").strip()
    if extra:
        import shlex
        sys.argv.extend(shlex.split(extra))
        log(f"KIT_EXTRA_ARGS -> {shlex.split(extra)}")

    env = None
    try:
        log("constructing og.Environment ...")
        env = og.Environment(configs=cfg)
        log("og.Environment built; reset ...")
        try:
            env.reset()
        except Exception:
            import traceback
            traceback.print_exc()
            log("reset failed -- dropping segmentation modalities and retrying")
            modalities = [m for m in modalities if not m.startswith("seg_")]
            for s in (list(env.robots[0].sensors.values())
                      + list((env.external_sensors or {}).values())):
                for m in ("seg_instance_id", "seg_instance", "seg_semantic"):
                    try:
                        s.remove_modality(m)
                    except Exception:
                        pass
            env.reset()
        log("reset done")

        robot = env.robots[0]
        jnames = list(robot.joints)
        lo, hi = _np(robot.joint_lower_limits), _np(robot.joint_upper_limits)
        log(f"robot={robot.name} joints={jnames}")
        log(f"  joint lower={np.round(lo, 3)}")
        log(f"  joint upper={np.round(hi, 3)}")

        cams = dict(robot.sensors)
        # env.external_sensors is None (not {}) when the config list is empty.
        cams.update(env.external_sensors or {})
        log(f"cameras found: {list(cams)}")
        wrist = [n for n in cams if "gripper" in n or "wrist" in n]
        if not wrist:
            log("!!! NO WRIST CAMERA was registered. Robot sensors: "
                f"{list(robot.sensors)}")
        meta = {"frames": [], "links": LINKS, "cameras": {}, "args": vars(a)}
        for n, s in cams.items():
            meta["cameras"][n] = {
                "prim_path": str(s.prim_path),
                "width": int(s.image_width), "height": int(s.image_height),
                "focal_length_mm": float(s.focal_length),
                "horizontal_aperture_mm": float(s.horizontal_aperture),
                "clipping_range": [float(v) for v in _np(s.clipping_range)],
                "modalities": sorted(list(s.modalities)),
            }
            w, h, f, ha = (s.image_width, s.image_height, s.focal_length, s.horizontal_aperture)
            va = ha * h / w
            meta["cameras"][n]["K"] = [w * f / ha, h * f / va, w / 2.0, h / 2.0]
            meta["cameras"][n]["fov_deg"] = [
                2 * np.degrees(np.arctan(ha / (2 * f))), 2 * np.degrees(np.arctan(va / (2 * f)))]
            log(f"  {n}: {meta['cameras'][n]}")
            (out / n).mkdir(exist_ok=True)

        # ---- motion plan: sweep every arm joint, and cycle the jaw ----------
        home = _np(robot.get_joint_positions())
        arm_idx = [jnames.index(j) for j in
                   ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]]
        way = [home[arm_idx].copy()]
        for k, ji in enumerate(arm_idx):
            for frac in (-0.30, 0.30, 0.0):
                w = home[arm_idx].copy()
                span = (hi[ji] - lo[ji])
                w[k] = float(np.clip(home[ji] + frac * span, lo[ji] + 0.02, hi[ji] - 0.02))
                way.append(w)
        per = max(2, a.steps // max(1, len(way)))
        log(f"{len(way)} waypoints x {per} steps = {len(way) * per} steps")

        adim = robot.action_space.shape[0]
        poses, cam_poses, jpos, probed = [], {n: [] for n in cams}, [], []
        step_dt = []
        nframe = 0
        for wi in range(1, len(way)):
            for k in range(per):
                alpha = (k + 1) / per
                target = way[wi - 1] * (1 - alpha) + way[wi] * alpha
                act = th.zeros(adim)
                act[: len(arm_idx)] = th.tensor(target, dtype=th.float32)
                # jaw: a full open/close cycle every waypoint
                act[-1] = float(np.sin(2 * np.pi * (nframe / max(1, per * 2))))
                t_s = time.time()
                env.step(act)
                step_dt.append(time.time() - t_s)

                poses.append(np.concatenate(
                    [np.concatenate(
                        [_np(robot.links[l].get_position_orientation()[0]),
                         _np(robot.links[l].get_position_orientation()[1])]) for l in LINKS]))
                jpos.append(_np(robot.get_joint_positions()))
                for n, s in cams.items():
                    p, q = s.get_position_orientation()
                    cam_poses[n].append(np.concatenate([_np(p), _np(q)]))

                save_probe = (nframe % a.probe_stride) == 0
                for n, s in cams.items():
                    r_obs = s.get_obs()
                    o, info = r_obs if isinstance(r_obs, tuple) else (r_obs, {})
                    if "rgb" in o:
                        Image.fromarray(_u8(o["rgb"])).save(
                            out / n / f"rgb_{nframe:04d}.png")
                    if save_probe:
                        rec = {"frame": nframe, "cam": n}
                        if "depth_linear" in o:
                            d = _np(o["depth_linear"]).astype(np.float32)
                            np.save(out / n / f"depth_{nframe:04d}.npy", d)
                        if "seg_instance_id" in o:
                            np.save(out / n / f"segid_{nframe:04d}.npy",
                                    _np(o["seg_instance_id"]).astype(np.int32))
                            lab = info.get("seg_instance_id", {})
                            rec["seg_labels"] = {str(k): str(v) for k, v in lab.items()}
                        probed.append(rec)
                if save_probe:
                    log(f"  frame {nframe}/{len(way) * per}  "
                        f"step={np.mean(step_dt[-a.probe_stride:]):.3f}s")
                nframe += 1

        np.savez_compressed(
            out / "poses.npz",
            links=np.asarray(LINKS), link_poses=np.asarray(poses),
            joint_names=np.asarray(jnames), joint_pos=np.asarray(jpos),
            cam_names=np.asarray(list(cams)),
            cam_poses=np.stack([np.asarray(cam_poses[n]) for n in cams]),
            step_dt=np.asarray(step_dt))
        meta["frames"] = probed
        meta["n_frames"] = nframe
        meta["step_dt_mean"] = float(np.mean(step_dt))
        meta["step_dt_p95"] = float(np.percentile(step_dt, 95))
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        log(f"DONE  {nframe} frames, {len(probed)} probe records, "
            f"mean step {np.mean(step_dt):.3f}s  p95 {np.percentile(step_dt, 95):.3f}s")
        (out / "OK").write_text(f"{nframe}\n")
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        (out / "FAILED").write_text(traceback.format_exc())
    finally:
        log("og.shutdown()")
        sys.stdout.flush()
        og.shutdown()


if __name__ == "__main__":
    main()
