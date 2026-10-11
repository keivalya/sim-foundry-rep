"""Load the SO-101 scene and (a) audit every link, (b) render high-res stills.

Why: the pick-place video looked like the robot had missing parts. Offline
rasterisation of the USD's visual meshes shows all 17 complete, winding-
consistent and correctly materialed, and they match NVIDIA's original in mesh
count and vertex totals -- so the conversion did not drop geometry. The two
remaining explanations are (1) a link genuinely missing or mislocated once
OmniGibson builds the articulation, or (2) plain undersampling: in the video the
arm is ~250 px tall at 1280x720, which aliases thin brackets and hole grids into
apparent holes.

This settles it: the audit catches (1), and the stills -- rendered at full
resolution with the camera close -- settle (2).
"""
from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path

import imageio, numpy as np, torch as th
from scipy.spatial.transform import Rotation
from simfoundry import CFG_DIR, REPO_DIR, import_og_dependencies
from simfoundry.utils.scene_utils import load_json_with_absolute_usd_paths
import_og_dependencies()
import omnigibson as og

EXPECTED_LINKS = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw"]
# Per-link visual-mesh extents measured straight from the USD, in metres.
USD_EXTENT = {          # measured from so101.usd's visual meshes, per link
    "base": 0.111, "shoulder": 0.111, "upper_arm": 0.142, "lower_arm": 0.157,
    "wrist": 0.078, "gripper": 0.105, "jaw": 0.092,
}


def _np(v):
    return np.asarray(v.cpu() if isinstance(v, th.Tensor) else v, dtype=float)


def look_at_quat(cam, target, up=(0.0, 0.0, 1.0)):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1440)
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--radius", type=float, default=0.62)
    ap.add_argument("--elev", type=float, default=22.0)
    ap.add_argument("--grip", type=float, default=None,
                    help="hold this gripper command before rendering. The static "
                         "rest-pose render looks correct; the video does not, and "
                         "the only difference is that the policy commands the jaw "
                         "OPEN. This reproduces that condition.")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    full = json.loads(Path(a.scene_json).read_text())
    base = np.asarray(
        full["state"]["registry"]["object_registry"]["robot0"]["root_link"]["pos"], float)
    tgt = base + np.array([0.08, 0.0, 0.14])

    cams = []
    for k in range(a.views):
        az = np.radians(35 + k * (360.0 / a.views))
        el = np.radians(a.elev)
        pos = tgt + np.array([a.radius*np.cos(az)*np.cos(el),
                              a.radius*np.sin(az)*np.cos(el),
                              a.radius*np.sin(el)])
        cams.append({
            "sensor_type": "VisionSensor", "name": f"insp_{k}",
            "relative_prim_path": f"/insp_{k}",
            "modalities": ["rgb"],
            "sensor_kwargs": {"image_height": a.height, "image_width": a.width,
                              "clipping_range": [0.01, 1000000.0]},
            "position": pos.tolist(),
            "orientation": look_at_quat(pos, tgt),
            "pose_frame": "scene",
        })

    os.chdir(CFG_DIR)
    cfg = {
        "env": {"external_sensors": cams, "action_frequency": 20,
                "rendering_frequency": 20, "physics_frequency": 120},
        "scene": {"type": "Scene",
                  "scene_file": load_json_with_absolute_usd_paths(a.scene_json),
                  "use_floor_plane": True, "floor_plane_visible": True,
                  "use_skybox": False, "include_robots": True},
    }

    try:
        env = og.Environment(configs=cfg)
        obs, _ = env.reset()
        adim = env.robots[0].action_space.shape[0]
        for _ in range(60):                      # settle, optionally holding the jaw open
            act = th.zeros(adim)
            if a.grip is not None:
                act[-1] = a.grip
            obs, _, _, _, _ = env.step(act)
        if a.grip is not None:
            rb = env.robots[0]
            jq = float(_np(rb.get_joint_positions())[list(rb.joints).index("Jaw")])
            print(f"  gripper cmd={a.grip} -> Jaw joint = {jq:+.4f} rad "
                  f"({np.degrees(jq):+.1f} deg)", flush=True)

        robot = env.robots[0]
        print(f"\n=== LINK AUDIT  robot={robot.name} ===", flush=True)
        print(f"{'link':12s} {'present':>8s} {'world pos':>26s} {'aabb extent (cm)':>22s} "
              f"{'usd max dim':>11s}", flush=True)
        missing, suspicious = [], []
        for l in EXPECTED_LINKS:
            if l not in robot.links:
                missing.append(l)
                print(f"{l:12s} {'MISSING':>8s}", flush=True)
                continue
            link = robot.links[l]
            pos = _np(link.get_position_orientation()[0])
            try:
                ext = _np(link.aabb_extent)
            except Exception as e:
                ext = np.full(3, np.nan)
                suspicious.append(f"{l}: aabb failed ({e})")
            got, exp = float(np.nanmax(ext)), USD_EXTENT[l]
            flag = ""
            if not np.isfinite(got):
                flag = "  <<< NO COLLISION GEOM"
            elif got < 0.5*exp or got > 2.0*exp:
                flag = f"  <<< {got/exp:.2f}x EXPECTED"
                suspicious.append(f"{l}: extent {got:.3f} vs usd {exp:.3f}")
            print(f"{l:12s} {'yes':>8s} {str(np.round(pos,4)):>26s} "
                  f"{str(np.round(ext*100,1)):>22s} {exp*100:9.1f}{flag}", flush=True)

        # Does OmniGibson actually have VISUAL meshes for each link? A link whose
        # visuals were not registered still simulates and still reports a sane
        # collision AABB -- it is simply invisible, which is exactly how a
        # connected jaw can read as a detached floating lever.
        print(f"\n{'link':12s} {'visual':>7s} {'collision':>10s}", flush=True)
        novis = []
        for l in EXPECTED_LINKS:
            if l not in robot.links:
                continue
            link = robot.links[l]
            nv = len(getattr(link, "visual_meshes", {}) or {})
            nc = len(getattr(link, "collision_meshes", {}) or {})
            if nv == 0:
                novis.append(l)
            print(f"{l:12s} {nv:7d} {nc:10d}" + ("   <<< NO VISUAL MESHES" if nv == 0 else ""),
                  flush=True)
        print(f"  links with no visual meshes: {novis or 'none'}", flush=True)

        # DECISIVE CHECK: where does each VISUAL mesh actually sit in-sim,
        # relative to its own link, versus where the USD says it should sit?
        # Link poses and joint offsets have all measured clean, so if a part
        # renders in the wrong place the displacement has to be here.
        import json as _json
        from pathlib import Path as _P
        exp = _json.loads(_P("/scratch/pandya.kei/simfoundry-repro/robot/so101_usd/visual_offsets.json").read_text())
        print(f"\n=== VISUAL MESH PLACEMENT (sim offset vs USD offset, per link) ===", flush=True)
        print(f"{'mesh':44s} {'sim |off|':>10s} {'usd |off|':>10s} {'err mm':>9s}", flush=True)
        worst = []
        for l in EXPECTED_LINKS:
            if l not in robot.links:
                continue
            link = robot.links[l]
            lp = _np(link.get_position_orientation()[0])
            lq = _np(link.get_position_orientation()[1])
            Rl = Rotation.from_quat(lq)
            for mname, mesh in (getattr(link, "visual_meshes", {}) or {}).items():
                mp = _np(mesh.get_position_orientation()[0])
                off_sim = Rl.inv().apply(mp - lp)          # into the link frame
                key = f"{l}/{mname}"
                e = exp.get(key)
                if e is None:
                    print(f"{key[:44]:44s} {np.linalg.norm(off_sim):10.4f} {'?':>10s} {'-':>9s}",
                          flush=True)
                    continue
                err = float(np.linalg.norm(off_sim - np.asarray(e)))*1000
                worst.append((err, key))
                flag = "   <<< DISPLACED" if err > 2.0 else ""
                print(f"{key[:44]:44s} {np.linalg.norm(off_sim):10.4f} "
                      f"{np.linalg.norm(e):10.4f} {err:9.2f}{flag}", flush=True)
        worst.sort(reverse=True)
        print(f"  worst visual displacement: "
              f"{worst[0][0]:.2f} mm on {worst[0][1]}" if worst else "  no data", flush=True)

        extras = [l for l in robot.links if l not in EXPECTED_LINKS]
        if extras:
            print(f"  extra links not in the expected set: {extras}", flush=True)
        print(f"\n  missing: {missing or 'none'}", flush=True)
        print(f"  suspicious: {suspicious or 'none'}", flush=True)

        # whole-robot AABB vs the sum of the parts: catches a stray/offset link
        allp = np.vstack([_np(robot.links[l].aabb_extent) for l in EXPECTED_LINKS
                          if l in robot.links])
        print(f"  robot aabb extent = {np.round(_np(robot.aabb_extent)*100,1)} cm", flush=True)
        print(f"  largest single link dim = {allp.max()*100:.1f} cm", flush=True)

        for k in range(a.views):
            f = obs.get("external", {}).get(f"insp_{k}", {}).get("rgb")
            if f is None:
                print(f"  view {k}: no rgb", flush=True); continue
            f = _np(f)[:, :, :3]
            f = (f*255).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8)
            imageio.imwrite(out / f"so101_closeup_{k}.png", f)
            print(f"  wrote so101_closeup_{k}.png  {f.shape[1]}x{f.shape[0]}", flush=True)
        json.dump({"missing": missing, "suspicious": suspicious},
                  open(out / "link_audit.json", "w"), indent=2)
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        (out / "FAILED").write_text(traceback.format_exc())
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
