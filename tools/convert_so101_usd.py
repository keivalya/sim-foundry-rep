"""Convert the SO-101 follower URDF to USD with Isaac Sim's URDF importer.

Step 1 of swapping the Franka for an SO-101. OmniGibson ships no SO-100/SO-101
(its arms are franka, vx300s, yam, r1/r1pro, a1, stretch, tiago, fetch), so the
asset has to be built from TheRobotStudio/SO-ARM100's
Simulation/SO101/so101_new_calib.urdf (relative `assets/*.stl` mesh paths,
17 visual + 17 collision geoms, 0.632 kg total, sane inertias).
"""
import os, sys, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--urdf", required=True)
ap.add_argument("--dest", required=True)
a = ap.parse_args()
os.makedirs(os.path.dirname(a.dest), exist_ok=True)

from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

import omni.kit.commands
from pxr import Usd, UsdPhysics, UsdGeom

status, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
# Keep fixed joints: gripper_frame_link is a massless frame joined by a fixed
# joint, and it is the natural EEF frame for the OmniGibson robot class.
cfg.merge_fixed_joints = False
cfg.fix_base = True                 # tabletop arm, bolted down
cfg.make_default_prim = True
cfg.create_physics_scene = False    # OmniGibson owns the physics scene
cfg.import_inertia_tensor = True    # the URDF's inertias are real, not guessed
cfg.distance_scale = 1.0            # URDF is already in metres
cfg.self_collision = False
cfg.convex_decomp = False           # convex hull per mesh is enough for AG
cfg.set_collision_from_visuals = False   # real collision meshes exist
for attr, val in (("default_drive_type", 1),            # 1 = position
                  ("default_position_drive_damping", 1e3),
                  ("default_drive_strength", 1e7),
                  ("parse_mimic", True)):
    try:
        setattr(cfg, attr, val)
    except Exception as e:
        print(f"  (ImportConfig.{attr} not settable: {e})", flush=True)

print(f">>> importing {a.urdf}", flush=True)
res = omni.kit.commands.execute("URDFParseAndImportFile",
                                urdf_path=a.urdf, import_config=cfg,
                                dest_path=a.dest)
print(f">>> import result: {res}", flush=True)

# Report what actually landed, so failures are visible rather than silent.
stage = Usd.Stage.Open(a.dest)
if stage is None:
    print("FATAL: no stage at dest", file=sys.stderr); sim_app.close(); sys.exit(1)
dp = stage.GetDefaultPrim()
print(f">>> default prim: {dp.GetPath() if dp else None}")
joints, links, arts = [], [], []
for p in stage.Traverse():
    t = p.GetTypeName()
    if "Joint" in str(t) and "Fixed" not in str(t):
        joints.append((p.GetName(), str(t)))
    if p.HasAPI(UsdPhysics.RigidBodyAPI):
        links.append(p.GetName())
    if p.HasAPI(UsdPhysics.ArticulationRootAPI):
        arts.append(str(p.GetPath()))
print(f">>> articulation roots: {arts}")
print(f">>> {len(links)} rigid bodies: {links}")
print(f">>> {len(joints)} movable joints:")
for n, t in joints:
    print(f"      {n:18s} {t}")
sim_app.close()
