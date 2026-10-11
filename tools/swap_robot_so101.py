"""Rewrite a SimFoundry reconstructed scene to use the SO-101 instead of the Franka.

Stage 14 always writes a FrankaPanda. Swapping it means three things, not one:
  * objects_info.init_info.robot0 -> class_name SO101, with SO-101-appropriate
    controller args (the Franka's end_effector/expected_file_hash are meaningless
    here and would be passed straight into the constructor)
  * state.registry.object_registry.robot0 -> 6 joint entries, not the Franka's 9,
    and the saved arm_0 IK goal invalidated (it was captured against the Franka)
  * the base moved: the SO-101 reaches ~0.32 m, not the Franka's 0.855 m, so the
    Franka standoff leaves every object far outside the workspace
"""
import json, argparse, numpy as np

SO101_USD = "/scratch/pandya.kei/simfoundry-repro/robot/so101_usd/so101.usd"


def base_sit_offset_and_yaw(usd=SO101_USD):
    """Measure, from the asset itself, two things the placement has to correct.

    1. How far the base link's ORIGIN sits below the robot's lowest geometry.
       It is +0.0301 m in this asset, and that origin is the articulation root
       whose position the scene sets -- so a root at z=0 leaves the base plate
       hovering 3 cm above the table, which is exactly what the renders showed.
    2. The yaw baked into the asset's root xform. /so101_new_calib carries
       [[0,-1,0],[1,0,0],[0,0,1]], a +90 deg rotation about Z, so the arm's
       forward +X points along world +Y. Placing the base along -X of the
       objects therefore put them off to the arm's SIDE, not in front of it.

    Measured rather than hardcoded so a different SO-101 export cannot silently
    reintroduce either error.
    """
    from pxr import Usd, UsdGeom
    PRED = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    st = Usd.Stage.Open(usd, Usd.Stage.LoadAll)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    lp = st.GetPrimAtPath("/so101_new_calib/base")
    Li = cache.GetLocalToWorldTransform(lp).GetInverse()
    pts = []
    for pr in Usd.PrimRange(lp, PRED):
        if pr.GetTypeName() != "Mesh" or "visuals" not in str(pr.GetPath()):
            continue
        v = np.array(UsdGeom.Mesh(pr).GetPointsAttr().Get() or [])
        if len(v):
            M = np.array(cache.GetLocalToWorldTransform(pr) * Li).T
            pts.append((np.c_[v, np.ones(len(v))] @ M.T)[:, :3])
    z_lift = float(np.vstack(pts)[:, 2].min())
    root = st.GetPrimAtPath("/so101_new_calib")
    R = np.array(cache.GetLocalToWorldTransform(root)).T[:3, :3]
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return z_lift, yaw

ap = argparse.ArgumentParser()
ap.add_argument("--scene-dir", required=True)
ap.add_argument("--out-suffix", default="_so101")
ap.add_argument("--standoff", type=float, default=0.20,
                help="metres from the object centroid to the base, along -x")
ap.add_argument("--reach", type=float, default=0.32)
ap.add_argument("--min-reach", type=float, default=0.08)
ap.add_argument("--controller", choices=["ik", "joint"], default="ik",
                help="arm_0 controller. 'ik' (default) is what policies/"
                     "pick_and_place.py drives -- it feeds EEF pose DELTAS. "
                     "'joint' is a JointController in position mode with "
                     "use_delta_commands=False, i.e. the command IS the vector "
                     "of absolute joint angles in radians. pi0.5 emits absolute "
                     "joint targets, so the VLA path needs 'joint'; running it "
                     "through the IK controller would read those targets as "
                     "Cartesian deltas and fling the arm.")
a = ap.parse_args()

sd = a.scene_dir.rstrip("/")
src = f"{sd}/s14_og/reconstructed_og_scene.json"
dst = f"{sd}/s14_og/reconstructed_og_scene{a.out_suffix}.json"
sp_src, sp_dst = f"{sd}/s14_og/settled_poses.json", f"{sd}/s14_og/settled_poses{a.out_suffix}.json"

s, sp = json.load(open(src)), json.load(open(sp_src))
cat = {k: v.get("args", {}).get("category") for k, v in s["objects_info"]["init_info"].items()}
objs = {k: np.asarray(v[0], float) for k, v in sp.items() if k != "robot" and cat.get(k)}
assert objs, "no categorised objects"

z_lift, baked_yaw = base_sit_offset_and_yaw()
print(f"asset: base origin sits {z_lift*100:.2f} cm below the lowest geometry")
print(f"asset: baked root yaw = {np.degrees(baked_yaw):+.1f} deg")

# Do NOT try to cancel the baked yaw. OmniGibson sets the articulation root's
# world pose, which REPLACES /so101_new_calib's authored xform rather than
# composing with it -- so the asset's +90 deg is already gone and an identity
# ori leaves the arm facing +X, which is what the -x standoff below assumes.
# Applying -90 deg "to compensate" (job 10870019/10870299) rotated the arm off
# its workspace instead: every waypoint stalled ~0.3 m short and nothing moved.
# baked_yaw is measured above for the record only.
new_ori = [0.0, 0.0, 0.0, 1.0]

centroid = np.mean(list(objs.values()), axis=0)
# -z_lift drops the root so the base plate rests ON the table instead of 3 cm
# above it.
new_base = np.array([centroid[0] - a.standoff, centroid[1], -z_lift])

ri = s["objects_info"]["init_info"]["robot0"]
# init_info records BOTH the module and the class name, and
# create_object_from_init_info does getattr(import_module(class_module),
# class_name). Changing only the name gave
# "module 'omnigibson.robots.franka' has no attribute 'SO101'".
print(f"robot class: {ri['class_module']}.{ri['class_name']} -> omnigibson.robots.so101.SO101")
print(f"arm_0 controller: {a.controller}")
ri["class_module"] = "omnigibson.robots.so101"
ri["class_name"] = "SO101"
# 5-DOF arm: a full 6-DOF pose target is rank-deficient, so IK can only
# least-squares the orientation. Fine for the top-down picks pick_and_place.py
# scripts, useless for a policy that emits joint angles.
ARM_IK = {"name": "InverseKinematicsController",
          "command_input_limits": None,
          "mode": "pose_delta_ori",
          "smoothing_filter_size": 5}
# Absolute joint positions in radians, straight through: no input/output
# scaling, no delta accumulation. This is the controller the pi0.5 bridge needs
# (see policies/pi05_closed_loop.py and vla/so101_units.py). Note there is NO
# smoothing filter here -- a filter would be sized to the DOF count and is the
# same trap the Franka's saved filters were, and the policy chunk is already
# smooth at 50 steps.
ARM_JOINT = {"name": "JointController", "motor_type": "position",
             "command_input_limits": None, "command_output_limits": None,
             "use_delta_commands": False}
# Gripper stays MultiFingerGripperController in "smooth" mode for BOTH paths.
# Its command_input_limits differ: pick_and_place.py drives it on [0,1], while
# the VLA path uses the controller's native [-1,1], where the controller's own
# map u = lo + (cmd+1)/2*(hi-lo) is EXACTLY the inverse of the LeRobot gripper
# normalization -- so a 0..100 gripper action converts with zero error.
ri["args"] = {
    "name": "robot0",
    "self_collisions": False,
    "controller_config": {
        "arm_0": ARM_IK if a.controller == "ik" else ARM_JOINT,
        "gripper_0": {"name": "MultiFingerGripperController",
                      "command_input_limits": [0.0, 1.0] if a.controller == "ik"
                                               else [-1.0, 1.0],
                      "mode": "smooth"},
    },
    "action_normalize": False,
    "obs_modalities": ["rgb", "depth_linear", "proprio"],
    "grasping_mode": "assisted",
}

reg = s["state"]["registry"]["object_registry"]["robot0"]
reg["root_link"]["pos"] = new_base.tolist()
reg["root_link"]["ori"] = new_ori
# NVIDIA's tuned rest pose, in the USD's joint order.
rest = [-0.2736, -0.6109, -0.0745, 1.5148, -1.6034, -0.1465]
reg["joint_pos"] = rest
reg["joint_vel"] = [0.0] * 6
reg["controllers"]["arm_0"]["goal_is_valid"] = False
reg["controllers"]["gripper_0"]["goal_is_valid"] = False
# The saved controller SMOOTHING FILTERS are sized to the Franka's DOF:
# arm control_filter.past_samples is (5, 7) and the gripper's vel_filter is
# (5, 2). Restoring those and then feeding SO-101 commands raised
# "could not broadcast input array from shape (5,) into shape (7,)" on every
# step, so the arm silently never moved and every waypoint stalled.
N_ARM, N_FINGER = 5, 1
_filters = [("gripper_0", "vel_filter", N_FINGER)]
if a.controller == "ik":
    # Only the IK controller carries a control_filter (smoothing_filter_size=5
    # above). The JointController has none, so leaving the Franka-sized one in
    # the saved state would be dead weight at best.
    _filters.insert(0, ("arm_0", "control_filter", N_ARM))
else:
    reg["controllers"]["arm_0"].pop("control_filter", None)
for key, sub, width in _filters:
    f = reg["controllers"][key].get(sub)
    if not f:
        continue
    hist = len(f["past_samples"])
    old = (hist, len(f["past_samples"][0]))
    f["past_samples"] = [[0.0] * width for _ in range(hist)]
    f["current_idx"] = 0
    f["fully_filled"] = False
    print(f"{key}.{sub}.past_samples: {old} -> ({hist}, {width}), reset")
print(f"joint_pos: 9 (Franka) -> {len(rest)} (SO-101)")

print(f"\nbase -> {np.round(new_base,4)} ori {np.round(new_ori,4)}"
      f"  (standoff {a.standoff} m, reach {a.reach} m)")
print(f"{'object':12s} {'category':20s}  dist   verdict")
ok = 0
for k, p in sorted(objs.items(), key=lambda kv: np.linalg.norm(kv[1] - new_base)):
    d = float(np.linalg.norm(p[:2] - new_base[:2]))
    v = ("reachable" if a.min_reach < d < a.reach
         else "TOO CLOSE" if d <= a.min_reach else "OUT OF REACH")
    ok += v == "reachable"
    print(f"{k:12s} {cat[k]:20s}  {d:.3f}  {v}")
assert ok, "repositioning brought nothing into reach"

json.dump(s, open(dst, "w"))
sp["robot"] = [new_base.tolist(), new_ori]
json.dump(sp, open(sp_dst, "w"), indent=2)
print(f"\nwrote {dst}\nwrote {sp_dst}\n{ok}/{len(objs)} reachable")
