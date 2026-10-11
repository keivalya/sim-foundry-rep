"""Exact SO-101 forward kinematics + the measured URDF->world transform.

Pure numpy/scipy (+ xml), so the CPU utility env can plan a scene layout without
Isaac. Needed because the blue-cube scene has to put objects where the pi0.5
checkpoint's own joint distribution can actually reach them -- guessing a
standoff is what put the reference scene's cup outside the workspace.

THE TRANSFORM IS MEASURED, NOT ASSUMED
--------------------------------------
robot/so101_src/so101_new_calib.urdf and NVIDIA's USD do NOT share a base frame.
Fitting the logged link poses of an actual 450-step rollout
(results/so101/vla/framed/{poses,episode}.npz, 450 frames x 5 links) against FK
from the logged joint angles gives, with max deviation 0.00000 m:

    p_world = root_pos + Rz(-90 deg) @ p_urdf + (0.020790, 0.015760, 0.032480)

i.e. the articulation root's world pose is the USD `base` link, the URDF base
frame is rotated -90 deg about Z inside it, and the URDF origin sits 32.48 mm
above the root. With the root dropped to z = -0.0301 (so the base plate rests on
the table, see tools/swap_robot_so101.py) the URDF origin lands 2.4 mm above the
table, so URDF z is table height to within 2.4 mm.

CONSEQUENCE, AND IT IS NOT COSMETIC: with an identity root orientation the arm's
forward (+x in the URDF) points along world **-Y**, not +X. The scripted
pick-and-place never noticed because InverseKinematicsController just yawed
shoulder_pan ~70 deg to reach the +X objects -- but pi0.5's shoulder_pan only
ever spans [-33.98, +28.09] normalized = [-37, +31] deg, so for the VLA the
objects have to be in FRONT of the arm, and front is -Y.
"""
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R

URDF = "/scratch/pandya.kei/simfoundry-repro/robot/so101_src/so101_new_calib.urdf"
CHAIN = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
# Fitted on 450 logged frames; max |residual| 0.0 m. See the docstring.
URDF_TO_ROOT_R = R.from_euler("z", -90, degrees=True).as_matrix()
URDF_TO_ROOT_T = np.array([0.020790, 0.015760, 0.032480])
ROOT_Z = -0.030081447776514957          # base plate resting on a table at z=0


def _load(urdf=URDF):
    t = ET.parse(urdf).getroot()
    J = {}
    for j in t.findall("joint"):
        o, ax, lim = j.find("origin"), j.find("axis"), j.find("limit")
        J[j.get("name")] = dict(
            xyz=np.array([float(v) for v in o.get("xyz").split()]) if o is not None else np.zeros(3),
            rpy=np.array([float(v) for v in (o.get("rpy") or "0 0 0").split()]) if o is not None else np.zeros(3),
            axis=np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.zeros(3),
            child=j.find("child").get("link"))
    return J


JOINTS = _load()


def fk_urdf(q5, tip="gripper_frame_joint"):
    """5 arm angles (rad, LeRobot joint order) -> {link: (pos, R)} in the URDF base frame."""
    T = np.eye(4)
    out = {"base_link": (T[:3, 3].copy(), T[:3, :3].copy())}
    for name, ang in zip(CHAIN, np.asarray(q5, float)):
        j = JOINTS[name]
        A = np.eye(4); A[:3, :3] = R.from_euler("xyz", j["rpy"]).as_matrix(); A[:3, 3] = j["xyz"]
        B = np.eye(4); B[:3, :3] = R.from_rotvec(j["axis"] * ang).as_matrix()
        T = T @ A @ B
        out[j["child"]] = (T[:3, 3].copy(), T[:3, :3].copy())
    j = JOINTS[tip]
    A = np.eye(4); A[:3, :3] = R.from_euler("xyz", j["rpy"]).as_matrix(); A[:3, 3] = j["xyz"]
    Tg = T @ A
    out["tip"] = (Tg[:3, 3].copy(), Tg[:3, :3].copy())
    return out


def urdf_to_world(p, root_pos=(0.0, 0.0, ROOT_Z)):
    return np.asarray(root_pos, float) + URDF_TO_ROOT_R @ np.asarray(p, float) + URDF_TO_ROOT_T


def world_to_urdf(p, root_pos=(0.0, 0.0, ROOT_Z)):
    return URDF_TO_ROOT_R.T @ (np.asarray(p, float) - np.asarray(root_pos, float) - URDF_TO_ROOT_T)


def fk_world(q5, root_pos=(0.0, 0.0, ROOT_Z), tip="gripper_frame_joint"):
    return urdf_to_world(fk_urdf(q5, tip)["tip"][0], root_pos)


def ik_tip(target_urdf, q0, lo, hi, restarts=12, seed=0):
    """Position IK on the 5 arm joints, BOUNDED BY THE POLICY'S OWN JOINT RANGE.

    5 DOF against a 3-D position target, so position is solvable and the two
    spare DOF are free; they are regularised (weakly) toward @q0 -- the
    checkpoint's median state -- so a solution that exists only in a far corner
    of the null space is not reported as "reachable". Bounds are the caller's,
    and the intended caller passes the checkpoint's own [min, max] per joint:
    a target the policy cannot reach WITHIN ITS OWN ACTION DISTRIBUTION is not
    a target it will ever go for, however reachable the hardware makes it.

    least_squares + random restarts rather than hand-rolled DLS: the planar
    3-link subchain has genuine local minima and a single damped descent stalls
    40 mm short on perfectly reachable points.
    """
    from scipy.optimize import least_squares
    rng = np.random.default_rng(seed)
    tgt = np.asarray(target_urdf, float)
    q0 = np.asarray(q0, float)[:5]
    lo5, hi5 = np.asarray(lo, float)[:5], np.asarray(hi, float)[:5]
    q0 = np.clip(q0, lo5, hi5)

    def res(q):
        return np.concatenate([fk_urdf(q)["tip"][0] - tgt, 1e-3 * (q - q0)])

    best, bq = np.inf, q0.copy()
    for k in range(restarts):
        s0 = q0 if k == 0 else np.clip(lo5 + rng.random(5) * (hi5 - lo5), lo5, hi5)
        try:
            r = least_squares(res, s0, bounds=(lo5, hi5), xtol=1e-12, ftol=1e-12, max_nfev=600)
        except Exception:
            continue
        e = float(np.linalg.norm(fk_urdf(r.x)["tip"][0] - tgt))
        if e < best:
            best, bq = e, r.x.copy()
        if best < 1e-4:
            break
    return bq, best
