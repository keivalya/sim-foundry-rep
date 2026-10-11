"""Build the scene the pi0.5 checkpoint was actually fine-tuned for:
a blue cube and an open-topped gray box on a table, in front of an SO-101.

WHY A NEW SCENE AT ALL
----------------------
`pdhinaka/pi05_so101_blue_cube_gray_box_cleaned_20k` is a task-specific
fine-tune on "put the blue cube in the gray box". The reconstructed
`reference_demo` scene has an orange block, a red boat toy and a white paper
cup -- no blue cube, no gray box -- so the policy has nothing to act on. This
writes a separate scene JSON; `reconstructed_og_scene_so101*.json` is NOT
touched, because every existing result reads it.

GEOMETRY IS PROCEDURAL ON PURPOSE
---------------------------------
A cube and a 5-slab open box are exactly representable as OmniGibson
`PrimitiveObject`s, which need no BEHAVIOR dataset asset, no reconstruction and
no USD authoring. The box is five fixed (kinematic) slabs rather than one
object because OmniGibson has no concave primitive; a convex-hull collider over
a hollow box would be a solid block and the cube could never go in.

PLACEMENT IS SOLVED, NOT GUESSED
--------------------------------
Two constraints, both easy to get silently wrong:

1. **Which way the arm faces.** With an identity root orientation the SO-101's
   forward (+x in the URDF) points along world **-Y** -- measured, see
   tools/so101_kin.py, which fits the transform against 450 logged frames to
   7e-6 m. The scripted pick-and-place never noticed because IK just yawed
   shoulder_pan ~70 deg to reach objects placed along +X. pi0.5 cannot do that:
   its shoulder_pan only ever spans [-33.98, +28.09] normalized = [-37, +31]
   deg. Objects go in FRONT of the arm, and front is -Y.

2. **Whether the policy can reach them.** Placement is checked by IK bounded by
   the CHECKPOINT'S OWN per-joint min/max (not the hardware limits): a pose the
   policy never visits in training is not a pose it will reach for. The solved
   joint vector for each target is printed in LeRobot units so it can be read
   against vla/so101_state_quantiles.json.

The robot starts at the checkpoint's median state (observation.state.q50), which
needs the widened joint limits from tools/widen_so101_limits.py -- q50's
shoulder_lift is -104.17, below the stock asset's -100 stop.

    conda_envs/simfoundry-editor/bin/python -I tools/make_bluecube_scene.py
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, "vla"))
import so101_kin as K                                           # noqa: E402
from so101_units import (CKPT_ACTION_MAX, CKPT_ACTION_MIN,      # noqa: E402
                         LEROBOT_JOINTS, NORM_LIMITS, SO101Units, WIDE_LIMITS)

# Checkpoint state distribution, LeRobot normalized units (vla/so101_state_quantiles.json).
CKPT_STATE_MIN = np.array([-33.758, -104.264, -24.967, 58.418, -34.066, 2.871])
CKPT_STATE_MAX = np.array([27.516, 41.407, 97.758, 98.593, 32.571, 35.644])
CKPT_STATE_Q50 = np.array([-5.180376, -104.1688, 1.0502148, 76.527885, -13.933344, 2.8711483])

# WHERE THE LAYOUT NUMBERS COME FROM
# ----------------------------------
# Not from eyeballing the training videos: from running the dataset's own
# recorded `observation.state` through tools/so101_kin.fk_urdf at the frames
# where the gripper closes (grasp) and opens (release). Over episodes 0,1,12,24
# of pdhinaka/so101_blue_cube_in_gray_box_pi05_cleaned that gives, in the
# robot's own frame (+x forward, +y left, z = table), with a spread of ~5 mm:
#
#     home  (every episode starts AND ends here)  tip (0.163, 0.000, 0.024)
#     grasp (the blue cube)                       tip (0.231, -0.086, 0.008)
#     release (over the gray box)                 tip (0.237, +0.116, 0.050)
#
# so the two objects sit side by side ~0.20 m apart on a line across the robot's
# forward axis, both ~0.24 m out. Episodes 25-49 are the same scene with the mat
# rotated 180 deg and the objects swapped (cube +0.137, box -0.069); the
# checkpoint saw both, and there is essentially NO per-episode randomisation in
# either -- object centroids repeat to ~1-2 px -- so the sim uses the recorded
# positions rather than sampling.
DATASET_HOME = np.array([-1.0, -104.2, 97.4, 76.7, -3.0, 3.0])

# The training prompt, verbatim from meta/tasks.parquet. It is NOT
# "put the blue cube in the gray box".
DATASET_TASK = "Pick up the blue cube and place it in the gray box"

VERSIONS = {"omnigibson": {"version": "3.8.0", "git_hash": None},
            "bddl": {"version": "3.7.0", "git_hash": None},
            "behavior-1k-assets": {"version": None},
            "omnigibson-robot-assets": {"version": None, "git_hash": None}}

ARM_JOINT = {"name": "JointController", "motor_type": "position",
             "command_input_limits": None, "command_output_limits": None,
             "use_delta_commands": False}


def body(name, scale, rgba, category, fixed):
    return {"class_module": "omnigibson.objects.primitive_object",
            "class_name": "PrimitiveObject",
            "args": {"name": name, "primitive_type": "Cube", "category": category,
                     "scale": [float(v) for v in scale], "rgba": [float(v) for v in rgba],
                     "fixed_base": bool(fixed), "visual_only": False}}


# Every placed body is given the robot's own yaw, so its `scale` (which is
# expressed in the body's local frame) means x=forward, y=left, z=up like the
# layout numbers do. Without this the box walls come out rotated 90 deg and the
# "thin" axis is the wrong one -- a square box hides the mistake, so it is done
# explicitly rather than relied on.
WORLD_ORI = (0.0, 0.0, -0.7071067811865476, 0.7071067811865476)   # Rz(-90 deg)


def st(pos, ori=WORLD_ORI):
    """One PrimitiveObject's saved state.

    `size` is NOT optional: PrimitiveObject._load_state reads state["size"] for
    every Cube and raises KeyError('size') on load without it (job 10893629).
    The value must be 1.0, not the edge length: because `scale` is given in the
    init args, _post_load leaves _extents at ones(3), so the size setter's
    scaling_factor is size/_extents[0] = size, and anything but 1.0 would
    rescale the mesh points on top of the scale and silently resize every box.
    radius/height are the -1 sentinels _dump_state writes for a Cube.
    """
    return {"is_asleep": False,
            "root_link": {"pos": [float(v) for v in pos], "ori": [float(v) for v in ori],
                          "lin_vel": [0.0, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.0]},
            "radius": -1, "height": -1, "size": 1.0,
            "non_kin": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{PROJ}/scene_data/bluecube/bluecube_so101_vla.json")
    # --- layout, in the ROBOT's own frame: +x forward, +y to the arm's left, z = table ---
    # Defaults are the TRAINING SET's own layout, recovered by running the
    # recorded joint angles through FK (see DATASET_LAYOUT above) -- not guessed.
    ap.add_argument("--cube-xy", default="0.231,-0.086", help="blue cube centre, robot frame (m)")
    ap.add_argument("--box-xy", default="0.237,0.116", help="gray box centre, robot frame (m)")
    ap.add_argument("--cube-size", type=float, default=0.028, help="edge length (m)")
    ap.add_argument("--box-scale", type=float, default=0.0,
                    help="if >0, box outer footprint = this x cube-size and the "
                         "wall height = 0.345 x outer, the proportions measured in "
                         "the training frames. 0 = use the explicit box-* values.")
    ap.add_argument("--box-inner", type=float, default=0.066, help="inner footprint, square (m)")
    ap.add_argument("--box-height", type=float, default=0.030, help="wall height above the table (m)")
    ap.add_argument("--box-wall", type=float, default=0.012, help="wall thickness (m)")
    ap.add_argument("--box-floor", type=float, default=0.004, help="base slab thickness (m)")
    # The training table is a large navy-slate desk mat on a sunlit cream desk.
    ap.add_argument("--mat", default="0.62,1.00,0.004", help="desk mat size (m)")
    ap.add_argument("--mat-x", type=float, default=0.22, help="mat centre forward of the base (m)")
    ap.add_argument("--desk", default="1.80,1.80,0.03", help="desk slab under the mat (m)")
    # THE SURFACE STACK. OmniGibson's floor plane is at z = 0 and is VISIBLE, with
    # a noisy default texture. The first version put the mat's top face at exactly
    # z = 0 to keep every height unchanged -- which makes the two coplanar, and
    # they Z-FIGHT: in the top_cam it showed as a dithered band, and in the
    # wrist_cam (a grazing, wide-angle view close to the table) the floor plane
    # won most of the frame, so the policy's wrist input was a field of speckled
    # grey instead of the uniform navy mat it was trained on. So nothing is
    # coplanar any more:
    #     floor plane  z = 0        (hidden)
    #     desk slab    top = 0.0015 (the cream desk the training frames show
    #                                beyond the mat's edge)
    #     desk mat     top = --surface-z
    # and every object height is measured from the mat, not from z = 0.
    ap.add_argument("--surface-z", type=float, default=0.005,
                    help="height of the working surface (the mat's top face), m")
    ap.add_argument("--desk-z", type=float, default=0.0015,
                    help="height of the desk slab's top face, m (must be > 0 so it "
                         "hides the floor plane, and < surface-z)")
    ap.add_argument("--cube-rgb", default="0.66,0.84,0.94",
                    help="pastel sky-blue, white-balanced out of the training frames "
                         "-- NOT a saturated primary blue")
    ap.add_argument("--box-rgb", default="0.54,0.54,0.55", help="neutral matte mid-gray")
    ap.add_argument("--mat-rgb", default="0.36,0.47,0.57", help="navy-slate desk mat")
    ap.add_argument("--desk-rgb", default="0.92,0.90,0.78", help="cream desk around the mat")
    ap.add_argument("--no-table", action="store_true", help="omit the mat and the desk")
    ap.add_argument("--grasping-mode", choices=["assisted", "sticky", "physical"],
                    default="assisted",
                    help="OmniGibson grasping mode.\n"
                         "'assisted' (the default everywhere else in this repo) "
                         "CANNOT BE RELEASED BY THIS POLICY. ManipulationRobot."
                         "_handle_assisted_grasping computes\n"
                         "    applying_grasp = control < joint_upper_limit\n"
                         "for a position-controlled finger, so the magic "
                         "constraint is dropped only when the jaw is commanded to "
                         "its FULL mechanical limit. pi0.5's gripper action never "
                         "exceeds 37.3/100 -- the real arm's gripper calibration "
                         "only ever covered a third of the travel, and its own "
                         "RELEASE commands in the training data are ~23/100 -- so "
                         "applying_grasp is permanently True, _ag_freeze_gripper "
                         "stays set and the jaw freezes. Measured in d1_shift: "
                         "the policy picks the cube up, carries it over the box "
                         "and then commands open 5..30 for 10 s while the achieved "
                         "jaw position sits pinned at 24.3 and the cube never "
                         "drops. 'sticky' uses the same test and has the same "
                         "problem. 'physical' has no magic constraint at all: the "
                         "jaw squeezes the cube and friction holds it, which is "
                         "the only mode whose release condition this policy can "
                         "actually express.")
    ap.add_argument("--finger-friction", type=float, default=0.0,
                    help="static/dynamic friction for the jaw pads; only worth "
                         "raising with --grasping-mode physical, where a 1-DOF "
                         "lever has to hold the cube on friction alone. 0 = leave "
                         "the asset's own material.")
    ap.add_argument("--wrist-focal", type=float, default=0.0,
                    help="override the wrist camera's focal length (mm over a "
                         "20.955 mm aperture). Needed because VisionSensor's "
                         "sensor_kwargs overwrite whatever the USD Camera prim "
                         "says, so re-authoring the prim alone does nothing. "
                         "13.5 = the NVIDIA default (75.6 deg), 10.5 = ~90 deg, "
                         "which is closer to the training rig's visibly "
                         "barrel-distorted wrist view. 0 = leave the class default.")
    ap.add_argument("--start-state", default="home",
                    help="'home' (the pose EVERY training episode starts and ends in), "
                         "'q50' (the marginal median, which is NOT a real pose), "
                         "'rest' (NVIDIA's), or 6 comma-separated LeRobot values")
    a = ap.parse_args()

    cube_xy = np.array([float(v) for v in a.cube_xy.split(",")])
    box_xy = np.array([float(v) for v in a.box_xy.split(",")])
    tbl = np.array([float(v) for v in a.mat.split(",")])
    dsk = np.array([float(v) for v in a.desk.split(",")])
    SURF = a.surface_z
    assert 0.0 < a.desk_z < SURF, "the desk must sit above the floor plane and below the mat"
    if a.box_scale > 0:
        outer_w = a.box_scale * a.cube_size
        a.box_inner = outer_w - 2 * a.box_wall
        a.box_height = 0.345 * outer_w
    units = SO101Units(norm_limits=NORM_LIMITS, phys_limits=WIDE_LIMITS)

    # ---- start pose --------------------------------------------------------
    if a.start_state == "home":
        start_n = DATASET_HOME.copy()
    elif a.start_state == "q50":
        start_n = CKPT_STATE_Q50.copy()
    elif a.start_state == "rest":
        start_n = units.rad_to_norm(np.array([-0.2736, -0.6109, -0.0745, 1.5148, -1.6034, -0.1465]))
    else:
        start_n = np.array([float(v) for v in a.start_state.split(",")])
    start_rad = units.norm_to_rad(start_n)
    clipped = units.clip_rad(start_rad)
    assert np.allclose(clipped, start_rad, atol=1e-9), (
        f"start pose {np.round(start_n,2)} is outside the WIDENED joint limits "
        f"{np.round(units.rad_to_norm(units.lo),1)}..{np.round(units.rad_to_norm(units.hi),1)}")
    print(f"start state ({a.start_state}) = {np.round(start_n, 2)} lerobot")
    print(f"            = {np.round(start_rad, 4)} rad = {np.round(np.degrees(start_rad), 1)} deg")
    tip0 = K.fk_urdf(start_rad[:5])["tip"][0]
    print(f"            tip at {np.round(tip0, 4)} m in the robot frame "
          f"(reach {np.linalg.norm(tip0[:2]):.3f} m, height {tip0[2]:.3f} m)")

    # ---- reachability, bounded by the CHECKPOINT's own joint range ---------
    lo_q = units.norm_to_rad(CKPT_STATE_MIN)
    hi_q = units.norm_to_rad(CKPT_STATE_MAX)
    print("\nreachability inside the checkpoint's own state range:")
    ok = True
    for lbl, xy, z in (("blue cube", cube_xy, SURF + a.cube_size / 2.0),
                       ("gray box", box_xy, SURF + a.box_height + 0.01)):
        tgt = np.array([xy[0], xy[1], z])
        q, err = K.ik_tip(tgt, start_rad[:5], lo_q, hi_q)
        n = units.rad_to_norm(np.r_[q, start_n[5]])
        good = err < 0.012
        ok &= good
        print(f"  {lbl:10s} target {np.round(tgt,3)}  IK residual {err*1000:5.1f} mm  "
              f"{'OK' if good else 'UNREACHABLE'}")
        print(f"  {'':10s} joints {np.round(n[:5], 1)} lerobot "
              f"(ckpt state range {np.round(CKPT_STATE_MIN[:5],0)} .. {np.round(CKPT_STATE_MAX[:5],0)})")
    assert ok, "a target is outside the pose set this checkpoint ever visits -- move it"
    d = float(np.linalg.norm(cube_xy - box_xy))
    print(f"  cube<->box centre distance {d:.3f} m  "
          f"(box inner {a.box_inner:.3f} m, cube {a.cube_size:.3f} m)")
    assert d > (a.box_inner / 2 + a.cube_size / 2 + 0.01), "cube starts inside/touching the box"

    # ---- world placement ---------------------------------------------------
    # p_world = root + Rz(-90) @ p_robotframe + URDF_TO_ROOT_T; table top at z=0.
    def W(xy, z):
        """Robot-frame (forward, left) + a WORLD height -> world position.

        z is taken as world height directly rather than run through the URDF
        transform, because every height in the layout is measured from the
        table (world z = 0) while the URDF base origin sits 2.4 mm above it.
        """
        p = K.urdf_to_world(np.array([xy[0], xy[1], 0.0]))
        p[2] = float(z)
        return p

    init, state = {}, {}

    init["robot0"] = {
        "class_module": "omnigibson.robots.so101", "class_name": "SO101",
        "args": {"name": "robot0", "self_collisions": False,
                 "controller_config": {
                     "arm_0": dict(ARM_JOINT),
                     "gripper_0": {"name": "MultiFingerGripperController",
                                   "command_input_limits": [-1.0, 1.0], "mode": "smooth"}},
                 "action_normalize": False,
                 "obs_modalities": ["rgb", "proprio"],
                 "grasping_mode": a.grasping_mode}}
    if a.finger_friction > 0:
        init["robot0"]["args"]["finger_static_friction"] = a.finger_friction
        init["robot0"]["args"]["finger_dynamic_friction"] = a.finger_friction
    if a.wrist_focal > 0:
        init["robot0"]["args"]["sensor_config"] = {
            "VisionSensor": {"sensor_kwargs": {
                "image_width": 640, "image_height": 480,
                "focal_length": float(a.wrist_focal), "horizontal_aperture": 20.955,
                "clipping_range": [0.01, 1000000.0]}}}
    state["robot0"] = {
        "is_asleep": False,
        "root_link": {"pos": [0.0, 0.0, K.ROOT_Z], "ori": [0.0, 0.0, 0.0, 1.0],
                      "lin_vel": [0.0, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.0]},
        "joint_pos": [float(v) for v in start_rad],
        "joint_vel": [0.0] * 6,
        # BOTH keys are required. ControllerBase._load_state does
        # `if state["goal"] is None`, so omitting "goal" raises KeyError('goal')
        # inside scene.initialize() and the whole load dies (job 10893572).
        # With goal=None the gripper's vel_filter is not read either, so it can
        # stay out -- see MultiFingerGripperController._load_state.
        "controllers": {"arm_0": {"goal_is_valid": False, "goal": None},
                        "gripper_0": {"goal_is_valid": False, "goal": None}},
        "non_kin": {}, "ag_obj_constraint_params": {"0": {}}}

    rgb = lambda s: [float(v) for v in s.split(",")] + [1.0]

    if not a.no_table:
        # The mat's top face is flush with z = 0, i.e. with the floor plane the
        # robot and the objects already rest on -- so adding it changes the LOOK
        # of the scene and nothing about its physics or any height in it. The
        # desk slab sits just under the mat so the mat's edge is visible, which
        # is one of the stronger cues in the training frames.
        init["desk"] = body("desk", dsk, rgb(a.desk_rgb), "desk", True)
        state["desk"] = st(W(np.array([a.mat_x, 0.0]), a.desk_z - dsk[2] / 2.0))
        init["desk_mat"] = body("desk_mat", tbl, rgb(a.mat_rgb), "desk_mat", True)
        state["desk_mat"] = st(W(np.array([a.mat_x, 0.0]), SURF - tbl[2] / 2.0))

    init["blue_cube"] = body("blue_cube", [a.cube_size] * 3, rgb(a.cube_rgb), "blue_cube", False)
    # 1 mm of clearance so it is not born interpenetrating the table.
    state["blue_cube"] = st(W(cube_xy, SURF + a.cube_size / 2.0 + 0.001))

    # Open-topped box: floor slab + four walls, all fixed/kinematic so the
    # target cannot be shoved out of frame and "cube inside box" stays a clean
    # geometric test.
    bi, bw, bh, bf = a.box_inner, a.box_wall, a.box_height, a.box_floor
    outer = bi + 2 * bw
    init["gray_box_floor"] = body("gray_box_floor", [outer, outer, bf], rgb(a.box_rgb),
                                  "gray_box_floor", True)
    state["gray_box_floor"] = st(W(box_xy, SURF + bf / 2.0))
    for nm, dx, dy, sx, sy in (("px", (bi + bw) / 2, 0.0, bw, outer),
                               ("nx", -(bi + bw) / 2, 0.0, bw, outer),
                               ("py", 0.0, (bi + bw) / 2, bi, bw),
                               ("ny", 0.0, -(bi + bw) / 2, bi, bw)):
        k = f"gray_box_wall_{nm}"
        init[k] = body(k, [sx, sy, bh], rgb(a.box_rgb), k, True)
        state[k] = st(W(box_xy + np.array([dx, dy]), SURF + bf + bh / 2.0))

    scene = {"versions": VERSIONS, "metadata": {},
             "state": {"registry": {"system_registry": {}, "object_registry": state}},
             "init_info": {"class_module": "omnigibson.scenes.scene_base", "class_name": "Scene",
                           "args": {"scene_file": None, "use_floor_plane": True,
                                    "floor_plane_visible": True, "use_skybox": True}},
             "objects_info": {"init_info": init}}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(scene, f, indent=1)

    print(f"\nworld placement (robot root at {np.round([0,0,K.ROOT_Z],4)}, identity ori; "
          f"arm faces world -Y):")
    for k in state:
        if k == "robot0":
            continue
        print(f"  {k:20s} {np.round(state[k]['root_link']['pos'], 4)}")
    # A layout summary the success metric can be checked against later.
    meta = {"cube_xy_robot": cube_xy.tolist(), "box_xy_robot": box_xy.tolist(),
            "cube_size": a.cube_size, "box_inner": bi, "box_wall": bw, "box_height": bh,
            "box_floor": bf, "box_centre_world": W(box_xy, 0.0).tolist(),
            "cube_start_world": state["blue_cube"]["root_link"]["pos"],
            "start_state_lerobot": start_n.tolist(),
            "start_joint_rad": start_rad.tolist(),
            "robot_root": [0.0, 0.0, K.ROOT_Z], "robot_forward_world": [0.0, -1.0, 0.0],
            "table_top_z": SURF}
    meta["task"] = DATASET_TASK
    meta["cube_rgb"] = [float(v) for v in a.cube_rgb.split(",")]
    mp = os.path.splitext(a.out)[0] + "_layout.json"
    with open(mp, "w") as f:
        json.dump(meta, f, indent=1)
    print(f"\nwrote {a.out}\nwrote {mp}")
    print(f"{len(init)} objects: " + ", ".join(init))


if __name__ == "__main__":
    main()
