"""OmniGibson robot class for the SO-101 (SO-ARM101) low-cost arm.

OmniGibson ships no SO-100/SO-101 (its arms are franka, vx300s, yam, r1/r1pro,
a1, stretch, tiago, fetch), so this wraps NVIDIA's asset from
isaac-sim/Sim-to-Real-SO-101-Workshop, de-instanced by deinstance_so101.py.
Modelled on vx300s.py.

Two things differ from every arm OmniGibson already supports, and both are
handled explicitly below:

1. ONE MOVING JAW. The SO-101 gripper is a single lever (`jaw`) closing against
   a static face that is part of the `gripper` body. ManipulationRobot's default
   assisted-grasp points are derived by walking finger_link_names and treating
   link 0 as the start and link 1 as the end, which needs exactly two fingers.
   We therefore set _ag_start_points/_ag_end_points explicitly (as franka.py
   does for the Panda hand): start on the static face, end on the moving jaw.

2. FIVE ARM DOF, not six or seven. Rotation (base yaw), then Pitch/Elbow/
   Wrist_Pitch as three parallel pitch axes forming a planar arm in the plane
   Rotation selects, then Wrist_Roll. Position is fully controllable; so is the
   gripper's pitch and its roll about its own approach axis. What is NOT
   independently controllable is the third orientation DOF. For TOP-DOWN picks
   this is not a limitation -- with the approach axis vertical, Wrist_Roll is
   the jaw yaw -- but a general 6-DOF pose target is rank-deficient, so
   InverseKinematicsController will only ever least-squares it. Drive this arm
   with top-down poses, or in joint space (which is also what LeRobot's real
   SO-101 interface consumes, so it is the better sim2real match).

Joint names come from the USD, NOT the upstream URDF: the workshop asset renames
shoulder_pan/shoulder_lift/elbow_flex/wrist_flex/wrist_roll/gripper to
Rotation/Pitch/Elbow/Wrist_Pitch/Wrist_Roll/Jaw.
"""

import math
import os
from functools import cached_property

import torch as th

from omnigibson.controllers import ControlType
from omnigibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot
from omnigibson.utils.transform_utils import euler2quat

# Where deinstance_so101.py wrote the editable asset.
SO101_USD_PATH = os.environ.get(
    "SO101_USD_PATH", "/scratch/pandya.kei/simfoundry-repro/robot/so101_usd/so101.usd"
)

# Wrist/gripper camera. NVIDIA's SO-ARM101-USD.usd (the "WITH camera" variant)
# has no UsdGeom.Camera in it at all -- only the camera BODY geometry -- so the
# sensor itself comes from their Isaac Lab config, task_env_cfg.py:49/106-109:
#   prim  "{ENV_REGEX_NS}/Robot/gripper/gripper_cam"
#   pos   (-0.005, 0.06, -0.062)   rot euler([-45,0,0]) deg, convention "opengl"
#   TiledCameraCfg(width=640, height=480, PinholeCameraCfg(focal_length=13.5))
# tools/add_wrist_cam.py authors that prim into so101_wristcam.usd; OmniGibson's
# BaseRobot._load_sensors then wraps any Camera prim that is a DIRECT child of a
# link (robot_base.py:212-259) into a VisionSensor -- but with VisionSensor's own
# 128x128 / f=17 mm defaults unless a sensor_config says otherwise. This is that
# config. Isaac Lab leaves horizontalAperture at the Kit default 20.955 mm, which
# with f=13.5 mm gives a 75.6 x 60.4 deg FOV and K = [412.31, 412.31, 320, 240].
# NOTE the nesting. robot_base._default_sensor_config is
#   {"VisionSensor": {"enabled":..., "noise_type":..., "noise_kwargs":...,
#                     "sensor_kwargs": {"image_height":..., "image_width":...}}}
# and _load_sensors hands the WHOLE per-class dict to create_sensor() as keyword
# arguments. create_sensor's signature is (sensor_type, relative_prim_path, name,
# modalities, enabled, sensor_kwargs, noise_type, noise_kwargs) -- it has no
# **kwargs -- so putting image_width at the top level raises
# "create_sensor() got an unexpected keyword argument 'image_width'"
# during scene load (job 10882579).
SO101_WRIST_CAM_CFG = {
    "VisionSensor": {
        "sensor_kwargs": {
            "image_width": 640,
            "image_height": 480,
            "focal_length": 13.5,
            "horizontal_aperture": 20.955,
            # 1 cm near plane: the jaws sit ~7 cm from the lens and grasped
            # objects come closer still.
            "clipping_range": (0.01, 1000000.0),
        },
    },
}


# ---------------------------------------------------------------------------
# 3. GRIPPER CALIBRATION FOR ASSISTED GRASPING
# ---------------------------------------------------------------------------
# ManipulationRobot._handle_assisted_grasping decides "is the gripper being
# commanded closed?" by comparing the commanded finger position against the
# joint's FULL MECHANICAL LIMIT (manipulation_robot.py, ~line 1620):
#
#     applying_grasp = control < joint_upper_limit        (non-inverted finger)
#
# For every robot OmniGibson ships that is fine, because their grippers run in
# `binary` mode where an open command literally IS the limit. It is wrong for a
# policy that commands the jaw in the REAL arm's own calibrated units. The
# pi0.5 SO-101 checkpoint (pdhinaka/pi05_so101_blue_cube_gray_box_cleaned_20k)
# never commands the gripper above 37.3/100, so `control` never reaches 1.7453
# rad, the test is permanently True, and TWO things go wrong:
#
#   1. the assisted grasp latches on ANY finger contact, even while the jaw is
#      being commanded WIDE OPEN -- in results/.../d1_shift it latched at step
#      ~98 with the jaw at 24.2/100, i.e. open;
#   2. it can never release. `_ag_freeze_gripper` stays set, deploy_control
#      overwrites the jaw target with the frozen position, and the jaw sits
#      pinned at 24.18 for the remaining 350 steps while the policy's gripper
#      target oscillates between 2 and 32 (d1_shift/vs_training.png).
#
# This is the same class of problem as the `shoulder_lift` mismatch handled in
# vla/so101_units.py: the real SO-101's gripper was calibrated over a different
# mechanical range than this USD's jaw, so LeRobot's [0, 100] covers only about
# a third of the simulated jaw's travel. The fix is a CALIBRATED CLOSE
# THRESHOLD -- the commanded jaw angle below which the command means "close".
#
# MEASURED, not guessed, from all 50 episodes / 18 736 frames of
# pdhinaka/so101_blue_cube_in_gray_box_pi05_cleaned. Each episode was segmented
# by binarising its gripper ACTION at the midpoint of its OWN two modes (its
# 10th vs 90th percentile) -- a per-episode band, so the measured levels are not
# a restatement of the threshold being tested. That finds exactly two open
# segments in all 50 of 50 episodes: the approach over the cube, and the release
# over the box. Across the 50:
#
#     phase                         min      median      max
#     carry (jaw shut on cube)    11.82      15.04     17.46
#     approach-open peak          23.41      30.81     37.12
#     release-open peak           23.41      29.91     37.27
#
# So the valid threshold window is (17.46, 23.41] -- ANY value in it keeps
# 100% of the carries AND fires 100% of the releases, while never reading an
# approach-open command as "closing". 20.0 sits in the middle of it. On this
# jaw's nominal range that is
#     -0.1745 + 0.200 * (1.7453 - -0.1745) = +0.2095 rad.
# (A first pass put the carry max at ~22 and so wanted T ~= 23; that was an
# artifact of segmenting with a GLOBAL 22-unit band, i.e. of assuming the answer.
# The per-episode segmentation above does not, and says 17.46.)
#
# Note what this does NOT change: the jaw's command -> angle map
# (SO101Units.gripper_norm_to_cmd) is untouched, so the achieved jaw trajectory,
# and therefore what the wrist camera sees, is bit-identical to the runs that
# produced the validated pick. Only the latch/release DECISION moves.
#
# The jaw's NOMINAL calibration range in radians -- the range LeRobot's [0, 100]
# gripper units are pinned to, i.e. vla/so101_units.py NORM_LIMITS[5]. Repeated
# here rather than imported because this file is copied to
# omnigibson/robots/so101.py, where vla/ is not on sys.path.
SO101_JAW_NORM_RANGE = (-0.1745, 1.7453)

# Gripper command, in LeRobot units, at/above which the command means "open".
# SO101_AG_CLOSE_NORM=100 restores OmniGibson's stock full-limit behaviour,
# which is what every result before 2026-10-07 was produced with.
SO101_AG_CLOSE_NORM = float(os.environ.get("SO101_AG_CLOSE_NORM", "20.0"))

# Print every latch / release to stdout. Cheap (a handful of lines per episode)
# and it is the only direct evidence that the fix fired, so it is on by default.
# Consecutive frames the jaw must be commanded OPEN before the grasp is actually
# dropped. MEASURED from c1007_1702_s2 (the best pick: 1805 mm carried, 185 mm
# lift, cube held over the box): after the grasp, the policy's commanded gripper
# does NOT sit low and then rise once. It oscillates, crossing any usable
# threshold in short bursts during the carry and in one sustained burst at the
# real release:
#
#     threshold   open frames   runs   longest run   other runs
#        22           25          6         10        5,4,3,2,1
#        24           11          3          8        2,1
#        26            5          2          4        1
#
# So a bare threshold cannot work: set it low enough to fire the release and the
# carry-time bursts drop the cube early (campaign 1007_2008, thresholds 18-22,
# every run <=127 mm path vs 1805 mm at threshold 27); set it high enough to
# survive the carry and the release never fires at all (threshold 27: zero
# post-grasp frames above it).
#
# Debouncing separates them cleanly -- at T=24 the real release is 8 frames and
# the longest carry burst is 2. 1 restores the old undebounced behaviour.
SO101_AG_OPEN_FRAMES = int(os.environ.get("SO101_AG_OPEN_FRAMES", "4"))

SO101_AG_VERBOSE = os.environ.get("SO101_AG_VERBOSE", "1") not in ("0", "", "no")


def so101_jaw_norm_to_rad(g):
    """LeRobot gripper units (0 = shut, 100 = nominal full open) -> radians."""
    lo, hi = SO101_JAW_NORM_RANGE
    return lo + (float(g) / 100.0) * (hi - lo)


class SO101(ManipulationRobot):
    """
    The SO-101 (SO-ARM101) 5-DOF arm + 1-DOF jaw from TheRobotStudio, using
    NVIDIA's Isaac asset. https://github.com/TheRobotStudio/SO-ARM100
    """

    def __init__(
        self,
        name,
        relative_prim_path=None,
        scale=None,
        visible=True,
        visual_only=False,
        self_collisions=True,
        link_physics_materials=None,
        load_config=None,
        fixed_base=True,
        abilities=None,
        control_freq=None,
        controller_config=None,
        action_type="continuous",
        action_normalize=True,
        reset_joint_pos=None,
        obs_modalities=("rgb", "proprio"),
        include_sensor_names=None,
        exclude_sensor_names=None,
        proprio_obs="default",
        sensor_config=None,
        grasping_mode="assisted",
        finger_static_friction=None,
        finger_dynamic_friction=None,
        **kwargs,
    ):
        # Static face of the jaw lives on `gripper`; the moving lever is `jaw`.
        # MEASURED, not guessed: the closed-jaw contact patch between the two
        # collision meshes (mean gap 2.3 mm over 400 verts). The pads sit 6.5-8.5
        # cm out from the hinge -- a first guess placed them near the hinge, where
        # the parts are always ~2 mm apart regardless of jaw angle, so the
        # apparent "opening" came out at 3 cm instead of the real ~5-6 cm.
        self._ag_start_points = [
            GraspingPoint(link_name="gripper", position=th.tensor([-0.0079, 0.0026, -0.1010])),
        ]
        self._ag_end_points = [
            GraspingPoint(link_name="jaw", position=th.tensor([-0.0122, -0.0805, 0.0169])),
        ]
        # The wrist camera is a Camera prim inside the USD, so it is discovered
        # automatically -- but VisionSensor would default it to 128x128 at
        # f=17 mm. Supply NVIDIA's intrinsics unless the caller overrides them.
        if sensor_config is None:
            sensor_config = {k: dict(v) for k, v in SO101_WRIST_CAM_CFG.items()}
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            link_physics_materials=link_physics_materials,
            load_config=load_config,
            abilities=abilities,
            control_freq=control_freq,
            controller_config=controller_config,
            action_type=action_type,
            action_normalize=action_normalize,
            reset_joint_pos=reset_joint_pos,
            obs_modalities=obs_modalities,
            include_sensor_names=include_sensor_names,
            exclude_sensor_names=exclude_sensor_names,
            proprio_obs=proprio_obs,
            sensor_config=sensor_config,
            grasping_mode=grasping_mode,
            finger_static_friction=finger_static_friction,
            finger_dynamic_friction=finger_dynamic_friction,
            **kwargs,
        )

    # ---- assisted grasping: calibrated close threshold -------------------
    # See the long note above SO101_JAW_NORM_RANGE for why this exists.

    @cached_property
    def _ag_close_threshold_rad(self):
        """Commanded jaw position (rad) BELOW which the command means 'close'.

        None means "use OmniGibson's stock rule", i.e. only a full-mechanical-
        limit command counts as open.
        """
        if SO101_AG_CLOSE_NORM >= 100.0:
            return None
        return so101_jaw_norm_to_rad(SO101_AG_CLOSE_NORM)

    def _handle_assisted_grasping(self):
        """ManipulationRobot's assisted-grasp state machine, but with the
        "is the jaw being commanded closed?" test taken against the CALIBRATED
        close threshold instead of the jaw's full mechanical limit.

        Implemented by substituting the gripper controller's stored `control`
        with a SATURATED value -- the closed limit where the real command is
        below the threshold, the open limit where it is above -- and delegating
        to super(). Those two values are exactly what OmniGibson's own limit
        test reads as True / False, so the whole grasp-window / release-window
        state machine stays in one place (OmniGibson's) instead of being forked
        into 70 duplicated lines here.

        This cannot reach the simulator. ManipulationRobot.deploy_control calls
        super().deploy_control(...) -- which writes the real control to the
        joints -- BEFORE it calls this method, so by now the actual jaw command
        has already been applied. The value is restored in a finally: anyway,
        so anything that reads controller.control later (e.g. is_grasping
        heuristics, state dumps) sees the real command.
        """
        thresh = self._ag_close_threshold_rad
        if thresh is None:
            return super()._handle_assisted_grasping()

        saved = {}
        try:
            for arm in self.arm_names:
                ctrl = self._controllers[f"gripper_{arm}"]
                if ctrl.control is None or ctrl.control_type != ControlType.POSITION:
                    continue
                # Per-finger inversion. The SO-101 jaw is non-inverted (closed =
                # lower limit), which is ManipulationRobot's default for
                # _grasping_direction == "lower"; an inverted finger would need
                # the threshold measured down from its upper limit instead, and
                # nothing here needs that, so refuse rather than guess.
                inverted = getattr(ctrl, "_inverted", None)
                if inverted is not None and bool(th.as_tensor(inverted).any()):
                    raise NotImplementedError(
                        "SO101's calibrated assisted-grasp threshold assumes a "
                        "non-inverted finger (closed == lower joint limit)")
                idx = ctrl.dof_idx
                lower = self.joint_lower_limits[idx]
                upper = self.joint_upper_limits[idx]
                control = ctrl.control
                closing = th.as_tensor(control) < thresh
                # Debounce: only report OPEN after SO101_AG_OPEN_FRAMES
                # consecutive open commands. Counter is per-arm and resets on
                # any closing command.
                n_open = getattr(self, "_so101_open_streak", {})
                if bool(th.as_tensor(closing).all()):
                    n_open[arm] = 0
                else:
                    n_open[arm] = n_open.get(arm, 0) + 1
                self._so101_open_streak = n_open
                if n_open[arm] < SO101_AG_OPEN_FRAMES:
                    # not yet a sustained open -> keep reporting "closing"
                    closing = th.ones_like(th.as_tensor(closing), dtype=th.bool)
                sub = th.where(closing, lower, upper)
                saved[arm] = control
                # Written through the private attribute because `control` is a
                # read-only property on ControllerBase. th.as_tensor / .numpy()
                # rather than omnigibson's `cb`, because `cb`'s methods are
                # None until the simulator picks a backend and this has to be
                # testable (and correct) either way.
                ctrl._control = sub if isinstance(control, th.Tensor) \
                    else sub.numpy().astype(control.dtype)
            return super()._handle_assisted_grasping()
        finally:
            for arm, c in saved.items():
                self._controllers[f"gripper_{arm}"]._control = c

    def _establish_grasp(self, arm="default", ag_data=None, contact_pos=None):
        out = super()._establish_grasp(arm=arm, ag_data=ag_data, contact_pos=contact_pos)
        a = self.default_arm if arm == "default" else arm
        obj = self._ag_obj_in_hand.get(a)
        if SO101_AG_VERBOSE and obj is not None:
            print(f"[so101-ag] GRASP   obj={obj.name} jaw={self.jaw_norm(a):.1f}/100",
                  flush=True)
        return out

    def _release_grasp(self, arm="default"):
        a = self.default_arm if arm == "default" else arm
        obj = self._ag_obj_in_hand.get(a)
        if SO101_AG_VERBOSE and obj is not None:
            print(f"[so101-ag] RELEASE obj={obj.name} jaw={self.jaw_norm(a):.1f}/100 "
                  f"open_streak={getattr(self, '_so101_open_streak', {}).get(a, 0)}",
                  flush=True)
        return super()._release_grasp(arm=arm)

    def jaw_norm(self, arm="default"):
        """Current jaw position in LeRobot gripper units (0..100)."""
        arm = self.default_arm if arm == "default" else arm
        q = float(self.get_joint_positions()[self.gripper_control_idx[arm]][0])
        lo, hi = SO101_JAW_NORM_RANGE
        return 100.0 * (q - lo) / (hi - lo)

    @property
    def model_name(self):
        return "SO101"

    @property
    def usd_path(self):
        # Not the standard models/<model>/usd/<model>.usda layout: this asset is
        # NVIDIA's, kept outside the BEHAVIOR robot-assets dataset.
        return SO101_USD_PATH

    @property
    def discrete_action_list(self):
        raise NotImplementedError()

    def _create_discrete_action_space(self):
        raise ValueError("SO101 does not support discrete actions!")

    @property
    def _raw_controller_order(self):
        return [f"arm_{self.default_arm}", f"gripper_{self.default_arm}"]

    @property
    def _default_controllers(self):
        controllers = super()._default_controllers
        controllers[f"arm_{self.default_arm}"] = "InverseKinematicsController"
        controllers[f"gripper_{self.default_arm}"] = "MultiFingerGripperController"
        return controllers

    @property
    def _default_joint_pos(self):
        # NVIDIA's tuned rest pose (assets/so101.py in the workshop repo), in the
        # USD's joint order: Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw.
        return th.tensor([-0.2736, -0.6109, -0.0745, 1.5148, -1.6034, -0.1465])

    @cached_property
    def arm_link_names(self):
        return {self.default_arm: ["base", "shoulder", "upper_arm", "lower_arm", "wrist"]}

    @cached_property
    def arm_joint_names(self):
        return {self.default_arm: ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]}

    @cached_property
    def eef_link_names(self):
        # The asset has no dedicated massless eef frame, so the hand body is it.
        # Its origin sits behind the jaws, which is why the policy closed-loops
        # on the assisted-grasp midpoint rather than on the EEF pose.
        return {self.default_arm: "gripper"}

    @cached_property
    def gripper_link_names(self):
        # Must be mutually exclusive from arm_link_names and finger_link_names,
        # and `gripper` is listed as a finger below, so this is empty.
        return {self.default_arm: []}

    @cached_property
    def finger_link_names(self):
        # BOTH surfaces, not just the moving lever. In "assisted" mode
        # ManipulationRobot requires contact on >= 2 finger links:
        #   len({finger_links} & robot_contacts) >= 2     (manipulation_robot.py)
        # With `jaw` alone that is unsatisfiable, so a grasp could never latch --
        # the arm reached the object and closed on it, and in_hand stayed None.
        # Listing `gripper` too is also physically honest: the SO-101's static
        # jaw face is part of that body, so it really is the second contact
        # surface. finger_joint_names stays one entry (only the lever is driven);
        # nothing ties the two lists' lengths together.
        return {self.default_arm: ["gripper", "jaw"]}

    @cached_property
    def finger_joint_names(self):
        return {self.default_arm: ["Jaw"]}

    @property
    def teleop_rotation_offset(self):
        return {self.default_arm: euler2quat([-math.pi, 0, 0])}

    @property
    def _assisted_grasp_start_points(self):
        return {self.default_arm: self._ag_start_points}

    @property
    def _assisted_grasp_end_points(self):
        return {self.default_arm: self._ag_end_points}
