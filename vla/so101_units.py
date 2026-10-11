"""LeRobot normalized SO-101 motor units  <->  OmniGibson joint radians.

Pure numpy, no torch / no lerobot / no omnigibson, so BOTH conda envs can import
it (openpi has numpy 2.2, simfoundry has numpy 1.26 -- see the wire format in
pi05_server.py for why that matters).

WHAT THE TWO UNIT SYSTEMS ARE
-----------------------------
* OmniGibson / Isaac drives the SO-101 in RADIANS, joint order taken from
  NVIDIA's USD: Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw.
* LeRobot drives a real SO-101 in NORMALIZED MOTOR UNITS. `so101_follower`
  declares every body joint as MotorNormMode.RANGE_M100_100 and the gripper as
  RANGE_0_100, and `Motor.normalize()` is an affine map from that motor's
  CALIBRATED MECHANICAL RANGE [range_min, range_max] onto that interval:

      body     n = 200 * (raw - range_min) / (range_max - range_min) - 100
      gripper  n = 100 * (raw - range_min) / (range_max - range_min)

  The pi0.5 checkpoint was trained on data recorded through exactly that path,
  so its state and action vectors live in those units.

THE MAPPING USED HERE, AND THE ASSUMPTION IN IT
-----------------------------------------------
The sim's equivalent of "the motor's calibrated mechanical range" is the joint's
limit pair in the USD. So the conversion is the same affine map with
[range_min, range_max] replaced by [joint_lower_limit, joint_upper_limit]:

      body     n = 200 * (q - lo) / (hi - lo) - 100
      gripper  n = 100 * (q - lo) / (hi - lo)

LIMITS are MEASURED IN-SIM, not hardcoded -- `from_sim_limits()` builds the
converter from `robot.joint_lower_limits` / `joint_upper_limits`. LIMITS below is
the set measured on this asset, used only as a default and as a cross-check.

ASSUMPTION, stated plainly: this assumes the sim joint's zero/sign convention
agrees with the convention of whatever physical arm recorded the training set.
That is NOT verifiable -- the checkpoint ships no calibration file, and a real
SO-101's calibration is per-unit. If the fine-tune's arm had, say, elbow_flex
wired with the opposite sign, every action would come back mirrored on that
joint and nothing here would detect it. The mapping is the only defensible
one available; it is not a proof that the policy sees what it saw in training.

Equally honest: the gripper map is EXACT under the sim's
MultiFingerGripperController in "smooth" mode, because that controller does
u = lo + (cmd+1)/2 * (hi - lo) over the jaw's own PHYSICAL limits. That is the
same affine map as the LeRobot gripper normalization only while the physical
limits equal the normalization range; `gripper_norm_to_cmd()` below inverts the
controller's map explicitly, so it stays exact if they are ever decoupled.

NORMALIZATION RANGE vs PHYSICAL LIMIT -- THEY ARE NOT THE SAME THING
--------------------------------------------------------------------
`Motor.normalize()` is NOT clamped. A real SO-101 that travels past the range
its calibration sweep covered therefore records values OUTSIDE [-100, 100], and
the pi0.5 blue-cube checkpoint's training set does exactly that:

    joint          state min .. max        ACTION min .. max
    shoulder_lift  -104.264 .. +41.407     -109.714 .. +40.791
    wrist_flex      +58.418 .. +98.593      +57.846 .. +102.681
    elbow_flex      -24.967 .. +97.758      -27.297 .. +99.033

So the policy's single most common shoulder_lift state (q50 = -104.169) and its
most common commanded shoulder_lift (action q50 = -108.380) are both BELOW the
-100 that the USD's -1.7453 rad limit maps to. Job 10885537 clipped the
shoulder_lift command on 200 of 450 steps and the arm parked on the stop.

This CANNOT be fixed by widening the limit alone, because the normalized value
is DEFINED by the range you normalize against: widen the range and -104 stops
meaning the same joint angle. The two roles have to be separated, and they are:

  * NORM_LIMITS -- the range the +-100 map is pinned to. Fixed, equal to the
    joint's NOMINAL limit pair (which is also what so101_new_calib.urdf
    declares as the mechanical range, i.e. what a calibration sweep covers).
    rad_to_norm / norm_to_rad use ONLY this. It never changes with the asset.
  * the PHYSICAL limits -- measured in-sim, used ONLY for clip_rad() and for
    the gripper controller's own map. tools/widen_so101_limits.py writes
    robot/so101_usd/so101_wristcam_wide.usd with
        Pitch        lower -100 -> -115 deg   (normalized -100 -> -115.0)
        Wrist_Pitch  upper  +95 -> +105 deg   (normalized +100 -> +110.5)
        Elbow        upper  +90 ->  +95 deg   (normalized +100 -> +105.3)
    which covers the dataset's action range with headroom.

from_sim_limits() builds exactly that: NORM_LIMITS for the units, the measured
limits for clipping. A physical limit WIDER than nominal is reported as the
deliberate fix; a NARROWER one is the dangerous case (the policy's range becomes
unreachable again) and warns loudly.
"""

import numpy as np

# USD joint order == LeRobot action/state order, element for element.
SIM_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
LEROBOT_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                  "wrist_flex", "wrist_roll", "gripper"]

# The NOMINAL limit pair per joint, in radians: the range the LeRobot
# [-100, 100] / [0, 100] map is pinned to. These are the limits authored in
# robot/so101_usd/so101_wristcam.usd, and they agree with the mechanical range
# so101_src/so101_new_calib.urdf declares to within the elbow's and wrist roll's
# retune. This array is the UNIT SYSTEM; it must not track a widened asset.
NORM_LIMITS = np.array([
    [-1.9199,  1.9199],   # Rotation     / shoulder_pan
    [-1.7453,  1.7453],   # Pitch        / shoulder_lift
    [-1.7453,  1.5708],   # Elbow        / elbow_flex      (asymmetric)
    [-1.6581,  1.6581],   # Wrist_Pitch  / wrist_flex
    [-2.7925,  2.7925],   # Wrist_Roll   / wrist_roll
    [-0.1745,  1.7453],   # Jaw          / gripper
], dtype=float)

LIMITS = NORM_LIMITS          # backwards-compatible alias

GRIPPER_IDX = 5
_EPS = 1e-9

# Dataset action min/max (LeRobot units) the physical limits must be able to
# execute. Dumped from the checkpoint's own normalizer safetensors; see
# vla/so101_state_quantiles.json and tools/widen_so101_limits.py.
CKPT_ACTION_MIN = np.array([-33.978, -109.714, -27.297, 57.846, -34.330, 0.000])
CKPT_ACTION_MAX = np.array([28.088, 40.791, 99.033, 102.681, 32.747, 37.275])


class SO101Units:
    """Affine converter: NORMALIZATION range and PHYSICAL limits kept separate.

    norm_limits -- defines the units. Pinned to NORM_LIMITS; do not vary it.
    phys_limits -- what the asset can actually reach. Used only by clip_rad()
                   and by the gripper controller's own map. Defaults to
                   norm_limits, which is the legacy behaviour.
    """

    def __init__(self, limits=None, norm_limits=NORM_LIMITS, phys_limits=None):
        # `limits=` is the old single-range signature; honour it as BOTH.
        if limits is not None:
            norm_limits = limits
            phys_limits = limits if phys_limits is None else phys_limits
        nl = np.asarray(norm_limits, float)
        pl = nl if phys_limits is None else np.asarray(phys_limits, float)
        self.nlo, self.nhi = nl[:, 0].copy(), nl[:, 1].copy()
        self.lo, self.hi = pl[:, 0].copy(), pl[:, 1].copy()
        assert self.nlo.shape == (6,) and np.all(self.nhi - self.nlo > _EPS)
        assert self.lo.shape == (6,) and np.all(self.hi - self.lo > _EPS)

    # ---- construction -----------------------------------------------------
    @classmethod
    def from_sim_limits(cls, joint_names, lower, upper, tol=2e-3, log=print):
        """Build from a live robot's limits, reordered into SIM_JOINTS order.

        The UNITS come from NORM_LIMITS regardless of what the asset says --
        that is the whole point (see the module docstring). The measured limits
        are used for clipping, and the comparison is reported both ways round:

          * WIDER than nominal  -> the deliberate fix from
            tools/widen_so101_limits.py. Reported with the normalized reach it
            buys, so you can see at a glance whether the checkpoint's commanded
            range is now executable.
          * NARROWER than nominal -> dangerous: part of the policy's action
            range becomes unreachable again. Warned about loudly.
        """
        joint_names = [str(j) for j in joint_names]
        idx = [joint_names.index(j) for j in SIM_JOINTS]
        lim = np.stack([np.asarray(lower, float)[idx],
                        np.asarray(upper, float)[idx]], axis=1)
        u = cls(norm_limits=NORM_LIMITS, phys_limits=lim)
        reach_lo = u.rad_to_norm(u.lo)
        reach_hi = u.rad_to_norm(u.hi)
        narrow = []
        log("   joint      nominal (rad)        physical (rad)       reachable (lerobot)   "
            "ckpt action range")
        for i, j in enumerate(SIM_JOINTS):
            dl, dh = lim[i, 0] - NORM_LIMITS[i, 0], lim[i, 1] - NORM_LIMITS[i, 1]
            tag = "same"
            if dl < -tol or dh > tol:
                tag = "WIDER"
            if dl > tol or dh < -tol:
                tag = "NARROWER"
                narrow.append(j)
            ok = (reach_lo[i] <= CKPT_ACTION_MIN[i] + 1e-6
                  and reach_hi[i] >= CKPT_ACTION_MAX[i] - 1e-6)
            log(f"   {j:12s}[{NORM_LIMITS[i,0]:+.4f},{NORM_LIMITS[i,1]:+.4f}] "
                f"[{lim[i,0]:+.4f},{lim[i,1]:+.4f}] "
                f"[{reach_lo[i]:+8.2f},{reach_hi[i]:+8.2f}] "
                f"[{CKPT_ACTION_MIN[i]:+8.2f},{CKPT_ACTION_MAX[i]:+8.2f}] "
                f"{tag:9s}{'ok' if ok else 'NOT REACHABLE'}")
        if narrow:
            log(f"!! sim joint limits are NARROWER than nominal on {narrow} -- part of "
                f"the checkpoint's action range is unreachable and will be clipped")
        unreach = [SIM_JOINTS[i] for i in range(6)
                   if reach_lo[i] > CKPT_ACTION_MIN[i] + 1e-6
                   or reach_hi[i] < CKPT_ACTION_MAX[i] - 1e-6]
        if unreach:
            log(f"!! the checkpoint's commanded range is NOT reachable on {unreach} -- "
                f"run tools/widen_so101_limits.py and set SO101_USD to its output")
        else:
            log("   the checkpoint's full commanded action range is physically reachable")
        return u, idx

    # ---- conversion (NORMALIZATION RANGE ONLY) ----------------------------
    def rad_to_norm(self, q):
        """radians (SIM_JOINTS order) -> LeRobot normalized units."""
        q = np.asarray(q, float)
        t = (q - self.nlo) / (self.nhi - self.nlo)         # 0..1 over NOMINAL
        n = t * 200.0 - 100.0
        n[..., GRIPPER_IDX] = t[..., GRIPPER_IDX] * 100.0  # gripper is 0..100
        return n

    def norm_to_rad(self, n):
        """LeRobot normalized units -> radians (SIM_JOINTS order). NOT clipped:
        a value outside +-100 is legitimate (the dataset has them) and whether
        it is reachable is clip_rad()'s business, not the unit map's."""
        n = np.asarray(n, float)
        t = (n + 100.0) / 200.0
        t = np.concatenate([t[..., :GRIPPER_IDX],
                            (n[..., GRIPPER_IDX:GRIPPER_IDX + 1] / 100.0)], axis=-1)
        return self.nlo + t * (self.nhi - self.nlo)

    # ---- physical limits --------------------------------------------------
    def clip_rad(self, q):
        """Clip to what the ASSET can reach (phys), not to the unit range."""
        return np.clip(np.asarray(q, float), self.lo, self.hi)

    def norm_reach(self):
        """The physical limits expressed in LeRobot units, i.e. how far outside
        +-100 this asset can actually be commanded."""
        return self.rad_to_norm(self.lo), self.rad_to_norm(self.hi)

    def gripper_norm_to_cmd(self, g):
        """LeRobot gripper (0..100) -> MultiFingerGripperController 'smooth'
        command in [-1, 1]. The controller computes
            u = phys_lo + (cmd+1)/2 * (phys_hi - phys_lo)
        so this inverts exactly that, which keeps the round trip exact even if
        the jaw's physical limits are ever widened away from the nominal range.
        """
        q = self.norm_to_rad(np.r_[np.zeros(5), float(g)])[GRIPPER_IDX]
        q = float(np.clip(q, self.lo[GRIPPER_IDX], self.hi[GRIPPER_IDX]))
        t = (q - self.lo[GRIPPER_IDX]) / (self.hi[GRIPPER_IDX] - self.lo[GRIPPER_IDX])
        return float(np.clip(t * 2.0 - 1.0, -1.0, 1.0))


# The widened asset's limits, in radians, in SIM_JOINTS order -- what
# tools/widen_so101_limits.py writes. Kept here so the self-test can check the
# fix without a simulator.
WIDE_LIMITS = np.array([
    [-1.9199,  1.9199],              # Rotation      unchanged
    [np.radians(-115.0),  1.7453],   # Pitch         lower widened
    [-1.7453, np.radians(95.0)],     # Elbow         upper widened
    [-1.6581, np.radians(105.0)],    # Wrist_Pitch   upper widened
    [-2.7925,  2.7925],              # Wrist_Roll    unchanged
    [-0.1745,  1.7453],              # Jaw           unchanged
], dtype=float)


DEFAULT = SO101Units()
rad_to_norm = DEFAULT.rad_to_norm
norm_to_rad = DEFAULT.norm_to_rad


def _selftest():
    u = SO101Units()
    rng = np.random.default_rng(0)

    # 1. named end points
    for i, j in enumerate(SIM_JOINTS):
        n_lo = u.rad_to_norm(u.lo)[i]
        n_hi = u.rad_to_norm(u.hi)[i]
        want = (0.0, 100.0) if i == GRIPPER_IDX else (-100.0, 100.0)
        assert abs(n_lo - want[0]) < 1e-9 and abs(n_hi - want[1]) < 1e-9, (j, n_lo, n_hi)
        print(f"  {j:12s} [{u.lo[i]:+.4f},{u.hi[i]:+.4f}] rad  ->  "
              f"[{n_lo:+.1f},{n_hi:+.1f}] lerobot")

    # 2. round-trip, 100k random configurations, both directions
    q = u.lo + rng.random((100000, 6)) * (u.hi - u.lo)
    e1 = np.abs(u.norm_to_rad(u.rad_to_norm(q)) - q).max()
    n = np.c_[rng.uniform(-100, 100, (100000, 5)), rng.uniform(0, 100, 100000)]
    e2 = np.abs(u.rad_to_norm(u.norm_to_rad(n)) - n).max()
    print(f"  rad->norm->rad  max err {e1:.3e} rad")
    print(f"  norm->rad->norm max err {e2:.3e} lerobot units")
    assert e1 < 1e-9 and e2 < 1e-9

    # 3. the rest pose, as a sanity anchor
    rest = np.array([-0.2736, -0.6109, -0.0745, 1.5148, -1.6034, -0.1465])
    print(f"  NVIDIA rest pose {np.round(rest, 4)} rad")
    print(f"               ->  {np.round(u.rad_to_norm(rest), 2)} lerobot")

    # 4. the gripper command path through MultiFingerGripperController(smooth)
    for g in (0.0, 25.0, 50.0, 100.0):
        cmd = u.gripper_norm_to_cmd(g)
        ctrl_q = u.lo[5] + (cmd + 1.0) / 2.0 * (u.hi[5] - u.lo[5])   # controller's own map
        back = u.rad_to_norm(np.r_[np.zeros(5), ctrl_q])[5]
        assert abs(back - g) < 1e-9
        print(f"  gripper {g:6.1f} -> cmd {cmd:+.4f} -> jaw {ctrl_q:+.4f} rad -> {back:6.1f}")

    # 5. THE CALIBRATION FIX. Two separate claims, checked separately.
    #    (a) the unit map does not move when the physical limits are widened
    #    (b) with the widened asset, the checkpoint's whole commanded range is
    #        reachable -- and with the original asset it is NOT (so the test
    #        would catch the widening silently not being applied)
    print("\n  --- calibration fix ---")
    w = SO101Units(norm_limits=NORM_LIMITS, phys_limits=WIDE_LIMITS)
    q = u.lo + rng.random((20000, 6)) * (u.hi - u.lo)
    e = np.abs(w.rad_to_norm(q) - u.rad_to_norm(q)).max()
    print(f"  (a) unit map unchanged by widening: max diff {e:.3e} lerobot units")
    assert e == 0.0, e
    e2 = np.abs(w.norm_to_rad(w.rad_to_norm(q)) - q).max()
    assert e2 < 1e-9, e2

    for lbl, cv in (("nominal", u), ("widened", w)):
        lo_n, hi_n = cv.norm_reach()
        bad = []
        for i, j in enumerate(LEROBOT_JOINTS):
            ok = lo_n[i] <= CKPT_ACTION_MIN[i] + 1e-6 and hi_n[i] >= CKPT_ACTION_MAX[i] - 1e-6
            if not ok:
                bad.append(j)
            print(f"  (b) {lbl:7s} {j:14s} reach [{lo_n[i]:+8.2f},{hi_n[i]:+8.2f}]  "
                  f"ckpt action [{CKPT_ACTION_MIN[i]:+8.2f},{CKPT_ACTION_MAX[i]:+8.2f}]  "
                  f"{'ok' if ok else 'CLIPPED'}")
        if lbl == "nominal":
            assert bad == ["shoulder_lift", "wrist_flex"], bad
            print(f"  (b) nominal asset clips {bad} -- this is the bug, as expected")
        else:
            assert not bad, bad
            print("  (b) widened asset: nothing clipped")

    # 6. the checkpoint's modal state must be reachable, and must round-trip
    q50 = np.array([-5.180376, -104.1688, 1.0502148, 76.527885, -13.933344, 2.8711483])
    r50 = w.norm_to_rad(q50)
    assert np.allclose(w.clip_rad(r50), r50, atol=1e-12), (r50, w.clip_rad(r50))
    print(f"  ckpt state q50 {np.round(q50,2)}")
    print(f"    -> {np.round(r50,4)} rad = {np.round(np.degrees(r50),1)} deg, "
          f"unclipped by the widened asset, round-trip err "
          f"{np.abs(w.rad_to_norm(r50)-q50).max():.2e}")
    a50 = np.array([-5.058455, -108.37973, 0.5531079, 77.627304, -13.137266, 1.4881524])
    ra = w.norm_to_rad(a50)
    assert np.allclose(w.clip_rad(ra), ra, atol=1e-12)
    print(f"  ckpt action q50 -> {np.round(ra,4)} rad, also unclipped")

    print("\nSO101_UNITS_OK")


if __name__ == "__main__":
    _selftest()
