"""Widen three SO-101 USD joint limits so the pi0.5 checkpoint's commanded range
is physically reachable, WITHOUT touching the unit normalization range.

WHY
---
The checkpoint `pdhinaka/pi05_so101_blue_cube_gray_box_cleaned_20k` was recorded
on a real SO-101 through LeRobot, whose `Motor.normalize()` is an UNCLAMPED
affine map from the motor's CALIBRATED range onto [-100, 100]. The recorded data
therefore runs outside +-100 wherever the physical arm travelled past the range
the calibration sweep happened to cover:

    joint          dataset state min/max      dataset ACTION min/max
    shoulder_lift  -104.264 ..  +41.407       -109.714 ..  +40.791   <-- past -100
    wrist_flex      +58.418 ..  +98.593        +57.846 .. +102.681   <-- past +100
    elbow_flex      -24.967 ..  +97.758        -27.297 ..  +99.033

The sim's affine map is the same map with the motor's calibrated range replaced
by the joint's NOMINAL limit pair (which is also the URDF's declared mechanical
range), so normalized +-100 is pinned to those nominal limits -- see
vla/so101_units.py. That map cannot be "fixed" by widening a limit, because
widening the range you normalize against redefines what -104 MEANS. The two
things have to be decoupled:

  * NORMALIZATION range: stays at the nominal limits.  (so101_units.NORM_LIMITS)
  * PHYSICAL limit:      widened here, so the arm can actually GO to -109.7.

Both halves are needed. Without the widening the command is clipped at -100 and
the arm parks on the stop (job 10885537: shoulder_lift clipped on 200/450
steps). Without the pinning the units silently change under the policy.

HOW MUCH
--------
Enough to cover the dataset's action min/max with ~5 normalized units of
headroom, since nothing stops the policy commanding slightly outside its
training range:

    Pitch        lower -100 deg -> -115 deg   (normalized -100 -> -115.0)
    Wrist_Pitch  upper  +95 deg -> +105 deg   (normalized +100 -> +110.5)
    Elbow        upper  +90 deg ->  +95 deg   (normalized +100 -> +105.3)

Nothing else in the dataset leaves [-100, 100], so nothing else is touched.
Self-collision is off on this robot and the widened poses keep every link above
the table (checked by FK), so the extra travel is free.

Written to a NEW usd; the original is left alone because tools/check_wrist_cam.py,
tools/episode_geom.py and every existing result read it.
"""
import argparse, shutil, os
from pxr import Usd

# joint prim -> (new lower deg or None, new upper deg or None)
WIDEN = {
    "Pitch":       (-115.0, None),
    "Wrist_Pitch": (None,   105.0),
    "Elbow":       (None,    95.0),
}

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/scratch/pandya.kei/simfoundry-repro/robot/so101_usd/so101_wristcam.usd")
ap.add_argument("--dst", default="/scratch/pandya.kei/simfoundry-repro/robot/so101_usd/so101_wristcam_wide.usd")
ap.add_argument("--root", default="/so101_new_calib/joints")
a = ap.parse_args()

assert os.path.abspath(a.src) != os.path.abspath(a.dst), "refusing to overwrite the source asset"
shutil.copyfile(a.src, a.dst)
st = Usd.Stage.Open(a.dst)
for j, (lo, hi) in WIDEN.items():
    p = st.GetPrimAtPath(f"{a.root}/{j}")
    assert p and p.IsValid(), f"missing joint prim {a.root}/{j}"
    for attr, new in (("physics:lowerLimit", lo), ("physics:upperLimit", hi)):
        if new is None:
            continue
        at = p.GetAttribute(attr)
        old = at.Get()
        at.Set(float(new))
        print(f"{j:12s} {attr:20s} {old:+8.3f} -> {at.Get():+8.3f} deg")
# Every other joint is printed too, so a silent drift in the source is visible.
for p in st.Traverse():
    if p.GetTypeName() == "PhysicsRevoluteJoint":
        print(f"  final {p.GetName():12s} [{p.GetAttribute('physics:lowerLimit').Get():+9.3f},"
              f" {p.GetAttribute('physics:upperLimit').Get():+9.3f}] deg")
st.GetRootLayer().Save()
print(f"\nwrote {a.dst}")
