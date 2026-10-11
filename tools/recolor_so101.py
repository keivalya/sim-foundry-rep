"""Repaint the SO-101's 3D-printed parts, so the arm the policy SEES in the sim
looks like the arm it was trained on.

The pi0.5 blue-cube checkpoint was fine-tuned on a real SO-101 printed in
off-white: in every training wrist_cam frame the two jaws occupy the bottom
~30% of the image, and in every top_cam frame the gripper enters from the top
edge. NVIDIA's Isaac asset is printed in bright yellow
(`material_a_3d_printed`, diffuse (1, 0.82, 0.12)). That is a large, constant,
in-frame appearance difference in both of the policy's inputs, on the one object
that is in shot 100% of the time -- a plausible reason a real-data VLA does
nothing in sim, and cheap to remove.

Only the diffuse constant of the printed-part material is touched; the servo
bodies (`material_sts3215`, near-black in both) and every other material are
left alone, because they already match. Writes a NEW usd.

    conda_envs/simfoundry-editor/bin/python -I tools/recolor_so101.py \
        --src robot/so101_usd/so101_wristcam_wide.usd \
        --dst robot/so101_usd/so101_wristcam_wide_white.usd
"""
import argparse, os, shutil
from pxr import Usd, Gf

ap = argparse.ArgumentParser()
P = "/scratch/pandya.kei/simfoundry-repro/robot/so101_usd"
ap.add_argument("--src", default=f"{P}/so101_wristcam_wide.usd")
ap.add_argument("--dst", default=f"{P}/so101_wristcam_wide_white.usd")
ap.add_argument("--material", default="/so101_new_calib/Looks/material_a_3d_printed/Shader")
ap.add_argument("--rgb", default="0.86,0.86,0.84", help="off-white, as the real arm photographs")
a = ap.parse_args()
assert os.path.abspath(a.src) != os.path.abspath(a.dst)
shutil.copyfile(a.src, a.dst)
st = Usd.Stage.Open(a.dst)
p = st.GetPrimAtPath(a.material)
assert p and p.IsValid(), f"no shader at {a.material}"
at = p.GetAttribute("inputs:diffuse_color_constant")
old = at.Get()
at.Set(Gf.Vec3f(*[float(v) for v in a.rgb.split(",")]))
print(f"{a.material}\n  diffuse {old} -> {at.Get()}")
st.GetRootLayer().Save()
print(f"wrote {a.dst}")
