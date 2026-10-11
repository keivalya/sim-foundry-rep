"""Check the offline wrist camera against the real sensor, and against physics.

Three independent things, none of which take the other's word:

1. EXTRINSIC/INTRINSIC MATCH. Re-derives the authored UsdGeom.Camera at
   /so101_new_calib/gripper/gripper_cam straight out of the asset -- its
   cam->gripper transform and its focalLength/aperture -- and diffs that against
   the constants the renderer actually uses.

2. AXIS CONVENTION. A USD camera looks along its local -Z with +Y up; sfrender
   is OpenCV (+x right, +y DOWN, +z forward). Getting that backwards is silent
   and catastrophic, so it is checked against geometry rather than asserted: the
   unit vector from the camera to the jaw pads at gripper-local (0,0,-0.101)
   must dot ~+1 with the -Z reading and ~-1 with the +Z reading. The OpenCV
   basis handed to sfrender is then checked for orthonormality, right-handedness
   and agreement with that view direction.

3. IT LOOKS AT THE OBJECT. For every frame in which an object is actually in the
   jaws (its origin within 2 cm of the pad point), the object is projected
   through the wrist camera and must land inside the image -- not on floor, not
   behind the chassis. Reported as pixel position and angle off the optical axis.

    $E tools/check_wrist_cam.py
"""
import os, sys, math, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import episode_geom as eg
import render_episode_multiview as rm

ap = argparse.ArgumentParser()
ap.add_argument("--usd", default=f"{rm.REPO}/robot/so101_usd/so101_wristcam.usd")
ap.add_argument("--prim", default="/so101_new_calib/gripper/gripper_cam")
ap.add_argument("--poses", default=f"{rm.REPO}/scene_data/reference_demo/pick_place/pick_place_so101_poses.npz")
ap.add_argument("--geom-cache", default=f"{rm.REPO}/results/so101/multiview/geom_cache.npz")
ap.add_argument("--width", type=int, default=720)
ap.add_argument("--height", type=int, default=450)
ap.add_argument("--grasp-tol", type=float, default=0.02)
a = ap.parse_args()

PADS = np.array([0.0, 0.0, -0.101])        # jaw pads, gripper-local
fail = []

# ---- 1. what the asset actually says -------------------------------------
print("1. authored camera vs renderer constants")
from pxr import Usd, UsdGeom
st = Usd.Stage.Open(a.usd, Usd.Stage.LoadAll)
xc = UsdGeom.XformCache(Usd.TimeCode.Default())
Gi = xc.GetLocalToWorldTransform(st.GetPrimAtPath("/so101_new_calib/gripper")).GetInverse()
prim = st.GetPrimAtPath(a.prim)
if not prim or prim.GetTypeName() != "Camera":
    print(f"   FAIL: no Camera prim at {a.prim} in {a.usd}")
    sys.exit(1)
M = np.array(xc.GetLocalToWorldTransform(prim) * Gi).T       # cam -> gripper
R_usd, t_usd = M[:3, :3], M[:3, 3]
cam = UsdGeom.Camera(prim)
fl = cam.GetPrim().GetAttribute("focalLength").Get()
ha = cam.GetPrim().GetAttribute("horizontalAperture").Get()
va = cam.GetPrim().GetAttribute("verticalAperture").Get()
sw, sh = eg.WRIST_CAM_INTRINSICS[4], eg.WRIST_CAM_INTRINSICS[5]
fx_usd, fy_usd = sw*fl/ha, sh*fl/va

for nm, got, want in (("mount", t_usd, eg.WRIST_CAM_MOUNT),
                      ("forward (USD -Z)", -R_usd[:, 2], eg.WRIST_CAM_FORWARD),
                      ("up (USD +Y)", R_usd[:, 1], eg.WRIST_CAM_UP),
                      ("fx, fy", np.array([fx_usd, fy_usd]),
                       np.array(eg.WRIST_CAM_INTRINSICS[:2]))):
    e = float(np.abs(np.asarray(got) - np.asarray(want)).max())
    print(f"   {nm:18s} asset {np.round(got, 6)}  renderer {np.round(want, 6)}"
          f"   max err {e:.2e}")
    if e > 1e-5:
        fail.append(f"{nm} differs by {e:.2e}")
print(f"   (focalLength {fl} mm, aperture {ha} x {va} mm at {sw:.0f}x{sh:.0f})")
print(f"   R orthonormal err {np.abs(R_usd.T@R_usd - np.eye(3)).max():.2e}, "
      f"det {np.linalg.det(R_usd):+.9f}")

# ---- 2. the convention, checked against geometry --------------------------
print("\n2. axis convention")
v = PADS - t_usd; v /= np.linalg.norm(v)
d_minus = float((-R_usd[:, 2]) @ v)
d_plus = float((R_usd[:, 2]) @ v)
print(f"   unit vector camera -> jaw pads {np.round(v, 4)}")
print(f"   dot with USD -Z  {d_minus:+.4f}   <- must be ~+1 (this is forward)")
print(f"   dot with USD +Z  {d_plus:+.4f}   <- must be ~-1")
if not (d_minus > 0.9 and d_plus < -0.9):
    fail.append("USD -Z is not the view direction")

B = eg.camera_basis(eg.WRIST_CAM_FORWARD, eg.WRIST_CAM_UP)   # [right|down|fwd]
orth = float(np.abs(B.T@B - np.eye(3)).max())
det = float(np.linalg.det(B))
dcheck = float(np.abs(np.cross(B[:, 2], B[:, 0]) - B[:, 1]).max())
print(f"   OpenCV basis [right|down|fwd]: orthonormal err {orth:.2e}, "
      f"det {det:+.6f}, (fwd x right == down) err {dcheck:.2e}")
print(f"   forward {B[:,2].round(6)}  down {B[:,1].round(6)}  "
      f"= -(USD up) {(-np.asarray(eg.WRIST_CAM_UP)).round(6)}")
if orth > 1e-9 or det < 0.999 or dcheck > 1e-9:
    fail.append("OpenCV basis is not a right-handed orthonormal frame")
if float(np.abs(B[:, 2] - np.asarray(eg.WRIST_CAM_FORWARD)).max()) > 1e-9:
    fail.append("OpenCV forward column != authored view direction")

# ---- 3. does it actually look at what is grasped? -------------------------
print("\n3. grasp frames: is the grasped object in the wrist image?")
lg, og = eg.cached_geom(None, None, a.geom_cache)
ep = eg.Episode(a.poses, lg, og)
wc = rm.WristCam(ep, w=a.width, h=a.height)
gi = ep.keys.index("link::gripper")
objs = [k for k in ep.keys if k.startswith("obj::") and k != "obj::agent"]

rows, misses = [], 0
for fr in range(len(ep)):
    p, R = ep.pose(fr, "link::gripper")
    pad = p + R @ PADS
    for k in objs:
        o = ep.poses[fr, ep.keys.index(k), :3]
        if np.linalg.norm(pad - o) > a.grasp_tol:
            continue
        R_wc, t, K = wc.pose(fr)
        c = (o - t) @ R_wc                      # world -> camera
        if c[2] <= 1e-6:
            rows.append((fr, k, None, None, None)); misses += 1; continue
        u = K[0, 0]*c[0]/c[2] + K[0, 2]
        vv = K[1, 1]*c[1]/c[2] + K[1, 2]
        ang = math.degrees(math.acos(np.clip(c[2]/np.linalg.norm(c), -1, 1)))
        inside = 0 <= u < a.width and 0 <= vv < a.height
        rows.append((fr, k, (u, vv), ang, inside))
        if not inside:
            misses += 1

if not rows:
    fail.append("no grasp frames found -- check --grasp-tol")
else:
    byobj = {}
    for fr, k, uv, ang, ins in rows:
        byobj.setdefault(k, []).append((fr, uv, ang, ins))
    for k, r in byobj.items():
        frs = [x[0] for x in r]
        ins = sum(1 for x in r if x[3])
        angs = [x[2] for x in r if x[2] is not None]
        us = [x[1][0] for x in r if x[1]]
        vs = [x[1][1] for x in r if x[1]]
        print(f"   {k:22s} frames {min(frs)}-{max(frs)} ({len(r)})  "
              f"in frame {ins}/{len(r)}  "
              f"px x {min(us):.0f}-{max(us):.0f} y {min(vs):.0f}-{max(vs):.0f}  "
              f"off-axis {min(angs):.1f}-{max(angs):.1f} deg")
    print(f"   total grasp frames {len(rows)}, outside image {misses}")
    if misses:
        fail.append(f"{misses} grasp frames have the object outside the wrist image")

print("\nRESULT:", "OK" if not fail else "FAIL: " + "; ".join(fail))
sys.exit(0 if not fail else 1)
