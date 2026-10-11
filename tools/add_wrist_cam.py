"""Author the SO-101 wrist (gripper) camera into the de-instanced USD.

WHY NOT NVIDIA'S "WITH CAMERA" ASSET
------------------------------------
`SO-ARM101-USD.usd` (the WITH-camera variant of the workshop asset) contains
**no UsdGeom.Camera prim at all** -- verified by traversing it with instance
proxies enabled: zero prims of type "Camera". All it adds over
`SO-ARM101-USD-NO-CAMERA.usd` is the physical camera *body* geometry, as
`gripper/visuals/camera_mount` and `gripper/collisions/camera_mount` references
into the `/meshes/camera_mount` prototype (bolts, pcb, rings, mount, cover,
focus, lens, socket, Cone, transistors).

Our `so101.usd` was flattened from the NO-CAMERA variant but, because both
variants share the same `/meshes` scope, it *already contains* that
`/meshes/camera_mount` prototype -- it is simply not referenced from the gripper
link. So nothing is gained by re-deinstancing the WITH-camera asset.

The actual sensor in NVIDIA's project is created in Isaac Lab code, not in the
USD -- `source/sim_to_real_so101/tasks/task_env_cfg.py`:

    camera_ego.prim_path   = "{ENV_REGEX_NS}/Robot/gripper/gripper_cam"
    camera_ego.offset.pos  = (-0.005, 0.06, -0.062)
    camera_ego.offset.rot  = euler_angles_to_quat([-45, 0, 0], degrees=True)
    camera_ego.offset.convention = "opengl"
    TiledCameraCfg(width=640, height=480,
                   spawn=PinholeCameraCfg(projection_type="pinhole",
                                          focal_length=13.5, f_stop=100,
                                          focus_distance=0.05))

This script authors exactly that prim into our asset. Cross-check: the lens
mesh of `camera_mount`, measured in the gripper link frame, has its centroid at
(-0.0039, 0.0518, -0.0599) -- within ~8 mm of NVIDIA's (-0.005, 0.06, -0.062),
so their offset really is "where the lens is".

WHY A USD PRIM RATHER THAN AN OmniGibson EXTERNAL SENSOR
-------------------------------------------------------
`BaseRobot._load_sensors` (robot_base.py:212-259) walks every link's DIRECT
children and wraps any prim whose type is in `SENSOR_PRIMS_TO_SENSOR_CLS`
({"Lidar": ScanSensor, "Camera": VisionSensor}) in a VisionSensor automatically.
So a Camera authored at `/so101_new_calib/gripper/gripper_cam` becomes a real
robot-owned sensor, observable at `obs[robot.name]["robot0:gripper:Camera:0"]`,
moves with the articulation for free, and needs no per-script camera plumbing.
That is the path a VLA wants. (An `external_sensors` entry with a
`relative_prim_path` under the robot also works and needs no asset change --
see `policies/wrist_cam_probe.py --external-wrist` -- but then the camera is
owned by the environment, not the robot.)

The camera's intrinsics are NOT taken from this prim at runtime: OmniGibson's
VisionSensor overwrites focalLength / horizontalAperture / clippingRange from
its own `sensor_config`. They are authored here anyway so the asset is
self-describing and so a non-OmniGibson consumer gets the right camera. The
matching runtime config lives in `robot/so101_og.py::SO101._default_sensor_cfg`.

Usage:
    python tools/add_wrist_cam.py robot/so101_usd/so101.usd \
                                  robot/so101_usd/so101_wristcam.usd
"""
import argparse
import math

import numpy as np
from pxr import Gf, Usd, UsdGeom

# NVIDIA's values, verbatim (task_env_cfg.py:106-109 + camera_object at :49).
NV_POS = (-0.005, 0.06, -0.062)
NV_EULER_XYZ_DEG = (-45.0, 0.0, 0.0)
NV_W, NV_H = 640, 480
NV_FOCAL_MM = 13.5
# Isaac Lab PinholeCameraCfg leaves the aperture at the USD/Kit default.
NV_H_APERTURE_MM = 20.955
NV_FOCUS_DISTANCE = 0.05
NV_F_STOP = 100.0


def euler_xyz_to_quatf(deg):
    """Extrinsic XYZ euler (degrees) -> Gf.Quatf, matching
    isaacsim.core.utils.rotations.euler_angles_to_quat(..., extrinsic=True)."""
    rx, ry, rz = (math.radians(a) for a in deg)
    def R(axis, a):
        c, s = math.cos(a), math.sin(a)
        if axis == 0:
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        if axis == 1:
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    # extrinsic xyz == Rz @ Ry @ Rx
    M = R(2, rz) @ R(1, ry) @ R(0, rx)
    from scipy.spatial.transform import Rotation as _R
    qx, qy, qz, qw = _R.from_matrix(M).as_quat()      # xyzw
    return M, Gf.Quatf(float(qw), Gf.Vec3f(float(qx), float(qy), float(qz)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--link", default="gripper")
    ap.add_argument("--name", default="gripper_cam")
    ap.add_argument("--pos", type=float, nargs=3, default=list(NV_POS))
    ap.add_argument("--euler", type=float, nargs=3, default=list(NV_EULER_XYZ_DEG),
                    help="extrinsic XYZ euler in degrees, USD/OpenGL camera frame "
                         "(-Z forward, +Y up), as Isaac Lab's convention='opengl'")
    ap.add_argument("--focal", type=float, default=NV_FOCAL_MM)
    ap.add_argument("--h-aperture", type=float, default=NV_H_APERTURE_MM)
    ap.add_argument("--width", type=int, default=NV_W)
    ap.add_argument("--height", type=int, default=NV_H)
    a = ap.parse_args()

    stage = Usd.Stage.Open(a.src, Usd.Stage.LoadAll)
    link_path = f"/so101_new_calib/{a.link}"
    assert stage.GetPrimAtPath(link_path).IsValid(), f"no link prim at {link_path}"
    cam_path = f"{link_path}/{a.name}"
    if stage.GetPrimAtPath(cam_path).IsValid():
        stage.RemovePrim(cam_path)

    cam = UsdGeom.Camera.Define(stage, cam_path)
    p = cam.GetPrim()
    # Author the full [translate, orient, scale] triple, in that order, exactly
    # as deinstance_so101.py does: OmniGibson rewrites xform properties on load
    # and a half-specified stack is what once displaced the gripper visuals.
    M, q = euler_xyz_to_quatf(a.euler)
    x = UsdGeom.Xformable(p)
    x.ClearXformOpOrder()
    x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*a.pos))
    x.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(q)
    x.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(1.0, 1.0, 1.0))

    v_aperture = a.h_aperture * a.height / a.width
    cam.CreateProjectionAttr().Set(UsdGeom.Tokens.perspective)
    cam.CreateFocalLengthAttr().Set(float(a.focal))
    cam.CreateHorizontalApertureAttr().Set(float(a.h_aperture))
    cam.CreateVerticalApertureAttr().Set(float(v_aperture))
    cam.CreateFocusDistanceAttr().Set(float(NV_FOCUS_DISTANCE))
    cam.CreateFStopAttr().Set(float(NV_F_STOP))
    # Stage is metres (metersPerUnit == 1.0), so clipping is in metres. 1 cm near
    # plane: the jaws are ~7 cm from the lens and objects come closer than that.
    cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 1.0e6))

    stage.Export(a.dst)

    # ---- report + verify on the exported file -------------------------------
    fwd = -M[:, 2]
    up = M[:, 1]
    right = M[:, 0]
    hfov = 2 * math.degrees(math.atan(a.h_aperture / (2 * a.focal)))
    vfov = 2 * math.degrees(math.atan(v_aperture / (2 * a.focal)))
    fx = a.width * a.focal / a.h_aperture
    fy = a.height * a.focal / v_aperture
    print(f"wrote {a.dst}")
    print(f"  camera prim      {cam_path}")
    print(f"  pos (link frame) {np.round(a.pos, 5)}")
    print(f"  quat xyzw        {np.round([q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2], q.GetReal()], 6)}")
    print(f"  forward (-Z)     {np.round(fwd, 4)}")
    print(f"  up      (+Y)     {np.round(up, 4)}")
    print(f"  right   (+X)     {np.round(right, 4)}")
    print(f"  {a.width}x{a.height}  f={a.focal} mm  aperture h={a.h_aperture} v={v_aperture:.5f} mm")
    print(f"  FOV  h={hfov:.2f} deg  v={vfov:.2f} deg")
    print(f"  K: fx={fx:.2f} fy={fy:.2f} cx={a.width/2:.1f} cy={a.height/2:.1f} px")

    v = Usd.Stage.Open(a.dst, Usd.Stage.LoadAll)
    vp = v.GetPrimAtPath(cam_path)
    assert vp.IsValid() and vp.GetTypeName() == "Camera", "camera prim did not survive export"
    link = v.GetPrimAtPath(link_path)
    kids = [c.GetName() for c in link.GetChildren() if c.GetTypeName() == "Camera"]
    assert kids == [a.name], f"camera is not a DIRECT child of the link: {kids}"
    print(f"  verified: Camera is a direct child of {link_path} "
          f"(OmniGibson _load_sensors will wrap it)")

    # The 7 articulation links and their visual/collision mesh counts must be
    # untouched -- this file is a superset of so101.usd, nothing else.
    from pxr import UsdPhysics
    for l in ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw"]:
        lp = v.GetPrimAtPath(f"/so101_new_calib/{l}")
        vis = [x for x in Usd.PrimRange(lp) if x.GetTypeName() == "Mesh" and "visuals" in str(x.GetPath())]
        col = [x for x in Usd.PrimRange(lp) if x.GetTypeName() == "Mesh" and "collisions" in str(x.GetPath())]
        api = [x for x in col if x.HasAPI(UsdPhysics.CollisionAPI)]
        print(f"  {l:11s} visual={len(vis):2d} collision={len(col):2d} CollisionAPI={len(api)}/{len(col)}")


if __name__ == "__main__":
    main()
