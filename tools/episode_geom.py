"""Shared geometry/IO for the offline episode renderers.

Pulled out of render_episode.py so the single-angle renderer and the multi-camera
one cannot drift apart. Nothing here touches a camera: it turns the USD asset +
the extraction manifest into per-link / per-object geometry in each body's OWN
frame, so a logged world pose places it directly.
"""
import os, json
import numpy as np
from pxr import Usd, UsdGeom, UsdShade
from scipy.spatial.transform import Rotation
import trimesh
import sfrender

PRED = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
MATCOL = {"material_a_3d_printed": [0.93, 0.80, 0.07],
          "material_sts3215": [0.11, 0.11, 0.12]}
LINKS = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw"]
ROOT = "/so101_new_calib"

# Verified from the asset (tools/probe output recorded in docs/STATUS.md):
# the gripper link's visual geometry spans gripper-local z = +0.001 .. -0.1044,
# i.e. the jaw pads sit at z ~= -0.101 and the approach axis is local -Z.
GRIPPER_APPROACH = np.array([0.0, 0.0, -1.0])
# The Jaw revolute axis resolves to gripper-local -Y, so the two fingers separate
# in the gripper's X-Z plane: putting image-right on +X shows both of them.
GRIPPER_RIGHT = np.array([1.0, 0.0, 0.0])

# ---- the live wrist sensor ------------------------------------------------
# Extrinsics and intrinsics of the UsdGeom.Camera authored at
# /so101_new_calib/gripper/gripper_cam in robot/so101_usd/so101_wristcam.usd,
# read out of that prim's cam->gripper transform (tools/check_wrist_cam.py
# re-derives these from the asset and fails if they drift).
#
# CONVENTION: a USD camera looks along its own local -Z with +Y up. sfrender is
# OpenCV (+x right, +y DOWN, +z forward). So the USD basis (X, Y, Z) maps to
# (right, down, forward) = (X, -Y, -Z). FORWARD/UP below are stored in USD terms
# (view direction and up), and WristCam does the flip.
#
# The resulting view direction is the gripper's -Z approach axis tilted 45 deg
# toward -Y, i.e. aimed at the jaw pads: the unit vector from the camera to the
# pads at gripper-local (0,0,-0.101) dots to +0.976 with it (and to -0.976 with
# the +Z reading, which is how the convention was checked).
WRIST_CAM_MOUNT = np.array([-0.005, 0.060, -0.062])
WRIST_CAM_FORWARD = np.array([0.0, -0.707106886, -0.707106677])   # USD -Z
WRIST_CAM_UP = np.array([0.0, 0.707106677, -0.707106886])         # USD +Y
# fx, fy, cx, cy, width, height. Taken from that same camera prim:
# focalLength 13.5 mm over a 20.954999923706055 x 15.7162504196167 mm aperture
# at 640x480 gives fx = 640*13.5/ha and fy = 480*13.5/va, i.e. 412.312 px both
# ways (they differ in the 5th decimal only because the apertures are stored as
# float32), for 75.6 deg horizontal x 60.4 deg vertical.
WRIST_CAM_INTRINSICS = (412.31209949773816, 412.3120856288367,
                        320.0, 240.0, 640.0, 480.0)


def camera_basis(forward, up):
    """USD camera (view dir, up) -> OpenCV cam2world basis [right | down | fwd]."""
    f = np.asarray(forward, float); f = f/np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, float)); r = r/np.linalg.norm(r)
    d = np.cross(f, r)                      # +y down, matches sfrender.look_at
    return np.stack([r, d, f], 1)


def K_from_sensor(intr, out_w, out_h):
    """Scale a real sensor's pinhole intrinsics to the render size.

    Scaled by HEIGHT, so the full vertical field of the sensor is kept and the
    wider render aspect only adds margin at the sides -- nothing the real
    camera sees is cropped away. Any principal-point offset is carried over."""
    fx, fy, cx, cy, sw, sh = intr
    s = out_h/sh
    return np.array([[fx*s, 0.0, out_w/2 + (cx - sw/2)*s],
                     [0.0, fy*s, out_h/2 + (cy - sh/2)*s],
                     [0.0, 0.0, 1.0]])


def load_link_geom(usd, decimate=0.0, verbose=True):
    """Per-link visual geometry expressed in that link's own frame."""
    st = Usd.Stage.Open(usd, Usd.Stage.LoadAll)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    out = {}
    for l in LINKS:
        lp = st.GetPrimAtPath(f"{ROOT}/{l}")
        Li = cache.GetLocalToWorldTransform(lp).GetInverse()
        V, F, C = [], [], []
        for p in Usd.PrimRange(lp, PRED):
            if p.GetTypeName() != "Mesh" or "visuals" not in str(p.GetPath()):
                continue
            m = UsdGeom.Mesh(p)
            pts = np.array(m.GetPointsAttr().Get() or [])
            cnts = np.array(m.GetFaceVertexCountsAttr().Get() or [])
            idx = np.array(m.GetFaceVertexIndicesAttr().Get() or [])
            if not len(pts):
                continue
            tris, o = [], 0
            for c in cnts:
                f = idx[o:o+c]; o += c
                for k in range(1, c-1):
                    tris.append([f[0], f[k], f[k+1]])
            M = np.array(cache.GetLocalToWorldTransform(p) * Li).T
            w = (np.c_[pts, np.ones(len(pts))] @ M.T)[:, :3]
            mat = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0]
            col = MATCOL.get(mat.GetPath().name if mat else None, [0.6, 0.6, 0.62])
            F.append(np.asarray(tris) + sum(len(x) for x in V))
            V.append(w); C.append(np.tile(col, (len(tris), 1)))
        Vl, Fl, Cl = np.vstack(V), np.vstack(F), np.vstack(C)
        if decimate and 0 < decimate < 1:
            # Kept only so the flag still resolves; see render_episode.py's help.
            # Each link is a CONCATENATION of disconnected parts and quadric
            # simplification across that shreds the arm. Do not enable.
            tm = trimesh.Trimesh(vertices=Vl, faces=Fl, process=False)
            tgt = max(64, int(len(Fl)*decimate))
            try:
                sm = tm.simplify_quadric_decimation(face_count=tgt)
                if len(sm.faces) >= 16:
                    from scipy.spatial import cKDTree as _KD
                    _, nn = _KD(tm.triangles_center).query(sm.triangles_center)
                    Vl, Fl, Cl = np.asarray(sm.vertices), np.asarray(sm.faces), Cl[nn]
            except Exception as e:
                print(f"    (decimation failed for {l}: {e})", flush=True)
        out[l] = (Vl, Fl, Cl)
        if verbose:
            print(f"  {l:11s} {len(Vl):7d} verts  {len(Fl):7d} tris", flush=True)
    return out


def load_obj_geom(extract, verbose=True):
    """Reconstructed scene objects, in their own frames."""
    out = {}
    man = json.load(open(os.path.join(extract, "manifest.json")))
    for o in man["objects"]:
        if not o.get("glb") or o["kind"] == "background" or o["name"] == "robot0":
            continue
        tm = trimesh.load(os.path.join(extract, o["glb"]), force="scene")
        V, F, C = [], [], []
        for name, g in tm.geometry.items():
            M = tm.graph.get(tm.graph.geometry_nodes[name][0])[0]
            v = trimesh.transform_points(g.vertices, M) * np.array(o["scale"])
            F.append(np.asarray(g.faces) + sum(len(x) for x in V))
            V.append(v); C.append(sfrender._face_colors(g))
        cat = o.get("category")
        out[cat] = (np.vstack(V), np.vstack(F), np.vstack(C))
        if verbose:
            print(f"  obj {cat:20s} {len(out[cat][0]):7d} verts", flush=True)
    return out


class Episode:
    """Logged poses + the geometry they index, with a per-frame world build."""

    def __init__(self, poses_npz, link_geom, obj_geom):
        d = np.load(poses_npz, allow_pickle=True)
        self.keys = [str(k) for k in d["keys"]]
        self.poses = d["poses"]
        self.link_geom, self.obj_geom = link_geom, obj_geom
        self.src = []
        for i, k in enumerate(self.keys):
            kind, name = k.split("::", 1)
            g = link_geom.get(name) if kind == "link" else obj_geom.get(name)
            if g is not None:
                self.src.append((i, g))

    def __len__(self):
        return self.poses.shape[0]

    def pose(self, fr, key):
        i = self.keys.index(key)
        return (self.poses[fr, i, :3],
                Rotation.from_quat(self.poses[fr, i, 3:7]).as_matrix())

    def build(self, fr):
        """World-space (V, F, C) for one frame."""
        V, F, C = [], [], []
        for i, g in self.src:
            p, q = self.poses[fr, i, :3], self.poses[fr, i, 3:7]
            V.append(Rotation.from_quat(q).apply(g[0]) + p)
            F.append(g[1] + sum(len(x) for x in V[:-1]))
            C.append(g[2])
        return np.vstack(V), np.vstack(F), np.vstack(C)

    def bounds(self, samples=12):
        step = max(1, len(self)//samples)
        p = np.vstack([self.build(f)[0] for f in range(0, len(self), step)])
        return p.min(0), p.max(0)


KEY = np.array([0.45, -0.65, 0.62]); KEY /= np.linalg.norm(KEY)


def face_normals(V, F):
    n = np.cross(V[F[:, 1]]-V[F[:, 0]], V[F[:, 2]]-V[F[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)


def base_shade(n):
    """View-INDEPENDENT shading term, so it is computed once per frame, not
    once per camera (same hemi+key model render_episode.py has always used)."""
    return 0.50 + 0.16*(0.5+0.5*n[:, 2]) + 0.34*np.clip(n @ KEY, 0, 1)


def cached_geom(usd, extract, cache):
    """Parse-once cache. Triangulating the USD is a Python loop over ~400k
    faces and costs ~100 s; the result is static, so keep it on disk."""
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        lg = {l: (z[f"L/{l}/V"], z[f"L/{l}/F"], z[f"L/{l}/C"]) for l in LINKS}
        names = [str(x) for x in z["objnames"]]
        og = {o: (z[f"O/{o}/V"], z[f"O/{o}/F"], z[f"O/{o}/C"]) for o in names}
        print(f"  geometry from cache {cache}", flush=True)
        return lg, og
    lg = load_link_geom(usd)
    og = load_obj_geom(extract)
    if cache:
        d = {"objnames": np.array(list(og.keys()))}
        for l, (V, F, C) in lg.items():
            d[f"L/{l}/V"], d[f"L/{l}/F"], d[f"L/{l}/C"] = V, F, C
        for o, (V, F, C) in og.items():
            d[f"O/{o}/V"], d[f"O/{o}/F"], d[f"O/{o}/C"] = V, F, C
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, **d)
        print(f"  cached geometry -> {cache}", flush=True)
    return lg, og
