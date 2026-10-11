"""Render a recorded SO-101 episode offline, from the USD meshes + logged poses.

Isaac's own capture of this scene draws the gripper wrong -- the jaw renders as a
detached lever -- even though every in-sim measurement comes back clean: link
poses correct, all 17 visual meshes at 0.00 mm offset from their links, 0.00 mm
joint drift, and a controlled still at the SAME gripper command renders fully
assembled. Rather than keep fighting that renderer, this rebuilds each frame from
first principles: take the per-frame world pose of every link, transform that
link's USD visual meshes by it, and rasterise. Same numpy rasteriser used for the
reconstructed scenes, so it has no OpenGL dependency and no Isaac involvement.
"""
import os, sys, json, math, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pxr import Usd, UsdGeom, UsdShade
from scipy.spatial.transform import Rotation
from PIL import Image
import trimesh
import sfrender

ap = argparse.ArgumentParser()
ap.add_argument("--poses", required=True)
ap.add_argument("--usd", default="/scratch/pandya.kei/simfoundry-repro/robot/so101_usd/so101.usd")
ap.add_argument("--extract", default="/scratch/pandya.kei/simfoundry-repro/results/reference_scene/extract",
                help="manifest dir for the reconstructed objects")
ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=int, default=15)
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=800)
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--decimate", type=float, default=0.0,
                help="DO NOT USE. Each link's geometry is a CONCATENATION of several "
                     "disconnected parts, and quadric simplification across that is "
                     "not watertight-preserving -- it shredded the arm into loose "
                     "triangles. Backface culling alone is the safe speedup and was "
                     "verified pixel-identical to full resolution.")
ap.add_argument("--only-frame", type=int, default=None,
                help="render just this frame to a PNG, for checking quality")
a = ap.parse_args()

PRED = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
MATCOL = {"material_a_3d_printed": [0.93, 0.80, 0.07],
          "material_sts3215": [0.11, 0.11, 0.12]}
LINKS = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw"]

st = Usd.Stage.Open(a.usd, Usd.Stage.LoadAll)
cache = UsdGeom.XformCache(Usd.TimeCode.Default())

# Per-link geometry, expressed in that LINK's own frame, so a logged world pose
# can place it directly.
link_geom = {}
for l in LINKS:
    lp = st.GetPrimAtPath(f"/so101_new_calib/{l}")
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
    if a.decimate and 0 < a.decimate < 1:
        tm = trimesh.Trimesh(vertices=Vl, faces=Fl, process=False)
        tgt = max(64, int(len(Fl)*a.decimate))
        try:
            sm = tm.simplify_quadric_decimation(face_count=tgt)
            if len(sm.faces) >= 16:
                # nearest-original-face colour, so materials survive decimation
                cent = sm.triangles_center
                from scipy.spatial import cKDTree as _KD
                _, nn = _KD(tm.triangles_center).query(cent)
                Vl, Fl, Cl = np.asarray(sm.vertices), np.asarray(sm.faces), Cl[nn]
        except Exception as e:
            print(f"    (decimation failed for {l}: {e})", flush=True)
    link_geom[l] = (Vl, Fl, Cl)
    print(f"  {l:11s} {len(link_geom[l][0]):7d} verts  {len(link_geom[l][1]):7d} tris", flush=True)

# reconstructed objects, in their own frames
obj_geom = {}
man = json.load(open(os.path.join(a.extract, "manifest.json")))
for o in man["objects"]:
    if not o.get("glb") or o["kind"] == "background" or o["name"] == "robot0":
        continue
    tm = trimesh.load(os.path.join(a.extract, o["glb"]), force="scene")
    V, F, C = [], [], []
    for name, g in tm.geometry.items():
        M = tm.graph.get(tm.graph.geometry_nodes[name][0])[0]
        v = trimesh.transform_points(g.vertices, M) * np.array(o["scale"])
        F.append(np.asarray(g.faces) + sum(len(x) for x in V))
        V.append(v); C.append(sfrender._face_colors(g))
    obj_geom[o.get("category")] = (np.vstack(V), np.vstack(F), np.vstack(C))
    print(f"  obj {o.get('category'):20s} {len(obj_geom[o.get('category')][0]):7d} verts", flush=True)

d = np.load(a.poses, allow_pickle=True)
keys = [str(k) for k in d["keys"]]
poses = d["poses"]
print(f"  {poses.shape[0]} frames, {len(keys)} prims", flush=True)


class Bare:
    pass


def build(fr):
    V, F, C = [], [], []
    for i, k in enumerate(keys):
        kind, name = k.split("::", 1)
        src = link_geom.get(name) if kind == "link" else obj_geom.get(name)
        if src is None:
            continue
        p, q = poses[fr, i, :3], poses[fr, i, 3:7]
        v = Rotation.from_quat(q).apply(src[0]) + p
        F.append(src[1] + sum(len(x) for x in V))
        V.append(v); C.append(src[2])
    return np.vstack(V), np.vstack(F), np.vstack(C)


# frame the whole episode once, so the camera never moves
allpts = np.vstack([build(f)[0] for f in range(0, len(poses), max(1, len(poses)//12))])
lo, hi = allpts.min(0), allpts.max(0)
tgt = (lo+hi)/2
dist = float(np.linalg.norm(hi-lo))*1.00   # 1.35 framed too wide; 0.95 cropped
az, el = math.radians(52), math.radians(20)
eye = tgt + np.array([dist*math.cos(az)*math.cos(el),
                      dist*math.sin(az)*math.cos(el), dist*math.sin(el)])
R_wc, t = sfrender.look_at(eye, tgt)
K = sfrender.K_from_fov(36.0, a.width, a.height)
floor_z = float(lo[2])

# imageio in this env has no ffmpeg backend, so write PNG frames and let the
# cluster's ffmpeg mux them.
import subprocess, tempfile
frame_dir = tempfile.mkdtemp(prefix="so101_frames_")
_n = 0
_frames = ([a.only_frame] if a.only_frame is not None
           else range(0, len(poses), a.stride))
for fr in _frames:
    V, F, C = build(fr)
    n = np.cross(V[F[:, 1]]-V[F[:, 0]], V[F[:, 2]]-V[F[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    KEY = np.array([0.45, -0.65, 0.62]); KEY /= np.linalg.norm(KEY)
    sh = 0.50 + 0.16*(0.5+0.5*n[:, 2]) + 0.34*np.clip(n@KEY, 0, 1)
    # Backface cull: these are closed solids, so inward-facing triangles can
    # never be seen. Halves the per-frame triangle loop at zero visual cost.
    cam_dir = V[F].mean(1) - t
    keep = (n * cam_dir).sum(1) < 0
    F, C, sh, n = F[keep], C[keep], sh[keep], n[keep]
    sc = Bare()
    sc.V, sc.F = V, F
    sc.RGB = np.clip(C*sh[:, None], 0, 1)
    sc.NF_MESH = len(F)
    sc.tree = None; sc.grid_on = False; sc.vignette = 0.0
    sc.shadow_strength = 0.0; sc.shadow_radius = 0.06
    sc.floor_z = floor_z; sc.ctr = tgt; sc.span = float((hi-lo)[:2].max())
    sc.render = sfrender.Scene.render.__get__(sc, Bare)
    img, _ = sc.render(R_wc, t, K, a.width, a.height)
    Image.fromarray((np.clip(img, 0, 1)*255).astype(np.uint8)).save(
        os.path.join(frame_dir, f"f{_n:05d}.png"))
    _n += 1
    if fr % 30 == 0:
        print(f"    frame {fr}/{len(poses)}", flush=True)
if a.only_frame is not None:
    import shutil
    shutil.copy(os.path.join(frame_dir, "f00000.png"), a.out)
    print(f"wrote single frame {a.out}")
    sys.exit(0)

FF = "/shared/EL9/explorer/FFmpeg/7.1.1"
env = dict(os.environ)
env["LD_LIBRARY_PATH"] = FF + "/lib:" + env.get("LD_LIBRARY_PATH", "")
subprocess.run([FF + "/bin/ffmpeg", "-y", "-v", "error", "-framerate", str(a.fps),
                "-i", os.path.join(frame_dir, "f%05d.png"),
                "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "18", a.out],
               check=True, env=env)
print(f"wrote {a.out}  ({_n} frames)")
