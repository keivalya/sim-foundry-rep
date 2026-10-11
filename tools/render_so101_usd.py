"""Rasterise the SO-101 USD's VISUAL meshes directly, bypassing Isaac entirely.

Diagnostic: the arm looked like it had missing parts in the Isaac-rendered
video, but there the robot is only ~250 px tall, so aliasing and genuine holes
are indistinguishable. This reads the link transforms and visual meshes straight
out of the USD and renders them big with the same numpy rasteriser used for the
reconstructed scenes, so what you see is exactly what the geometry is.

Reports per-mesh watertightness and winding consistency too, since
doubleSided=False means any inward-facing triangle is culled and reads as a hole.
"""
import sys, os, math, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pxr import Usd, UsdGeom, UsdShade
import trimesh
import sfrender

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--views", type=int, default=4)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

PRED = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
s = Usd.Stage.Open(a.usd, Usd.Stage.LoadAll)
cache = UsdGeom.XformCache(Usd.TimeCode.Default())
LINKS = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw"]
MATCOL = {"material_a_3d_printed": [0.93, 0.80, 0.07],   # yellow PLA
          "material_sts3215": [0.10, 0.10, 0.11]}        # black servo

V, F, C = [], [], []
print(f"{'mesh':48s} {'verts':>8s} {'tris':>8s} {'watertight':>11s} {'winding':>9s}")
for link in LINKS:
    lp = s.GetPrimAtPath(f"/so101_new_calib/{link}")
    for p in Usd.PrimRange(lp, PRED):
        if p.GetTypeName() != "Mesh" or "visuals" not in str(p.GetPath()):
            continue
        m = UsdGeom.Mesh(p)
        pts = np.array(m.GetPointsAttr().Get() or [])
        cnts = np.array(m.GetFaceVertexCountsAttr().Get() or [])
        idx = np.array(m.GetFaceVertexIndicesAttr().Get() or [])
        if not len(pts) or not len(cnts):
            continue
        # triangulate (fan) any non-triangular faces
        tris, o = [], 0
        for c in cnts:
            f = idx[o:o+c]; o += c
            for k in range(1, c-1):
                tris.append([f[0], f[k], f[k+1]])
        tris = np.asarray(tris)
        Mw = np.array(cache.GetLocalToWorldTransform(p)).T
        w = (np.c_[pts, np.ones(len(pts))] @ Mw.T)[:, :3]
        tm = trimesh.Trimesh(vertices=w, faces=tris, process=False)
        mat = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0]
        name = mat.GetPath().name if mat else None
        col = MATCOL.get(name, [0.6, 0.6, 0.62])
        print(f"{str(p.GetPath())[-48:]:48s} {len(pts):8d} {len(tris):8d} "
              f"{str(tm.is_watertight):>11s} {str(tm.is_winding_consistent):>9s}")
        F.append(tris + sum(len(x) for x in V)); V.append(w)
        C.append(np.tile(col, (len(tris), 1)))

V = np.vstack(V); F = np.vstack(F); C = np.vstack(C)
print(f"\ntotal: {len(V)} verts, {len(F)} tris")

# shade with the same model as sfrender, then hand the rasteriser the arrays
n = np.cross(V[F[:, 1]]-V[F[:, 0]], V[F[:, 2]]-V[F[:, 0]])
n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
KEY = np.array([0.45, -0.65, 0.62]); KEY /= np.linalg.norm(KEY)
FILL = np.array([-0.7, 0.35, 0.3]); FILL /= np.linalg.norm(FILL)
hemi = 0.5 + 0.5*n[:, 2]
shade = 0.46 + 0.16*hemi + 0.34*np.clip(n@KEY, 0, 1) + 0.13*np.clip(n@FILL, 0, 1)


class Bare:
    """Minimal stand-in so sfrender.Scene.render can be reused as-is."""
    pass


sc = Bare()
sc.V, sc.F = V, F
sc.RGB = np.clip(C*shade[:, None], 0, 1)
sc.NF_MESH = len(F)          # no floor -> nothing is ever "floor"
sc.tree = None
sc.grid_on = False
sc.vignette = 0.0
sc.shadow_strength = 0.0
sc.shadow_radius = 0.06
sc.floor_z = float(V[:, 2].min())
sc.ctr = (V.min(0)+V.max(0))/2
sc.span = float(max(V.max(0)[0]-V.min(0)[0], V.max(0)[1]-V.min(0)[1]))
sc.render = sfrender.Scene.render.__get__(sc, Bare)

W, H, SS = 900, 1100, 2
lo, hi = V.min(0), V.max(0)
tgt = (lo+hi)/2
d = float(np.linalg.norm(hi-lo))*1.25
for k in range(a.views):
    az = math.radians(30 + k*(360.0/a.views))
    eye = tgt + np.array([d*math.cos(az)*math.cos(math.radians(14)),
                          d*math.sin(az)*math.cos(math.radians(14)),
                          d*math.sin(math.radians(14))])
    R_wc, t = sfrender.look_at(eye, tgt)
    K = sfrender.K_from_fov(36.0, W*SS, H*SS)
    img, _ = sc.render(R_wc, t, K, W*SS, H*SS)
    sfrender.save(img, os.path.join(a.out_dir, f"usd_view_{k}.png"), size=(W, H))
    print(f"  wrote usd_view_{k}.png  az={math.degrees(az):.0f}")
