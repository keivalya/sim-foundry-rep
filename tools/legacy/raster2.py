"""Pure-numpy z-buffer rasteriser with real texture sampling. No OpenGL."""
import json, os, sys, numpy as np, trimesh
from scipy.spatial.transform import Rotation
from PIL import Image

data_dir, out_png = sys.argv[1], sys.argv[2]
SKIP_BG = os.environ.get("SKIP_BG", "1") == "1"
W, H = 1600, 1000
m = json.load(open(os.path.join(data_dir, "manifest.json")))

def face_colors(g):
    """Average the basecolour texture over each face's UVs."""
    try:
        vis = g.visual
        img = getattr(getattr(vis, "material", None), "baseColorTexture", None)
        uv  = getattr(vis, "uv", None)
        if img is not None and uv is not None and len(uv) == len(g.vertices):
            tex = np.asarray(img.convert("RGB"), dtype=float) / 255.0
            th, tw = tex.shape[:2]
            fuv = uv[g.faces].mean(1)                     # per-face UV centroid
            u = np.clip((fuv[:,0] % 1.0) * (tw-1), 0, tw-1).astype(int)
            v = np.clip((1.0 - (fuv[:,1] % 1.0)) * (th-1), 0, th-1).astype(int)
            return tex[v, u]
        fc = np.asarray(vis.to_color().face_colors)[:, :3].astype(float)/255
        if fc.std() > 1e-6: return fc
    except Exception as e:
        print("   tex fallback:", type(e).__name__, e)
    return np.tile([0.62,0.62,0.66], (len(g.faces),1))

V, F, C, objpts = [], [], [], []
for o in m["objects"]:
    if not o.get("glb"): continue
    if SKIP_BG and o["kind"] == "background":
        print(f"  {o['name']:30s} SKIPPED (background)"); continue
    tm = trimesh.load(os.path.join(data_dir, o["glb"]), force="scene")
    T = np.eye(4)
    T[:3,:3] = Rotation.from_quat(o["orientation"]).as_matrix() * np.array(o["scale"])
    T[:3, 3] = o["position"]
    for name, g in tm.geometry.items():
        M = T @ tm.graph.get(tm.graph.geometry_nodes[name][0])[0]
        v = trimesh.transform_points(g.vertices, M)
        F.append(np.asarray(g.faces) + sum(len(x) for x in V)); V.append(v)
        C.append(face_colors(g)); objpts.append(v)
    print(f"  {o['name']:30s} {o['kind']}")

V = np.vstack(V); F = np.vstack(F); C = np.vstack(C)
print(f"total {len(V)} verts / {len(F)} tris")

allp = np.vstack(objpts); lo, hi = allp.min(0), allp.max(0)
centre = (lo+hi)/2; ext = float(np.linalg.norm(hi-lo)); d = ext*1.15
eye = centre + np.array([d*0.70, -d*0.95, d*0.55])
fwd = centre-eye; fwd/=np.linalg.norm(fwd)
right = np.cross(fwd,[0,0,1.]); right/=np.linalg.norm(right)
up = np.cross(right,fwd); Rm = np.stack([right,up,-fwd])
cam = (V-eye) @ Rm.T; z = -cam[:,2]
f = (H/2)/np.tan(np.deg2rad(42)/2)
x =  cam[:,0]*f/np.maximum(z,1e-6) + W/2
y = -cam[:,1]*f/np.maximum(z,1e-6) + H/2

n = np.cross(V[F[:,1]]-V[F[:,0]], V[F[:,2]]-V[F[:,0]])
n /= np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9)
L = np.array([0.35,-0.55,0.76]); L/=np.linalg.norm(L)
sh = np.clip(np.abs(n@L),0,1)*0.70 + 0.30
rgb = np.clip(C*sh[:,None],0,1)

img = np.full((H,W,3), 0.11); zb = np.full((H,W), np.inf)
zf = z[F]; px, py = x[F], y[F]
ok = (zf>0).all(1)
order = np.argsort(-zf.mean(1)); order = order[ok[order]]
drawn = 0
for i in order:
    tx, ty, tz = px[i], py[i], zf[i]
    x0,x1 = int(max(0,np.floor(tx.min()))), int(min(W-1,np.ceil(tx.max())))
    y0,y1 = int(max(0,np.floor(ty.min()))), int(min(H-1,np.ceil(ty.max())))
    if x1<x0 or y1<y0: continue
    gx,gy = np.meshgrid(np.arange(x0,x1+1), np.arange(y0,y1+1))
    ax,ay,bx,by,cx,cy = tx[0],ty[0],tx[1],ty[1],tx[2],ty[2]
    den = (by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
    if abs(den)<1e-12: continue
    w0 = ((by-cy)*(gx-cx)+(cx-bx)*(gy-cy))/den
    w1 = ((cy-ay)*(gx-cx)+(ax-cx)*(gy-cy))/den
    w2 = 1-w0-w1
    ins = (w0>=0)&(w1>=0)&(w2>=0)
    if not ins.any(): continue
    zz = w0*tz[0]+w1*tz[1]+w2*tz[2]
    sub = zb[y0:y1+1, x0:x1+1]
    upd = ins & (zz < sub)
    if not upd.any(): continue
    sub[upd] = zz[upd]; img[y0:y1+1, x0:x1+1][upd] = rgb[i]; drawn += 1
print("rasterised", drawn, "triangles")
Image.fromarray((img*255).astype(np.uint8)).save(out_png)
print("wrote", out_png)
