"""Pure-numpy z-buffer rasteriser: no OpenGL, no display, no GPU."""
import json, os, sys, numpy as np, trimesh
from scipy.spatial.transform import Rotation
from PIL import Image

data_dir, out_png = sys.argv[1], sys.argv[2]
W, H = 1600, 1000
m = json.load(open(os.path.join(data_dir, "manifest.json")))

V, F, C = [], [], []          # verts, faces, per-face rgb
objpts = []
for o in m["objects"]:
    if not o.get("glb"): continue
    tm = trimesh.load(os.path.join(data_dir, o["glb"]), force="scene")
    T = np.eye(4)
    T[:3,:3] = Rotation.from_quat(o["orientation"]).as_matrix() * np.array(o["scale"])
    T[:3, 3] = o["position"]
    for name, g in tm.geometry.items():
        M = T @ tm.graph.get(tm.graph.geometry_nodes[name][0])[0]
        v = trimesh.transform_points(g.vertices, M)
        try:                       # bake texture -> per-face colour
            col = np.asarray(g.visual.to_color().face_colors)[:, :3].astype(float)/255
        except Exception:
            col = np.tile([0.62,0.62,0.66], (len(g.faces),1))
        F.append(np.asarray(g.faces) + sum(len(x) for x in V)); V.append(v); C.append(col)
        if o["kind"] != "background": objpts.append(v)
    print(f"  {o['name']:30s} {o['kind']}")

V = np.vstack(V); F = np.vstack(F); C = np.vstack(C)
print(f"total {len(V)} verts / {len(F)} tris")

lo, hi = np.vstack(objpts).min(0), np.vstack(objpts).max(0)
centre = (lo+hi)/2; ext = float(np.linalg.norm(hi-lo))
d = ext*1.25
eye = centre + np.array([d*0.75, -d*1.0, d*0.62])
fwd = centre-eye; fwd/=np.linalg.norm(fwd)
right = np.cross(fwd,[0,0,1.]); right/=np.linalg.norm(right)
up = np.cross(right,fwd)
Rm = np.stack([right,up,-fwd])
cam = (V-eye) @ Rm.T                       # camera space, looking down -z
z = -cam[:,2]
f = (H/2)/np.tan(np.deg2rad(45)/2)
x =  cam[:,0]*f/np.maximum(z,1e-6) + W/2
y = -cam[:,1]*f/np.maximum(z,1e-6) + H/2

# flat shading from face normal
n = np.cross(V[F[:,1]]-V[F[:,0]], V[F[:,2]]-V[F[:,0]])
n /= np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9)
L = np.array([0.4,-0.6,0.7]); L/=np.linalg.norm(L)
sh = np.clip(np.abs(n@L),0,1)*0.75 + 0.25
rgb = np.clip(C*sh[:,None],0,1)

img = np.full((H,W,3), 0.10); zb = np.full((H,W), np.inf)
zf = z[F]; keep = (zf>0).all(1)
order = np.argsort(-zf.mean(1))            # far -> near
order = order[keep[order]]
px, py = x[F], y[F]
drawn = 0
for i in order:
    tri_x, tri_y, tri_z = px[i], py[i], zf[i]
    x0,x1 = int(max(0,np.floor(tri_x.min()))), int(min(W-1,np.ceil(tri_x.max())))
    y0,y1 = int(max(0,np.floor(tri_y.min()))), int(min(H-1,np.ceil(tri_y.max())))
    if x1<x0 or y1<y0: continue
    gx,gy = np.meshgrid(np.arange(x0,x1+1), np.arange(y0,y1+1))
    ax,ay = tri_x[0],tri_y[0]; bx,by = tri_x[1],tri_y[1]; cx,cy = tri_x[2],tri_y[2]
    den = (by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
    if abs(den)<1e-12: continue
    w0 = ((by-cy)*(gx-cx)+(cx-bx)*(gy-cy))/den
    w1 = ((cy-ay)*(gx-cx)+(ax-cx)*(gy-cy))/den
    w2 = 1-w0-w1
    inside = (w0>=0)&(w1>=0)&(w2>=0)
    if not inside.any(): continue
    zz = w0*tri_z[0]+w1*tri_z[1]+w2*tri_z[2]
    sub = zb[y0:y1+1, x0:x1+1]
    upd = inside & (zz < sub)
    if not upd.any(): continue
    sub[upd] = zz[upd]
    img[y0:y1+1, x0:x1+1][upd] = rgb[i]
    drawn += 1
print("rasterised", drawn, "triangles")
Image.fromarray((img*255).astype(np.uint8)).save(out_png)
print("wrote", out_png)
