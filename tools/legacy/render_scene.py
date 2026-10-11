import json, os, sys, numpy as np, trimesh, pyrender
from scipy.spatial.transform import Rotation

data_dir = sys.argv[1]
out_png  = sys.argv[2]

m = json.load(open(os.path.join(data_dir, "manifest.json")))
scene = pyrender.Scene(bg_color=[0.12,0.13,0.16,1.0], ambient_light=[0.35]*3)

allpts, objpts = [], []
for o in m["objects"]:
    if not o.get("glb"): continue
    tm = trimesh.load(os.path.join(data_dir, o["glb"]), force="scene")
    T = np.eye(4)
    T[:3,:3] = Rotation.from_quat(o["orientation"]).as_matrix() * np.array(o["scale"])
    T[:3, 3] = o["position"]
    for name, g in tm.geometry.items():
        M = T @ tm.graph.get(tm.graph.geometry_nodes[name][0])[0]
        scene.add(pyrender.Mesh.from_trimesh(g, smooth=False), pose=M)
        pts_ = trimesh.transform_points(g.bounds, M)
        allpts.append(pts_)
        if o["kind"]!="background": objpts.append(pts_)
    print(f"  placed {o['name']:30s} {o['kind']}")

pts = np.vstack(objpts); lo, hi = pts.min(0), pts.max(0)   # frame the tabletop, not the room
centre = (lo+hi)/2; extent = float(np.linalg.norm(hi-lo))
print("scene bounds", np.round(lo,3), np.round(hi,3), "extent", round(extent,3))

cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(45))
# oblique downward view, matching the README gallery framing
d = extent*0.55
eye = centre + np.array([d*0.95, -d*1.15, d*0.75])
fwd = (centre-eye); fwd/=np.linalg.norm(fwd)
right = np.cross(fwd,[0,0,1]); right/=np.linalg.norm(right)
up = np.cross(right,fwd)
P = np.eye(4); P[:3,0]=right; P[:3,1]=up; P[:3,2]=-fwd; P[:3,3]=eye
scene.add(cam, pose=P)
scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=4.0), pose=P)
scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=2.0),
          pose=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,3],[0,0,0,1]],dtype=float))

r = pyrender.OffscreenRenderer(1600, 1000)
color, _ = r.render(scene)
trimesh.util.np.asarray(color)
from PIL import Image; Image.fromarray(color).save(out_png)
print("wrote", out_png)
