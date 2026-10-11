"""De-instance NVIDIA's SO-101 USD so OmniGibson can author on its meshes.

The workshop asset attaches each link's geometry as an INSTANCEABLE reference
to an internal prototype (/visuals/<link>, /colliders/<link>). Isaac Lab is fine
with that, but USD instance proxies are read-only: OmniGibson applies physics
materials to finger collision meshes and enumerates mesh children for AABBs
(which our pick-and-place policy reads via aabb_extent/aabb_center), and both
fail on a proxy. Clearing the instanceable flag composes the references into
ordinary, editable prims.
"""
import sys
from pxr import Usd, UsdGeom, UsdPhysics, Gf

src, dst = sys.argv[1], sys.argv[2]
s = Usd.Stage.Open(src, Usd.Stage.LoadAll)

n = 0
for p in s.Traverse():
    if p.IsInstanceable():
        p.SetInstanceable(False)
        n += 1
print(f"cleared instanceable on {n} prims")

# OmniGibson's XformPrim._post_load does th.tensor(get_attribute("xformOp:scale"))
# with no None guard (xform_prim.py:78), so every Xformable needs an EXPLICIT
# scale authored. NVIDIA's asset leaves identity scale unauthored, which is
# legal USD but makes that call raise "Could not infer dtype of NoneType".
from pxr import UsdGeom as _UG, Gf as _Gf
added = 0
for p in s.Traverse():
    x = _UG.Xformable(p)
    if not x:
        continue
    if p.GetAttribute("xformOp:scale").IsValid():
        continue
    # Author the FULL standard triple, not scale alone. Every prim the asset
    # already transforms uses order [translate, orient, scale]; leaving a prim
    # with only xformOp:scale is a half-specified stack, and OmniGibson rewrites
    # xform properties on load, which is the one structural difference between
    # this asset and NVIDIA's original.
    x.AddTranslateOp(_UG.XformOp.PrecisionDouble).Set(_Gf.Vec3d(0.0, 0.0, 0.0))
    x.AddOrientOp(_UG.XformOp.PrecisionFloat).Set(_Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    x.AddScaleOp(_UG.XformOp.PrecisionFloat).Set(_Gf.Vec3f(1.0, 1.0, 1.0))
    added += 1
print(f"authored identity translate/orient/scale on {added} prims")

# OmniGibson decides visual-vs-collision PER MESH:
#   is_collision = prim.HasAPI(UsdPhysics.CollisionAPI)   (rigid_prim.py:220)
# NVIDIA applies CollisionAPI to the per-link `collisions` GROUP Xform and lets
# PhysX propagate down the subtree. That is valid USD and fine for Isaac Lab,
# but it leaves every Mesh without the API, so OmniGibson classifies all 92
# meshes as visual, finds zero collision meshes, and entity_prim.aabb dies with
# "torch.cat(): expected a non-empty list of Tensors". Push the API down onto
# the meshes and drop it from the group so there is exactly one representation.
groups = [p for p in s.Traverse()
          if p.HasAPI(UsdPhysics.CollisionAPI) and p.GetTypeName() != "Mesh"]
nmesh = 0
for g in groups:
    for d in Usd.PrimRange(g):
        if d.GetTypeName() != "Mesh":
            continue
        UsdPhysics.CollisionAPI.Apply(d)
        mc = UsdPhysics.MeshCollisionAPI.Apply(d)
        mc.CreateApproximationAttr().Set("convexHull")
        nmesh += 1
    g.RemoveAPI(UsdPhysics.CollisionAPI)
print(f"applied CollisionAPI to {nmesh} meshes under {len(groups)} collision groups")

# Replace collision geometry with convex hulls. NVIDIA ships the raw 3D-print
# STLs as colliders: 1,196,652 verts over 17 meshes (the wrist alone is 161,982).
# Isaac Lab copes because it cooks once and caches; OmniGibson additionally runs
# get_mesh_volume_and_com and check_extent_radius_ratio per mesh in Python, and
# the combination did not finish loading the scene inside the 1 h `sharing` cap
# (job 10845375 timed out still in og.Environment()). Hulls are safe here because
# the robot runs with self_collisions=False, so jaw-vs-gripper hull overlap is
# never evaluated, and assisted grasping uses an AG-point ray test rather than
# precise contact geometry. VISUAL meshes are left at full resolution.
import trimesh as _tm
import numpy as _np
before = after = 0
for p in s.Traverse():
    if p.GetTypeName() != "Mesh" or not p.HasAPI(UsdPhysics.CollisionAPI):
        continue
    m = UsdGeom.Mesh(p)
    pts = _np.array(m.GetPointsAttr().Get() or [])
    cnts = _np.array(m.GetFaceVertexCountsAttr().Get() or [])
    idx = _np.array(m.GetFaceVertexIndicesAttr().Get() or [])
    if len(pts) == 0 or len(cnts) == 0 or not (cnts == 3).all():
        continue
    hull = _tm.Trimesh(vertices=pts, faces=idx.reshape(-1, 3), process=False).convex_hull
    before += len(pts); after += len(hull.vertices)
    m.GetPointsAttr().Set([Gf.Vec3f(*v) for v in _np.asarray(hull.vertices, dtype=float)])
    m.GetFaceVertexIndicesAttr().Set(_np.asarray(hull.faces).ravel().tolist())
    m.GetFaceVertexCountsAttr().Set([3] * len(hull.faces))
    for a in ("normals", "primvars:st", "primvars:normals"):
        at = p.GetAttribute(a)
        if at and at.IsValid():
            at.Block()                 # stale: indexed against the old topology
print(f"hulled collision meshes: {before} -> {after} verts "
      f"({100.0*after/max(before,1):.1f}%)")

flat = s.Flatten()
out = Usd.Stage.Open(flat)
out.Export(dst)
print(f"exported {dst}")

# verify
v = Usd.Stage.Open(dst, Usd.Stage.LoadAll)
links = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw"]
print(f"\n{'link':12s} {'vis':>4s} {'coll':>5s}  collision-mesh child paths")
ok = True
for l in links:
    lp = v.GetPrimAtPath(f"/so101_new_calib/{l}")
    vis = [x for x in Usd.PrimRange(lp) if x.GetTypeName() == "Mesh" and "visuals" in str(x.GetPath())]
    col = [x for x in Usd.PrimRange(lp) if x.GetTypeName() == "Mesh" and "collisions" in str(x.GetPath())]
    api = [x for x in col if x.HasAPI(UsdPhysics.CollisionAPI)]
    editable = all(not x.IsInstanceProxy() for x in col)
    ok &= bool(col) and editable and len(api) == len(col)
    print(f"{l:12s} {len(vis):4d} {len(col):5d}  editable={editable} "
          f"CollisionAPI={len(api)}/{len(col)}")
missing = [str(p.GetPath()) for p in v.Traverse()
           if UsdGeom.Xformable(p) and not p.GetAttribute("xformOp:scale").IsValid()]
print(f"\nprims still missing xformOp:scale: {len(missing)}")
for m in missing[:5]:
    print("   ", m)
ok &= not missing
print("\nASSET READY FOR OMNIGIBSON" if ok else "\nSTILL BROKEN")
