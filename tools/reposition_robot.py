"""Move the Franka base so the reconstructed objects fall inside its workspace.

Stage 14 drops the robot at a fixed default pose (x=-1.0 here), which in this
scene leaves every object past the 0.855 m nominal reach. This rewrites the
scene state so the base sits a comfortable distance in FRONT of the objects
(+x is the Franka's forward direction at identity orientation), and writes a
matching settled_poses copy so the policy's reach report stays truthful.

Only state.registry.object_registry.robot0.root_link.pos is touched -- that is
the single place the base pose lives (init_info carries no position). The saved
arm_0 IK goal is invalidated, because it was captured relative to the OLD base
and would otherwise yank the arm on the first step.
"""
import json, sys, argparse, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--scene-dir", required=True)
ap.add_argument("--out-suffix", default="_closer")
ap.add_argument("--standoff", type=float, default=0.52,
                help="metres from the object centroid to the base, along -x")
ap.add_argument("--reach", type=float, default=0.855)
ap.add_argument("--min-reach", type=float, default=0.25)
a = ap.parse_args()

sd = a.scene_dir.rstrip("/")
src = f"{sd}/s14_og/reconstructed_og_scene.json"
dst = f"{sd}/s14_og/reconstructed_og_scene{a.out_suffix}.json"
sp_src = f"{sd}/s14_og/settled_poses.json"
sp_dst = f"{sd}/s14_og/settled_poses{a.out_suffix}.json"

s = json.load(open(src))
sp = json.load(open(sp_src))
cat = {k: v.get("args", {}).get("category")
       for k, v in s["objects_info"]["init_info"].items()}
objs = {k: np.asarray(v[0], float) for k, v in sp.items()
        if k != "robot" and cat.get(k)}
assert objs, "no categorised objects in settled_poses"

centroid = np.mean(list(objs.values()), axis=0)
new_base = np.array([centroid[0] - a.standoff, centroid[1], 0.0])

old = np.asarray(s["state"]["registry"]["object_registry"]["robot0"]["root_link"]["pos"], float)
print(f"object centroid : {np.round(centroid,3)}")
print(f"base  {np.round(old,3)} -> {np.round(new_base,3)}  (standoff {a.standoff} m)\n")
print(f"{'object':22s} {'category':20s}  old_d   new_d   verdict")
ok = 0
for k, p in sorted(objs.items(), key=lambda kv: np.linalg.norm(kv[1] - new_base)):
    od, nd = np.linalg.norm(p - old), np.linalg.norm(p - new_base)
    v = "reachable" if a.min_reach < nd < a.reach else ("TOO CLOSE" if nd <= a.min_reach else "OUT OF REACH")
    ok += v == "reachable"
    print(f"{k:22s} {cat[k]:20s}  {od:.3f}   {nd:.3f}   {v}")
assert ok, "repositioning did not bring anything into reach"

s["state"]["registry"]["object_registry"]["robot0"]["root_link"]["pos"] = new_base.tolist()
# Stale: captured against the old base pose.
s["state"]["registry"]["object_registry"]["robot0"]["controllers"]["arm_0"]["goal_is_valid"] = False
json.dump(s, open(dst, "w"))
sp["robot"] = [new_base.tolist(), [0.0, 0.0, 0.0, 1.0]]
json.dump(sp, open(sp_dst, "w"), indent=2)
print(f"\nwrote {dst}\nwrote {sp_dst}\n{ok}/{len(objs)} objects now reachable")
