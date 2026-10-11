"""Geometry cache for the offline multi-angle renderer, for a PROCEDURAL scene.

tools/episode_geom.load_obj_geom() reads the reconstruction's `manifest.json`
and its per-object .glb files. The blue-cube scene has neither: its objects are
OmniGibson `PrimitiveObject` cubes that exist only as numbers in the scene JSON.
So this writes the same .npz that `episode_geom.cached_geom()` would have
written -- robot link meshes straight from the USD, object meshes generated from
the scene JSON's own `scale` and `rgba` -- and
tools/render_episode_multiview.py picks it up with --geom-cache, unchanged.

Each object's mesh is built in its OWN frame (a box centred on the origin with
half-extents scale/2), which is what Episode.build() expects: it applies the
logged world pose on top.

    conda_envs/simfoundry-editor/bin/python -I tools/make_bluecube_geom_cache.py \
        --scene scene_data/bluecube/bc_small.json \
        --out results/so101/vla/bluecube/v1_small/geom_cache.npz
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import episode_geom as eg  # noqa: E402

# Unit cube centred on the origin: 8 verts, 12 tris, outward winding.
_V = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
               [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float) * 0.5
_F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
               [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
               [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], int)


def box(scale, rgb):
    return (_V * np.asarray(scale, float), _F.copy(),
            np.tile(np.asarray(rgb, float)[:3], (len(_F), 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default="desk",
                    help="comma-separated categories to leave OUT of the cache. "
                         "The default drops the 1.8 m desk slab: the offline "
                         "renderers frame every camera on the bounding box of "
                         "the geometry they are given, and a desk that big "
                         "shrinks the robot to a few dozen pixels in every "
                         "static view. It is backdrop in the live cameras only.")
    ap.add_argument("--usd", default="/scratch/pandya.kei/simfoundry-repro/robot/so101_usd/so101_wristcam_wide.usd")
    a = ap.parse_args()

    drop = {c.strip() for c in a.exclude.split(",") if c.strip()}
    s = json.load(open(a.scene))
    objs = {}
    for name, info in s["objects_info"]["init_info"].items():
        if info.get("class_name") != "PrimitiveObject":
            continue
        args = info["args"]
        cat = args.get("category", name)
        if cat in drop:
            print(f"  {cat:22s} SKIPPED")
            continue
        objs[cat] = box(args["scale"], args["rgba"])
        print(f"  {cat:22s} scale {np.round(args['scale'], 4)}  rgba {np.round(args['rgba'], 3)}")
    assert objs, "no PrimitiveObjects in that scene"

    lg = eg.load_link_geom(a.usd)
    d = {"objnames": np.array(list(objs.keys()))}
    for l, (V, F, C) in lg.items():
        d[f"L/{l}/V"], d[f"L/{l}/F"], d[f"L/{l}/C"] = V, F, C
    for o, (V, F, C) in objs.items():
        d[f"O/{o}/V"], d[f"O/{o}/F"], d[f"O/{o}/C"] = V, F, C
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, **d)
    print(f"wrote {a.out}  ({len(lg)} links, {len(objs)} objects)")


if __name__ == "__main__":
    main()
