"""Prove sfrender's fast_small batch rasteriser is pixel-identical to the loop.

Renders real episode frames from several cameras twice -- once with
fast_small=0 (the original per-triangle Python loop, unchanged) and once with
fast_small=N -- and reports the exact pixel difference and the speed ratio.
Also re-checks that backface culling changes nothing, since that claim is the
other thing this pipeline leans on.

    $E tools/check_fast_raster.py [--frames 0,140,300] [--fast-small 16]
"""
import os, sys, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sfrender, episode_geom as eg
import render_episode_multiview as rm

ap = argparse.ArgumentParser()
ap.add_argument("--poses", default=f"{rm.REPO}/scene_data/reference_demo/pick_place/pick_place_so101_poses.npz")
ap.add_argument("--usd", default=f"{rm.REPO}/robot/so101_usd/so101.usd")
ap.add_argument("--extract", default=f"{rm.REPO}/results/reference_scene/extract")
ap.add_argument("--geom-cache", default=f"{rm.REPO}/results/so101/multiview/geom_cache.npz")
ap.add_argument("--frames", default="0,140,300")
ap.add_argument("--fast-small", type=int, default=16)
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=400)
a = ap.parse_args()

lg, og = eg.cached_geom(a.usd, a.extract, a.geom_cache)
ep = eg.Episode(a.poses, lg, og)
lo, hi = ep.bounds()
tgt = (lo+hi)/2; tgt[2] = lo[2] + (hi[2]-lo[2])*0.5
diag = float(np.linalg.norm(hi-lo))

cams = {}
for nm, (az, el, pad, lift, fov) in rm.STATIC_CAMS.items():
    if nm == "closeup":
        continue
    R, t = rm.orbit_pose(tgt, diag*pad, az, el)
    cams[nm] = (R, t, sfrender.K_from_fov(fov, a.width, a.height))
wc = rm.WristCam(ep, w=a.width, h=a.height)


def shot(fr, R_wc, t, K, fast, cull=True):
    V, F, C = ep.build(fr)
    n = eg.face_normals(V, F)
    RGB = np.clip(C*eg.base_shade(n)[:, None], 0, 1)
    if cull:
        keep = (n*(V[F].mean(1)-t)).sum(1) < 0
        F, RGB = F[keep], RGB[keep]
    sc = rm.Bare()
    sc.V, sc.F, sc.RGB = V, F, RGB
    sc.NF_MESH = len(F)
    sc.tree = None; sc.grid_on = False; sc.vignette = 0.0
    sc.shadow_strength = 0.0; sc.shadow_radius = 0.06
    sc.floor_z = float(lo[2]); sc.ctr = tgt; sc.span = float((hi-lo)[:2].max())
    sc.render = sfrender.Scene.render.__get__(sc, rm.Bare)
    t0 = time.time()
    img, al = sc.render(R_wc, t, K, a.width, a.height, fast_small=fast)
    return img, al, time.time()-t0, len(F)


def u8(x):
    return (np.clip(x, 0, 1)*255).astype(np.uint8)


bad = 0
for fr in [int(x) for x in a.frames.split(",")]:
    for nm in list(cams) + ["wrist"]:
        R_wc, t, K = cams[nm] if nm in cams else wc.pose(fr)
        ref, al0, t_ref, ntri = shot(fr, R_wc, t, K, 0)
        fast, al1, t_fast, _ = shot(fr, R_wc, t, K, a.fast_small)
        d = np.abs(ref-fast)
        du = np.abs(u8(ref).astype(int)-u8(fast).astype(int))
        npx = int((du.max(2) > 0).sum())
        bad += npx
        print(f"f{fr:3d} {nm:9s} {ntri:7d} tris  loop {t_ref:6.2f}s -> batch "
              f"{t_fast:5.2f}s ({t_ref/max(t_fast,1e-6):4.1f}x)   "
              f"max float diff {d.max():.3e}  differing 8-bit px {npx} / "
              f"{a.width*a.height}", flush=True)

# backface culling re-check, on the view where it matters most
fr = int(a.frames.split(",")[len(a.frames.split(","))//2])
R_wc, t, K = cams["hero"]
c_on, _, t_on, n_on = shot(fr, R_wc, t, K, a.fast_small, cull=True)
c_off, _, t_off, n_off = shot(fr, R_wc, t, K, a.fast_small, cull=False)
dc = int((np.abs(u8(c_on).astype(int)-u8(c_off).astype(int)).max(2) > 0).sum())
print(f"\nbackface cull f{fr} hero: {n_off} -> {n_on} tris, "
      f"{t_off:.2f}s -> {t_on:.2f}s, differing 8-bit px {dc} / {a.width*a.height}")
print("\nRESULT:", "IDENTICAL" if bad == 0 and dc == 0 else f"DIFFERS ({bad} px batch, {dc} px cull)")
sys.exit(0 if bad == 0 and dc == 0 else 1)
