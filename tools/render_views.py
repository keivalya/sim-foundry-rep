"""Multi-angle presentation renders of a SimFoundry scene extraction.

Thin driver over sfrender (the rasteriser lives there, so this and the aligned
real-vs-sim renderer cannot drift apart).

    render_views.py <extract_dir> <out_dir> [n_orbit_views]
"""
import os, sys, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sfrender

data_dir, out_dir = sys.argv[1], sys.argv[2]
NVIEW = int(sys.argv[3]) if len(sys.argv) > 3 else 4
W, H, SS, FOV = 960, 720, 2, 34.0
os.makedirs(out_dir, exist_ok=True)

sc = sfrender.Scene(data_dir, include_robot=True)


def shoot(name, pts, az, el, pad=1.35, lift=0.35):
    p = np.vstack(pts)
    lo, hi = p.min(0), p.max(0)
    tgt = (lo + hi) / 2
    tgt[2] = lo[2] + (hi[2] - lo[2]) * lift
    d = float(np.linalg.norm(hi - lo)) * pad
    eye = tgt + np.array([d*math.cos(az)*math.cos(el),
                          d*math.sin(az)*math.cos(el),
                          d*math.sin(el)])
    R_wc, t = sfrender.look_at(eye, tgt)
    K = sfrender.K_from_fov(FOV, W*SS, H*SS)
    img, _ = sc.render(R_wc, t, K, W*SS, H*SS)
    sfrender.save(img, os.path.join(out_dir, name), size=(W, H))  # SS -> LANCZOS
    print(f"   wrote {name}  az={math.degrees(az):.0f} el={math.degrees(el):.0f}", flush=True)


both = sc.objpts + sc.robotpts
for k in range(NVIEW):
    shoot(f"scene_{k}.png", both, math.radians(40 + k*(360.0/NVIEW)),
          math.radians(24 if k % 2 == 0 else 34))

# Objects alone: in the full-scene frame they are tiny, because the bound is
# dominated by the robot standing off to one side.
for k in range(2):
    shoot(f"objects_{k}.png", sc.objpts, math.radians(45 + k*140),
          math.radians(28), pad=1.9, lift=0.5)

# At high elevation the footprint projects wider, so the top-down needs MORE
# pad, not less, and no target lift -- otherwise the cup falls out of frame.
shoot("scene_top.png", both, math.radians(40), math.radians(70), pad=1.75, lift=0.0)
shoot("hero.png", both, math.radians(62), math.radians(17), pad=1.25, lift=0.30)
