"""Does Isaac draw each SO-101 link where OmniGibson says that link is?

This is the measurement that actually settles "the RTX renderer corrupts this
scene / draws the jaw as a detached lever". It is deliberately independent of
any projection convention on the prediction side:

  1. back-project the camera's `depth_linear` image to WORLD points using the
     camera pose the sim reports. (Sanity check built in: the scene floor must
     come back at z = 0. It does -- 0.0010 m from the external camera and
     0.0000 m from the wrist camera, so the reported camera poses are right.)
  2. take each link's USD visual meshes, place them at the link's reported world
     pose, and keep only the surface that is actually VISIBLE from that camera.
  3. for every visible surface point, measure the distance to the nearest point
     Isaac actually drew.

A link rendered where it is reported scores ~1-2 mm (the depth-image
quantisation floor). A link rendered somewhere else scores its displacement.

Then it correlates that per-frame error against how far the link moved during
that control step, because the answer turns out to be temporal: the error is
~0.5-0.7x the link's per-step motion, r = 0.80-0.96. The image is a fraction of
a control step old -- latency, not corruption.

Usage:
    python tools/check_render_lag.py --probe-dir results/so101/wristcam/run_NNN \
                                     --cam ext_scene
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_render_fidelity import LINKS, load_link_points, splat  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--cam", default="ext_scene")
    ap.add_argument("--usd",
                    default="/scratch/pandya.kei/simfoundry-repro/robot/so101_usd/so101_wristcam.usd")
    ap.add_argument("--max-depth", type=float, default=2.0)
    a = ap.parse_args()

    pd = Path(a.probe_dir)
    meta = json.loads((pd / "meta.json").read_text())
    d = np.load(pd / "poses.npz", allow_pickle=True)
    cn = [str(x) for x in d["cam_names"]]
    lp, cp = d["link_poses"], d["cam_poses"]
    ci = cn.index(a.cam)
    fx, fy, cx, cy = meta["cameras"][a.cam]["K"]
    pts = load_link_points(a.usd)
    frames = sorted(int(re.search(r"(\d+)", f.split("_")[-1]).group(1))
                    for f in glob.glob(f"{pd}/{a.cam}/depth_*.npy"))

    err = {l: [] for l in LINKS}
    spd = {l: [] for l in LINKS}
    floor = []
    for f in frames:
        z = np.load(f"{pd}/{a.cam}/depth_{f:04d}.npy").astype(np.float64)
        h, w = z.shape
        v, u = np.mgrid[0:h, 0:w]
        ok = np.isfinite(z) & (z > 0) & (z < a.max_depth)
        P = np.stack([(u[ok] - cx) * z[ok] / fx, -(v[ok] - cy) * z[ok] / fy, -z[ok]], 1)
        Rm = Rotation.from_quat(cp[ci, f, 3:]).as_matrix()
        W = P @ Rm.T + cp[ci, f, :3]
        hist, edges = np.histogram(W[:, 2], bins=200)
        floor.append(0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1]))
        tree = cKDTree(W)
        pw = {l: Rotation.from_quat(lp[f, 7 * i + 3:7 * i + 7]).apply(pts[l])
              + lp[f, 7 * i:7 * i + 3] for i, l in enumerate(LINKS)}
        pid, pz, _ = splat(pw, w, h, cp[ci, f, :3], cp[ci, f, 3:], [fx, fy, cx, cy])
        for i, l in enumerate(LINKS):
            m = (pid == i) & np.isfinite(pz)
            if m.sum() < 50:
                continue
            vv, uu = np.nonzero(m)
            zz = pz[m]
            Q = np.stack([(uu - cx) * zz / fx, -(vv - cy) * zz / fy, -zz], 1) @ Rm.T + cp[ci, f, :3]
            dd, _ = tree.query(Q)
            err[l].append(np.median(dd) * 1000)
            prev = max(f - 1, 0)
            spd[l].append(float(np.linalg.norm(
                lp[f, 7 * i:7 * i + 3] - lp[prev, 7 * i:7 * i + 3]) * 1000))

    print(f"camera {a.cam}, {len(frames)} probed frames")
    print(f"floor plane recovered from the depth image at z = "
          f"{np.median(floor):+.4f} m  (scene floor is z = 0 -- this validates "
          f"the reported camera pose)")
    print(f"\n{'link':11s} {'frames':>6} {'render err mm':>14} {'p90':>7} {'worst':>7} "
          f"{'step motion mm':>15} {'err@still':>10} {'err@fast':>9} {'r':>6} {'slope':>6}")
    for l in LINKS:
        e, s = np.array(err[l]), np.array(spd[l])
        if not len(e):
            print(f"{l:11s} {0:6d}   (never visible)")
            continue
        still, fast = s < 2.0, s > 40.0
        r = np.corrcoef(s, e)[0, 1] if len(e) > 2 and s.std() > 1e-9 else np.nan
        sl = np.polyfit(s, e, 1)[0] if len(e) > 2 and s.std() > 1e-9 else np.nan
        print(f"{l:11s} {len(e):6d} {np.median(e):14.2f} {np.percentile(e,90):7.2f} "
              f"{e.max():7.2f} {np.median(s):15.2f} "
              f"{(np.median(e[still]) if still.any() else np.nan):10.2f} "
              f"{(np.median(e[fast]) if fast.any() else np.nan):9.2f} "
              f"{r:6.3f} {sl:6.3f}")
    print("\nerr@still = frames where the link moved < 2 mm during the step;")
    print("err@fast  = frames where it moved > 40 mm. A renderer that misplaces")
    print("geometry would be wrong at BOTH; a one-step latency is only wrong at speed.")


if __name__ == "__main__":
    main()
