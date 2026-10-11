"""Measure whether Isaac actually drew the SO-101 where the SO-101 is.

>>> PREFER tools/check_render_lag.py. <<<
This file compares depths in IMAGE space, over the pixels where a link is
predicted to be the nearest surface. That is fine for chunky links but it
badly over-reports error on the thin, self-occluding gripper and jaw: a
predicted silhouette that is a pixel too wide reads as "Isaac drew nothing
here", and this file reported 400 mm for the gripper where the world-space
measurement in check_render_lag.py gives 16 mm. Use it for the IoU /
segmentation cross-check, not for the verdict.


The claim "the RTX renderer corrupts this scene -- it draws the jaw as a
detached floating lever" has to be settled with numbers, not with a look at one
still. This is the measurement.

policies/wrist_cam_probe.py logs, per frame and per camera:
  * `seg_instance_id`  -- every pixel labelled with the PRIM PATH of the visual
    mesh that drew it, so the pixels belonging to each link are known exactly;
  * `depth_linear`     -- distance to the image plane, in metres;
  * the camera's world pose and intrinsics, and every link's world pose.

This script independently predicts, for each frame, which pixels each link
*should* cover: it takes that link's visual meshes straight out of the USD, puts
them at the logged world pose, projects them through the logged camera and
z-buffers them. Then, per link:

  IoU            overlap between Isaac's mask and the predicted mask
  d_centroid     distance between the two masks' centroids, in px and in mm
                 (px converted at the link's median depth, so it is a real
                 physical displacement)
  d_depth        median |Isaac depth - predicted depth| over the overlap

A link rendered 10 cm from where it is cannot hide from d_centroid: at the wrist
camera's fx = 412 px and a 0.15 m range, 10 cm is ~275 px, i.e. wider than the
whole 160 px analysis grid. Agreement is therefore a strong statement, and it is
made on every probed frame, not on one.

Prediction is by vertex splatting rather than triangle rasterisation: the visual
meshes are dense 3D-print STLs, so at the reduced analysis resolution every
covered pixel gets hit, and it is ~100x faster than rasterising 17 full-res
meshes per frame per camera. (It is also immune to the trap in
tools/render_episode.py -- no decimation, so nothing gets shredded.)

Usage:
    python tools/check_render_fidelity.py --probe-dir results/so101/wristcam/run_NNN
"""
import argparse
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation

LINKS = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw"]
PRED = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)


def load_link_points(usd_path):
    """Visual-mesh vertices of each link, expressed in that LINK's own frame."""
    st = Usd.Stage.Open(usd_path, Usd.Stage.LoadAll)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    out = {}
    for l in LINKS:
        lp = st.GetPrimAtPath(f"/so101_new_calib/{l}")
        Li = cache.GetLocalToWorldTransform(lp).GetInverse()
        V = []
        for p in Usd.PrimRange(lp, PRED):
            if p.GetTypeName() != "Mesh" or "visuals" not in str(p.GetPath()):
                continue
            pts = np.array(UsdGeom.Mesh(p).GetPointsAttr().Get() or [])
            if not len(pts):
                continue
            M = np.array(cache.GetLocalToWorldTransform(p) * Li).T
            V.append((np.c_[pts, np.ones(len(pts))] @ M.T)[:, :3])
        out[l] = np.vstack(V)
    return out


def project(pts_w, cam_p, cam_q, K):
    """World points -> (u, v, z) with a USD camera (-Z forward, +Y up)."""
    R = Rotation.from_quat(cam_q).as_matrix()       # columns: right, up, -forward
    d = pts_w - cam_p
    x = d @ R[:, 0]
    y = -(d @ R[:, 1])
    z = -(d @ R[:, 2])
    fx, fy, cx, cy = K
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * x / z + cx
        v = fy * y / z + cy
    return u, v, z


def splat(points_by_link, w, h, cam_p, cam_q, K):
    """Nearest-surface z-buffer by vertex splatting.

    Returns (link index per pixel, depth per pixel, raw-coverage mask).
    """
    idbuf = np.full(h * w, -1, dtype=np.int16)
    zbuf = np.full(h * w, np.inf, dtype=np.float64)
    us, vs, zs, ids = [], [], [], []
    for i, l in enumerate(LINKS):
        u, v, z = project(points_by_link[l], cam_p, cam_q, K)
        ok = np.isfinite(z) & (z > 1e-4)
        us.append(u[ok]); vs.append(v[ok]); zs.append(z[ok])
        ids.append(np.full(ok.sum(), i, dtype=np.int16))
    u = np.concatenate(us); v = np.concatenate(vs)
    z = np.concatenate(zs); i = np.concatenate(ids)
    px = np.round(u).astype(np.int64)
    py = np.round(v).astype(np.int64)
    m = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    px, py, z, i = px[m], py[m], z[m], i[m]
    flat = py * w + px
    order = np.lexsort((z, flat))                    # nearest first within a pixel
    flat, z, i = flat[order], z[order], i[order]
    first = np.ones(len(flat), bool)
    first[1:] = flat[1:] != flat[:-1]
    idbuf[flat[first]] = i[first]
    zbuf[flat[first]] = z[first]
    idbuf = idbuf.reshape(h, w)
    zbuf = zbuf.reshape(h, w)

    # Splatting leaves pin-holes in flat, sparsely-tessellated faces, and through
    # a hole the z-buffer happily accepts a vertex of the part's BACK surface --
    # which reads as "predicted depth much too far" and fakes a disagreement with
    # the renderer. (That bias is invisible to a self-test whose reference is
    # generated by the same splat.) Closing it with a 3x3 min over the depth, and
    # carrying the winning id along, makes the near surface win every pixel.
    # NOTE the returned `cov`: the evaluation mask stays the RAW splat coverage,
    # so this fills holes without dilating the silhouette by a pixel.
    cov = idbuf >= 0
    z4 = np.where(np.isfinite(zbuf), zbuf, np.inf)
    bz, bi = z4.copy(), idbuf.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            sz = np.roll(np.roll(z4, dy, axis=0), dx, axis=1)
            si = np.roll(np.roll(idbuf, dy, axis=0), dx, axis=1)
            take = sz < bz
            bz[take] = sz[take]
            bi[take] = si[take]
    bz[~np.isfinite(bz)] = np.inf
    return np.where(cov, bi, -1).astype(np.int16), bz, cov


def centroid(mask):
    ys, xs = np.nonzero(mask)
    return None if not len(xs) else np.array([xs.mean(), ys.mean()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--usd",
                    default="/scratch/pandya.kei/simfoundry-repro/robot/so101_usd/so101_wristcam.usd")
    ap.add_argument("--scale", type=int, default=1,
                    help="analysis downsample factor. Keep at 1: validated against\n                         synthetic data, scale=1 gives IoU 1.000 and 1.5e-8 m\n                         depth error on a perfect render, while scale=4 drops\n                         the clean baseline to IoU ~0.3 purely from sampling.")
    ap.add_argument("--depth-tol", type=float, default=0.005,
                    help="a predicted-link pixel counts as DRAWN WHERE PREDICTED "
                         "if Isaac's linear depth there is within this many metres "
                         "of the depth predicted from the logged link pose")
    ap.add_argument("--iou-fail", type=float, default=0.5)
    ap.add_argument("--centroid-fail-mm", type=float, default=15.0)
    ap.add_argument("--overlay", default=None,
                    help="also write a contact sheet to this PNG: Isaac's own rgb "
                         "with the INDEPENDENTLY predicted per-link masks tinted "
                         "over it, for 6 evenly spaced probed frames of --overlay-cam")
    ap.add_argument("--overlay-cam", default=None)
    a = ap.parse_args()

    pd = Path(a.probe_dir)
    meta = json.loads((pd / "meta.json").read_text())
    d = np.load(pd / "poses.npz", allow_pickle=True)
    link_poses = d["link_poses"]                      # [T, 7*7]
    cam_names = [str(x) for x in d["cam_names"]]
    cam_poses = d["cam_poses"]                        # [C, T, 7]
    pts = load_link_points(a.usd)
    print(f"USD visual verts per link: "
          f"{ {k: len(v) for k, v in pts.items()} }")

    s = a.scale
    rows, drows, report = [], [], {"probe_dir": str(pd), "scale": s, "cameras": {}}
    for ci, cam in enumerate(cam_names):
        cm = meta["cameras"][cam]
        W, H = cm["width"], cm["height"]
        w, h = W // s, H // s
        K = [cm["K"][0] / s, cm["K"][1] / s, cm["K"][2] / s, cm["K"][3] / s]
        recs = [r for r in meta["frames"] if r["cam"] == cam]
        per_link = {l: {"iou": [], "dpx": [], "dmm": [], "ddepth": [],
                        "n_isaac": [], "n_pred": [],
                        "hit": [], "behind": [], "infront": [], "absdz": [],
                        "npx": []} for l in LINKS}
        nseg = 0
        depth_stats = []
        for r in recs:
            f = r["frame"]
            segp = pd / cam / f"segid_{f:04d}.npy"
            depp = pd / cam / f"depth_{f:04d}.npy"
            cp, cq = cam_poses[ci, f, :3], cam_poses[ci, f, 3:7]
            pw = {l: Rotation.from_quat(link_poses[f, 7 * i + 3:7 * i + 7]).apply(pts[l])
                  + link_poses[f, 7 * i:7 * i + 3] for i, l in enumerate(LINKS)}
            pid, pz, _cov = splat(pw, w, h, cp, cq, K)

            if depp.exists():
                di = np.load(depp)[::s, ::s][:h, :w].astype(np.float64)
                m = (pid >= 0) & np.isfinite(di) & (di > 0)
                if m.sum():
                    depth_stats.append(float(np.median(np.abs(di[m] - pz[m]))))
                # THE PRIMARY TEST, and it needs no segmentation. For every pixel
                # where link L should be the nearest surface, is Isaac's measured
                # depth the depth L's own geometry would produce?
                #   hit      |dz| <= tol          -> L really was drawn there
                #   behind   isaac > pred + tol   -> nothing of L there; the ray
                #                                   reached something further away
                #                                   (this is what a detached or
                #                                   missing link looks like)
                #   infront  isaac < pred - tol   -> something nearer occludes it
                #                                   (a scene object, legitimately)
                dz = di - pz
                for li, l in enumerate(LINKS):
                    mm = (pid == li) & np.isfinite(di) & (di > 0)
                    n = int(mm.sum())
                    st = per_link[l]
                    st["npx"].append(n)
                    if n < 20:
                        continue
                    d = dz[mm]
                    st["hit"].append(float((np.abs(d) <= a.depth_tol).mean()))
                    st["behind"].append(float((d > a.depth_tol).mean()))
                    st["infront"].append(float((d < -a.depth_tol).mean()))
                    st["absdz"].append(float(np.median(np.abs(d))))

            if not segp.exists() or "seg_labels" not in r:
                continue
            nseg += 1
            seg = np.load(segp)[::s, ::s][:h, :w]
            lab = r["seg_labels"]
            id2link = {}
            for k, v in lab.items():
                for li, l in enumerate(LINKS):
                    if f"/{l}/" in str(v) or str(v).endswith(f"/{l}"):
                        id2link[int(k)] = li
                        break
            iid = np.full(seg.shape, -1, dtype=np.int16)
            for k, li in id2link.items():
                iid[seg == k] = li
            di = np.load(depp)[::s, ::s][:h, :w].astype(np.float64) if depp.exists() else None
            for li, l in enumerate(LINKS):
                mi, mp = (iid == li), (pid == li)
                st = per_link[l]
                st["n_isaac"].append(int(mi.sum()))
                st["n_pred"].append(int(mp.sum()))
                if not mi.any() and not mp.any():
                    continue
                inter = (mi & mp).sum()
                st["iou"].append(float(inter / max(1, (mi | mp).sum())))
                ci_, cp_ = centroid(mi), centroid(mp)
                if ci_ is not None and cp_ is not None:
                    dpx = float(np.linalg.norm(ci_ - cp_))
                    zmed = float(np.median(pz[mp])) if mp.any() else np.nan
                    st["dpx"].append(dpx)
                    st["dmm"].append(dpx * zmed / K[0] * 1000.0)
                if di is not None and inter:
                    st["ddepth"].append(float(np.median(np.abs(di[mi & mp] - pz[mi & mp]))))

        cr = {"n_frames": len(recs), "n_with_seg": nseg,
              "resolution": [W, H], "analysis_resolution": [w, h],
              "K_full": cm["K"], "fov_deg": cm["fov_deg"],
              "median_depth_err_m": (float(np.median(depth_stats)) if depth_stats else None),
              "p95_depth_err_m": (float(np.percentile(depth_stats, 95)) if depth_stats else None),
              "links": {}}
        for l in LINKS:
            st = per_link[l]
            dep = {"depth_frames": len(st["hit"]),
                   "px_pred_median": float(np.median(st["npx"])) if st["npx"] else 0.0,
                   "hit_frac_median": float(np.median(st["hit"])) if st["hit"] else None,
                   "hit_frac_min": float(np.min(st["hit"])) if st["hit"] else None,
                   "behind_frac_median": float(np.median(st["behind"])) if st["behind"] else None,
                   "behind_frac_max": float(np.max(st["behind"])) if st["behind"] else None,
                   "infront_frac_median": float(np.median(st["infront"])) if st["infront"] else None,
                   "abs_dz_m_median": float(np.median(st["absdz"])) if st["absdz"] else None}
            if not st["iou"]:
                cr["links"][l] = dep
                drows.append((cam, l, dep))
                continue
            dep_extra = dep
            bad = sum(1 for i, v in enumerate(st["iou"])
                      if v < a.iou_fail
                      or (i < len(st["dmm"]) and st["dmm"][i] > a.centroid_fail_mm))
            e = {"frames": len(st["iou"]),
                 "iou_median": float(np.median(st["iou"])),
                 "iou_min": float(np.min(st["iou"])),
                 "d_centroid_px_median": float(np.median(st["dpx"])) if st["dpx"] else None,
                 "d_centroid_px_max": float(np.max(st["dpx"])) if st["dpx"] else None,
                 "d_centroid_mm_median": float(np.median(st["dmm"])) if st["dmm"] else None,
                 "d_centroid_mm_max": float(np.max(st["dmm"])) if st["dmm"] else None,
                 "d_depth_m_median": float(np.median(st["ddepth"])) if st["ddepth"] else None,
                 "px_isaac_median": float(np.median(st["n_isaac"])),
                 "px_pred_median": float(np.median(st["n_pred"])),
                 "bad_frames": bad}
            e.update(dep_extra)
            cr["links"][l] = e
            rows.append((cam, l, e))
            drows.append((cam, l, e))
        report["cameras"][cam] = cr

    def _f(v):
        return -1.0 if v is None else float(v)

    print(f"\n=== DEPTH AGREEMENT (no segmentation needed) ===")
    print(f"{'camera':14s} {'link':11s} {'frm':>4s} {'px/frame':>9s} {'hit%':>7s} "
          f"{'hit% min':>9s} {'BEHIND%':>8s} {'behind max':>11s} {'occl%':>7s} "
          f"{'|dz| m':>9s}")
    for cam, l, e in drows:
        if not e.get("depth_frames"):
            print(f"{cam:14s} {l:11s} {0:4d}   (never visible)")
            continue
        print(f"{cam:14s} {l:11s} {e['depth_frames']:4d} {e['px_pred_median']:9.0f} "
              f"{100*e['hit_frac_median']:7.2f} {100*e['hit_frac_min']:9.2f} "
              f"{100*e['behind_frac_median']:8.2f} {100*e['behind_frac_max']:11.2f} "
              f"{100*e['infront_frac_median']:7.2f} {e['abs_dz_m_median']:9.5f}")
    if not rows:
        out = Path(a.probe_dir) / "fidelity.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {out}  (no seg_instance_id in this run; depth table above "
              f"is the measurement)")

    if rows:
        print(f"\n=== SEGMENTATION CROSS-CHECK ===")
    print(f"{'camera':14s} {'link':11s} {'frm':>4s} {'IoU med':>8s} {'IoU min':>8s} "
          f"{'dC px':>7s} {'dC mm':>7s} {'dC mm max':>10s} {'ddepth m':>9s} "
          f"{'px isaac':>9s} {'px pred':>8s} {'bad':>4s}")
    for cam, l, e in rows:
        print(f"{cam:14s} {l:11s} {e['frames']:4d} {e['iou_median']:8.3f} {e['iou_min']:8.3f} "
              f"{_f(e['d_centroid_px_median']):7.2f} {_f(e['d_centroid_mm_median']):7.2f} "
              f"{_f(e['d_centroid_mm_max']):10.2f} "
              f"{_f(e['d_depth_m_median']):9.4f} "
              f"{e['px_isaac_median']:9.0f} {e['px_pred_median']:8.0f} {e['bad_frames']:4d}")
    for cam, cr in report["cameras"].items():
        print(f"\n{cam}: {cr['n_frames']} probe frames ({cr['n_with_seg']} with segmentation), "
              f"{cr['resolution'][0]}x{cr['resolution'][1]}, "
              f"FOV {cr['fov_deg'][0]:.1f}x{cr['fov_deg'][1]:.1f} deg, "
              f"whole-image median depth error {cr['median_depth_err_m']}")

    out = Path(a.probe_dir) / "fidelity.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")

    if a.overlay:
        from PIL import Image
        cam = a.overlay_cam or cam_names[0]
        ci = cam_names.index(cam)
        cm = meta["cameras"][cam]
        W, H = cm["width"], cm["height"]
        recs = [r for r in meta["frames"] if r["cam"] == cam]
        pick = [recs[i] for i in np.linspace(0, len(recs) - 1, 6).astype(int)]
        # one hue per link, so a link drawn in the wrong place is obvious
        COL = np.array([[255, 80, 80], [255, 180, 60], [240, 240, 60], [90, 220, 90],
                        [80, 200, 255], [150, 120, 255], [255, 90, 220]], float)
        panels = []
        for r in pick:
            f = r["frame"]
            rgb = np.asarray(Image.open(pd / cam / f"rgb_{f:04d}.png").convert("RGB"),
                             dtype=float)
            cp, cq = cam_poses[ci, f, :3], cam_poses[ci, f, 3:7]
            pw = {l: Rotation.from_quat(link_poses[f, 7 * i + 3:7 * i + 7]).apply(pts[l])
                  + link_poses[f, 7 * i:7 * i + 3] for i, l in enumerate(LINKS)}
            pid, _, _ = splat(pw, W, H, cp, cq, cm["K"])
            for li in range(len(LINKS)):
                m = pid == li
                if m.any():
                    rgb[m] = 0.45 * rgb[m] + 0.55 * COL[li]
            panels.append(np.clip(rgb, 0, 255).astype(np.uint8))
        g = np.concatenate([np.concatenate(panels[:3], axis=1),
                            np.concatenate(panels[3:], axis=1)], axis=0)
        Image.fromarray(g).save(a.overlay)
        print(f"wrote overlay {a.overlay} "
              f"(frames {[r['frame'] for r in pick]}, cam {cam})")

    worst = [(_f(e["d_centroid_mm_max"]), cam, l) for cam, l, e in rows]
    worst.sort(reverse=True)
    print("\nworst link displacements (mm):")
    for v, cam, l in worst[:6]:
        print(f"  {v:9.2f}  {cam}/{l}")
    nbad = sum(e["bad_frames"] for _, _, e in rows)
    print(f"\nTOTAL bad (link, frame) pairs: {nbad} out of "
          f"{sum(e['frames'] for _, _, e in rows)}")


if __name__ == "__main__":
    main()
