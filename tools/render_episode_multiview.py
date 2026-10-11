"""Render one recorded episode from N cameras at once, including a wrist-eye view.

Same first-principles approach as render_episode.py (transform each link's USD
visual meshes by its logged world pose and rasterise in numpy -- these nodes have
no OpenGL/EGL at all), but the expensive per-frame work is done ONCE and shared
across every camera:

    build world geometry -> face normals -> view-independent shading
        \\-> per camera: backface cull + rasterise

Only the cull and the rasterisation are per-camera, which is unavoidable: both
depend on where the camera is.

Cameras are the five static rig positions below plus `wrist`, a camera rigidly
bolted to the `gripper` link and posed from that link's logged pose each frame.
By default it reproduces the extrinsic AND intrinsic of the live sensor authored
at /so101_new_calib/gripper/gripper_cam in robot/so101_usd/so101_wristcam.usd,
so the offline view stands in for what a policy's wrist camera would see;
--wrist-mount / --wrist-forward / --wrist-up / --wrist-intrinsics override it.
That camera looks down the gripper's -Z approach axis tilted 45 deg toward -Y.
The -Z approach axis is not taken on faith: the gripper
link's visual geometry spans local z = +0.001 .. -0.1044 (jaw pads at ~-0.101)
with nothing on the +Z side, and over the recorded episode the point
`gripper_pos + R_gripper @ (0,0,-0.101)` closes to 5 mm of orange_block_toy and
9 mm of red_boat_toy at the grasp frames, while the +Z mirror of it stays ~200 mm
away. The Jaw revolute axis resolves to gripper-local -Y, so image-right is put
on gripper +X and the two fingers separate left/right in frame.

NOT USED, deliberately: --decimate (render_episode.py explains; it shreds the
arm). Backface culling IS used and is the thing that makes this tractable.
"""
import os, sys, math, json, argparse, time, subprocess, shutil
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw, ImageFont
import sfrender
import episode_geom as eg
import episode_floor

REPO = "/scratch/pandya.kei/simfoundry-repro"
FF = "/shared/EL9/explorer/FFmpeg/7.1.1"

# name -> (azimuth deg, elevation deg, distance x bbox-diagonal, target lift, fov)
STATIC_CAMS = {
    "hero":    (52,  20, 1.00, 0.50, 36.0),   # same framing as render_episode.py
    # 1.05 cropped the base plate off the bottom in both orbits -- the whole
    # link chain and the plate sitting flat on the floor have to stay in frame.
    "orbit_b": (142, 20, 1.22, 0.42, 36.0),
    "orbit_c": (238, 26, 1.22, 0.42, 36.0),
    "top":     (52,  72, 1.15, 0.50, 36.0),
    "closeup": (104, 30, 0.95, 0.35, 34.0),   # framed on the work area, not the rig
}
DEFAULT_ORDER = ["hero", "orbit_b", "orbit_c", "closeup", "top", "wrist"]
LABEL = {"hero": "hero 3/4", "orbit_b": "orbit 142 deg", "orbit_c": "orbit 238 deg",
         "closeup": "work area, close", "top": "top-down", "wrist": "wrist-eye (gripper cam)"}


def font(sz, bold=True):
    for p in (f"/usr/share/fonts/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
              f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def orbit_pose(tgt, dist, az_deg, el_deg):
    az, el = math.radians(az_deg), math.radians(el_deg)
    eye = tgt + np.array([dist*math.cos(az)*math.cos(el),
                          dist*math.sin(az)*math.cos(el), dist*math.sin(el)])
    return sfrender.look_at(eye, tgt)


class WristCam:
    """Rigidly attached to the gripper link; re-posed from its logged pose.

    Defaults reproduce the live sensor: the UsdGeom.Camera authored at
    /so101_new_calib/gripper/gripper_cam in robot/so101_usd/so101_wristcam.usd,
    i.e. gripper-local position (-0.005, 0.060, -0.062) looking along
    (0, -0.7071, -0.7071) -- the gripper's -Z approach axis tilted 45 deg toward
    -Y, onto the jaw pads -- with the sensor's own 412.31 px focal length.
    `forward`/`up` are in USD camera terms (view direction, up); the flip to
    sfrender's OpenCV axes is done by eg.camera_basis. tools/check_wrist_cam.py
    re-derives all of this from the asset.

    A camera at the bare link origin pointing straight down -Z is NOT usable:
    the chassis occupies local x -0.035..0.030, y -0.028..0.024, z -0.104..0.001
    and fills the frame with yellow. The authored mount sits clear of it.
    """

    def __init__(self, ep, mount=eg.WRIST_CAM_MOUNT, forward=eg.WRIST_CAM_FORWARD,
                 up=eg.WRIST_CAM_UP, intr=eg.WRIST_CAM_INTRINSICS, fov=0.0,
                 w=0, h=0):
        self.ep = ep
        self.mount = np.asarray(mount, float)
        self.basis = eg.camera_basis(forward, up)
        self.K = (sfrender.K_from_fov(fov, w, h) if fov
                  else eg.K_from_sensor(intr, w, h))
        fovx = 2*math.degrees(math.atan((w/2)/self.K[0, 0]))
        fovy = 2*math.degrees(math.atan((h/2)/self.K[1, 1]))
        print(f"  wrist cam: mount {self.mount}, view dir "
              f"{self.basis[:, 2].round(4)} (gripper frame), "
              f"fx={self.K[0,0]:.2f} fy={self.K[1,1]:.2f} -> "
              f"{fovx:.1f} deg x {fovy:.1f} deg", flush=True)

    def pose(self, fr):
        p, R = self.ep.pose(fr, "link::gripper")
        return R @ self.basis, p + R @ self.mount, self.K


class StaticCam:
    def __init__(self, R_wc, t, K):
        self.R_wc, self.t, self.K = R_wc, t, K

    def pose(self, fr):
        return self.R_wc, self.t, self.K


class Bare:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", default=f"{REPO}/scene_data/reference_demo/pick_place/pick_place_so101_poses.npz")
    ap.add_argument("--usd", default=f"{REPO}/robot/so101_usd/so101.usd")
    ap.add_argument("--extract", default=f"{REPO}/results/reference_scene/extract")
    ap.add_argument("--outdir", default=f"{REPO}/results/so101/multiview")
    ap.add_argument("--cameras", default=",".join(DEFAULT_ORDER))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="render at most N frames (0 = all)")
    ap.add_argument("--floor", dest="floor", action="store_true", default=True)
    ap.add_argument("--no-floor", dest="floor", action="store_false")
    ap.add_argument("--grid-cols", type=int, default=3)
    ap.add_argument("--sheet-frames", default="", help="comma frame indices for the contact sheet")
    ap.add_argument("--only-frame", type=int, default=None,
                    help="render just this frame for every camera, as PNGs, and stop")
    ap.add_argument("--keep-frames", action="store_true", help="do not delete the PNG frame dirs")
    ap.add_argument("--geom-cache", default=f"{REPO}/results/so101/multiview/geom_cache.npz",
                    help="cache the parsed USD/GLB geometry here (the USD parse is ~100 s)")
    ap.add_argument("--fast-small", type=int, default=16,
                    help="sfrender batch-raster threshold in px; 0 disables (see check_fast_raster.py)")
    ap.add_argument("--wrist-mount", default=",".join(map(str, eg.WRIST_CAM_MOUNT)),
                    help="wrist camera mount offset in GRIPPER-LOCAL metres")
    ap.add_argument("--wrist-forward", default=",".join(map(str, eg.WRIST_CAM_FORWARD)),
                    help="wrist camera VIEW DIRECTION in gripper-local coords "
                         "(USD camera convention: this is the camera's local -Z)")
    ap.add_argument("--wrist-up", default=",".join(map(str, eg.WRIST_CAM_UP)),
                    help="wrist camera UP in gripper-local coords (USD local +Y)")
    ap.add_argument("--wrist-intrinsics",
                    default=",".join(map(str, eg.WRIST_CAM_INTRINSICS)),
                    help="live sensor fx,fy,cx,cy,width,height; scaled to the "
                         "render size by height so nothing the sensor sees is cropped")
    ap.add_argument("--wrist-fov", type=float, default=0.0,
                    help="override the sensor intrinsics with a plain vertical FOV")
    ap.add_argument("--title", default="SO-101 pick-and-place",
                    help="caption on the contact sheet")
    ap.add_argument("--skip-render", action="store_true",
                    help="reuse existing frames/ PNGs and only rebuild videos + sheet")
    a = ap.parse_args()

    cams_req = [c for c in a.cameras.split(",") if c]
    os.makedirs(a.outdir, exist_ok=True)
    frame_root = os.path.join(a.outdir, "frames")

    print("loading geometry...", flush=True)
    lg, og = eg.cached_geom(a.usd, a.extract, a.geom_cache)
    ep = eg.Episode(a.poses, lg, og)
    nf = len(ep)
    print(f"  {nf} frames, {len(ep.keys)} prims", flush=True)

    frames = ([a.only_frame] if a.only_frame is not None
              else list(range(a.start, nf, a.stride)))
    if a.limit:
        frames = frames[:a.limit]

    # ---- framing -----------------------------------------------------------
    lo, hi = ep.bounds()
    tgt = (lo + hi) / 2
    diag = float(np.linalg.norm(hi - lo))
    floor_z = float(lo[2])
    span = float((hi - lo)[:2].max())

    # work-area bound: the objects + where the gripper actually goes
    okeys = [k for k in ep.keys if k.startswith("obj::") and k != "obj::agent"]
    wpts = np.vstack([ep.poses[:, ep.keys.index(k), :3] for k in okeys]
                     + [ep.poses[:, ep.keys.index("link::gripper"), :3]])
    wlo, whi = wpts.min(0) - 0.08, wpts.max(0) + 0.08
    wlo[2] = min(wlo[2], floor_z)

    cams = {}
    for name in cams_req:
        if name == "wrist":
            vec = lambda t: [float(v) for v in t.split(",")]
            cams[name] = WristCam(ep, w=a.width, h=a.height, fov=a.wrist_fov,
                                  mount=vec(a.wrist_mount),
                                  forward=vec(a.wrist_forward),
                                  up=vec(a.wrist_up),
                                  intr=vec(a.wrist_intrinsics))
            continue
        az, el, pad, lift, fov = STATIC_CAMS[name]
        if name == "closeup":
            c = (wlo + whi) / 2
            c[2] = wlo[2] + (whi[2] - wlo[2]) * lift
            R_wc, t = orbit_pose(c, float(np.linalg.norm(whi - wlo)) * pad, az, el)
        else:
            c = tgt.copy()
            c[2] = lo[2] + (hi[2] - lo[2]) * lift
            R_wc, t = orbit_pose(c, diag * pad, az, el)
        cams[name] = StaticCam(R_wc, t, sfrender.K_from_fov(fov, a.width, a.height))
    for name in cams_req:
        os.makedirs(os.path.join(frame_root, name), exist_ok=True)

    # The ground plane is composited analytically after rasterising (see
    # tools/episode_floor.py) -- tessellating it costs more than the whole robot.
    # ---- render ------------------------------------------------------------
    t0 = time.time()
    per_cam_t = {c: 0.0 for c in cams_req}
    build_t = 0.0
    if not a.skip_render:
        for i, fr in enumerate(frames):
            tb = time.time()
            V, F, C = ep.build(fr)
            n = eg.face_normals(V, F)
            sh = eg.base_shade(n)
            RGB = np.clip(C * sh[:, None], 0, 1)
            cen = V[F].mean(1)
            if a.floor:
                low = V[V[:, 2] < floor_z + 0.12]
                tree = cKDTree(low[:, :2]) if len(low) else None
            else:
                tree = None
            build_t += time.time() - tb

            for name in cams_req:
                tc = time.time()
                R_wc, t, K = cams[name].pose(fr)
                # Backface cull: these are closed solids, so inward-facing
                # triangles can never be seen. This is the only geometry
                # reduction used -- decimation shreds the arm, see module docs.
                keep = (n * (cen - t)).sum(1) < 0
                sc = Bare()
                sc.V, sc.F = V, F[keep]
                sc.RGB = RGB[keep]
                sc.NF_MESH = len(sc.F)
                sc.tree = None; sc.grid_on = False; sc.vignette = 0.0
                sc.shadow_strength = 0.0; sc.shadow_radius = 0.06
                sc.floor_z = floor_z; sc.ctr = tgt; sc.span = span
                sc.render = sfrender.Scene.render.__get__(sc, Bare)
                img, alpha = sc.render(R_wc, t, K, a.width, a.height,
                                       fast_small=a.fast_small)
                if a.floor:
                    episode_floor.composite(img, alpha, R_wc, t, K, floor_z,
                                            tgt, span, tree=tree)
                Image.fromarray((np.clip(img, 0, 1)*255).astype(np.uint8)).save(
                    os.path.join(frame_root, name, f"f{i:05d}.png"))
                per_cam_t[name] += time.time() - tc
            if i % 10 == 0 or i == len(frames)-1:
                el_ = time.time()-t0
                print(f"    frame {fr:4d}  [{i+1}/{len(frames)}]  "
                      f"{el_:.0f}s elapsed, {el_/(i+1):.2f}s/frame, "
                      f"eta {el_/(i+1)*(len(frames)-i-1)/60:.1f} min", flush=True)

        print(f"  render done in {time.time()-t0:.0f}s "
              f"(geometry build {build_t:.0f}s shared across {len(cams_req)} cameras)", flush=True)
        for name in cams_req:
            print(f"    {name:10s} {per_cam_t[name]:7.0f}s rasterise", flush=True)

    if a.only_frame is not None:
        for name in cams_req:
            dst = os.path.join(a.outdir, f"frame{a.only_frame:05d}_{name}.png")
            shutil.copy(os.path.join(frame_root, name, "f00000.png"), dst)
            print(f"  wrote {dst}")
        return

    # ---- videos ------------------------------------------------------------
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = FF + "/lib:" + env.get("LD_LIBRARY_PATH", "")

    def mux(pattern, out, fps):
        subprocess.run([FF + "/bin/ffmpeg", "-y", "-v", "error", "-framerate", str(fps),
                        "-i", pattern, "-pix_fmt", "yuv420p", "-c:v", "libx264",
                        "-crf", "18", out], check=True, env=env)
        print(f"  wrote {out}", flush=True)

    vids = os.path.join(a.outdir, "videos")
    os.makedirs(vids, exist_ok=True)
    for name in cams_req:
        mux(os.path.join(frame_root, name, "f%05d.png"),
            os.path.join(vids, f"so101_{name}.mp4"), a.fps)

    # ---- grid video --------------------------------------------------------
    cols = a.grid_cols
    rows = int(math.ceil(len(cams_req)/cols))
    tw, th = a.width, a.height
    pad, lab = 6, 26
    GW, GH = cols*tw + (cols+1)*pad, rows*(th+lab) + (rows+1)*pad
    GW += GW % 2; GH += GH % 2
    grid_dir = os.path.join(frame_root, "_grid")
    os.makedirs(grid_dir, exist_ok=True)
    f14 = font(15)
    for i in range(len(frames)):
        g = Image.new("RGB", (GW, GH), (24, 24, 26))
        dr = ImageDraw.Draw(g)
        for j, name in enumerate(cams_req):
            r_, c_ = divmod(j, cols)
            x, y = pad + c_*(tw+pad), pad + r_*(th+lab+pad)
            g.paste(Image.open(os.path.join(frame_root, name, f"f{i:05d}.png")), (x, y))
            dr.text((x+2, y+th+5), LABEL.get(name, name), fill=(215, 215, 215), font=f14)
        g.save(os.path.join(grid_dir, f"f{i:05d}.png"))
    mux(os.path.join(grid_dir, "f%05d.png"),
        os.path.join(vids, f"so101_grid_{cols}x{rows}.mp4"), a.fps)

    # ---- contact sheet -----------------------------------------------------
    if a.sheet_frames:
        want = [int(x) for x in a.sheet_frames.split(",")]
    else:
        want = [frames[int(len(frames)*f)] for f in (0.12, 0.42, 0.78)]
    rowsi = [min(range(len(frames)), key=lambda k: abs(frames[k]-wf)) for wf in want]
    sc_ = 0.46
    tw2, th2 = int(tw*sc_), int(th*sc_)
    pad, lab, top, left = 10, 20, 74, 62
    SW = left + len(cams_req)*(tw2+pad) + pad
    SH = top + len(rowsi)*(th2+lab+pad) + pad
    sheet = Image.new("RGB", (SW, SH), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    # The episode is not necessarily the scripted pick-and-place any more --
    # policies/pi05_closed_loop.py writes the same poses.npz format -- so the
    # caption is a flag, with the old text as the default.
    dr.text((pad, 12), f"{a.title}, one recorded episode from "
            f"{len(cams_req)} camera angles", fill=(20, 20, 20), font=font(21))
    dr.text((pad, 40), f"offline numpy render of {os.path.basename(a.poses)} "
            f"({nf} frames) - tools/render_episode_multiview.py",
            fill=(110, 110, 110), font=font(14, bold=False))
    for ci, name in enumerate(cams_req):
        x = left + ci*(tw2+pad)
        dr.text((x+2, top-16), LABEL.get(name, name), fill=(40, 40, 40), font=font(13))
    for ri, k in enumerate(rowsi):
        y = top + ri*(th2+lab+pad)
        dr.text((6, y+th2//2-8), f"f{frames[k]:03d}", fill=(40, 40, 40), font=font(15))
        for ci, name in enumerate(cams_req):
            x = left + ci*(tw2+pad)
            im = Image.open(os.path.join(frame_root, name, f"f{k:05d}.png"))
            sheet.paste(im.resize((tw2, th2), Image.LANCZOS), (x, y))
            dr.rectangle([x, y, x+tw2-1, y+th2-1], outline=(200, 200, 200))
    sp = os.path.join(a.outdir, "contact_sheet.png")
    sheet.save(sp)
    print(f"  wrote {sp}  {sheet.size}", flush=True)

    # A --skip-render pass re-muxes from existing PNGs, so its timings are
    # meaningless; leave the previous run's render_info.json alone.
    if not a.skip_render:
        json.dump({"cameras": cams_req, "frames_rendered": len(frames),
                   "stride": a.stride, "width": a.width, "height": a.height,
                   "fps": a.fps, "floor": a.floor, "poses": a.poses,
                   "wrist_cam": {"mount": a.wrist_mount, "forward": a.wrist_forward,
                                 "up": a.wrist_up, "intrinsics": a.wrist_intrinsics,
                                 "fov_override": a.wrist_fov},
                   "seconds_total": round(time.time()-t0, 1),
                   "seconds_geometry_build": round(build_t, 1),
                   "seconds_per_camera": {k: round(v, 1)
                                          for k, v in per_cam_t.items()}},
                  open(os.path.join(a.outdir, "render_info.json"), "w"), indent=2)

    if not a.keep_frames:
        shutil.rmtree(grid_dir, ignore_errors=True)
    print(f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
