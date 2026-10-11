"""Contact sheets for a pi0.5 blue-cube rollout.

Two figures, both built only from frames that were actually recorded:

  live_filmstrip.png   the policy's OWN two input cameras (top_cam, wrist_cam)
                       at N evenly spaced steps, labelled with the step index,
                       the commanded gripper value and the grasp-centre-to-cube
                       distance, so the strip can be read against score.json
                       instead of squinted at.
  real_vs_sim.png      one training frame beside the matching sim top_cam frame,
                       which is the only honest way to judge the appearance gap
                       that decides whether a real-data VLA transfers at all.

    conda_envs/simfoundry-editor/bin/python -I tools/bluecube_sheet.py \
        --run results/so101/vla/bluecube/v2_match
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import so101_kin  # noqa: F401,E402  (kept so the module import path is identical)
from score_bluecube import AG_MOVING, AG_STATIC, xform  # noqa: E402

REF_DIR = "/scratch/pandya.kei/simfoundry-repro/results/so101/vla/bluecube/dataset_ref"


def label(img, lines, pad=4):
    im = img.convert("RGB")
    d = ImageDraw.Draw(im)
    h = 12 * len(lines) + 2 * pad
    d.rectangle([0, 0, im.width, h], fill=(0, 0, 0))
    for i, t in enumerate(lines):
        d.text((pad, pad + 12 * i), t, fill=(255, 255, 255))
    return im


def grid(rows, gap=4, bg=(24, 24, 28)):
    w = max(sum(i.width for i in r) + gap * (len(r) - 1) for r in rows)
    h = sum(max(i.height for i in r) for r in rows) + gap * (len(rows) - 1)
    out = Image.new("RGB", (w, h), bg)
    y = 0
    for r in rows:
        x = 0
        for i in r:
            out.paste(i, (x, y))
            x += i.width + gap
        y += max(i.height for i in r) + gap
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--ref", default=f"{REF_DIR}/ep000_top_cam_00_t0.10s.png")
    a = ap.parse_args()
    run = a.run.rstrip("/")

    tops = sorted(glob.glob(f"{run}/top_cam/rgb_*.png"))
    wrists = sorted(glob.glob(f"{run}/wrist_cam/rgb_*.png"))
    assert tops, f"no top_cam frames in {run}"
    sel = np.linspace(0, len(tops) - 1, a.n).round().astype(int)

    # per-frame annotations, when the pose/episode logs are there
    dist = grip = None
    pz = f"{run}/poses.npz"
    if os.path.exists(pz):
        P = np.load(pz, allow_pickle=True)
        keys = [str(k) for k in P["keys"]]
        po = P["poses"]
        if "obj::blue_cube" in keys:
            i = {k: n for n, k in enumerate(keys)}
            c = 0.5 * (xform(po[:, i["link::gripper"]], AG_STATIC)
                       + xform(po[:, i["link::jaw"]], AG_MOVING))
            dist = np.linalg.norm(c - po[:, i["obj::blue_cube"], :3], axis=1)
    ez = f"{run}/episode.npz"
    if os.path.exists(ez):
        grip = np.load(ez, allow_pickle=True)["action_lerobot"][:, 5]

    rows = [[], []]
    for k in sel:
        step = int(os.path.basename(tops[k]).split("_")[1].split(".")[0])
        lines = [f"step {step}"]
        if grip is not None and step < len(grip):
            lines[0] += f"  jaw cmd {grip[step]:.1f}"
        if dist is not None and step < len(dist):
            lines.append(f"grip->cube {dist[step]*1000:.0f} mm")
        rows[0].append(label(Image.open(tops[k]), ["top_cam  " + lines[0]]
                             + lines[1:]).resize((400, 300)))
        wk = min(k, len(wrists) - 1)
        rows[1].append(label(Image.open(wrists[wk]), ["wrist_cam"]).resize((400, 300)))
    out = f"{run}/live_filmstrip.png"
    grid(rows).save(out)
    print(f"wrote {out}")

    # 2x2: both of the policy's cameras, real beside sim, at the home pose and
    # at the moment the jaw is widest open over the cube.
    pairs = [("ep000_top_cam_00_t0.10s.png", "top_cam", 0, "home"),
             ("ep000_wrist_cam_00_t0.10s.png", "wrist_cam", 0, "home"),
             ("ep000_top_cam_ev_t2.00s_gripopen_over_cube.png", "top_cam", None, "jaw open over the cube"),
             ("ep000_wrist_cam_ev_t2.00s_gripopen_over_cube.png", "wrist_cam", None, "jaw open over the cube")]
    kopen = int(np.nanargmax(grip)) if grip is not None else len(tops) // 4
    rows2 = []
    for fn, cam, fr, what in pairs:
        rp = os.path.join(REF_DIR, fn)
        files = sorted(glob.glob(f"{run}/{cam}/rgb_*.png"))
        k = 0 if fr == 0 else min(kopen, len(files) - 1)
        if not os.path.exists(rp) or not files:
            continue
        rows2.append([
            label(Image.open(rp).convert("RGB").resize((480, 360)),
                  [f"REAL  {cam}  {what}"]),
            label(Image.open(files[k]).convert("RGB").resize((480, 360)),
                  [f"SIM   {cam}  {what} (step {k})"])])
    if rows2:
        out3 = f"{run}/real_vs_sim_both_cams.png"
        grid(rows2).save(out3)
        print(f"wrote {out3}")

    if os.path.exists(a.ref):
        ref = label(Image.open(a.ref).convert("RGB").resize((640, 480)),
                    ["REAL  training set, episode 0 t=0.1 s  (top_cam)"])
        sim = label(Image.open(tops[0]).convert("RGB").resize((640, 480)),
                    ["SIM   this rollout, step 0  (top_cam)"])
        out2 = f"{run}/real_vs_sim.png"
        grid([[ref, sim]]).save(out2)
        print(f"wrote {out2}")
    else:
        print(f"(no reference frame at {a.ref}; skipped real_vs_sim.png)")


if __name__ == "__main__":
    main()
