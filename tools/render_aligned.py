"""Render the reconstructed scene from the ACTUAL camera that filmed it.

Stage 4 writes image_<i>_cam2world.npy -- the pose, in the reconstructed world
frame, of the frame stage 3 selected -- and Depth-Anything-3 exports the
matching per-frame intrinsics. Feeding both to a real pinhole camera puts the
sim render in pixel correspondence with the source frame, so the real-vs-sim
pair actually tests the reconstruction instead of just sitting next to it.

The robot is excluded: stage 14 adds it at a default pose, so it is not in the
real frame and including it would be dishonest in an overlay.
"""
import json, sys, os, argparse, numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sfrender

ap = argparse.ArgumentParser()
ap.add_argument("--extract", required=True)
ap.add_argument("--scene-data", required=True, help="Data/<scene> produced by pipeline A")
ap.add_argument("--out-dir", required=True)
ap.add_argument("--upscale", type=int, default=4)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

sd = a.scene_data.rstrip("/")
sel = json.load(open(f"{sd}/s3_ground/frame_selection.json"))
idx = sel["selected_idx"]
T = np.load(f"{sd}/s4_frame/image_{idx}_cam2world.npy")
R_wc, t = T[:3, :3], T[:3, 3]
npz = np.load(f"{sd}/s2_da/da/exports/npz/results.npz")
K = npz["intrinsics"][idx].astype(float)
real = Image.open(f"{sd}/s3_ground/raw_img.png").convert("RGB")
w0, h0 = real.size
print(f"frame {idx} ({sel['decided_by']}), real {w0}x{h0}")
print(f"  cam at {np.round(t,3)}  fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
      f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

u = a.upscale
w, h = w0 * u, h0 * u
Ku = K.copy(); Ku[:2] *= u
print(f"  rendering {w}x{h} (x{u})")

sc = sfrender.Scene(a.extract, include_robot=False)
# Match the real table: no studio grid, no vignette, and a softer/wider contact
# shadow. The presentation defaults read as heavy dark halos against a photo.
sc.grid_on = False
sc.vignette = 0.0
sc.shadow_strength = 0.30
sc.shadow_radius = 0.035
img, alpha = sc.render(R_wc, t, Ku, w, h)
cover = 100.0 * alpha.mean()
print(f"  geometry covers {cover:.1f}% of frame")

sim = sfrender.save(img, f"{a.out_dir}/aligned_sim.png")
realU = real.resize((w, h), Image.LANCZOS)
realU.save(f"{a.out_dir}/aligned_real.png")

# 50/50 blend: if the reconstruction is geometrically right, silhouettes line up.
blend = Image.blend(realU, sim, 0.5)
blend.save(f"{a.out_dir}/aligned_blend.png")

def font(sz):
    for p in ("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

pad, lab, top = 14, 34, 46
sheet = Image.new("RGB", (3*w + 4*pad, top + h + lab + pad), (255, 255, 255))
dr = ImageDraw.Draw(sheet)
dr.text((pad, 12), "SimFoundry Pipeline A - reference_video.mp4: real vs reconstructed sim, "
                   f"same camera (frame {idx}, DA3 intrinsics)", fill=(20, 20, 20), font=font(21))
for i, (im, cap) in enumerate([(realU, "REAL - source video frame"),
                               (sim, "SIM - reconstructed scene, same camera pose"),
                               (blend, "50/50 overlay")]):
    x = pad + i*(w + pad)
    sheet.paste(im, (x, top))
    dr.rectangle([x, top, x+w-1, top+h-1], outline=(200, 200, 200))
    dr.text((x+3, top+h+7), cap, fill=(70, 70, 70), font=font(17))
sheet.save(f"{a.out_dir}/real_vs_sim_aligned.png")
print(f"  wrote {a.out_dir}/real_vs_sim_aligned.png {sheet.size}")
