"""Real-vs-sim comparison, matching the framing of SimFoundry's own Scene Gallery."""
from PIL import Image, ImageDraw, ImageFont
import numpy as np

D = "/scratch/pandya.kei/repos/SimFoundry/Data/fruits_demo"
real = Image.open(f"{D}/s3_ground/raw_img.png").convert("RGB")
sim  = Image.open(f"{D}/s14_og/reconstructed_scene.png").convert("RGB")
print("real:", real.size, " sim:", sim.size)

def autocrop_content(im, pad=70):
    """Isaac renders the scene small on a near-white gradient backdrop, and a
    median-background test fails on that gradient. Crop on SATURATION instead:
    the reconstructed objects are strongly coloured, the backdrop is not.
    The robot is desaturated too, so include dark pixels to keep it in frame."""
    import colorsys
    a = np.asarray(im).astype(float) / 255.0
    mx, mn = a.max(2), a.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    dark = mx < 0.80
    mask = (sat > 0.25) | dark
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return im
    x0, x1 = max(0, xs.min()-pad), min(im.width,  xs.max()+pad)
    y0, y1 = max(0, ys.min()-pad), min(im.height, ys.max()+pad)
    return im.crop((x0, y0, x1, y1))

sim_c = autocrop_content(sim)
print("sim cropped:", sim_c.size)

# match heights
H = 560
def fit(im):
    w = int(im.width * H / im.height)
    return im.resize((w, H), Image.LANCZOS)
real_f, sim_f = fit(real), fit(sim_c)

GAP, TOP, BOT = 24, 54, 44
W = real_f.width + GAP + sim_f.width
canvas = Image.new("RGB", (W, H + TOP + BOT), (250, 250, 250))
canvas.paste(real_f, (0, TOP))
canvas.paste(sim_f, (real_f.width + GAP, TOP))

d = ImageDraw.Draw(canvas)
try:
    ft = ImageFont.truetype("DejaVuSans-Bold.ttf", 25)
    fs = ImageFont.truetype("DejaVuSans.ttf", 16)
except Exception:
    ft = fs = ImageFont.load_default()

d.text((4, 14), "Real world  —  Fruits.mp4, frame 12", fill=(20, 20, 20), font=ft)
d.text((real_f.width + GAP + 4, 14), "SimFoundry reconstruction  —  OmniGibson / Isaac Sim",
       fill=(20, 20, 20), font=ft)
d.text((4, TOP + H + 12),
       "6 objects: teal plate, orange plate, red apple, yellow banana, orange, yellow pear   |   "
       "SAM3 -> FLUX -> Hunyuan3D (res 1536) -> FoundationPose -> CoACD -> PyBullet",
       fill=(90, 90, 90), font=fs)
d.line([(real_f.width + GAP//2, TOP), (real_f.width + GAP//2, TOP + H)], fill=(205, 205, 205), width=2)

out = "/scratch/pandya.kei/simfoundry-repro/results/comparison_real_vs_sim.png"
canvas.save(out)
print("wrote", out, canvas.size)
