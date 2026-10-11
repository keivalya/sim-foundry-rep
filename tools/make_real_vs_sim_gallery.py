"""Real-vs-sim pairs. Sources:
 - 6 pairs shipped in the repo's own gallery (real .png + sim .gif)
 - 1 pair from OUR reproduction (source video frame + our Isaac Sim render)
The 8 published DROID/YAM asset scenes are deliberately NOT here: that dataset
ships no source imagery, and the example videos are different captures
(PutMarkerInCup.mp4 has a paper cup + scissors; the scene has a mug + marker).
"""
from PIL import Image, ImageDraw, ImageFont, ImageSequence
import numpy as np, os

G = "/scratch/pandya.kei/repos/SimFoundry/docs/gallery"
D = "/scratch/pandya.kei/repos/SimFoundry/Data/fruits_demo"
OUT = "/scratch/pandya.kei/simfoundry-repro/results/gallery_real_vs_sim.png"

def gif_frame(path, frac=0.5):
    g = Image.open(path)
    n = getattr(g, "n_frames", 1)
    g.seek(min(n-1, int(n*frac)))
    return g.convert("RGB")

def crop_sat(im, pad=40):
    """Crop Isaac's near-white gradient backdrop via saturation+darkness."""
    a = np.asarray(im).astype(float)/255.0
    mx, mn = a.max(2), a.min(2)
    sat = np.where(mx>0, (mx-mn)/np.maximum(mx,1e-6), 0)
    mask = (sat > 0.25) | (mx < 0.80)
    ys, xs = np.where(mask)
    if len(xs)==0: return im
    return im.crop((max(0,xs.min()-pad), max(0,ys.min()-pad),
                    min(im.width,xs.max()+pad), min(im.height,ys.max()+pad)))

PAIRS = []
for s in ["bathroom_1","dining_1","home_coffee_4","kitchen_2","outdoor_1","toys_1"]:
    PAIRS.append((s.replace("_"," "), Image.open(f"{G}/{s}_real.png").convert("RGB"),
                  gif_frame(f"{G}/{s}_seq.gif"), "NVIDIA"))
PAIRS.append(("fruits_demo  (our run)",
              Image.open(f"{D}/s3_ground/raw_img.png").convert("RGB"),
              crop_sat(Image.open(f"{D}/s14_og/reconstructed_scene.png").convert("RGB")),
              "OURS"))

PH, GAPin, GAPout, LBL = 230, 8, 20, 30     # pane height, inner/outer gaps, label strip
COLS, HDR = 2, 96

def fit(im, h):
    return im.resize((max(1,int(im.width*h/im.height)), h), Image.LANCZOS)

cells = []
for name, real, sim, who in PAIRS:
    r, s = fit(real, PH), fit(sim, PH)
    w = r.width + GAPin + s.width
    cell = Image.new("RGB", (w, PH+LBL), (248,248,249))
    cell.paste(r, (0,0)); cell.paste(s, (r.width+GAPin, 0))
    cells.append((cell, name, who, r.width))

CW = max(c[0].width for c in cells)
rows = (len(cells)+COLS-1)//COLS
W = COLS*CW + (COLS+1)*GAPout
H = HDR + rows*(PH+LBL+GAPout) + GAPout
canvas = Image.new("RGB", (W,H), (248,248,249))
d = ImageDraw.Draw(canvas)
try:
    f_t = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
    f_s = ImageFont.truetype("DejaVuSans.ttf", 16)
    f_l = ImageFont.truetype("DejaVuSans-Bold.ttf", 17)
    f_p = ImageFont.truetype("DejaVuSans.ttf", 13)
except Exception:
    f_t=f_s=f_l=f_p=ImageFont.load_default()

d.text((GAPout,16), "SimFoundry — real world vs. reconstruction", fill=(18,18,20), font=f_t)
d.text((GAPout,56), "left = real capture   ·   right = simulation.   "
                    "Six pairs are NVIDIA's own published gallery; the last is our reproduction "
                    "of Fruits.mp4 rendered in Isaac Sim.", fill=(95,95,100), font=f_s)

for i,(cell,name,who,rw) in enumerate(cells):
    r,c = divmod(i,COLS)
    x = GAPout + c*(CW+GAPout); y = HDR + r*(PH+LBL+GAPout)
    canvas.paste(cell,(x,y))
    d.text((x+2, y+PH+7), name, fill=(25,25,28), font=f_l)
    col = (175,70,25) if who=="OURS" else (120,120,126)
    tw = d.textlength(who, font=f_p)
    d.text((x+cell.width-tw-2, y+PH+10), who, fill=col, font=f_p)
    d.line([(x+rw+GAPin//2, y),(x+rw+GAPin//2, y+PH)], fill=(210,210,212), width=1)

canvas.save(OUT); print("wrote", OUT, canvas.size)
