"""Contact sheet of all 8 published SimFoundry scenes (rendered earlier by raster2.py)."""
from PIL import Image, ImageDraw, ImageFont
import numpy as np, os

R = "/scratch/pandya.kei/simfoundry-repro/results"
SCENES = [
    ("DROID_droid_desk_serve_fruits.png",      "serve_fruits",      "DROID"),
    ("DROID_droid_desk_put_marker_in_cup.png", "put_marker_in_cup", "DROID"),
    ("DROID_droid_desk_stack_dishware.png",    "stack_dishware",    "DROID"),
    ("DROID_droid_desk_store_marker.png",      "store_marker",      "DROID"),
    ("DROID_droid_desk_throw_away_trash.png",  "throw_away_trash",  "DROID"),
    ("YAM_bowls.png",                          "bowls",             "YAM"),
    ("YAM_pot_on_stove.png",                   "pot_on_stove",      "YAM"),
    ("YAM_stack_dishware.png",                 "stack_dishware",    "YAM"),
]

def crop_content(im, pad=28):
    """These renders are objects on a near-black ground; crop on brightness."""
    a = np.asarray(im.convert("RGB")).astype(float)
    lum = a.mean(2)
    mask = lum > (lum.min() + 14)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return im
    x0, x1 = max(0, xs.min()-pad), min(im.width,  xs.max()+pad)
    y0, y1 = max(0, ys.min()-pad), min(im.height, ys.max()+pad)
    return im.crop((x0, y0, x1, y1))

TW, TH = 480, 320                     # tile content box
COLS, PAD, LBL = 4, 14, 34
HDR = 92

tiles = []
for fn, name, plat in SCENES:
    im = crop_content(Image.open(os.path.join(R, fn)))
    # letterbox into TWxTH preserving aspect
    s = min(TW / im.width, TH / im.height)
    im = im.resize((max(1,int(im.width*s)), max(1,int(im.height*s))), Image.LANCZOS)
    box = Image.new("RGB", (TW, TH), (16, 16, 18))
    box.paste(im, ((TW-im.width)//2, (TH-im.height)//2))
    tiles.append((box, name, plat))

rows = (len(tiles) + COLS - 1) // COLS
W = COLS*TW + (COLS+1)*PAD
H = HDR + rows*(TH+LBL+PAD) + PAD
canvas = Image.new("RGB", (W, H), (248, 248, 249))
d = ImageDraw.Draw(canvas)
try:
    f_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
    f_sub   = ImageFont.truetype("DejaVuSans.ttf", 17)
    f_lbl   = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    f_plat  = ImageFont.truetype("DejaVuSans.ttf", 15)
except Exception:
    f_title = f_sub = f_lbl = f_plat = ImageFont.load_default()

d.text((PAD, 18), "SimFoundry — published example scenes", fill=(18,18,20), font=f_title)
d.text((PAD, 58),
       "nadunRanawaka1/simfoundry-assets  ·  8 scenes  ·  rendered with a pure-numpy "
       "z-buffer rasteriser (these nodes have no OpenGL)",
       fill=(95,95,100), font=f_sub)

PLAT_COL = {"DROID": (30,110,190), "YAM": (170,70,25)}
for i,(box,name,plat) in enumerate(tiles):
    r,c = divmod(i, COLS)
    x = PAD + c*(TW+PAD)
    y = HDR + r*(TH+LBL+PAD)
    canvas.paste(box, (x,y))
    d.text((x+2, y+TH+6), name, fill=(25,25,28), font=f_lbl)
    tw = d.textlength(plat, font=f_plat)
    d.text((x+TW-tw-4, y+TH+8), plat, fill=PLAT_COL[plat], font=f_plat)

out = os.path.join(R, "gallery_published_scenes.png")
canvas.save(out)
print("wrote", out, canvas.size)
