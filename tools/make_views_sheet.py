import os, sys, numpy as np
from PIL import Image, ImageDraw, ImageFont
d, out = sys.argv[1], sys.argv[2]
tiles = [("hero.png","3/4 view"),("scene_1.png","orbit 130 deg"),
         ("scene_2.png","orbit 220 deg"),("scene_3.png","orbit 310 deg"),
         ("scene_top.png","top-down"),("objects_0.png","objects, close")]
def font(sz):
    for p in ("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p): return ImageFont.truetype(p, sz)
    return ImageFont.load_default()
ims=[(Image.open(os.path.join(d,f)).convert("RGB"),c) for f,c in tiles]
w,h=ims[0][0].size; sc=0.62; w,h=int(w*sc),int(h*sc)
pad,lab,top=16,30,56
W=3*w+4*pad; H=top+2*(h+lab)+3*pad
sheet=Image.new("RGB",(W,H),(255,255,255)); dr=ImageDraw.Draw(sheet)
dr.text((pad,14),"SimFoundry Pipeline A - reference_video.mp4 reconstructed scene",
        fill=(20,20,20), font=font(23))
for i,(im,cap) in enumerate(ims):
    r,c=divmod(i,3); x=pad+c*(w+pad); y=top+r*(h+lab+pad)
    sheet.paste(im.resize((w,h),Image.LANCZOS),(x,y))
    dr.rectangle([x,y,x+w-1,y+h-1],outline=(205,205,205))
    dr.text((x+3,y+h+6),cap,fill=(70,70,70),font=font(17))
sheet.save(out); print(out, sheet.size)
