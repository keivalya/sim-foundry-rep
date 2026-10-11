Superseded renderers, kept for provenance.

* `raster.py`, `raster2.py` — the first pure-numpy rasterisers. `raster2.py`
  added real texture sampling and produced the early scene renders. Both are
  replaced by `../sfrender.py`, which has a proper pinhole camera, a tessellated
  floor, per-pixel contact shadows and configurable styling.
* `render_scene.py` — a pyrender attempt. It cannot work here: these nodes ship
  the CUDA stack only, with no OpenGL/EGL userspace at all. Kept as the reason
  everything downstream is rasterised in numpy.

Nothing current imports these.
