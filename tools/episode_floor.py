"""Analytic ground plane for the episode renders.

sfrender.Scene tessellates its floor into 7200 triangles. Those are the biggest
triangles in the frame, so they dominate the per-triangle rasteriser loop (they
cost ~2x the whole robot). The floor is a plane, though, and the robot never goes
below it, so every background pixel that the camera ray meets below the horizon
IS floor -- no z-test, no triangles. This does exactly the same grid/shadow/
vignette modulation sfrender does per floor pixel, just without the geometry.
"""
import numpy as np

BASE = np.array([0.855, 0.845, 0.825])


def composite(img, alpha, R_wc, t, K, floor_z, ctr, span, tree=None,
              shade=0.87, shadow=0.55, radius=0.06, grid=True, vignette=0.28,
              extent=2.2, fade=0.14):
    """Paint the ground plane into the un-covered pixels of an sfrender image."""
    h, w = alpha.shape
    yy, xx = np.nonzero(~alpha)
    if not len(yy):
        return img
    dc = np.stack([(xx-K[0, 2])/K[0, 0], (yy-K[1, 2])/K[1, 1], np.ones(len(xx))], 1)
    dw = dc @ R_wc.T
    dz = dw[:, 2]
    tt = (floor_z - t[2]) / np.where(np.abs(dz) < 1e-9, -1e-9, dz)
    hit = tt > 1e-3
    if not hit.any():
        return img
    yy, xx, tt, dw = yy[hit], xx[hit], tt[hit], dw[hit]
    wp = t[None, :2] + dw[:, :2]*tt[:, None]

    R = span*extent
    rad = np.linalg.norm(wp - ctr[None, :2], axis=1)
    # Soft edge instead of sfrender's hard tessellated square: the plane is
    # infinite, so it has to be faded out or it fills the frame to the horizon.
    cov = np.clip((R - rad)/(R*fade), 0, 1)
    keep = cov > 0
    yy, xx, wp, rad, cov = yy[keep], xx[keep], wp[keep], rad[keep], cov[keep]
    if not len(yy):
        return img

    mod = np.ones(len(yy))
    if tree is not None and shadow > 0:
        dist, _ = tree.query(wp, workers=-1)
        mod *= 1.0 - shadow*np.clip(1.0 - dist/radius, 0, 1)**1.6
    if grid:
        gl = np.minimum(np.minimum(wp[:, 0] % 0.1, 0.1 - wp[:, 0] % 0.1),
                        np.minimum(wp[:, 1] % 0.1, 0.1 - wp[:, 1] % 0.1))
        mod *= 1.0 - 0.045*np.clip(1.0 - gl/0.0016, 0, 1)
    if vignette > 0:
        mod *= 1.0 - vignette*np.clip((rad - span*0.9)/(span*1.6), 0, 1)

    col = BASE[None, :]*shade*mod[:, None]
    img[yy, xx] = img[yy, xx]*(1-cov[:, None]) + col*cov[:, None]
    return img
