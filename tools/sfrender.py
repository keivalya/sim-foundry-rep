"""Pure-numpy scene renderer for SimFoundry extractions.

These nodes have no OpenGL/EGL userspace at all (see explorer-cluster-quirks),
so everything here -- projection, z-buffer, shading, shadows -- is numpy.

Camera model is a real pinhole (fx, fy, cx, cy) with OpenCV axes
(+x right, +y DOWN, +z forward). That matters: it lets us render the sim from
the exact cam2world/intrinsics that Depth-Anything-3 recovered for the frame
the pipeline reconstructed from, which is what makes a real-vs-sim overlay
meaningful rather than decorative.
"""
import json, os, math, numpy as np, trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from PIL import Image


def _face_colors(g):
    """Per-face colour: average the basecolour texture over each face's UVs."""
    try:
        vis = g.visual
        img = getattr(getattr(vis, "material", None), "baseColorTexture", None)
        uv = getattr(vis, "uv", None)
        if img is not None and uv is not None and len(uv) == len(g.vertices):
            tex = np.asarray(img.convert("RGB"), float) / 255.0
            th, tw = tex.shape[:2]
            fuv = uv[g.faces].mean(1)
            u = np.clip((fuv[:, 0] % 1.0) * (tw - 1), 0, tw - 1).astype(int)
            v = np.clip((1.0 - (fuv[:, 1] % 1.0)) * (th - 1), 0, th - 1).astype(int)
            return tex[v, u]
        fc = np.asarray(vis.to_color().face_colors)[:, :3].astype(float) / 255
        if fc.std() > 1e-6:
            return fc
    except Exception:
        pass
    return np.tile([0.70, 0.70, 0.73], (len(g.faces), 1))


class Scene:
    def __init__(self, data_dir, include_robot=True, floor=True, verbose=True):
        m = json.load(open(os.path.join(data_dir, "manifest.json")))
        V, F, C = [], [], []
        self.objpts, self.robotpts = [], []
        for o in m["objects"]:
            if not o.get("glb") or o["kind"] == "background":
                continue
            is_robot = o["name"] == "robot0"
            if is_robot and not include_robot:
                continue
            tm = trimesh.load(os.path.join(data_dir, o["glb"]), force="scene")
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat(o["orientation"]).as_matrix() * np.array(o["scale"])
            T[:3, 3] = o["position"]
            for name, g in tm.geometry.items():
                M = T @ tm.graph.get(tm.graph.geometry_nodes[name][0])[0]
                v = trimesh.transform_points(g.vertices, M)
                F.append(np.asarray(g.faces) + sum(len(x) for x in V))
                V.append(v); C.append(_face_colors(g))
                (self.robotpts if is_robot else self.objpts).append(v)
            if verbose:
                print(f"   {o['name']:8s} {o.get('category','')}", flush=True)

        self.V = np.vstack(V); self.F = np.vstack(F); self.C = np.vstack(C)
        self.NF_MESH = len(self.F)
        allp = np.vstack(self.objpts + self.robotpts)
        self.floor_z = float(np.vstack(self.objpts)[:, 2].min())
        lo, hi = allp.min(0), allp.max(0)
        self.ctr = (lo + hi) / 2
        self.span = float(max(hi[0] - lo[0], hi[1] - lo[1]))
        self.tree = None
        # Floor styling knobs. The orbit/contact-sheet look wants a visible
        # studio floor; an aligned real-vs-sim overlay wants none of it, because
        # the real table has no grid and a much softer shadow.
        self.grid_on = True
        self.shadow_strength = 0.62
        self.shadow_radius = 0.06
        self.vignette = 0.30

        if floor:
            # Tessellated, not one big quad: a single quad is dropped wholesale by
            # the near-plane cull because its far corners sit behind the camera.
            r, N = self.span * 2.2, 60
            gx = np.linspace(self.ctr[0]-r, self.ctr[0]+r, N+1)
            gy = np.linspace(self.ctr[1]-r, self.ctr[1]+r, N+1)
            GX, GY = np.meshgrid(gx, gy, indexing="ij")
            gv = np.stack([GX.ravel(), GY.ravel(), np.full(GX.size, self.floor_z)], 1)
            idx = np.arange((N+1)**2).reshape(N+1, N+1)
            a, b, c, d = (idx[:-1,:-1].ravel(), idx[1:,:-1].ravel(),
                          idx[1:,1:].ravel(), idx[:-1,1:].ravel())
            gf = np.vstack([np.stack([a,b,c],1), np.stack([a,c,d],1)])
            # Shadow casters: only vertices sitting LOW. Using every vertex
            # smears a contact shadow under the robot's raised arm.
            low = allp[allp[:, 2] < self.floor_z + 0.12]
            self.tree = cKDTree(low[:, :2]) if len(low) else None
            self.V = np.vstack([self.V, gv])
            self.F = np.vstack([self.F, gf + (len(self.V) - len(gv))])
            self.C = np.vstack([self.C, np.tile([0.855,0.845,0.825], (len(gf),1))])
            if verbose:
                print(f"   floor: {len(gf)} tris at z={self.floor_z:.3f}, "
                      f"{len(low)} shadow casters", flush=True)

        n = np.cross(self.V[self.F[:,1]]-self.V[self.F[:,0]],
                     self.V[self.F[:,2]]-self.V[self.F[:,0]])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        KEY = np.array([0.45,-0.65,0.62]); KEY /= np.linalg.norm(KEY)
        FILL = np.array([-0.7,0.35,0.3]); FILL /= np.linalg.norm(FILL)
        hemi = 0.5 + 0.5*n[:,2]                       # sky above, bounce below
        shade = (0.46 + 0.16*hemi + 0.34*np.clip(n@KEY,0,1)
                      + 0.13*np.clip(n@FILL,0,1))
        self.RGB = np.clip(self.C * shade[:,None], 0, 1)

    # ---- rendering ---------------------------------------------------------
    def render(self, R_wc, t, K, w, h, backdrop=True, fast_small=0):
        """R_wc, t: cam2world rotation/translation. K: 3x3 pinhole. OpenCV axes.

        fast_small: OFF (0) by default, so every existing caller is unchanged.
        When set to n, triangles whose screen bbox spans <= n pixels in both axes
        are taken out of the Python per-triangle loop and rasterised in one
        vectorised batch instead. Same coverage test, same z-buffer semantics
        (the z-buffer is order independent for opaque geometry, which is why the
        batch can be resolved separately from the painter-ordered loop), so the
        result is bit-identical -- verified on episode frames, see
        tools/check_fast_raster.py. Typical episode frames are >99% sub-pixel
        triangles, so this is where the speedup comes from."""
        V = self.V
        cm = (V - t) @ R_wc                           # world -> cam (R_wc^T @ p)
        z = cm[:, 2]
        zs = np.where(np.abs(z) < 1e-6, 1e-6, z)
        x = K[0,0]*cm[:,0]/zs + K[0,2]
        y = K[1,1]*cm[:,1]/zs + K[1,2]

        if backdrop:
            img = np.empty((h, w, 3)); g = np.linspace(0.965, 0.855, h)[:,None]
            img[:,:,0] = g; img[:,:,1] = g; img[:,:,2] = np.clip(g*1.01, 0, 1)
        else:
            img = np.zeros((h, w, 3))
        alpha = np.zeros((h, w), bool)
        zb = np.full((h, w), np.inf)
        isfloor = np.zeros((h, w), bool)

        zf = z[self.F]; px, py = x[self.F], y[self.F]
        ok = ((zf > 1e-3).all(1) & (px.max(1) >= 0) & (px.min(1) < w)
              & (py.max(1) >= 0) & (py.min(1) < h))
        order = np.argsort(-zf.mean(1)); order = order[ok[order]]
        small = None
        if fast_small:
            small = ok & ((np.ceil(px.max(1)) - np.floor(px.min(1))) <= fast_small) \
                       & ((np.ceil(py.max(1)) - np.floor(py.min(1))) <= fast_small)
            order = order[~small[order]]
        for i in order:
            tx, ty, tz = px[i], py[i], zf[i]
            x0, x1 = int(max(0, np.floor(tx.min()))), int(min(w-1, np.ceil(tx.max())))
            y0, y1 = int(max(0, np.floor(ty.min()))), int(min(h-1, np.ceil(ty.max())))
            if x1 < x0 or y1 < y0:
                continue
            gxx, gyy = np.meshgrid(np.arange(x0,x1+1), np.arange(y0,y1+1))
            ax,ay,bx,by,cx,cy = tx[0],ty[0],tx[1],ty[1],tx[2],ty[2]
            den = (by-cy)*(ax-cx) + (cx-bx)*(ay-cy)
            if abs(den) < 1e-12:
                continue
            w0 = ((by-cy)*(gxx-cx) + (cx-bx)*(gyy-cy))/den
            w1 = ((cy-ay)*(gxx-cx) + (ax-cx)*(gyy-cy))/den
            w2 = 1-w0-w1
            ins = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not ins.any():
                continue
            zz = w0*tz[0] + w1*tz[1] + w2*tz[2]
            sub = zb[y0:y1+1, x0:x1+1]; upd = ins & (zz < sub)
            if not upd.any():
                continue
            sub[upd] = zz[upd]
            img[y0:y1+1, x0:x1+1][upd] = self.RGB[i]
            alpha[y0:y1+1, x0:x1+1][upd] = True
            isfloor[y0:y1+1, x0:x1+1][upd] = (i >= self.NF_MESH)

        if small is not None and small.any():
            _raster_small(np.nonzero(small)[0], px, py, zf, self.RGB,
                          self.NF_MESH, img, alpha, zb, isfloor, w, h)

        # Floor detail per PIXEL, not baked per triangle: an 11 cm floor cell
        # against a 6 cm shadow falloff gave blocky triangles, and grid lines
        # came out as zigzags. For a plane the per-pixel solve is exact.
        if isfloor.any():
            yy, xx = np.nonzero(isfloor)
            dc = np.stack([(xx-K[0,2])/K[0,0], (yy-K[1,2])/K[1,1], np.ones(len(xx))], 1)
            dw = dc @ R_wc.T
            tt = (self.floor_z - t[2]) / np.where(np.abs(dw[:,2]) < 1e-9, -1e-9, dw[:,2])
            wp = t[None,:2] + dw[:,:2]*tt[:,None]
            mod = np.ones(len(xx))
            if self.tree is not None and self.shadow_strength > 0:
                dist, _ = self.tree.query(wp, workers=-1)
                mod *= 1.0 - self.shadow_strength*np.clip(1.0 - dist/self.shadow_radius, 0, 1)**1.6
            if self.grid_on:
                gl = np.minimum(np.minimum(wp[:,0] % 0.1, 0.1 - wp[:,0] % 0.1),
                                np.minimum(wp[:,1] % 0.1, 0.1 - wp[:,1] % 0.1))
                mod *= 1.0 - 0.045*np.clip(1.0 - gl/0.0016, 0, 1)
            if self.vignette > 0:
                rad = np.linalg.norm(wp - self.ctr[None,:2], axis=1)
                mod *= 1.0 - self.vignette*np.clip((rad - self.span*0.9)/(self.span*1.6), 0, 1)
            img[yy, xx] *= mod[:,None]
        return img, alpha


def _raster_small(S, px, py, zf, RGB, nf_mesh, img, alpha, zb, isfloor, w, h,
                  budget=3_000_000):
    """Batch-rasterise triangles whose screen bbox is only a few pixels across.

    Triangles are grouped into power-of-two window sizes first, so a 1-pixel
    triangle is not evaluated over a 17x17 window just because some other
    triangle in the batch needs one.

    Chunking and grouping are both safe: every chunk z-tests against, and then
    updates, the same z-buffer, and a z-buffer does not care in what order
    opaque fragments arrive. Within a chunk, ties at a pixel are resolved by
    taking the nearest fragment (lexsort by pixel, then depth)."""
    if not len(S):
        return
    imgf, zbf, alf, flf = (img.reshape(-1, 3), zb.ravel(), alpha.ravel(),
                           isfloor.ravel())
    x0 = np.floor(px[S].min(1))
    y0 = np.floor(py[S].min(1))
    need = np.maximum(np.ceil(px[S].max(1)) - x0, np.ceil(py[S].max(1)) - y0) + 1
    tier = 1 << np.ceil(np.log2(np.maximum(need, 1))).astype(int)   # 1,2,4,8,16
    for k in np.unique(tier):
        G = S[tier == k]
        gx0 = x0[tier == k].astype(np.int64)
        gy0 = y0[tier == k].astype(np.int64)
        DX, DY = np.meshgrid(np.arange(k), np.arange(k))
        DX, DY = DX.ravel()[None, :], DY.ravel()[None, :]
        chunk = max(1024, int(budget // (k*k)))
        for c0 in range(0, len(G), chunk):
            T = G[c0:c0+chunk]
            XX = gx0[c0:c0+chunk][:, None] + DX
            YY = gy0[c0:c0+chunk][:, None] + DY
            ax, ay = px[T, 0][:, None], py[T, 0][:, None]
            bx, by = px[T, 1][:, None], py[T, 1][:, None]
            cx, cy = px[T, 2][:, None], py[T, 2][:, None]
            den = (by-cy)*(ax-cx) + (cx-bx)*(ay-cy)
            den = np.where(np.abs(den) < 1e-12, np.nan, den)
            w0 = ((by-cy)*(XX-cx) + (cx-bx)*(YY-cy))/den
            w1 = ((cy-ay)*(XX-cx) + (ax-cx)*(YY-cy))/den
            w2 = 1 - w0 - w1
            ins = ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)
                   & (XX >= 0) & (XX < w) & (YY >= 0) & (YY < h))
            if not ins.any():
                continue
            zz = w0*zf[T, 0][:, None] + w1*zf[T, 1][:, None] + w2*zf[T, 2][:, None]
            ti, pj = np.nonzero(ins)
            flat = YY[ti, pj]*w + XX[ti, pj]
            zs = zz[ti, pj]
            o = np.lexsort((zs, flat))
            fs = flat[o]
            first = np.empty(len(fs), bool)
            first[0] = True
            first[1:] = fs[1:] != fs[:-1]
            pix, zwin, twin = fs[first], zs[o][first], T[ti[o][first]]
            upd = zwin < zbf[pix]
            pix, zwin, twin = pix[upd], zwin[upd], twin[upd]
            zbf[pix] = zwin
            imgf[pix] = RGB[twin]
            alf[pix] = True
            flf[pix] = twin >= nf_mesh


def look_at(eye, target):
    """cam2world R,t in OpenCV axes (+z forward, +y down) for a look-at."""
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, [0,0,1.]); r /= np.linalg.norm(r)
    d = np.cross(f, r)                      # +y down
    return np.stack([r, d, f], 1), np.asarray(eye, float)


def K_from_fov(fov_deg, w, h):
    f = (h/2)/math.tan(math.radians(fov_deg)/2)
    return np.array([[f,0,w/2],[0,f,h/2],[0,0,1.]])


def save(img, path, size=None):
    im = Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8))
    if size:
        im = im.resize(size, Image.LANCZOS)
    im.save(path)
    return im
