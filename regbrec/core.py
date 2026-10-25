from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, sparse

MASK_NAMES = ["mean img", "corr img", "max proj", "ROI labels", "ROI weights", "red img"]
N_MASKS = len(MASK_NAMES)
DEFAULT_OVERLAP_THOLD = 0.6
DEFAULT_CONTRAST = 5

STATUS_MATCHED, STATUS_DAY1_ONLY, STATUS_DAY2_ONLY = 0, 1, 2
STATUS_NAMES = {STATUS_MATCHED: "matched", STATUS_DAY1_ONLY: "day1_only", STATUS_DAY2_ONLY: "day2_only"}


def get_lims(I, cont=DEFAULT_CONTRAST):
    I = np.asarray(I, dtype=float).ravel()
    I = I[np.isfinite(I)]
    if I.size == 0:
        return np.array([0.0, 1.0])
    mu = np.median(I)
    lo = I[I < mu + 1e-7]
    hi = I[I > mu - 1e-7]
    sd1 = np.mean(np.abs(lo - mu)) if lo.size else 0.0
    sd2 = 1e-7 + (np.mean(np.abs(hi - mu)) if hi.size else 0.0)
    return mu + cont * np.array([-sd1, sd2])


def slider_to_contrast(value):
    value = min(max(value, 0.001), 0.999)
    return 10.0 * (1.0 - value)


_CROSS = ndimage.generate_binary_structure(2, 1)


def bwperim(mask):
    mask = np.asarray(mask, dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=_CROSS, border_value=0)
    return mask & ~eroded


def fit_affine(moving_pts, fixed_pts):
    mv = np.asarray(moving_pts, dtype=float).reshape(-1, 2)
    fx = np.asarray(fixed_pts, dtype=float).reshape(-1, 2)
    if mv.shape[0] != fx.shape[0]:
        raise ValueError("moving and fixed point counts differ")
    if mv.shape[0] < 3:
        raise ValueError("an affine fit needs at least 3 control-point pairs")
    X = np.column_stack([mv, np.ones(len(mv))])
    if np.linalg.matrix_rank(X) < 3:
        raise ValueError("control points are collinear; add points that are spread out")
    sol, *_ = np.linalg.lstsq(X, fx, rcond=None)
    A = np.eye(3)
    A[:2, :] = sol.T
    return A


def apply_affine(A, pts):
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    return pts @ A[:2, :2].T + A[:2, 2]


def affine_to_matlab(A):
    S = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1.0]])
    return (S @ A @ np.linalg.inv(S)).T


def affine_from_matlab(T):
    S = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1.0]])
    return np.linalg.inv(S) @ np.asarray(T, dtype=float).T @ S


def warp_image(img, A, out_shape=None, fill=0.0):
    img = np.asarray(img, dtype=float)
    if out_shape is None:
        out_shape = img.shape
    Ainv = np.linalg.inv(A)
    M = np.array([[Ainv[1, 1], Ainv[1, 0]], [Ainv[0, 1], Ainv[0, 0]]])
    off = np.array([Ainv[1, 2], Ainv[0, 2]])
    return ndimage.affine_transform(img, M, offset=off, output_shape=tuple(out_shape),
                                    order=1, mode="constant", cval=fill, prefilter=False)


@dataclass
class ROISet:
    dims: tuple
    ids: np.ndarray
    pix: list
    border: list
    centroids: np.ndarray
    _lookup: object = field(default=None, repr=False)

    @property
    def n(self):
        return len(self.ids)

    @property
    def npix(self):
        return np.array([len(p) for p in self.pix])

    def index_of(self, raw_id):
        hit = np.flatnonzero(self.ids == raw_id)
        return int(hit[0]) if hit.size else None

    def matrix(self):
        if self._lookup is None:
            npx = self.dims[0] * self.dims[1]
            rows = np.repeat(np.arange(self.n), self.npix)
            cols = np.concatenate(self.pix) if self.n else np.zeros(0, int)
            self._lookup = sparse.csr_matrix(
                (np.ones(len(cols), dtype=np.float64), (rows, cols)), shape=(self.n, npx))
        return self._lookup

    def rois_at(self, lin_idx):
        col = self.matrix()[:, lin_idx]
        return col.nonzero()[0]

    @staticmethod
    def _from_masks(dims, ids, pix_list):
        ids_out, pix_out, border_out, cents = [], [], [], []
        Ly, Lx = dims
        for rid, lin in zip(ids, pix_list):
            lin = np.unique(np.asarray(lin, dtype=np.int64))
            if lin.size == 0:
                continue
            yy, xx = np.divmod(lin, Lx)
            y0, x0 = yy.min(), xx.min()
            sub = np.zeros((yy.max() - y0 + 1, xx.max() - x0 + 1), bool)
            sub[yy - y0, xx - x0] = True
            per = bwperim(sub)[yy - y0, xx - x0]
            ids_out.append(rid)
            pix_out.append(lin)
            border_out.append(per)
            cents.append([yy.mean(), xx.mean()])
        return ROISet(tuple(dims), np.array(ids_out, dtype=int), pix_out, border_out,
                      np.array(cents, dtype=float).reshape(-1, 2))

    @classmethod
    def from_stat(cls, stat, raw_ids, dims):
        Ly, Lx = dims
        pix = []
        for r in raw_ids:
            y = np.asarray(stat[r]["ypix"]).astype(np.int64)
            x = np.asarray(stat[r]["xpix"]).astype(np.int64)
            ok = (y >= 0) & (y < Ly) & (x >= 0) & (x < Lx)
            pix.append(y[ok] * Lx + x[ok])
        return cls._from_masks(dims, raw_ids, pix)

    def transform(self, A, out_dims):
        Ly, Lx = self.dims
        Ho, Wo = out_dims
        Ainv = np.linalg.inv(A)
        new_pix = []
        new_ids = []
        for rid, lin, per in zip(self.ids, self.pix, self.border):
            inner = lin[~per]
            if inner.size == 0:
                continue
            yy, xx = np.divmod(inner, Lx)
            r0, r1, c0, c1 = yy.min(), yy.max(), xx.min(), xx.max()
            corners = np.array([[c0 - 1, r0 - 1], [c1 + 1, r0 - 1], [c0 - 1, r1 + 1], [c1 + 1, r1 + 1]], float)
            oc = apply_affine(A, corners)
            ox0 = max(int(np.floor(oc[:, 0].min())) - 1, 0)
            ox1 = min(int(np.ceil(oc[:, 0].max())) + 1, Wo - 1)
            oy0 = max(int(np.floor(oc[:, 1].min())) - 1, 0)
            oy1 = min(int(np.ceil(oc[:, 1].max())) + 1, Ho - 1)
            if ox1 < ox0 or oy1 < oy0:
                continue
            gy, gx = np.mgrid[oy0:oy1 + 1, ox0:ox1 + 1]
            gx = gx.ravel().astype(float)
            gy = gy.ravel().astype(float)
            sx = Ainv[0, 0] * gx + Ainv[0, 1] * gy + Ainv[0, 2]
            sy = Ainv[1, 0] * gx + Ainv[1, 1] * gy + Ainv[1, 2]
            valid = (sx >= 0) & (sx <= Lx - 1) & (sy >= 0) & (sy <= Ly - 1)
            pad = 2
            local = np.zeros((r1 - r0 + 1 + 2 * pad, c1 - c0 + 1 + 2 * pad))
            local[yy - r0 + pad, xx - c0 + pad] = 1.0
            vals = ndimage.map_coordinates(local, [sy - r0 + pad, sx - c0 + pad], order=1,
                                           mode="constant", cval=0.0, prefilter=False)
            on = valid & (vals > 0)
            if not on.any():
                continue
            new_ids.append(rid)
            new_pix.append(gy[on].astype(np.int64) * Wo + gx[on].astype(np.int64))
        return ROISet._from_masks(tuple(out_dims), new_ids, new_pix)


def find_plane_dir(path):
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        path = os.path.dirname(path)
    if os.path.isfile(os.path.join(path, "stat.npy")):
        return path
    for sub in ("plane0", os.path.join("suite2p", "plane0")):
        p = os.path.join(path, sub)
        if os.path.isfile(os.path.join(p, "stat.npy")):
            return p
    raise FileNotFoundError(f"no stat.npy found in {path} (or its plane0 / suite2p/plane0 folder)")


def session_label(plane_dir):
    parts = os.path.normpath(plane_dir).split(os.sep)
    while parts and (parts[-1].startswith("plane") or parts[-1] == "suite2p"):
        parts.pop()
    return parts[-1] if parts else "session"


def _embed(img, ops, dims):
    img = np.asarray(img, dtype=float)
    if img.shape == tuple(dims):
        return img
    out = np.zeros(dims)
    yr = ops.get("yrange", [0, dims[0]])
    xr = ops.get("xrange", [0, dims[1]])
    h = min(img.shape[0], yr[1] - yr[0])
    w = min(img.shape[1], xr[1] - xr[0])
    out[yr[0]:yr[0] + h, xr[0]:xr[0] + w] = img[:h, :w]
    return out


def _crop(img, ops):
    yr = ops.get("yrange", [0, img.shape[0]])
    xr = ops.get("xrange", [0, img.shape[1]])
    return img[yr[0]:yr[1], xr[0]:xr[1]]


class Session:

    def __init__(self, path):
        self.plane_dir = find_plane_dir(path)
        self.stat = np.load(os.path.join(self.plane_dir, "stat.npy"), allow_pickle=True)
        self.ops = np.load(os.path.join(self.plane_dir, "ops.npy"), allow_pickle=True).item()
        ic_path = os.path.join(self.plane_dir, "iscell.npy")
        if os.path.isfile(ic_path):
            self.iscell = np.load(ic_path)
        else:
            self.iscell = np.ones((len(self.stat), 2))
        self.dims = (int(self.ops["Ly"]), int(self.ops["Lx"]))
        self.n_total = len(self.stat)
        self.iscell_raw = np.flatnonzero(self.iscell[:, 0] > 0.5)
        self.rois_default = ROISet.from_stat(self.stat, self.iscell_raw, self.dims)
        self._build_images()

    @property
    def name(self):
        return self.plane_dir

    def raw_to_iscell(self, raw):
        hit = np.flatnonzero(self.iscell_raw == raw)
        return int(hit[0]) if hit.size else None

    def _build_images(self):
        ops, dims = self.ops, self.dims
        Ly, Lx = dims
        imgs = np.zeros((2, N_MASKS, Ly, Lx))

        def put(m, raw, proc):
            for p, im in enumerate((raw, proc)):
                if im is not None:
                    imgs[p, m] = _embed(im, ops, dims)

        mean = ops.get("meanImg")
        put(0, mean, ops.get("meanImgE", mean))
        put(1, ops.get("Vcorr"), ops.get("Vcorr"))
        mp = ops.get("sdmov", ops.get("max_proj"))
        put(2, mp, mp)
        labels = np.zeros(dims)
        lam_raw = np.zeros(dims)
        lam_norm = np.zeros(dims)
        for k, s in enumerate(self.stat):
            y = np.asarray(s["ypix"]).astype(int)
            x = np.asarray(s["xpix"]).astype(int)
            ok = (y >= 0) & (y < Ly) & (x >= 0) & (x < Lx)
            lam = np.asarray(s["lam"], float)[ok] if "lam" in s else np.ones(ok.sum())
            y, x = y[ok], x[ok]
            labels[y, x] = k + 1
            lam_raw[y, x] = np.maximum(lam_raw[y, x], lam)
            if lam.size and lam.max() > 0:
                lam_norm[y, x] = np.maximum(lam_norm[y, x], lam / lam.max())
        put(3, labels, labels)
        put(4, lam_raw, lam_norm)
        red = ops.get("meanImg_chan2")
        put(5, red, ops.get("meanImg_chan2_corrected", red))
        self.images_default = imgs
        self.lims = np.zeros((2, N_MASKS, 2))
        for p in range(2):
            for m in range(N_MASKS):
                self.lims[p, m] = self.lims_for(imgs[p, m], m, DEFAULT_CONTRAST)

    def lims_for(self, img, mask, cont):
        crop = _crop(np.asarray(img), self.ops)
        if crop.size == 0:
            crop = np.asarray(img)
        if mask == 3:
            lims = np.array([crop.min(), crop.max()], float)
        else:
            lims = get_lims(crop, cont)
        if not lims[1] > lims[0]:
            lims = np.array([lims[0] - 1, lims[0] + 1])
        return lims


def overlap_matrix(rois1, rois2):
    C = (rois1.matrix() @ rois2.matrix().T).tocsr()
    C.eliminate_zeros()
    p = sparse.diags(1.0 / np.maximum(rois1.npix, 1)) @ C
    return C, p.tocsr()


def detect_overlap(rois1, rois2, thold=DEFAULT_OVERLAP_THOLD):
    _, P = overlap_matrix(rois1, rois2)
    pairs = []
    order1 = np.argsort(rois1.ids, kind="stable")
    for j in order1:
        row = P.getrow(j)
        if row.nnz == 0:
            continue
        cand_pos = row.indices
        cand_p = row.data
        o = np.argsort(rois2.ids[cand_pos], kind="stable")
        cand_pos, cand_p = cand_pos[o], cand_p[o]
        k = int(np.argmax(cand_p))
        p_best = float(cand_p[k])
        id1 = int(rois1.ids[j])
        id2 = int(rois2.ids[cand_pos[k]])
        if p_best > thold:
            taken = [i for i, pr in enumerate(pairs) if pr[1] == id2]
            if not taken:
                pairs.append([id1, id2, p_best])
            elif p_best > pairs[taken[0]][2]:
                pairs = [pr for pr in pairs if pr[1] != id2]
                pairs.append([id1, id2, p_best])
    return pairs


def toggle_pair_at(pairs, rois1, rois2, click_yx):
    rois = (rois1, rois2)
    dims = rois1.dims
    y = int(round(click_yx[0]))
    x = int(round(click_yx[1]))
    if not (0 <= y < dims[0] and 0 <= x < dims[1]) or rois1.n == 0 or rois2.n == 0:
        return pairs, False
    lin = y * dims[1] + x
    selected = []
    for f in range(2):
        for pos in rois[f].rois_at(lin):
            selected.append((f, int(pos)))
    if not selected:
        return pairs, False
    cents = np.array([rois[f].centroids[pos] for f, pos in selected])
    d = np.sqrt(((cents - np.array([click_yx[0], click_yx[1]])) ** 2).sum(1))
    this_file, this_pos = selected[int(np.argmin(d))]
    other_file = 1 - this_file
    this_cent = rois[this_file].centroids[this_pos]
    d2 = np.sqrt(((rois[other_file].centroids - this_cent) ** 2).sum(1))
    cand_pos = int(np.argmin(d2))
    sel_id = int(rois[this_file].ids[this_pos])
    cand_id = int(rois[other_file].ids[cand_pos])
    a = rois[this_file].pix[this_pos]
    b = rois[other_file].pix[cand_pos]
    overlap_p = len(np.intersect1d(a, b)) / len(a)
    if overlap_p <= 0:
        return pairs, False
    pairs = [list(p) for p in pairs]
    in_other = [pr[other_file] == cand_id for pr in pairs]
    in_this = [pr[this_file] == sel_id for pr in pairs]
    if all(io == it for io, it in zip(in_other, in_this)):
        if any(in_other):
            pairs = [pr for pr, it in zip(pairs, in_this) if not it]
        else:
            new = [0, 0, overlap_p]
            new[this_file], new[other_file] = sel_id, cand_id
            pairs.append(new)
    else:
        pairs = [pr for pr, io, it in zip(pairs, in_other, in_this) if not (io or it)]
        new = [0, 0, overlap_p]
        new[this_file], new[other_file] = sel_id, cand_id
        pairs.append(new)
    return pairs, True


def _prep_for_reg(img, sigma_hp=10.0):
    img = np.nan_to_num(np.asarray(img, dtype=np.float64))
    lo, hi = np.percentile(img, [1, 99.5])
    img = np.clip(img, lo, hi)
    img = img - ndimage.gaussian_filter(img, sigma_hp)
    sd = img.std()
    return (img - img.mean()) / (sd if sd > 0 else 1.0)


def image_correlation(fixed, moving_warped, valid=None):
    f = np.asarray(fixed, float)
    m = np.asarray(moving_warped, float)
    if valid is None:
        valid = np.ones(f.shape, bool)
    f, m = f[valid], m[valid]
    f = f - f.mean()
    m = m - m.mean()
    den = np.sqrt((f ** 2).sum() * (m ** 2).sum())
    return float((f * m).sum() / den) if den > 0 else 0.0


def auto_register(fixed_img, moving_img, motion="affine", init=None):
    f = _prep_for_reg(fixed_img)
    m = _prep_for_reg(moving_img)
    if f.shape != m.shape:
        H = max(f.shape[0], m.shape[0])
        W = max(f.shape[1], m.shape[1])
        f2 = np.zeros((H, W)); f2[:f.shape[0], :f.shape[1]] = f
        m2 = np.zeros((H, W)); m2[:m.shape[0], :m.shape[1]] = m
        f, m = f2, m2
    if init is None:
        from skimage.registration import phase_cross_correlation
        shift, _, _ = phase_cross_correlation(f, m, upsample_factor=10, normalization=None)
        A0 = np.eye(3)
        A0[0, 2], A0[1, 2] = shift[1], shift[0]
    else:
        A0 = np.asarray(init, float)
    info = {"init": A0.copy(), "method": None}
    try:
        import cv2
        mot = {"translation": cv2.MOTION_TRANSLATION, "rigid": cv2.MOTION_EUCLIDEAN,
               "affine": cv2.MOTION_AFFINE}[motion]
        W = np.linalg.inv(A0)[:2].astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-7)
        for scale in (0.5, 1.0):
            if scale != 1.0:
                fs = cv2.resize(f.astype(np.float32), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                ms = cv2.resize(m.astype(np.float32), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                Ws = W.copy(); Ws[:, 2] *= scale
            else:
                fs, ms, Ws = f.astype(np.float32), m.astype(np.float32), W.copy()
            _, Ws = cv2.findTransformECC(fs, ms, Ws, mot, crit, None, 5)
            W = Ws.copy()
            if scale != 1.0:
                W[:, 2] /= scale
        Wf = np.eye(3); Wf[:2] = W
        A = np.linalg.inv(Wf)
        info["method"] = "opencv ECC (" + motion + ")"
    except ImportError:
        A = _scipy_affine(f, m, A0)
        info["method"] = "scipy correlation (affine)"
    warped = warp_image(m, A, f.shape)
    valid = warp_image(np.ones(m.shape), A, f.shape) > 0.999
    info["corr_before"] = image_correlation(f, warp_image(m, A0, f.shape), valid)
    info["corr_after"] = image_correlation(f, warped, valid)
    return A, info


def refine_with_centroids(A, rois1, rois2, rmax=4.0, n_iter=20):
    from scipy.spatial import cKDTree
    c1 = rois1.centroids[:, ::-1]
    c2 = rois2.centroids[:, ::-1]
    t1 = cKDTree(c1)
    A = np.asarray(A, float).copy()
    n_prev = -1
    for _ in range(n_iter):
        w = apply_affine(A, c2)
        d, j = t1.query(w)
        back = cKDTree(w).query(c1)[1]
        mutual = (back[j] == np.arange(len(c2))) & (d < rmax)
        if mutual.sum() < 3:
            break
        A_new = fit_affine(c2[mutual], c1[j[mutual]])
        if mutual.sum() == n_prev and np.allclose(A_new, A, atol=1e-6):
            A = A_new
            break
        A, n_prev = A_new, mutual.sum()
    w = apply_affine(A, c2)
    d, j = t1.query(w)
    back = cKDTree(w).query(c1)[1]
    mutual = (back[j] == np.arange(len(c2))) & (d < rmax)
    info = {"n_centroid_pairs": int(mutual.sum()),
            "median_centroid_residual": float(np.median(d[mutual])) if mutual.any() else np.nan}
    return A, info


def _scipy_affine(f, m, A0):
    from scipy.optimize import minimize
    fs = ndimage.zoom(f, 0.5, order=1)
    ms = ndimage.zoom(m, 0.5, order=1)
    S = np.diag([0.5, 0.5, 1.0])
    Si = np.linalg.inv(S)

    def unpack(v):
        A = np.eye(3); A[:2] = v.reshape(2, 3); return A

    def cost(v, ff, mm):
        A = unpack(v)
        w = warp_image(mm, A, ff.shape)
        val = warp_image(np.ones(mm.shape), A, ff.shape) > 0.999
        return -image_correlation(ff, w, val)

    v = (S @ A0 @ Si)[:2].ravel()
    v = minimize(cost, v, args=(fs, ms), method="Powell").x
    v = (Si @ unpack(v) @ S)[:2].ravel()
    v = minimize(cost, v, args=(f, m), method="Powell").x
    return unpack(v)


class Registration:

    def __init__(self, day1=None, day2=None):
        self.day1 = None
        self.day2 = None
        self.A = None
        self.moving_pts = np.zeros((0, 2))
        self.fixed_pts = np.zeros((0, 2))
        self.transform_source = None
        self.thold = DEFAULT_OVERLAP_THOLD
        self.pairs = []
        self.rois2 = None
        self.images2 = None
        if day1 is not None:
            self.load_day(1, day1)
        if day2 is not None:
            self.load_day(2, day2)

    def load_day(self, day, path_or_session):
        s = path_or_session if isinstance(path_or_session, Session) else Session(path_or_session)
        if day == 1:
            self.day1 = s
        else:
            self.day2 = s
        self.pairs = []
        self._apply()
        return s

    @property
    def ready(self):
        return self.day1 is not None and self.day2 is not None

    @property
    def rois1(self):
        return self.day1.rois_default if self.day1 is not None else None

    def set_transform(self, A, moving_pts=None, fixed_pts=None, source="manual"):
        self.A = None if A is None else np.asarray(A, float)
        self.moving_pts = np.zeros((0, 2)) if moving_pts is None else np.asarray(moving_pts, float).reshape(-1, 2)
        self.fixed_pts = np.zeros((0, 2)) if fixed_pts is None else np.asarray(fixed_pts, float).reshape(-1, 2)
        self.transform_source = source if A is not None else None
        self.pairs = []
        self._apply()

    def set_control_points(self, moving_pts, fixed_pts):
        A = fit_affine(moving_pts, fixed_pts)
        self.set_transform(A, moving_pts, fixed_pts, "manual control points")
        return A

    def auto_transform(self, mask=0, proc=0, motion="affine", refine=True):
        A, info = auto_register(self.day1.images_default[proc, mask],
                                self.day2.images_default[proc, mask], motion=motion)
        src = f"auto: {info['method']} on {MASK_NAMES[mask]}{' (proc)' if proc else ''}"
        if refine and motion == "affine":
            A, rinfo = refine_with_centroids(A, self.rois1, self.day2.rois_default)
            info.update(rinfo)
            src += " + ROI-centroid refinement"
        self.set_transform(A, source=src)
        info["source"] = src
        return A, info

    def reset_transform(self):
        self.set_transform(None)

    def control_point_residuals(self):
        if self.A is None or len(self.moving_pts) == 0:
            return np.zeros(0)
        return np.sqrt(((apply_affine(self.A, self.moving_pts) - self.fixed_pts) ** 2).sum(1))

    def _apply(self):
        if self.day2 is None:
            self.rois2, self.images2 = None, None
            return
        if self.A is None:
            self.rois2 = self.day2.rois_default
            self.images2 = self.day2.images_default
            return
        out = self.day1.dims if self.day1 is not None else self.day2.dims
        self.rois2 = self.day2.rois_default.transform(self.A, out)
        imgs = np.zeros((2, N_MASKS) + tuple(out))
        for p in range(2):
            for m in range(N_MASKS):
                imgs[p, m] = warp_image(self.day2.images_default[p, m], self.A, out)
        self.images2 = imgs

    def _check_dims(self):
        if self.A is None and self.day1.dims != self.day2.dims:
            raise ValueError("day 1 and day 2 have different frame sizes; register them first")

    def detect(self, thold=None):
        if thold is not None:
            self.thold = float(thold)
        self._check_dims()
        self.pairs = detect_overlap(self.rois1, self.rois2, self.thold)
        return self.pairs

    def reset_overlap(self):
        self.pairs = []

    def click(self, y, x):
        self._check_dims()
        self.pairs, changed = toggle_pair_at(self.pairs, self.rois1, self.rois2, (y, x))
        return changed

    def result(self):
        return build_result(self)

    def save(self, path):
        return save_result(self.result(), path)

    def save_transform(self, path):
        d = {
            "A_moving_to_fixed": self.A.tolist(),
            "convention": "column vectors [x, y, 1], 0-based pixel centers, maps day 2 (moving) to day 1 (fixed)",
            "T_matlab_affine2d": affine_to_matlab(self.A).tolist(),
            "moving_points_xy": self.moving_pts.tolist(),
            "fixed_points_xy": self.fixed_pts.tolist(),
            "source": self.transform_source,
            "day1": self.day1.plane_dir if self.day1 else None,
            "day2": self.day2.plane_dir if self.day2 else None,
        }
        with open(path, "w") as fh:
            json.dump(d, fh, indent=1)

    def load_transform(self, path):
        with open(path) as fh:
            d = json.load(fh)
        self.set_transform(np.array(d["A_moving_to_fixed"]), d.get("moving_points_xy"),
                           d.get("fixed_points_xy"), d.get("source") or "loaded")

    def save_session(self, path):
        d = {
            "day1": self.day1.plane_dir if self.day1 else None,
            "day2": self.day2.plane_dir if self.day2 else None,
            "A_moving_to_fixed": None if self.A is None else self.A.tolist(),
            "moving_points_xy": self.moving_pts.tolist(),
            "fixed_points_xy": self.fixed_pts.tolist(),
            "transform_source": self.transform_source,
            "overlap_thold": self.thold,
            "pairs": [[int(a), int(b), float(p)] for a, b, p in self.pairs],
        }
        with open(path, "w") as fh:
            json.dump(d, fh, indent=1)

    @classmethod
    def load_session(cls, path):
        with open(path) as fh:
            d = json.load(fh)
        reg = cls(d.get("day1"), d.get("day2"))
        if d.get("A_moving_to_fixed") is not None:
            reg.set_transform(np.array(d["A_moving_to_fixed"]), d.get("moving_points_xy"),
                              d.get("fixed_points_xy"), d.get("transform_source"))
        reg.thold = d.get("overlap_thold", DEFAULT_OVERLAP_THOLD)
        reg.pairs = [list(p) for p in d.get("pairs", [])]
        return reg


def build_result(reg):
    s1, s2 = reg.day1, reg.day2
    pairs = reg.pairs
    idcs = np.array([[p[0], p[1]] for p in pairs], dtype=int).reshape(-1, 2)
    p_ov = np.array([p[2] for p in pairs], dtype=float)
    iscell_idcs = np.array([[s1.raw_to_iscell(a), s2.raw_to_iscell(b)] for a, b in idcs],
                           dtype=int).reshape(-1, 2)

    match_of_1 = {int(a): (int(b), float(p)) for a, b, p in pairs}
    matched_2 = {int(b) for _, b, _ in pairs}
    n1 = len(s1.iscell_raw)
    rows = []
    for i1, raw1 in enumerate(s1.iscell_raw):
        if int(raw1) in match_of_1:
            raw2, p = match_of_1[int(raw1)]
            rows.append((i1, raw1, i1, raw2, s2.raw_to_iscell(raw2), p, STATUS_MATCHED))
        else:
            rows.append((i1, raw1, i1, np.nan, np.nan, np.nan, STATUS_DAY1_ONLY))
    next_id = n1
    for i2, raw2 in enumerate(s2.iscell_raw):
        if int(raw2) not in matched_2:
            rows.append((next_id, np.nan, np.nan, raw2, i2, np.nan, STATUS_DAY2_ONLY))
            next_id += 1
    tab = np.array(rows, dtype=float).reshape(-1, 7)

    day2_to_unified = np.full(len(s2.iscell_raw), -1, dtype=int)
    for r in tab:
        if np.isfinite(r[4]):
            day2_to_unified[int(r[4])] = int(r[0])
    in_reg = set(int(i) for i in reg.rois2.ids) if reg.rois2 is not None else set()
    lost = np.array([int(r) for r in s2.iscell_raw if int(r) not in in_reg], dtype=int)

    return {
        "day1_path": s1.plane_dir,
        "day2_path": s2.plane_dir,
        "A_moving_to_fixed": np.eye(3) if reg.A is None else reg.A,
        "transform_source": reg.transform_source or "none (identity)",
        "overlap_thold": reg.thold,
        "rois_idcs": idcs,
        "rois_iscell_idcs": iscell_idcs,
        "p_overlaps": p_ov,
        "unified_id": tab[:, 0].astype(int),
        "day1_raw": tab[:, 1],
        "day1_iscell": tab[:, 2],
        "day2_raw": tab[:, 3],
        "day2_iscell": tab[:, 4],
        "p_overlap": tab[:, 5],
        "status": tab[:, 6].astype(int),
        "day1_to_unified": np.arange(n1),
        "day2_to_unified": day2_to_unified,
        "day1_iscell_raw": s1.iscell_raw,
        "day2_iscell_raw": s2.iscell_raw,
        "day2_lost_in_transform_raw": lost,
    }


TABLE_COLS = ["unified_id", "day1_raw", "day1_iscell", "day2_raw", "day2_iscell", "p_overlap", "status"]


def save_result(res, path):
    base = os.path.splitext(path)[0]
    np.savez(base + ".npz", **{k: (np.array(v) if not isinstance(v, str) else np.array(v)) for k, v in res.items()})
    with open(base + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(TABLE_COLS)
        for i in range(len(res["unified_id"])):
            row = []
            for c in TABLE_COLS:
                v = res[c][i]
                if c == "status":
                    row.append(STATUS_NAMES[int(v)])
                elif c == "p_overlap":
                    row.append("" if not np.isfinite(v) else f"{v:.4f}")
                else:
                    row.append("" if not np.isfinite(v) else str(int(v)))
            w.writerow(row)
    return base + ".npz", base + ".csv"


def load_result(path):
    d = np.load(os.path.splitext(path)[0] + ".npz", allow_pickle=False)
    return {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}


def align_traces(res, X1, X2):
    n = len(res["unified_id"])
    out = []
    for X, col in ((X1, "day1_iscell"), (X2, "day2_iscell")):
        X = np.asarray(X, dtype=float)
        Y = np.full((n,) + X.shape[1:], np.nan)
        idx = res[col]
        ok = np.isfinite(idx)
        Y[res["unified_id"][ok]] = X[idx[ok].astype(int)]
        out.append(Y)
    return out[0], out[1]


def _intersect_stable(a, b):
    a = np.asarray(a).ravel()
    bset = set(np.asarray(b).ravel().tolist())
    seen, out = set(), []
    for v in a.tolist():
        if v in bset and v not in seen:
            out.append(v); seen.add(v)
    return np.array(out, dtype=a.dtype)


def _intersect_n(arrs):
    a, b = arrs[0], arrs[1]
    for i in range(2, len(arrs)):
        b = _intersect_stable(a, b)
        a = arrs[i]
    return _intersect_stable(a, b)


def find_consistent_rois(roi_list):
    if len(roi_list) == 1:
        return np.asarray(roi_list[0])
    firsts = [np.asarray(x)[:, 0] for x in roi_list]
    init = _intersect_n(firsts)
    out = np.zeros((len(init), len(roi_list)), dtype=int)
    for c, r in enumerate(init):
        out[c] = [np.asarray(x)[np.asarray(x)[:, 0] == r, 1][0] for x in roi_list]
    return np.column_stack([init, out])


def cat_overlap(results):
    results = [load_result(r) if isinstance(r, str) else r for r in results]
    fnames = [results[0]["day1_path"]] + [r["day2_path"] for r in results]
    return {
        "fnames": fnames,
        "rois_idcs": find_consistent_rois([r["rois_idcs"] for r in results]),
        "rois_iscell_idcs": find_consistent_rois([r["rois_iscell_idcs"] for r in results]),
    }
