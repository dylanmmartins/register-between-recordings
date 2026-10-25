from __future__ import annotations

import os
import sys
import traceback

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from regbrec import core as rc

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

QShortcut = getattr(QtGui, "QShortcut", None) or QtWidgets.QShortcut
GREEN = "#00ff00"
WHITE = "#ffffff"


def _gray64(v):
    idx = np.clip(np.floor(v), 1, 64)
    return ((idx - 1) / 63.0 * 255).astype(np.uint8)


class _ROICache:

    def __init__(self, rois):
        self.rois = rois
        n = rois.npix
        self.owner = np.repeat(np.arange(rois.n), n)
        self.pix = np.concatenate(rois.pix) if rois.n else np.zeros(0, int)
        self.border = np.concatenate(rois.border).astype(float) if rois.n else np.zeros(0)
        self.area = 1.0 - self.border

    def status(self, matched_ids):
        m = np.isin(self.rois.ids, list(matched_ids))
        return m[self.owner]


class ControlPointDialog(QtWidgets.QDialog):

    COLORS = [(255, 80, 80), (80, 200, 255), (255, 220, 0), (120, 255, 120), (255, 120, 255), (255, 160, 60)]

    def __init__(self, parent, moving_img, fixed_img, moving_lims, fixed_lims,
                 moving_pts=None, fixed_pts=None, A=None):
        super().__init__(parent)
        self.setWindowTitle("Control point selection")
        self.resize(1400, 800)
        self.A = A
        self.mv = [list(p) for p in (moving_pts if moving_pts is not None else [])]
        self.fx = [list(p) for p in (fixed_pts if fixed_pts is not None else [])]
        n = min(len(self.mv), len(self.fx))
        self.mv, self.fx = self.mv[:n], self.fx[:n]
        self.items = {0: [], 1: []}

        lay = QtWidgets.QVBoxLayout(self)
        tip = QtWidgets.QLabel(
            "Click a landmark in the LEFT image (day 2, moving), then the same landmark in the RIGHT "
            "image (day 1, fixed). Repeat for at least 3 (ideally 6-10, spread out) pairs. "
            "Drag points to adjust. Drag empty space to pan, wheel to zoom.")
        tip.setWordWrap(True)
        lay.addWidget(tip)
        glw = pg.GraphicsLayoutWidget()
        lay.addWidget(glw, 1)
        self.views = []
        for col, (img, lims, title) in enumerate(((moving_img, moving_lims, "Day 2 (moving)"),
                                                  (fixed_img, fixed_lims, "Day 1 (fixed)"))):
            vb = glw.addViewBox(row=0, col=col, lockAspect=True, invertY=True)
            vb.setMenuEnabled(False)
            it = pg.ImageItem(img)
            it.setLevels(lims)
            vb.addItem(it)
            lbl = pg.TextItem(title, color="w", anchor=(0, 0))
            vb.addItem(lbl)
            lbl.setPos(2, 2)
            self.views.append(vb)
        glw.scene().sigMouseClicked.connect(self._clicked)

        row = QtWidgets.QHBoxLayout()
        lay.addLayout(row)
        self.predict_cb = QtWidgets.QCheckBox("Pre-place partner point using current transform")
        self.predict_cb.setChecked(A is not None)
        self.predict_cb.setEnabled(A is not None)
        row.addWidget(self.predict_cb)
        row.addStretch(1)
        self.info = QtWidgets.QLabel()
        row.addWidget(self.info)
        btns = QtWidgets.QHBoxLayout()
        lay.addLayout(btns)
        for text, fn in (("Undo last point", self.undo), ("Clear all", self.clear)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            btns.addWidget(b)
        btns.addStretch(1)
        self.ok_btn = QtWidgets.QPushButton("Fit affine && apply")
        self.ok_btn.clicked.connect(self.accept)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btns.addWidget(self.ok_btn)
        btns.addWidget(cancel)
        self._redraw()

    def _clicked(self, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton or ev.isAccepted():
            return
        for side, vb in enumerate(self.views):
            if vb.sceneBoundingRect().contains(ev.scenePos()):
                p = vb.mapSceneToView(ev.scenePos())
                xy = [p.x() - 0.5, p.y() - 0.5]
                self._add(side, xy)
                return

    def _add(self, side, xy):
        lists = (self.mv, self.fx)
        mine, other = lists[side], lists[1 - side]
        if len(mine) > len(other):
            mine[-1] = xy
        else:
            mine.append(xy)
            if self.predict_cb.isChecked() and self.A is not None and len(other) < len(mine):
                A = self.A if side == 0 else np.linalg.inv(self.A)
                other.append(rc.apply_affine(A, xy)[0].tolist())
        self._redraw()

    def undo(self):
        if len(self.mv) > len(self.fx):
            self.mv.pop()
        elif len(self.fx) > len(self.mv):
            self.fx.pop()
        elif self.mv:
            self.mv.pop(); self.fx.pop()
        self._redraw()

    def clear(self):
        self.mv, self.fx = [], []
        self._redraw()

    def _moved(self, side, k, item):
        pos = item.pos()
        (self.mv, self.fx)[side][k] = [pos.x() - 0.5, pos.y() - 0.5]
        self._update_info()

    def _redraw(self):
        for side in (0, 1):
            for it in self.items[side]:
                self.views[side].removeItem(it)
            self.items[side] = []
            for k, (x, y) in enumerate((self.mv, self.fx)[side]):
                c = self.COLORS[k % len(self.COLORS)]
                it = pg.TargetItem(pos=(x + 0.5, y + 0.5), size=14, symbol="crosshair",
                                   pen=pg.mkPen(c, width=2), movable=True, label=str(k + 1),
                                   labelOpts={"color": c, "offset": (8, -8)})
                it.sigPositionChangeFinished.connect(lambda item, s=side, kk=k: self._moved(s, kk, item))
                self.views[side].addItem(it)
                self.items[side].append(it)
        self._update_info()

    def _update_info(self):
        n = min(len(self.mv), len(self.fx))
        txt = f"{n} pair(s)"
        if n >= 3:
            try:
                A = rc.fit_affine(self.mv[:n], self.fx[:n])
                res = np.sqrt(((rc.apply_affine(A, self.mv[:n]) - np.array(self.fx[:n])) ** 2).sum(1))
                worst = int(np.argmax(res)) + 1
                txt += f" | residual mean {res.mean():.2f} px, max {res.max():.2f} px (pair {worst})"
            except ValueError as e:
                txt += f" | {e}"
        self.info.setText(txt)
        self.ok_btn.setEnabled(n >= 3)

    def points(self):
        n = min(len(self.mv), len(self.fx))
        return np.array(self.mv[:n]), np.array(self.fx[:n])


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, day1=None, day2=None):
        super().__init__()
        self.setWindowTitle("regbrec")
        self.resize(1500, 950)
        self.reg = rc.Registration()
        self.displayed = 1
        self.mask = 0
        self.proc = 0
        self.im_flag = True
        self.cursor_mirror = False
        self.overlay = False
        self.lims = {1: None, 2: None}
        self._cache = {}
        self._auto_msg = ""
        self._build_ui()
        self._build_shortcuts()
        for d, p in ((1, day1), (2, day2)):
            if p:
                self._load_day(d, p)
        self.update_displays()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)

        files = QtWidgets.QGridLayout()
        outer.addLayout(files)
        self.path_lbl = {}
        for r, d in enumerate((1, 2)):
            b = QtWidgets.QPushButton(f"Load day {d}...")
            b.clicked.connect(lambda _, dd=d: self.on_load_day(dd))
            files.addWidget(b, r, 0)
            lbl = QtWidgets.QLabel("...")
            lbl.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            files.addWidget(lbl, r, 1, 1, 3)
            self.path_lbl[d] = lbl
        files.setColumnStretch(1, 1)

        bar = QtWidgets.QHBoxLayout()
        outer.addLayout(bar)
        tbox = QtWidgets.QGroupBox("Transform (day 2 → day 1)")
        tl = QtWidgets.QGridLayout(tbox)
        for i, (text, fn, tip) in enumerate((
                ("Register (control points)", self.on_register, "Click matching landmarks (cpselect); least-squares affine"),
                ("Auto register", self.on_auto, "Affine from the displayed image (ECC), refined with ROI centroids"),
                ("Load transform...", self.on_load_tform, ""),
                ("Save transform...", self.on_save_tform, ""),
                ("Reset", self.on_reset, "Remove transform and all matches"))):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            tl.addWidget(b, 0, i)
        self.refine_cb = QtWidgets.QCheckBox("refine auto with ROI centroids")
        self.refine_cb.setChecked(True)
        tl.addWidget(self.refine_cb, 1, 0, 1, 2)
        self.tform_lbl = QtWidgets.QLabel("no transform")
        self.tform_lbl.setWordWrap(True)
        tl.addWidget(self.tform_lbl, 1, 2, 1, 3)
        bar.addWidget(tbox, 3)

        obox = QtWidgets.QGroupBox("ROIs")
        ol = QtWidgets.QGridLayout(obox)
        ol.addWidget(QtWidgets.QLabel("overlap threshold"), 0, 0)
        self.thold_spin = QtWidgets.QDoubleSpinBox()
        self.thold_spin.setRange(0.01, 1.0)
        self.thold_spin.setSingleStep(0.05)
        self.thold_spin.setDecimals(2)
        self.thold_spin.setValue(rc.DEFAULT_OVERLAP_THOLD)
        self.thold_spin.setToolTip("fraction of the day-1 ROI's pixels covered by the day-2 ROI (must be exceeded)")
        ol.addWidget(self.thold_spin, 0, 1)
        b = QtWidgets.QPushButton("Detect overlap")
        b.clicked.connect(self.on_detect)
        ol.addWidget(b, 1, 0)
        b = QtWidgets.QPushButton("Reset overlap")
        b.clicked.connect(self.on_reset_overlap)
        ol.addWidget(b, 1, 1)
        bar.addWidget(obox, 1)

        sbox = QtWidgets.QGroupBox("Save")
        sl = QtWidgets.QGridLayout(sbox)
        for i, (text, fn) in enumerate((("Save analysis...", self.on_save_analysis),
                                        ("Save session...", self.on_save_session),
                                        ("Load session...", self.on_load_session))):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            sl.addWidget(b, i // 2, i % 2)
        bar.addWidget(sbox, 1)

        ind = QtWidgets.QHBoxLayout()
        outer.addLayout(ind)
        self.mask_lbl = []
        for m, name in enumerate(rc.MASK_NAMES):
            l = QtWidgets.QLabel(f" {m + 1}: {name} ")
            self.mask_lbl.append(l)
            ind.addWidget(l)
        ind.addSpacing(20)
        self.ind_lbl = {}
        for key, text in (("day1", "Q: day 1"), ("day2", "W: day 2"), ("roi1", "A: ROIs 1"),
                          ("roi2", "S: ROIs 2"), ("proc", "P: processed"), ("cursor", "O: cursor"),
                          ("overlay", "V: overlay")):
            l = QtWidgets.QLabel(f" {text} ")
            l.setFrameShape(QtWidgets.QFrame.Shape.Box)
            self.ind_lbl[key] = l
            ind.addWidget(l)
        ind.addStretch(1)

        self.glw = pg.GraphicsLayoutWidget()
        outer.addWidget(self.glw, 1)
        self.img_vb = self.glw.addViewBox(row=0, col=0, lockAspect=True, invertY=True)
        self.roi_vb = self.glw.addViewBox(row=0, col=1, lockAspect=True, invertY=True)
        for vb in (self.img_vb, self.roi_vb):
            vb.setMenuEnabled(False)
            vb.setBackgroundColor("k")
        self.roi_vb.setXLink(self.img_vb)
        self.roi_vb.setYLink(self.img_vb)
        self.img_item = pg.ImageItem()
        self.roi_item = pg.ImageItem()
        self.img_vb.addItem(self.img_item)
        self.roi_vb.addItem(self.roi_item)
        self.cursors = []
        for vb in (self.img_vb, self.roi_vb):
            c = pg.ScatterPlotItem(size=10, pen=pg.mkPen((0, 128, 255)), brush=pg.mkBrush(0, 128, 255))
            vb.addItem(c)
            self.cursors.append(c)
        self.glw.scene().sigMouseClicked.connect(self._scene_clicked)
        self.glw.scene().sigMouseMoved.connect(self._scene_moved)

        crow = QtWidgets.QHBoxLayout()
        outer.addLayout(crow)
        crow.addWidget(QtWidgets.QLabel("contrast"))
        self.contrast = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.contrast.setRange(0, 1000)
        self.contrast.setValue(500)
        self.contrast.valueChanged.connect(self.on_contrast)
        crow.addWidget(self.contrast, 1)
        crow.addSpacing(20)
        crow.addWidget(QtWidgets.QLabel(
            "Right panel: open contour = unmatched, filled = matched; bright = day on the left, dim = other day. "
            "Click to toggle a match."), 3)
        self.status = self.statusBar()
        for w in self.findChildren(QtWidgets.QPushButton):
            w.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.contrast.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

    def _build_shortcuts(self):
        keymap = {"Q": lambda: self._show(1, True), "W": lambda: self._show(2, True),
                  "A": lambda: self._show(1, False), "S": lambda: self._show(2, False),
                  "P": self._toggle_proc, "O": self._toggle_cursor, "V": self._toggle_overlay}
        for m in range(rc.N_MASKS):
            keymap[str(m + 1)] = lambda mm=m: self._set_mask(mm)
        self._shortcuts = []
        for key, fn in keymap.items():
            sc = QShortcut(QtGui.QKeySequence(key), self)
            sc.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(fn)
            self._shortcuts.append(sc)

    def _err(self, title, exc):
        traceback.print_exc()
        QtWidgets.QMessageBox.critical(self, title, f"{type(exc).__name__}: {exc}")

    def _busy(self, on):
        if on:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        else:
            QtWidgets.QApplication.restoreOverrideCursor()
        QtWidgets.QApplication.processEvents()

    def _need_both(self):
        if not self.reg.ready:
            QtWidgets.QMessageBox.information(self, "regbrec", "Load day 1 and day 2 first.")
            return False
        return True

    def _session(self, d):
        return self.reg.day1 if d == 1 else self.reg.day2

    def _images(self, d):
        if d == 1:
            return None if self.reg.day1 is None else self.reg.day1.images_default
        return self.reg.images2

    def _rois(self, d):
        r = self.reg.rois1 if d == 1 else self.reg.rois2
        if r is None:
            return None
        c = self._cache.get(d)
        if c is None or c.rois is not r:
            c = _ROICache(r)
            self._cache[d] = c
        return c

    def _default_dir(self):
        s = self.reg.day1 or self.reg.day2
        return s.plane_dir if s else os.getcwd()

    def _default_name(self, suffix):
        names = [rc.session_label(s.plane_dir) for s in (self.reg.day1, self.reg.day2) if s]
        return os.path.join(self._default_dir(), "__".join(names) + suffix)

    def _reset_lims(self):
        for d in (1, 2):
            s = self._session(d)
            self.lims[d] = None if s is None else s.lims.copy()

    def on_load_day(self, d):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, f"Choose day {d} suite2p folder (plane0, suite2p or experiment folder)", self._default_dir())
        if path:
            self._load_day(d, path)
            self.update_displays()

    def _load_day(self, d, path):
        self._busy(True)
        try:
            self.reg.load_day(d, path)
            self.path_lbl[d].setText(self._session(d).plane_dir)
            self._reset_lims()
            self.displayed = d
        except Exception as e:
            self._busy(False)
            self._err(f"Could not load day {d}", e)
            return
        self._busy(False)

    def on_register(self):
        if not self._need_both():
            return
        p, m = self.proc, self.mask
        mv_img = self.reg.day2.images_default[p, m]
        fx_img = self.reg.day1.images_default[p, m]
        dlg = ControlPointDialog(self, mv_img, fx_img, self.reg.day2.lims[p, m], self.lims[1][p, m],
                                 self.reg.moving_pts, self.reg.fixed_pts, self.reg.A)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        mv, fx = dlg.points()
        self._busy(True)
        try:
            self.reg.set_control_points(mv, fx)
        except Exception as e:
            self._busy(False)
            self._err("Registration failed", e)
            return
        self._busy(False)
        self._after_transform()

    def on_auto(self):
        if not self._need_both():
            return
        self._busy(True)
        try:
            _, info = self.reg.auto_transform(mask=self.mask, proc=self.proc, refine=self.refine_cb.isChecked())
        except Exception as e:
            self._busy(False)
            self._err("Auto registration failed", e)
            return
        self._busy(False)
        msg = f"image correlation {info['corr_before']:.3f} → {info['corr_after']:.3f}"
        if "n_centroid_pairs" in info:
            msg += (f"; {info['n_centroid_pairs']} centroid pairs, median residual "
                    f"{info['median_centroid_residual']:.2f} px")
        self._auto_msg = msg
        self._after_transform()
        self.overlay = True
        self.update_displays()

    def _after_transform(self):
        self.update_displays()

    def on_load_tform(self):
        if not self._need_both():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load transform", self._default_dir(), "JSON (*.json)")
        if path:
            self._busy(True)
            try:
                self.reg.load_transform(path)
            except Exception as e:
                self._busy(False)
                self._err("Could not load transform", e)
                return
            self._busy(False)
            self.update_displays()

    def on_save_tform(self):
        if self.reg.A is None:
            QtWidgets.QMessageBox.information(self, "regbrec", "No transform to save yet.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save transform", self._default_name("_tform.json"),
                                                        "JSON (*.json)")
        if path:
            self.reg.save_transform(path)

    def on_reset(self):
        self.reg.reset_transform()
        self._reset_lims()
        self.update_displays()

    def on_detect(self):
        if not self._need_both():
            return
        try:
            self.reg.detect(self.thold_spin.value())
        except Exception as e:
            self._err("Detect overlap failed", e)
        self.update_displays()

    def on_reset_overlap(self):
        self.reg.reset_overlap()
        self.update_displays()

    def _scene_clicked(self, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton or not self.reg.ready:
            return
        if not self.roi_vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        p = self.roi_vb.mapSceneToView(ev.scenePos())
        try:
            if self.reg.click(p.y() - 0.5, p.x() - 0.5):
                self.update_displays()
        except Exception as e:
            self._err("Click failed", e)

    def _scene_moved(self, pos):
        if not self.cursor_mirror:
            return
        for k, vb in enumerate((self.img_vb, self.roi_vb)):
            if vb.sceneBoundingRect().contains(pos):
                p = vb.mapSceneToView(pos)
                self.cursors[1 - k].setData([p.x()], [p.y()])
                self.cursors[k].setData([], [])
                return

    def _show(self, d, im_flag):
        self.displayed, self.im_flag = d, im_flag
        self.update_displays()

    def _set_mask(self, m):
        self.mask = m
        self.update_displays()

    def _toggle_proc(self):
        self.proc = 1 - self.proc
        self.update_displays()

    def _toggle_cursor(self):
        self.cursor_mirror = not self.cursor_mirror
        if not self.cursor_mirror:
            for c in self.cursors:
                c.setData([], [])
        self.update_displays()

    def _toggle_overlay(self):
        self.overlay = not self.overlay
        self.update_displays()

    def on_contrast(self, value):
        d = self.displayed
        s = self._session(d)
        imgs = self._images(d)
        if s is None or imgs is None or not self.im_flag:
            return
        cont = rc.slider_to_contrast(value / 1000.0)
        self.lims[d][self.proc, self.mask] = s.lims_for(imgs[self.proc, self.mask], self.mask, cont)
        self.update_displays()

    def on_save_analysis(self):
        if not self._need_both():
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save analysis", self._default_name("_reg.npz"),
                                                        "NumPy archive (*.npz)")
        if path:
            try:
                npz, csvp = self.reg.save(path)
                self.status.showMessage(f"saved {npz} and {csvp}", 15000)
            except Exception as e:
                self._err("Save failed", e)

    def on_save_session(self):
        if self.reg.day1 is None and self.reg.day2 is None:
            QtWidgets.QMessageBox.critical(self, "Error", "No data loaded - please load at least 1 session before trying to save")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save session", self._default_name("_session.json"),
                                                        "JSON (*.json)")
        if path:
            self.reg.thold = self.thold_spin.value()
            self.reg.save_session(path)

    def on_load_session(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load session", self._default_dir(), "JSON (*.json)")
        if not path:
            return
        self._busy(True)
        try:
            self.reg = rc.Registration.load_session(path)
        except Exception as e:
            self._busy(False)
            self._err("Could not load session", e)
            return
        self._busy(False)
        self._cache = {}
        for d in (1, 2):
            s = self._session(d)
            self.path_lbl[d].setText(s.plane_dir if s else "...")
        self.thold_spin.setValue(self.reg.thold)
        self._reset_lims()
        self.update_displays()

    def _roi_value_map(self):
        reg = self.reg
        dims = (reg.day1 or reg.day2).dims
        C = np.zeros(dims[0] * dims[1])
        matched = {1: {p[0] for p in reg.pairs}, 2: {p[1] for p in reg.pairs}}
        d = self.displayed
        for i in (1, 2):
            if i == d:
                continue
            rc_ = self._rois(i)
            if rc_ is None or len(rc_.pix) == 0:
                continue
            st = rc_.status(matched[i])
            C[rc_.pix[~st]] = rc_.border[~st] * 20
            C[rc_.pix[st]] = rc_.area[st] * 20
        rc_ = self._rois(d)
        if rc_ is not None and len(rc_.pix):
            st = rc_.status(matched[d])
            for _ in range(2):
                C[rc_.pix[~st]] = C[rc_.pix[~st]] + rc_.border[~st] * 50
            C[rc_.pix[st]] = C[rc_.pix[st]] + rc_.area[st] * 50
        return C.reshape(dims)

    def _roi_mask_image(self, d):
        dims = (self.reg.day1 or self.reg.day2).dims
        C = np.zeros(dims[0] * dims[1])
        rc_ = self._rois(d)
        if rc_ is not None and len(rc_.pix):
            matched = {p[0] if d == 1 else p[1] for p in self.reg.pairs}
            st = rc_.status(matched)
            C[rc_.pix[~st]] = C[rc_.pix[~st]] + rc_.border[~st] * 8
            C[rc_.pix[st]] = C[rc_.pix[st]] + rc_.area[st] * 8
        return np.minimum(C, 10).reshape(dims)

    def _norm(self, img, lims):
        return np.clip((img - lims[0]) / max(lims[1] - lims[0], 1e-12), 0, 1)

    def update_displays(self):
        reg = self.reg
        self._update_indicators()
        if reg.day1 is None and reg.day2 is None:
            return
        if self.displayed == 1 and reg.day1 is None:
            self.displayed = 2
        if self.displayed == 2 and reg.day2 is None:
            self.displayed = 1
        d, p, m = self.displayed, self.proc, self.mask
        if self.overlay and reg.ready and reg.day1.dims == reg.images2.shape[2:]:
            a = self._norm(reg.day1.images_default[p, m], self.lims[1][p, m])
            b = self._norm(reg.images2[p, m], self.lims[2][p, m])
            rgb = np.stack([a, b, a], axis=-1)
            self.img_item.setImage((rgb * 255).astype(np.uint8), autoLevels=False, levels=(0, 255))
        elif self.im_flag:
            img = self._images(d)[p, m]
            self.img_item.setImage(img, autoLevels=False, levels=tuple(self.lims[d][p, m]))
        else:
            self.img_item.setImage(self._roi_mask_image(d), autoLevels=False, levels=(0, 10))
        self.roi_item.setImage(_gray64(self._roi_value_map()), autoLevels=False, levels=(0, 255))
        self._update_status()

    def _update_indicators(self):
        def hl(lbl, on):
            lbl.setStyleSheet(f"background-color: {GREEN if on else WHITE}; color: black;")
        for k, l in enumerate(self.mask_lbl):
            hl(l, k == self.mask)
        hl(self.ind_lbl["day1"], self.im_flag and self.displayed == 1)
        hl(self.ind_lbl["day2"], self.im_flag and self.displayed == 2)
        hl(self.ind_lbl["roi1"], not self.im_flag and self.displayed == 1)
        hl(self.ind_lbl["roi2"], not self.im_flag and self.displayed == 2)
        hl(self.ind_lbl["proc"], self.proc == 1)
        hl(self.ind_lbl["cursor"], self.cursor_mirror)
        hl(self.ind_lbl["overlay"], self.overlay)

    def _update_status(self):
        reg = self.reg
        parts = []
        if reg.day1:
            parts.append(f"day 1: {len(reg.day1.iscell_raw)} cells")
        if reg.day2:
            lost = len(reg.day2.iscell_raw) - (reg.rois2.n if reg.rois2 is not None else 0)
            parts.append(f"day 2: {len(reg.day2.iscell_raw)} cells" + (f" ({lost} lost in transform)" if lost else ""))
        if reg.ready:
            parts.append(f"matched: {len(reg.pairs)}")
        self.status.showMessage("   |   ".join(parts))
        if reg.A is None:
            self.tform_lbl.setText("no transform (day 2 ROIs used as-is)")
        else:
            res = reg.control_point_residuals()
            t = reg.transform_source or ""
            if res.size:
                t += f" | {res.size} pts, residual mean {res.mean():.2f} px"
            elif reg.transform_source and reg.transform_source.startswith("auto") and self._auto_msg:
                t += f" | {self._auto_msg}"
            self.tform_lbl.setText(t)
        self.tform_lbl.setToolTip(np.array2string(reg.A, precision=4) if reg.A is not None else "")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = MainWindow(*(argv + [None, None])[:2])
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
