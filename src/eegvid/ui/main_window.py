"""Main window. Workflow for lab assistants: Connect -> check offsets and signals -> Record -> Stop."""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QSpinBox, QSplitter, QTabWidget,
    QVBoxLayout, QWidget,
)

from ..biosemi import AcquisitionError, BioSemiDevice, Chunk, RingBufferLibrary, SimulatedRingBuffer
from ..biosemi.channel_names import apply_names, parse_actiview_cfg
from ..biosemi.offsets import DEFAULT_LIMIT_MV, OffsetMonitor, describe, summarize
from ..biosemi.protocol import UV_PER_LSB_24, build_layout
from ..dsp import Decimator, display_filter
from ..recording import RecordingSession
from ..video import Camera, list_cameras
from .offset_grid import COLORS, OffsetGrid
from .review_window import ReviewWindow

DISPLAY_SECONDS = 5.0
DISPLAY_RATE = 256  # Hz after decimation, for plotting only
SAVE_RATES = [("Full rate (as set on the AD-box)", 0), ("2048 Hz", 2048), ("1024 Hz", 1024),
              ("512 Hz", 512), ("256 Hz", 256)]
HIGHPASS = [("Off", 0.0), ("0.16 Hz", 0.16), ("0.5 Hz", 0.5), ("1 Hz", 1.0)]
LOWPASS = [("Off", 0.0), ("100 Hz", 100.0), ("70 Hz", 70.0), ("40 Hz", 40.0), ("30 Hz", 30.0)]
NOTCH = [("Off", 0.0), ("50 Hz", 50.0), ("60 Hz", 60.0)]


class DisplayBuffer:
    """Rolling copy of the displayed channels (already decimated), for plotting only."""

    def __init__(self, n_ch: int, rate: int):
        self.data = np.zeros((int(DISPLAY_SECONDS * rate), n_ch))
        self.pos = 0
        self.filled = 0
        self._lock = threading.Lock()

    def push(self, sel: np.ndarray) -> None:
        n, L = sel.shape[0], self.data.shape[0]
        if n == 0:
            return
        with self._lock:
            self.filled = min(L, self.filled + n)
            if n >= L:
                self.data[:] = sel[-L:]
                self.pos = 0
            else:
                idx = (self.pos + np.arange(n)) % L
                self.data[idx] = sel
                self.pos = (self.pos + n) % L

    def snapshot(self) -> np.ndarray:
        """The filled part of the buffer, oldest first."""
        with self._lock:
            return np.roll(self.data, -self.pos, axis=0)[self.data.shape[0] - self.filled:]


class _DisplayPath:
    """Everything the acquisition thread needs to feed the plot; swapped atomically."""

    def __init__(self, idx: np.ndarray, labels: list[str], ref: str, ref_idx: np.ndarray, fs: int):
        self.idx, self.labels, self.ref, self.ref_idx = idx, labels, ref, ref_idx
        self.decim = Decimator(max(1, fs // DISPLAY_RATE), len(idx), integer=False)
        self.buffer = DisplayBuffer(len(idx), DISPLAY_RATE)

    def push(self, adc: np.ndarray) -> None:
        sig = adc[:, self.idx].astype(np.float64)
        if self.ref != "none" and self.ref_idx.size:
            sig -= adc[:, self.ref_idx].mean(axis=1, keepdims=True)
        self.buffer.push(self.decim.process(sig * UV_PER_LSB_24))


def _combo(items, saved=None) -> QComboBox:
    box = QComboBox()
    for text, data in items:
        box.addItem(text, data)
    if saved is not None:
        i = box.findData(type(items[0][1])(saved))
        if i >= 0:
            box.setCurrentIndex(i)
    return box


class _ClickableLabel(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self.clicked.emit()
        super().mousePressEvent(event)


class _Bridge(QObject):
    error = Signal(str)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Video Recorder")
        self.settings = QSettings()
        self.device: BioSemiDevice | None = None
        self.camera: Camera | None = None
        self.session: RecordingSession | None = None
        self.channels = []
        self.offsets: OffsetMonitor | None = None
        self._disp: _DisplayPath | None = None
        self._last_trigger = 0
        self._tick = 0
        self.bridge = _Bridge()
        self.bridge.error.connect(self._on_async_error)
        self._last_folder: Path | None = None
        self._review_windows: list[ReviewWindow] = []
        self._build_ui()
        review = QAction("Review a session…", self, shortcut=QKeySequence.Open,
                         triggered=lambda: self.open_review(choose=True))
        self.menuBar().addMenu("&File").addAction(review)
        self._timer = QTimer(self, interval=33, timeout=self._refresh)
        self._timer.start()

    # ---------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        s = self.settings
        self.source = QComboBox()
        self.source.addItems(["Simulator (no hardware)", "BioSemi ActiveTwo"])
        self.source.setCurrentIndex(int(s.value("source", 0)))
        self.lib_path = QLineEdit(s.value("lib_path", ""))
        self.lib_path.setPlaceholderText("auto-detect (drivers/ folder)")
        lib_btn = QPushButton("…", clicked=self._browse_lib)
        lib_row = QHBoxLayout()
        lib_row.addWidget(self.lib_path)
        lib_row.addWidget(lib_btn)

        self.n_eeg = QComboBox()
        self.n_eeg.addItems(["16", "32", "64", "128", "256"])
        self.n_eeg.setCurrentText(str(s.value("n_eeg", "64")))
        self.names_cfg = s.value("names_cfg", "") or ""
        self.names_lbl = QLabel()
        self.names_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.names_btn = QPushButton("Load .cfg…", clicked=self._browse_names)
        self.names_btn.setToolTip("Use electrode names (Fp1, AF3, …) from an ActiView configuration file's\n"
                                  "[Labels] section, the same file ActiView loads. Without one, channels\n"
                                  "keep BioSemi's input names (A1…A32, B1…, EXG1–8).")
        self.names_clear = QPushButton("×", clicked=lambda: self._set_names_cfg(""))
        self.names_clear.setToolTip("Use BioSemi input names (A1, A2, …)")
        self.names_clear.setMaximumWidth(28)
        names_row = QHBoxLayout()
        names_row.addWidget(self.names_lbl, 1)
        names_row.addWidget(self.names_btn)
        names_row.addWidget(self.names_clear)
        self._set_names_cfg(self.names_cfg)
        self.include_exg = QCheckBox("Include EXG1–8")
        self.include_exg.setChecked(s.value("include_exg", "true") == "true")
        self.save_rate = _combo(SAVE_RATES, s.value("save_rate", 0))
        self.save_rate.setToolTip(
            "The AD-box's speed-mode switch sets the acquisition rate (2048 Hz or more).\n"
            "Choosing a lower rate here low-pass filters and downsamples before saving,\n"
            "like ActiView's Decimation setting. Triggers and video sync are kept exact.")
        self.save_rate.currentIndexChanged.connect(self._update_rate_labels)

        self.camera_box = QComboBox()
        refresh = QPushButton("↻", clicked=self._refresh_cameras)
        cam_row = QHBoxLayout()
        cam_row.addWidget(self.camera_box)
        cam_row.addWidget(refresh)
        self._refresh_cameras()

        self.subject = QLineEdit()
        self.subject.setPlaceholderText("e.g. P001")
        self.out_dir = QLineEdit(s.value("out_dir", str(Path.home() / "EEG Recordings")))
        out_btn = QPushButton("…", clicked=self._browse_out)
        out_row = QHBoxLayout()
        out_row.addWidget(self.out_dir)
        out_row.addWidget(out_btn)
        self.notes = QLineEdit()

        setup = QGroupBox("Setup")
        form = QFormLayout(setup)
        form.addRow("EEG source", self.source)
        form.addRow("Driver library", lib_row)
        form.addRow("EEG channels", self.n_eeg)
        form.addRow("", self.include_exg)
        form.addRow("Channel names", names_row)
        form.addRow("Save sample rate", self.save_rate)
        form.addRow("Camera", cam_row)
        form.addRow("Subject ID", self.subject)
        form.addRow("Save to", out_row)
        form.addRow("Notes", self.notes)

        self.connect_btn = QPushButton("Connect", clicked=self._toggle_connect)
        self.record_btn = QPushButton("● Record", clicked=self._toggle_record, enabled=False)
        self.record_btn.setMinimumHeight(48)
        self.record_btn.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.marker_text = QLineEdit(placeholderText="marker label")
        self.marker_btn = QPushButton("Add marker", clicked=self._add_marker, enabled=False)
        mk_row = QHBoxLayout()
        mk_row.addWidget(self.marker_text)
        mk_row.addWidget(self.marker_btn)

        rows = [("device", "Device"), ("rate", "Acquisition rate"), ("save", "Saving at"),
                ("cms", "CMS/DRL"), ("battery", "Battery"), ("offsets", "Electrode offsets"),
                ("trigger", "Last trigger"), ("recording", "Recording"), ("video", "Video")]
        self.status_lbl = {k: (_ClickableLabel if k == "offsets" else QLabel)("–") for k, _ in rows}
        status = QGroupBox("Status")
        sform = QFormLayout(status)
        for k, title in rows:
            sform.addRow(title, self.status_lbl[k])
        self.status_lbl["offsets"].setCursor(Qt.PointingHandCursor)
        self.status_lbl["offsets"].setToolTip("Show the Electrode offsets tab")
        self.status_lbl["offsets"].clicked.connect(lambda: self.tabs.setCurrentIndex(1))

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(setup)
        lv.addWidget(self.connect_btn)
        lv.addWidget(self.record_btn)
        lv.addLayout(mk_row)
        self.review_btn = QPushButton("Review a session…", clicked=lambda: self.open_review())
        self.review_btn.setToolTip("Open a recording folder to scrub through EEG and video together\n"
                                   "and edit its markers.")
        lv.addWidget(self.review_btn)
        lv.addWidget(status)
        lv.addStretch(1)
        left.setMaximumWidth(400)

        self.video_lbl = QLabel("No camera", alignment=Qt.AlignCenter)
        self.video_lbl.setMinimumSize(320, 240)
        self.video_lbl.setStyleSheet("background: #222; color: #aaa;")

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_signals_tab(), "Signals")
        self.tabs.addTab(self._build_offsets_tab(), "Electrode offsets")

        right = QSplitter(Qt.Vertical)
        right.addWidget(self.video_lbl)
        right.addWidget(self.tabs)
        right.setSizes([300, 500])

        central = QSplitter(Qt.Horizontal)
        central.addWidget(left)
        central.addWidget(right)
        central.setStretchFactor(1, 1)
        self.setCentralWidget(central)
        self.resize(1280, 860)

    def _build_signals_tab(self) -> QWidget:
        s = self.settings
        self.plot = pg.PlotWidget()
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setLabel("bottom", "time", "s")
        self.plot.setXRange(-DISPLAY_SECONDS, 0, padding=0)
        self.plot.hideAxis("left")
        self.curves = []

        self.n_show = _combo([("8", 8), ("16", 16), ("32", 32)], s.value("n_show", 8))
        self.first_ch = QSpinBox(minimum=1, maximum=1, prefix="from ch ")
        self.reference = QComboBox()
        self.reference.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.reference.setMinimumContentsLength(12)
        self.reference.setToolTip("Display only. The saved data is always the raw (CMS-referenced)\n"
                                  "signal, like ActiView; re-reference during analysis.")
        self.scale = QSpinBox(minimum=10, maximum=10000, value=int(s.value("scale", 100)),
                              suffix=" µV/div", singleStep=10)
        self.highpass = _combo(HIGHPASS, s.value("highpass", 0.16))
        self.lowpass = _combo(LOWPASS, s.value("lowpass", 100.0))
        self.notch = _combo(NOTCH, s.value("notch", 0.0))
        for w in (self.n_show, self.reference):
            w.currentIndexChanged.connect(self._update_display_channels)
        self.first_ch.valueChanged.connect(self._update_display_channels)

        row1 = QHBoxLayout()
        for text, w in [("Show", self.n_show), (None, self.first_ch), ("Reference", self.reference)]:
            if text:
                row1.addWidget(QLabel(text))
            row1.addWidget(w)
        row1.addStretch(1)
        row1.addWidget(self.scale)
        row2 = QHBoxLayout()
        for text, w in [("High-pass", self.highpass), ("Low-pass", self.lowpass), ("Notch", self.notch)]:
            row2.addWidget(QLabel(text))
            row2.addWidget(w)
        row2.addStretch(1)
        hint = QLabel("Filters and reference affect the display only; data is saved unfiltered.")
        hint.setStyleSheet("color: #666;")
        row2.addWidget(hint)

        box = QWidget()
        v = QVBoxLayout(box)
        v.addLayout(row1)
        v.addLayout(row2)
        v.addWidget(self.plot)
        return box

    def _build_offsets_tab(self) -> QWidget:
        self.offset_grid = OffsetGrid()
        self.offset_limit = QDoubleSpinBox(minimum=5, maximum=150, decimals=0, singleStep=5,
                                           value=float(self.settings.value("offset_limit", DEFAULT_LIMIT_MV)),
                                           prefix="±", suffix=" mV")
        self.offset_limit.valueChanged.connect(lambda: self._refresh_offsets(force=True))
        legend = QHBoxLayout()
        for key, text in [("ok", "within limit"), ("warn", "above limit"), ("bad", "above 2× limit"),
                          ("open", "no contact")]:
            bg, fg = COLORS[key]
            lab = QLabel(text)
            lab.setStyleSheet(f"background: {bg}; color: {fg}; padding: 2px 6px; border-radius: 3px;")
            legend.addWidget(lab)
        legend.addStretch(1)
        top = QHBoxLayout()
        top.addWidget(QLabel("Acceptable offset"))
        top.addWidget(self.offset_limit)
        top.addSpacing(16)
        top.addLayout(legend)

        self.cms_banner = QLabel()
        self.cms_banner.setWordWrap(True)
        self.cms_banner.setToolTip(
            "CMS and DRL replace a ground electrode on the ActiveTwo. Every offset is measured\n"
            "against CMS, so CMS/DRL have no offset of their own; the hardware only reports\n"
            "whether CMS is in range (the blue CMS/DRL light on the AD-box, ActiView's 'CM in Range').")
        self.cms_banner.hide()
        self.offset_summary = QLabel("Connect to see electrode offsets.")
        self.offset_summary.setWordWrap(True)
        self.offset_summary.setStyleSheet("font-size: 14px;")
        info = QLabel(
            "The ActiveTwo has no impedance check. Instead, BioSemi uses each electrode's DC offset "
            "relative to CMS: lower and steadier is better contact. Add gel or reseat electrodes "
            "outside the limit, and make sure CMS/DRL are in place first (offsets are meaningless "
            "while CMS is out of range). Values are averaged over the last 0.5 s, in mV.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #555;")

        box = QWidget()
        v = QVBoxLayout(box)
        v.addLayout(top)
        v.addWidget(self.cms_banner)
        v.addWidget(self.offset_summary)
        v.addWidget(self.offset_grid, 1)
        v.addWidget(info)
        return box

    def _refresh_cameras(self) -> None:
        self.camera_box.clear()
        self.camera_box.addItem("None", None)
        for i in list_cameras():
            self.camera_box.addItem(f"Camera {i}", i)
        saved = self.settings.value("camera")
        if saved is not None:
            idx = self.camera_box.findData(int(saved))
            if idx >= 0:
                self.camera_box.setCurrentIndex(idx)

    def _browse_lib(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "BioSemi RingBuffer library")
        if path:
            self.lib_path.setText(path)

    def _browse_names(self) -> None:
        start = str(Path(self.names_cfg).parent) if self.names_cfg else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "ActiView configuration file", start,
                                              "ActiView config (*.cfg);;All files (*)")
        if path:
            try:
                parse_actiview_cfg(path)
            except (OSError, ValueError) as e:
                QMessageBox.warning(self, "Channel names", f"Could not read channel names:\n{e}")
                return
            self._set_names_cfg(path)

    def _set_names_cfg(self, path: str) -> None:
        self.names_cfg = path
        if path:
            ok = Path(path).exists()
            self.names_lbl.setText(Path(path).name if ok else f"{Path(path).name} (missing!)")
            self.names_lbl.setToolTip(path)
            self.names_lbl.setStyleSheet("" if ok else "color: #c0392b;")
        else:
            self.names_lbl.setText("BioSemi inputs (A1, A2, …)")
            self.names_lbl.setToolTip("")
            self.names_lbl.setStyleSheet("color: #666;")
        self.names_clear.setVisible(bool(path))

    def _browse_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Save recordings to", self.out_dir.text())
        if path:
            self.out_dir.setText(path)

    def _save_settings(self) -> None:
        s = self.settings
        s.setValue("source", self.source.currentIndex())
        s.setValue("lib_path", self.lib_path.text())
        s.setValue("n_eeg", self.n_eeg.currentText())
        s.setValue("include_exg", "true" if self.include_exg.isChecked() else "false")
        s.setValue("names_cfg", self.names_cfg)
        s.setValue("save_rate", self.save_rate.currentData())
        s.setValue("camera", self.camera_box.currentData())
        s.setValue("out_dir", self.out_dir.text())
        s.setValue("n_show", self.n_show.currentData())
        s.setValue("scale", self.scale.value())
        s.setValue("highpass", self.highpass.currentData())
        s.setValue("lowpass", self.lowpass.currentData())
        s.setValue("notch", self.notch.currentData())
        s.setValue("offset_limit", self.offset_limit.value())
        if self.reference.count():
            s.setValue("reference", self.reference.currentText())

    # ---------------------------------------------------------------- connect
    def _toggle_connect(self) -> None:
        if self.device is None:
            self._connect()
        else:
            self._disconnect()

    def _connect(self) -> None:
        self._save_settings()
        try:
            if self.source.currentIndex() == 0:
                driver = SimulatedRingBuffer()
            else:
                driver = RingBufferLibrary(self.lib_path.text() or None)
            device = BioSemiDevice(driver)
            # Set up before start(): chunks arrive as soon as streaming begins.
            self.offsets = None
            info = device.start(self._on_chunk, lambda e: self.bridge.error.emit(str(e)))
            self.offsets = OffsetMonitor(info.sample_rate)
            self.channels = build_layout(info.speed_mode, info.mk2, int(self.n_eeg.currentText()),
                                         self.include_exg.isChecked())
            names = parse_actiview_cfg(self.names_cfg) if self.names_cfg else None
            self.channels, name_notes = apply_names(self.channels, names)
        except (OSError, AcquisitionError, ValueError, FileNotFoundError) as e:
            if "device" in locals():
                device.stop()
            QMessageBox.critical(self, "Could not connect", str(e))
            return
        self.device = device
        labels = [c.label for c in self.channels]
        self._chan_idx = np.array([c.adc_index for c in self.channels], dtype=np.intp)
        self.offset_grid.set_channels(labels, [c.is_exg for c in self.channels])
        if name_notes:
            QMessageBox.information(
                self, "Channel names",
                "Some names in the .cfg file are used twice, so these channels were renamed to keep "
                "every name unique:\n\n" + "\n".join(name_notes[:20])
                + (f"\n… and {len(name_notes) - 20} more" if len(name_notes) > 20 else ""))
        self.reference.blockSignals(True)
        self.reference.clear()
        self.reference.addItems(["None (CMS)", "Average of all"] + labels)
        i = self.reference.findText(str(self.settings.value("reference", "None (CMS)")))
        self.reference.setCurrentIndex(max(0, i))
        self.reference.blockSignals(False)
        self._update_display_channels()

        cam_index = self.camera_box.currentData()
        if cam_index is not None:
            try:
                self.camera = Camera(cam_index)
                self.camera.start()
            except RuntimeError as e:
                self.camera = None
                QMessageBox.warning(self, "Camera", str(e))

        self.status_lbl["device"].setText(
            f"{'Mk2' if info.mk2 else 'Mk1'}, speed mode {info.speed_mode.mode}, {len(self.channels)} ch")
        self._update_rate_labels()
        self.connect_btn.setText("Disconnect")
        self.record_btn.setEnabled(True)
        self._set_setup_enabled(False)

    def _disconnect(self) -> None:
        if self.session:
            self._stop_recording()
        if self.device:
            self.device.stop()
        if self.camera:
            self.camera.stop()
        self.device = self.camera = None
        self._disp = None
        self.offsets = None
        self.offset_grid.set_values(None, self.offset_limit.value())
        self.offset_summary.setText("Connect to see electrode offsets.")
        self.cms_banner.hide()
        self.connect_btn.setText("Connect")
        self.record_btn.setEnabled(False)
        self._set_setup_enabled(True)
        self.video_lbl.setText("No camera")
        for k in self.status_lbl:
            self.status_lbl[k].setText("–")
            self.status_lbl[k].setStyleSheet("")

    def _set_setup_enabled(self, on: bool) -> None:
        for w in (self.source, self.lib_path, self.n_eeg, self.include_exg, self.camera_box,
                  self.names_btn, self.names_clear):
            w.setEnabled(on)

    def _decimation(self) -> int:
        """Factor from the acquisition rate to the chosen save rate (1 = full rate)."""
        target = self.save_rate.currentData()
        if not target or self.device is None or self.device.info is None:
            return 1
        fs = self.device.info.sample_rate
        return fs // target if target < fs and fs % target == 0 else 1

    def _update_rate_labels(self) -> None:
        if self.device is None or self.device.info is None:
            return
        fs = self.device.info.sample_rate
        self.status_lbl["rate"].setText(f"{fs} Hz (AD-box switch)")
        self.status_lbl["rate"].setToolTip("Set by the speed-mode switch on the AD-box, not by software.")
        d = self._decimation()
        target = self.save_rate.currentData()
        if d == 1:
            note = f" (can't save at {target} Hz from {fs} Hz)" if target and target < fs else ""
            self.status_lbl["save"].setText(f"{fs} Hz, full rate{note}")
        else:
            self.status_lbl["save"].setText(f"{fs // d} Hz (÷{d}, anti-alias filtered)")

    def _update_display_channels(self) -> None:
        if self.device is None or self.device.info is None or not self.channels:
            return
        n = self.n_show.currentData()
        self.first_ch.blockSignals(True)
        self.first_ch.setMaximum(max(1, len(self.channels) - n + 1))
        self.first_ch.setSingleStep(n)
        self.first_ch.blockSignals(False)
        start = self.first_ch.value() - 1
        chans = self.channels[start:start + n]
        ri = self.reference.currentIndex()
        if ri <= 0:
            ref, ref_idx = "none", np.array([], dtype=np.intp)
        elif ri == 1:
            ref, ref_idx = "average", self._chan_idx
        else:
            ref, ref_idx = "channel", self._chan_idx[ri - 2:ri - 1]
        self._disp = _DisplayPath(np.array([c.adc_index for c in chans], dtype=np.intp),
                                  [c.label for c in chans], ref, ref_idx, self.device.info.sample_rate)
        for c in self.curves:
            self.plot.removeItem(c)
        self.curves = [self.plot.plot(pen=pg.mkPen(pg.intColor(i, max(len(chans), 1), maxValue=180), width=1))
                       for i in range(len(chans))]

    # ---------------------------------------------------------------- data (acq thread)
    def _on_chunk(self, chunk: Chunk) -> None:
        session = self.session
        if session is not None:
            try:
                session.add_eeg_chunk(chunk)
            except Exception as e:  # disk full etc.
                self.bridge.error.emit(f"Writing EEG failed: {e}")
        adc = chunk.adc24()
        if self.offsets is not None:
            self.offsets.push(adc)
        disp = self._disp  # one read: the GUI thread may swap in a new display path
        if disp is not None:
            disp.push(adc)
        trig = chunk.status24() & 0xFFFF
        nz = trig[trig != 0]
        if nz.size:
            self._last_trigger = int(nz[-1])

    def _on_async_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Acquisition stopped", msg)
        self._disconnect()

    def _current_offsets(self) -> dict[str, float] | None:
        if self.offsets is None or not self.channels:
            return None
        mv = self.offsets.offsets_mv(self._chan_idx)
        return None if mv is None else dict(zip((c.label for c in self.channels), mv.tolist()))

    # ---------------------------------------------------------------- recording
    def _toggle_record(self) -> None:
        if self.session is None:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self) -> None:
        subject = self.subject.text().strip()
        if not subject:
            QMessageBox.warning(self, "Subject ID", "Please enter a subject ID first.")
            return
        st = self.device.last_status
        if st is not None and not st.cms_in_range:
            if QMessageBox.question(self, "CMS/DRL out of range",
                                    "CMS is not in range (electrode contact problem). Record anyway?") \
                    != QMessageBox.Yes:
                return
        offsets = self._current_offsets()
        if offsets:
            limit = self.offset_limit.value()
            summary = summarize(list(offsets), np.array(list(offsets.values())), limit)
            if not summary.all_ok:
                names = ", ".join(describe(lab, mv) for lab, mv in summary.bad[:12])
                more = f" and {len(summary.bad) - 12} more" if len(summary.bad) > 12 else ""
                if QMessageBox.question(
                        self, "Electrode offsets",
                        f"{len(summary.bad)} of {summary.n_total} electrodes are outside ±{limit:.0f} mV:\n"
                        f"{names}{more}.\n\nRecord anyway?") != QMessageBox.Yes:
                    self.tabs.setCurrentIndex(1)
                    return
        self._save_settings()
        try:
            self.session = RecordingSession(Path(self.out_dir.text()), subject, self.device.info,
                                            self.channels, self.camera, self.notes.text(),
                                            decimation=self._decimation(), electrode_offsets_mv=offsets,
                                            names_cfg=self.names_cfg or None)
        except Exception as e:
            QMessageBox.critical(self, "Could not start recording", str(e))
            return
        self.record_btn.setText("■ Stop")
        self.record_btn.setStyleSheet("font-size: 16px; font-weight: bold; color: white; background: #c0392b;")
        self.marker_btn.setEnabled(True)
        self.subject.setEnabled(False)
        self.save_rate.setEnabled(False)

    def _stop_recording(self) -> None:
        session, self.session = self.session, None
        if session:
            session.close(electrode_offsets_mv=self._current_offsets())
            self.statusBar().showMessage(f"Saved to {session.folder}", 15000)
            self._last_folder = session.folder
            self.review_btn.setText("Review last recording…")
        self.record_btn.setText("● Record")
        self.record_btn.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.marker_btn.setEnabled(False)
        self.subject.setEnabled(True)
        self.save_rate.setEnabled(True)

    def open_review(self, folder: Path | None = None, choose: bool = False) -> None:
        """Open a review window: on ``folder``, else the latest recording, else ask."""
        folder = folder or (None if choose else self._last_folder)
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, "Open a recording folder", self.out_dir.text())
            if not folder:
                return
        win = ReviewWindow(start_dir=self.out_dir.text())
        if win.open_session(folder):
            self._review_windows.append(win)  # keep a reference, or Python garbage-collects it
            win.destroyed.connect(lambda *_, w=win: self._review_windows.remove(w)
                                  if w in self._review_windows else None)
            win.show()
        else:
            win.close()
        self._last_folder = None
        self.review_btn.setText("Review a session…")

    def _add_marker(self) -> None:
        if self.session:
            label = self.marker_text.text().strip() or "marker"
            self.session.add_marker(label)
            self.statusBar().showMessage(f"Marker '{label}' added", 3000)

    # ---------------------------------------------------------------- refresh (GUI thread)
    def _refresh(self) -> None:
        self._tick += 1
        if self.camera:
            frame = self.camera.latest_frame()
            if frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, _ = rgb.shape
                img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
                self.video_lbl.setPixmap(QPixmap.fromImage(img).scaled(
                    self.video_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            vid = f"{self.camera.measured_fps:.1f} fps"
            if self.session:
                vid += f", {self.camera.frames_written} frames, {self.camera.frames_dropped} dropped"
            self.status_lbl["video"].setText(vid)

        if self.device is not None:
            disp = self._disp
            if disp is not None and self.tabs.currentIndex() == 0:
                self._plot_signals(disp, disp.buffer.snapshot())
            st = self.device.last_status
            if st is not None:
                self._badge("cms", "in range" if st.cms_in_range else "OUT OF RANGE", st.cms_in_range)
                self._badge("battery", "LOW" if st.battery_low else "ok", not st.battery_low)
            self.status_lbl["trigger"].setText(str(self._last_trigger))
            if self._tick % 8 == 0:  # ~4 Hz keeps the numbers readable
                self._refresh_offsets()

        if self.session:
            m, s = divmod(int(self.session.elapsed), 60)
            self.status_lbl["recording"].setText(f"● {m:02d}:{s:02d} → {self.session.folder.name}")
        elif self.device:
            self.status_lbl["recording"].setText("not recording")

    def _refresh_offsets(self, force: bool = False) -> None:
        offsets = self._current_offsets()
        limit = self.offset_limit.value()
        if offsets is None:
            if force:
                self.offset_grid.set_values(None, limit)
            return
        mv = np.array(list(offsets.values()))
        st = self.device.last_status if self.device else None
        cms_ok = st is None or st.cms_in_range
        self._show_cms_banner(cms_ok)
        self.offset_grid.set_values(mv, limit, valid=cms_ok)
        if not cms_ok:
            self.offset_summary.setText("Electrode offsets can't be judged until CMS/DRL are in range.")
            self._badge("offsets", "waiting for CMS/DRL", False)
            return
        s = summarize(list(offsets), mv, limit)
        if s.all_ok:
            self.offset_summary.setText(f"All {s.n_total} electrodes within ±{limit:.0f} mV.")
            self._badge("offsets", f"all {s.n_total} within ±{limit:.0f} mV", True)
        else:
            worst = ", ".join(describe(lab, v) for lab, v in s.bad[:10])
            more = f", +{len(s.bad) - 10} more" if len(s.bad) > 10 else ""
            self.offset_summary.setText(
                f"<b>{s.n_ok} of {s.n_total}</b> within ±{limit:.0f} mV. Check: {worst}{more}")
            self._badge("offsets", f"{len(s.bad)} outside ±{limit:.0f} mV (click)", False)

    def _show_cms_banner(self, ok: bool) -> None:
        b = self.cms_banner
        b.show()
        if ok:
            b.setText("<b>CMS/DRL: in range</b> — the reference electrodes are making contact.")
            b.setStyleSheet("background: #e8f5e9; color: #1b5e20; padding: 6px; border-radius: 4px;")
        else:
            b.setText("<b>CMS/DRL: OUT OF RANGE</b> — fix these first: check that the CMS and DRL "
                      "electrodes are plugged in, gelled and touching the skin (the blue light on the "
                      "AD-box blinks until they are). Every offset below is measured against CMS, so "
                      "none of them mean anything yet.")
            b.setStyleSheet("background: #c62828; color: white; padding: 6px; border-radius: 4px;")

    def _plot_signals(self, disp: _DisplayPath, data: np.ndarray) -> None:
        if data.shape[0] < 4:
            return
        t = (np.arange(data.shape[0]) - data.shape[0]) / DISPLAY_RATE
        data = display_filter(data, DISPLAY_RATE, self.highpass.currentData(),
                              self.lowpass.currentData(), self.notch.currentData())
        spacing = self.scale.value()
        n = len(disp.labels)
        for i, c in enumerate(self.curves[:n]):
            c.setData(t, data[:, i] - i * spacing)
        ax = self.plot.getAxis("left")
        ax.setTicks([[(-i * spacing, lab) for i, lab in enumerate(disp.labels)]])
        self.plot.showAxis("left")
        self.plot.setYRange(-(n - 0.5) * spacing, 0.5 * spacing, padding=0)

    def _badge(self, key: str, text: str, good: bool) -> None:
        lbl = self.status_lbl[key]
        lbl.setText(text)
        lbl.setStyleSheet("color: #1e8449;" if good else "color: white; background: #c0392b; padding: 2px;")

    def closeEvent(self, event) -> None:
        self._save_settings()
        self._disconnect()
        super().closeEvent(event)
