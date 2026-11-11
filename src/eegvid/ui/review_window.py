"""Session review: scrub through a recording with EEG and video in sync, and edit markers."""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QSettings, Qt, QTimer
from PySide6.QtGui import QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QAbstractSpinBox, QApplication, QComboBox, QDialog, QDoubleSpinBox,
    QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from ..dsp import display_filter
from ..review import Marker, ReviewSession, format_time

PLOT_RATE = 512  # Hz; EEG is block-averaged to about this rate before display
MARKER_COLOR = "#e67e22"
TRIGGER_COLOR = "#2471a3"
PLAYHEAD_COLOR = "#c0392b"
SELECTION_BRUSH = (36, 113, 163, 55)
WINDOW_BRUSH = (230, 126, 34, 50)       # labelled time windows (markers with a duration)
WINDOW_BRUSH_SELECTED = (230, 126, 34, 95)
HIGHPASS = [("Off", 0.0), ("0.16 Hz", 0.16), ("0.5 Hz", 0.5), ("1 Hz", 1.0)]
LOWPASS = [("Off", 0.0), ("100 Hz", 100.0), ("70 Hz", 70.0), ("40 Hz", 40.0), ("30 Hz", 30.0)]
NOTCH = [("Off", 0.0), ("50 Hz", 50.0), ("60 Hz", 60.0)]
WINDOW_MIN_S, WINDOW_MAX_S = 0.25, 600.0
SCALE_MIN, SCALE_MAX = 1, 20000  # µV/div
SHOW = [("4", 4), ("8", 8), ("16", 16), ("32", 32), ("All", 0)]
SPEEDS = [("0.25×", 0.25), ("0.5×", 0.5), ("1×", 1.0), ("2×", 2.0), ("4×", 4.0)]
COL_TIME, COL_LEN, COL_LABEL, COL_NOTE, COL_SOURCE = range(5)


def _combo(items, saved=None) -> QComboBox:
    box = QComboBox()
    for text, data in items:
        box.addItem(text, data)
    if saved is not None:
        i = box.findData(type(items[0][1])(saved))
        if i >= 0:
            box.setCurrentIndex(i)
    return box


def _button(text: str, slot, tip: str = "") -> QPushButton:
    b = QPushButton(text)
    if slot is not None:
        b.clicked.connect(slot)
    b.setFocusPolicy(Qt.NoFocus)  # so Space/arrows always drive playback, not the last button
    if tip:
        b.setToolTip(tip)
    return b


class _Keys(QObject):
    """Window-wide playback keys that still let text fields receive their keys."""

    # Widgets that need these keys for typing. Number boxes (Window, µV/div, latency…) are
    # NOT listed: after clicking one, Space and ←/→ must still drive playback (↑/↓ still
    # step the box, and digits still type into it).
    TEXT = (QLineEdit, QPlainTextEdit, QTextEdit)

    def __init__(self, win: "ReviewWindow"):
        super().__init__(win)
        self.win = win

    def eventFilter(self, obj, ev) -> bool:  # noqa: N802 (Qt API)
        if ev.type() != QEvent.KeyPress or not self.win.isActiveWindow() or self.win.s is None:
            return False
        focus = QApplication.focusWidget()
        if isinstance(focus, self.TEXT) and not isinstance(focus.parent(), QAbstractSpinBox):
            return False
        w, k, mod = self.win, ev.key(), ev.modifiers()
        shift = bool(mod & Qt.ShiftModifier)
        actions = {
            Qt.Key_Space: w.toggle_play,
            Qt.Key_Left: lambda: w.step_seconds(-1) if shift else w.step_frame(-1),
            Qt.Key_Right: lambda: w.step_seconds(+1) if shift else w.step_frame(+1),
            Qt.Key_L: w.loop_btn.toggle,
            Qt.Key_M: w.add_marker,
            Qt.Key_BracketLeft: lambda: w.jump_marker(-1),
            Qt.Key_BracketRight: lambda: w.jump_marker(+1),
            Qt.Key_Home: lambda: w.set_time(0),
            Qt.Key_End: lambda: w.set_time(w.s.duration),
            Qt.Key_I: lambda: w.set_selection_edge(start=True),
            Qt.Key_O: lambda: w.set_selection_edge(start=False),
            Qt.Key_Escape: w.clear_selection,
        }
        if k in actions and not (mod & (Qt.ControlModifier | Qt.MetaModifier)):
            actions[k]()
            return True
        if ev.matches(QKeySequence.Save):
            w.save()
            return True
        return False


def _format_len(d: float) -> str:
    return "" if d <= 0 else f"{d:.2f} s" if d < 60 else format_time(d)


def _parse_len(text: str) -> float | None:
    """'1.5', '1.5 s', '0:02.5' -> seconds; '' -> 0 (a point marker); None if unreadable."""
    text = text.strip().lower().removesuffix("s").strip()
    if not text:
        return 0.0
    try:
        secs = 0.0
        for part in text.split(":"):
            secs = secs * 60 + float(part)
        return secs if secs >= 0 else None
    except ValueError:
        return None


class _EegViewBox(pg.ViewBox):
    """EEG plot interaction: the zoom lives in the window's Window/µV-per-div boxes (so
    clicking around never resets it). Scroll = zoom time, Shift/Option/Ctrl+scroll or
    scrolling over the channel names = zoom amplitude, sideways scroll = move in time,
    drag = select a time span to label."""

    def __init__(self, win: "ReviewWindow"):
        super().__init__(enableMenu=False)
        self.win = win
        self.setMouseEnabled(False, False)

    def wheelEvent(self, ev, axis=None):  # noqa: N802 (Qt API)
        if self.win.s is None:
            ev.ignore()
            return
        mods = ev.modifiers() & (Qt.ShiftModifier | Qt.AltModifier | Qt.ControlModifier | Qt.MetaModifier)
        delta = ev.delta()
        if axis == 1 or mods:
            self.win.zoom_amplitude(delta)
        elif axis is None and ev.orientation() == Qt.Horizontal:
            self.win.scroll_time(delta)
        else:
            self.win.zoom_time(delta)
        ev.accept()

    def mouseDragEvent(self, ev, axis=None):  # noqa: N802 (Qt API)
        if ev.button() != Qt.LeftButton or axis is not None or self.win.s is None:
            ev.ignore()
            return
        ev.accept()
        x0 = self.mapSceneToView(ev.buttonDownScenePos()).x()
        x1 = self.mapSceneToView(ev.scenePos()).x()
        self.win.set_selection(x0, x1)


class _ChannelPicker(QDialog):
    """Tick the channels to show. Typing names (comma-separated) filters the list."""

    def __init__(self, parent: "ReviewWindow", labels: list[str], chosen: list[str] | None):
        super().__init__(parent)
        self.win = parent
        self.setWindowTitle("Channels to show")
        self.filter = QLineEdit(placeholderText="Type names, e.g.  Fp1, Fp2   (Enter = show only these)")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self._filter_changed)
        self.filter.returnPressed.connect(self.only_matches)
        self.list = QListWidget()
        on = set(labels if chosen is None else chosen)
        for lab in labels:
            it = QListWidgetItem(lab)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if lab in on else Qt.Unchecked)
            self.list.addItem(it)
        self.list.itemChanged.connect(self._apply)
        btns = QHBoxLayout()
        for text, slot in [("All", lambda: self._check_all(True)), ("None", lambda: self._check_all(False)),
                           ("Only matches", self.only_matches)]:
            b = QPushButton(text, clicked=slot)
            b.setAutoDefault(False)
            btns.addWidget(b)
        btns.addStretch(1)
        done = QPushButton("Done", clicked=self.close)
        done.setAutoDefault(False)
        btns.addWidget(done)
        v = QVBoxLayout(self)
        v.addWidget(self.filter)
        v.addWidget(self.list, 1)
        v.addLayout(btns)
        self.resize(300, 520)

    def _terms(self) -> list[str]:
        return [x.strip().lower() for x in self.filter.text().split(",") if x.strip()]

    def _matches(self, label: str, terms: list[str]) -> bool:
        lab = label.lower()
        return not terms or any(lab == x or lab.startswith(x) for x in terms)

    def _filter_changed(self) -> None:
        terms = self._terms()
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setHidden(not self._matches(it.text(), terms))

    def _set_states(self, want) -> None:
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setCheckState(Qt.Checked if want(it) else Qt.Unchecked)
        self.list.blockSignals(False)
        self._apply()

    def _check_all(self, on: bool) -> None:
        self._set_states(lambda it: on)

    def only_matches(self) -> None:
        terms = self._terms()
        if terms:
            self._set_states(lambda it: self._matches(it.text(), terms))

    def _apply(self, *_) -> None:
        items = [self.list.item(i) for i in range(self.list.count())]
        chosen = [it.text() for it in items if it.checkState() == Qt.Checked]
        self.win.set_channels(None if len(chosen) == len(items) else chosen)


class ReviewWindow(QMainWindow):
    def __init__(self, folder: str | Path | None = None, start_dir: str | None = None):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.settings = QSettings()
        self.start_dir = start_dir or str(Path.home())
        self.s: ReviewSession | None = None
        self.t = 0.0
        self.dirty = False
        self._video = None  # review.video_reader.VideoReader (PyAV is imported on first use)
        self._frame_idx: int | None = None
        self._overlay_items: list = []
        self._timeline_items: list = []
        self._filling_table = False
        self.selection: tuple[float, float] | None = None  # session-time span being labelled
        self._chosen: list[str] | None = None               # channel labels to show (None = all)
        self._picker: _ChannelPicker | None = None
        self._scale_exact = 100.0                           # µV/div before rounding (smooth zoom)
        self._play_timer = QTimer(self, interval=30, timeout=self._play_tick)
        self._play_t0 = self._play_wall0 = 0.0
        self._build_ui()
        self._keys = _Keys(self)
        QApplication.instance().installEventFilter(self._keys)
        self.setWindowTitle("Review session")
        self.resize(1440, 920)
        if folder:
            self.open_session(folder)

    # ---------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        s = self.settings
        open_btn = _button("Open session…", self.choose_session)
        self.info_lbl = QLabel("Open a recording folder to review it.")
        self.info_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.latency = QDoubleSpinBox(minimum=-500, maximum=1000, decimals=0, singleStep=5, suffix=" ms")
        self.latency.setToolTip("Delay between the camera seeing something and the computer receiving\n"
                                "the frame. Measure it once per camera with 'eegvid-align calibrate';\n"
                                "a session's calibration.json is used automatically when present.")
        self.latency.valueChanged.connect(self._latency_changed)
        top = QHBoxLayout()
        top.addWidget(open_btn)
        top.addWidget(self.info_lbl, 1)
        top.addWidget(QLabel("Camera latency"))
        top.addWidget(self.latency)

        # video + markers
        self.video_lbl = QLabel("No video", alignment=Qt.AlignCenter)
        self.video_lbl.setMinimumSize(320, 240)
        self.video_lbl.setStyleSheet("background: #222; color: #aaa;")
        self.video_lbl.setFocusPolicy(Qt.NoFocus)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Time", "Length", "Label", "Note", "Source"])
        self.table.horizontalHeaderItem(COL_LEN).setToolTip(
            "Blank = a point marker. A length (e.g. 2.5) makes it a labelled time window.")
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(COL_TIME, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(COL_LEN, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(COL_LABEL, QHeaderView.Interactive)
        hh.setSectionResizeMode(COL_NOTE, QHeaderView.Stretch)
        hh.setSectionResizeMode(COL_SOURCE, QHeaderView.ResizeToContents)
        self.table.setColumnWidth(COL_LABEL, 160)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
                                   | QAbstractItemView.SelectedClicked)
        self.table.cellChanged.connect(self._cell_changed)
        self.table.cellClicked.connect(self._row_clicked)
        self.table.itemSelectionChanged.connect(self._selection_row_changed)
        mk_btns = QHBoxLayout()
        self.add_btn = _button("+ Add at playhead", self.add_marker,
                               "M. With a time span selected, labels the whole span.")
        mk_btns.addWidget(self.add_btn)
        for text, slot, tip in [("Move to playhead", self.move_marker, "Keeps its length"),
                                ("Set to span", self.span_marker,
                                 "Make the selected marker cover the selected time span"),
                                ("Delete", self.delete_marker, "")]:
            mk_btns.addWidget(_button(text, slot, tip))
        mk_btns.addStretch(1)
        self.save_btn = _button("Save", self.save, "Ctrl+S / ⌘S")
        mk_btns.addWidget(self.save_btn)
        self.notes = QPlainTextEdit()
        self.notes.setPlaceholderText("Review notes for this session (saved in session.json)")
        self.notes.setMaximumHeight(90)
        self.notes.textChanged.connect(self._notes_changed)
        mk_panel = QWidget()
        mv = QVBoxLayout(mk_panel)
        mv.setContentsMargins(0, 0, 0, 0)
        mv.addWidget(QLabel("Markers  (double-click a label, note or length to edit; click a row to jump)"))
        mv.addWidget(self.table, 1)
        self.sel_lbl = QLabel()
        self.sel_lbl.setStyleSheet("color: #2471a3;")
        self.sel_lbl.setWordWrap(True)
        mv.addWidget(self.sel_lbl)
        mv.addLayout(mk_btns)
        mv.addWidget(self.notes)
        upper = QSplitter(Qt.Horizontal)
        upper.addWidget(self.video_lbl)
        upper.addWidget(mk_panel)
        upper.setSizes([620, 760])

        # EEG
        self.n_show = _combo(SHOW, s.value("review/n_show", 16))
        self.first_ch = QSpinBox(minimum=1, maximum=1, prefix="from ch ")
        self.ch_btn = _button("Channels…", self.pick_channels, "Choose which channels to show")
        self.reference = QComboBox()
        self.reference.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.reference.setMinimumContentsLength(12)
        self.scale = QSpinBox(minimum=SCALE_MIN, maximum=SCALE_MAX, value=int(s.value("review/scale", 100)),
                              suffix=" µV/div", singleStep=10)
        self.scale.setToolTip("Amplitude zoom. Also: Shift+scroll (or scroll over the channel names).")
        self._scale_exact = float(self.scale.value())
        self.highpass = _combo(HIGHPASS, s.value("review/highpass", 0.16))
        self.lowpass = _combo(LOWPASS, s.value("review/lowpass", 40.0))
        self.notch = _combo(NOTCH, s.value("review/notch", 0.0))
        self.window = QDoubleSpinBox(minimum=WINDOW_MIN_S, maximum=WINDOW_MAX_S, decimals=2,
                                     singleStep=1.0, suffix=" s",
                                     value=float(s.value("review/window", 10.0)))
        self.window.setToolTip("Time zoom: seconds of EEG shown. Also: scroll (or pinch) on the EEG.")
        for w in (self.n_show, self.reference, self.highpass, self.lowpass, self.notch):
            w.currentIndexChanged.connect(self._eeg_settings_changed)
        for w in (self.first_ch, self.scale, self.window):
            w.valueChanged.connect(self._eeg_settings_changed)
        eeg_bar = QHBoxLayout()
        eeg_bar.addWidget(self.ch_btn)
        for text, w in [("Show", self.n_show), (None, self.first_ch), ("Reference", self.reference),
                        ("High-pass", self.highpass), ("Low-pass", self.lowpass), ("Notch", self.notch),
                        ("Window", self.window)]:
            if text:
                eeg_bar.addWidget(QLabel(text))
            eeg_bar.addWidget(w)
        eeg_bar.addStretch(1)
        eeg_bar.addWidget(self.scale)
        self.eeg_plot = pg.PlotWidget(viewBox=_EegViewBox(self))
        self.eeg_plot.hideButtons()  # no auto-range "A": the zoom is the Window / µV-per-div boxes
        self.eeg_plot.setToolTip("Click: move the playhead · Drag: select a time span to label\n"
                                 "Scroll or pinch: zoom time · Shift+scroll: zoom amplitude")
        self.eeg_plot.viewport().installEventFilter(self)
        self.eeg_plot.setLabel("bottom", "session time", "s")
        self.eeg_plot.setFocusPolicy(Qt.NoFocus)
        self.eeg_plot.scene().sigMouseClicked.connect(
            lambda ev: self._plot_clicked(self.eeg_plot, ev))
        self.eeg_curves: list[pg.PlotDataItem] = []
        self.eeg_playhead = pg.InfiniteLine(angle=90, pen=pg.mkPen(PLAYHEAD_COLOR, width=2))
        self.eeg_plot.addItem(self.eeg_playhead)
        self.sel_region = pg.LinearRegionItem(movable=True, brush=pg.mkBrush(*SELECTION_BRUSH),
                                              pen=pg.mkPen("#2471a3", width=1))
        self.sel_region.setZValue(5)
        self.sel_region.sigRegionChanged.connect(self._sel_region_dragged)
        self.sel_region.hide()
        self.eeg_plot.addItem(self.sel_region)
        eeg_box = QWidget()
        ev_ = QVBoxLayout(eeg_box)
        ev_.setContentsMargins(0, 0, 0, 0)
        ev_.addLayout(eeg_bar)
        ev_.addWidget(self.eeg_plot)

        # timeline + transport
        self.timeline = pg.PlotWidget()
        self.timeline.setMouseEnabled(x=False, y=False)
        self.timeline.setMenuEnabled(False)
        self.timeline.hideAxis("left")
        self.timeline.setYRange(0, 1, padding=0)
        self.timeline.setFixedHeight(110)
        self.timeline.setFocusPolicy(Qt.NoFocus)
        self.timeline.scene().sigMouseClicked.connect(lambda ev: self._plot_clicked(self.timeline, ev))
        self.view_region = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(36, 113, 163, 40))
        self.view_region.setZValue(-10)
        self.timeline.addItem(self.view_region)
        self.tl_sel_region = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(*SELECTION_BRUSH),
                                                 pen=pg.mkPen("#2471a3"))
        self.tl_sel_region.setZValue(-5)
        self.tl_sel_region.hide()
        self.timeline.addItem(self.tl_sel_region)
        self.playhead = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen(PLAYHEAD_COLOR, width=3),
                                        hoverPen=pg.mkPen(PLAYHEAD_COLOR, width=5))
        self.playhead.setZValue(10)
        self.playhead.sigDragged.connect(lambda line: self.set_time(line.value()))
        self.timeline.addItem(self.playhead)

        self.play_btn = _button("▶ Play", self.toggle_play, "Space")
        self.speed = _combo(SPEEDS, 1.0)
        self.speed.setFocusPolicy(Qt.NoFocus)
        self.speed.currentIndexChanged.connect(self._restart_play_clock)
        self.time_lbl = QLabel("–")
        self.time_lbl.setStyleSheet("font-family: Menlo, Consolas, monospace; font-size: 14px;")
        transport = QHBoxLayout()
        for text, slot, tip in [("⏮", lambda: self.jump_marker(-1), "Previous marker  ["),
                                ("−1 s", lambda: self.step_seconds(-1), "Shift+←"),
                                ("−1 frame", lambda: self.step_frame(-1), "←")]:
            transport.addWidget(_button(text, slot, tip))
        transport.addWidget(self.play_btn)
        for text, slot, tip in [("+1 frame", lambda: self.step_frame(+1), "→"),
                                ("+1 s", lambda: self.step_seconds(+1), "Shift+→"),
                                ("⏭", lambda: self.jump_marker(+1), "Next marker  ]")]:
            transport.addWidget(_button(text, slot, tip))
        transport.addWidget(self.speed)
        self.loop_btn = _button("⟲ Loop selection", None,
                                "L. While a time span is selected, playback repeats it until you pause.")
        self.loop_btn.setCheckable(True)
        self.loop_btn.setChecked(self.settings.value("review/loop", "true") in (True, "true"))
        self.loop_btn.toggled.connect(self._loop_toggled)
        transport.addWidget(self.loop_btn)
        transport.addSpacing(12)
        transport.addWidget(self.time_lbl)
        transport.addStretch(1)
        legend = QLabel(f"<span style='color:{MARKER_COLOR}'>■</span> markers  "
                        f"<span style='color:{TRIGGER_COLOR}'>■</span> triggers  "
                        f"<span style='color:{PLAYHEAD_COLOR}'>■</span> playhead (drag, or click anywhere)  "
                        f"· drag on the EEG to select a span (or I / O)")
        transport.addWidget(legend)
        bottom = QWidget()
        bv = QVBoxLayout(bottom)
        bv.setContentsMargins(0, 0, 0, 0)
        bv.addWidget(self.timeline)
        bv.addLayout(transport)

        main = QSplitter(Qt.Vertical)
        main.addWidget(upper)
        main.addWidget(eeg_box)
        main.setSizes([360, 420])
        central = QWidget()
        cv = QVBoxLayout(central)
        cv.addLayout(top)
        cv.addWidget(main, 1)
        cv.addWidget(bottom)
        self.setCentralWidget(central)
        self.statusBar()
        self._set_enabled(False)

    def _set_enabled(self, on: bool) -> None:
        for w in self.centralWidget().findChildren(QWidget):
            if w is not self.info_lbl and not isinstance(w, QSplitter):
                w.setEnabled(on)
        for b in self.centralWidget().findChildren(QPushButton):
            if b.text() == "Open session…":
                b.setEnabled(True)

    # ---------------------------------------------------------------- loading
    def choose_session(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open a recording folder", self.start_dir)
        if folder:
            self.open_session(folder)

    def open_session(self, folder: str | Path) -> bool:
        if not self._confirm_discard():
            return False
        try:
            s = ReviewSession(folder)
        except Exception as e:  # noqa: BLE001 - show anything that stops loading
            QMessageBox.critical(self, "Could not open session", str(e))
            return False
        self._stop_play()
        if self._video is not None:
            self._video.close()
            self._video = None
        if self._picker is not None:
            self._picker.close()
            self._picker = None
        self.s = s
        self.selection = None
        self.start_dir = str(Path(folder).parent)
        video_problem = ""
        if s.has_video:
            try:
                from ..review.video_reader import VideoReader
                self._video = VideoReader(s.video_path)
                if self._video.n_frames != len(s.frame_times):
                    video_problem = (f"video.mp4 has {self._video.n_frames} frames but video_frames.csv "
                                     f"lists {len(s.frame_times)}; using the first "
                                     f"{min(self._video.n_frames, len(s.frame_times))}.")
                    s.limit_video_frames(self._video.n_frames)
            except Exception as e:  # noqa: BLE001 - e.g. an mp4 never finalised after a crash
                self._video = None
                video_problem = f"The video file can't be read ({e}). The recording may not have finished."
        self._frame_idx = None
        self._set_enabled(True)

        meta = s.meta
        started = dt.datetime.fromisoformat(meta["started_wall_clock"])
        warn = "" if s.complete else ("  <span style='color:#c0392b'><b>Recording did not finish "
                                      "cleanly</b> (data up to the last saved second is shown)</span>")
        self.info_lbl.setText(
            f"<b>{meta.get('subject', '?')}</b> · {started:%Y-%m-%d %H:%M} · "
            f"{format_time(s.duration, ms=False)} · {s.fs:g} Hz · {len(s.eeg_channels)} channels · "
            f"{'video' if s.has_video else 'no video'} · {len(s.markers)} markers · "
            f"{len(s.trigger_times)} triggers{warn}")
        self.setWindowTitle(f"Review – {s.folder.name}")
        self.latency.blockSignals(True)
        if s.calibrated_latency_ms is not None:
            self.latency.setValue(s.calibrated_latency_ms)
            self.latency.setToolTip(self.latency.toolTip().split("\n\n")[0] +
                                    "\n\nFrom this session's calibration.json.")
        else:
            self.latency.setValue(float(self.settings.value("review/latency_ms", 0.0)))
        s.camera_latency_s = self.latency.value() / 1000
        self.latency.blockSignals(False)
        if self._video is None:
            self.video_lbl.setPixmap(QPixmap())
            self.video_lbl.setText(video_problem or "This session has no video")
            self.video_lbl.setWordWrap(True)
        if video_problem:
            self.statusBar().showMessage(video_problem, 20000)

        labels = [s.bdf.labels[i] for i in s.eeg_channels]
        self.reference.blockSignals(True)
        self.reference.clear()
        self.reference.addItems(["None (CMS)", "Average of all"] + labels)
        i = self.reference.findText(str(self.settings.value("review/reference", "None (CMS)")))
        self.reference.setCurrentIndex(max(0, i))
        self.reference.blockSignals(False)
        saved = str(self.settings.value("review/channels", "") or "")
        chosen = [x for x in saved.split("\n") if x in labels]
        self._chosen = chosen or None  # names not in this session: show everything
        self._update_ch_btn()
        self.notes.blockSignals(True)
        self.notes.setPlainText(s.review_notes)
        self.notes.blockSignals(False)
        self.timeline.setXRange(0, s.duration, padding=0.01)
        self.playhead.setBounds([0, s.duration])
        self._fill_table()
        self._draw_timeline()
        self._show_selection()
        self._set_dirty(False)
        self.set_time(0.0)
        return True

    # ---------------------------------------------------------------- time
    def set_time(self, t: float) -> None:
        if self.s is None:
            return
        self.t = float(np.clip(t, 0.0, self.s.duration))
        if abs(self.playhead.value() - self.t) > 1e-9:
            self.playhead.blockSignals(True)
            self.playhead.setValue(self.t)
            self.playhead.blockSignals(False)
        self._update_time_label()
        self._show_frame()
        self._draw_eeg()

    def _update_time_label(self) -> None:
        s = self.s
        clock_txt = ""
        try:
            started = dt.datetime.fromisoformat(s.meta["started_wall_clock"])
            wall = started + dt.timedelta(seconds=s.t_to_host(self.t) - s.meta["started_host_time"])
            clock_txt = f"   ({wall:%H:%M:%S})"
        except (KeyError, ValueError, TypeError):
            pass
        self.time_lbl.setText(f"{format_time(self.t)} / {format_time(s.duration, ms=False)}{clock_txt}")

    def toggle_play(self) -> None:
        if self._play_timer.isActive():
            self._stop_play()
        elif self.s is not None:
            loop = self._loop_span()
            if loop is not None and not loop[0] <= self.t < loop[1] - 1e-3:
                self.set_time(loop[0])  # looping starts at the beginning of the selection
            elif self.t >= self.s.duration - 1e-3:
                self.set_time(0)
            self._restart_play_clock()
            self._play_timer.start()
            self.play_btn.setText("❚❚ Pause")

    def _stop_play(self) -> None:
        self._play_timer.stop()
        self.play_btn.setText("▶ Play")

    def _restart_play_clock(self) -> None:
        self._play_t0, self._play_wall0 = self.t, time.perf_counter()

    def _play_tick(self) -> None:
        t = self._play_t0 + (time.perf_counter() - self._play_wall0) * self.speed.currentData()
        loop = self._loop_span()
        if loop is not None and t >= loop[1]:
            self.set_time(loop[0])
            self._restart_play_clock()
            return
        if t >= self.s.duration:
            self.set_time(self.s.duration)
            self._stop_play()
        else:
            self.set_time(t)

    def _loop_span(self) -> tuple[float, float] | None:
        sel = self.selection
        if self.loop_btn.isChecked() and sel is not None and sel[1] - sel[0] >= 0.05:
            return sel
        return None

    def _loop_toggled(self, on: bool) -> None:
        self.settings.setValue("review/loop", on)
        self.statusBar().showMessage("Loop selection on" if on else "Loop selection off", 3000)

    def step_seconds(self, direction: int) -> None:
        self.set_time(self.t + direction)
        if self._play_timer.isActive():
            self._restart_play_clock()

    def step_frame(self, direction: int) -> None:
        """Pause, then move to the previous/next video frame (or 1/30 s without video)."""
        self._stop_play()
        s = self.s
        if s is None:
            return
        ft = s.frame_times if self._video is not None else np.empty(0)
        if ft.size and ft[0] - 0.5 <= self.t <= ft[-1] + 0.5:
            # nearest frame strictly before / after the playhead (robust to rounding)
            k = (np.searchsorted(ft, self.t + 1e-6, side="right") if direction > 0
                 else np.searchsorted(ft, self.t - 1e-6, side="left") - 1)
            if 0 <= k < ft.size:
                self.set_time(float(ft[k]))
                return
        self.set_time(self.t + direction / 30)

    def _plot_clicked(self, plot: pg.PlotWidget, ev) -> None:
        if self.s is None or ev.button() != Qt.LeftButton:
            return
        vb = plot.getPlotItem().vb
        if vb.sceneBoundingRect().contains(ev.scenePos()):
            self.set_time(vb.mapSceneToView(ev.scenePos()).x())
            if self._play_timer.isActive():
                self._restart_play_clock()

    def _latency_changed(self, ms: float) -> None:
        if self.s is not None:
            self.s.camera_latency_s = ms / 1000
            if self.s.calibrated_latency_ms is None:
                self.settings.setValue("review/latency_ms", ms)
            self._frame_idx = None
            self._show_frame()

    # ---------------------------------------------------------------- video
    def _show_frame(self) -> None:
        if self._video is None:
            return
        idx = self.s.frame_at(self.t)
        if idx is None:
            self._frame_idx = None
            self.video_lbl.setPixmap(QPixmap())
            self.video_lbl.setText("No video at this time")
            return
        if idx == self._frame_idx:
            return
        rgb = self._video.read(idx)  # frame-accurate; decodes forward or seeks as needed
        if rgb is None:
            self._frame_idx = None
            return
        self._frame_idx = idx
        h, w, _ = rgb.shape
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.video_lbl.setPixmap(QPixmap.fromImage(img).scaled(
            self.video_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # ---------------------------------------------------------------- EEG
    def _eeg_settings_changed(self) -> None:
        s = self.settings
        s.setValue("review/n_show", self.n_show.currentData())
        s.setValue("review/scale", self.scale.value())
        s.setValue("review/highpass", self.highpass.currentData())
        s.setValue("review/lowpass", self.lowpass.currentData())
        s.setValue("review/notch", self.notch.currentData())
        s.setValue("review/window", self.window.value())
        if round(self._scale_exact) != self.scale.value():
            self._scale_exact = float(self.scale.value())
        if self.reference.count():
            s.setValue("review/reference", self.reference.currentText())
        self._draw_eeg()

    def _draw_eeg(self) -> None:
        s = self.s
        if s is None:
            return
        pool = self._channel_pool()
        n_show = self.n_show.currentData() or max(1, len(pool))
        self.first_ch.blockSignals(True)
        self.first_ch.setMaximum(max(1, len(pool) - n_show + 1))
        self.first_ch.setSingleStep(n_show)
        self.first_ch.setEnabled(len(pool) > n_show)
        self.first_ch.blockSignals(False)
        first = self.first_ch.value() - 1
        shown = pool[first:first + n_show]
        labels = [s.bdf.labels[i] for i in shown]

        win = self.window.value()
        t0 = self.t - win / 2
        self.view_region.setRegion([max(0.0, t0), min(s.duration, t0 + win)])
        margin = int(s.fs)  # extra data either side so display filters have no edge effects
        a = max(0, int(np.floor(t0 * s.fs)) - margin)
        b = min(s.n_samples, int(np.ceil((t0 + win) * s.fs)) + margin)
        ri = self.reference.currentIndex()
        ref_ch = s.eeg_channels[ri - 2] if ri >= 2 else None
        if ri == 1:
            read = list(s.eeg_channels)  # average reference uses every channel, shown or not
        else:
            read = list(dict.fromkeys(shown + ([ref_ch] if ref_ch is not None else [])))
        col = {c: k for k, c in enumerate(read)}
        if shown and b > a:
            data = s.bdf.read_physical(read, a, b)
            ref = (data.mean(axis=1, keepdims=True) if ri == 1
                   else data[:, [col[ref_ch]]] if ref_ch is not None else 0.0)
            data = data[:, [col[c] for c in shown]] - ref
        else:
            data = np.empty((0, len(shown)))

        f = max(1, int(s.fs // PLOT_RATE))
        n = (data.shape[0] // f) * f
        times = np.empty(0)
        if n >= 4 * f:
            data = data[:n].reshape(-1, f, data.shape[1]).mean(axis=1)
            times = (a + np.arange(data.shape[0]) * f + (f - 1) / 2) / s.fs
            data = display_filter(data, s.fs / f, self.highpass.currentData(),
                                  self.lowpass.currentData(), self.notch.currentData())
            keep = (times >= t0) & (times <= t0 + win)
            times, data = times[keep], data[keep]

        spacing = self.scale.value()
        while len(self.eeg_curves) < len(shown):
            c = self.eeg_plot.plot(pen=pg.mkPen(pg.intColor(len(self.eeg_curves), 9, maxValue=170), width=1))
            c.setDownsampling(auto=True, method="peak")
            c.setClipToView(True)
            self.eeg_curves.append(c)
        for i, c in enumerate(self.eeg_curves):
            if i < len(shown) and times.size:
                c.setData(times, data[:, i] - i * spacing)
            else:
                c.clear()
        ax = self.eeg_plot.getAxis("left")
        ax.setTicks([[(-i * spacing, lab) for i, lab in enumerate(labels)]])
        self.eeg_plot.setYRange(-(len(shown) - 0.5) * spacing, 0.5 * spacing, padding=0)
        self.eeg_plot.setXRange(t0, t0 + win, padding=0)
        self.eeg_playhead.setValue(self.t)

        for item in self._overlay_items:
            self.eeg_plot.removeItem(item)
        self._overlay_items = []
        tt = s.trigger_times
        sel = np.flatnonzero((tt >= t0) & (tt <= t0 + win))
        for k in sel[:200]:
            line = pg.InfiniteLine(tt[k], angle=90, pen=pg.mkPen(TRIGGER_COLOR, style=Qt.DashLine),
                                   label=str(int(s.trigger_values[k])),
                                   labelOpts={"position": 0.03, "color": TRIGGER_COLOR})
            self.eeg_plot.addItem(line)
            self._overlay_items.append(line)
        picked = self._selected_marker()
        top = 0.5 * spacing
        for m in s.markers:
            mt, me = s.marker_time(m), s.marker_end(m)
            if me < t0 or mt > t0 + win:
                continue
            sel = m is picked
            if m.duration > 0:
                item = pg.LinearRegionItem(
                    [mt, me], movable=sel,
                    brush=pg.mkBrush(*(WINDOW_BRUSH_SELECTED if sel else WINDOW_BRUSH)),
                    pen=pg.mkPen(MARKER_COLOR, width=2 if sel else 1))
                item.setZValue(-20)
                if sel:
                    item.setToolTip("Drag the edges to change this window, or the middle to move it")
                    item.sigRegionChangeFinished.connect(
                        lambda it, m=m: self._marker_dragged(m, *it.getRegion()))
                text = pg.TextItem(m.label, color=MARKER_COLOR, fill=(255, 255, 255, 200), anchor=(0, 0))
                text.setPos(max(mt, t0), top)
                self.eeg_plot.addItem(text)
                self._overlay_items.append(text)
            else:
                item = pg.InfiniteLine(mt, angle=90, movable=sel, pen=pg.mkPen(MARKER_COLOR, width=3 if sel else 2),
                                       hoverPen=pg.mkPen(MARKER_COLOR, width=5), label=m.label,
                                       labelOpts={"position": 0.97, "color": MARKER_COLOR,
                                                  "fill": (255, 255, 255, 200)})
                if sel:
                    item.setToolTip("Drag to move this marker")
                    item.sigPositionChangeFinished.connect(lambda it, m=m: self._marker_dragged(m, it.value()))
            self.eeg_plot.addItem(item)
            self._overlay_items.append(item)

    # ---------------------------------------------------------------- timeline
    def _draw_timeline(self) -> None:
        s = self.s
        for item in self._timeline_items:
            self.timeline.removeItem(item)
        self._timeline_items = []
        tt = s.trigger_times
        if tt.size:
            x = np.repeat(tt, 2)
            y = np.tile([0.0, 0.3], tt.size)
            ticks = pg.PlotDataItem(x, y, connect="pairs", pen=pg.mkPen(TRIGGER_COLOR, width=1))
            self.timeline.addItem(ticks)
            self._timeline_items.append(ticks)
        for i, m in enumerate(s.markers):
            if m.duration > 0:
                region = pg.LinearRegionItem([s.marker_time(m), s.marker_end(m)], movable=False,
                                             brush=pg.mkBrush(*WINDOW_BRUSH), pen=pg.mkPen(MARKER_COLOR))
                region.setZValue(-15)
                self.timeline.addItem(region)
                self._timeline_items.append(region)
            line = pg.InfiniteLine(s.marker_time(m), angle=90, pen=pg.mkPen(MARKER_COLOR, width=2),
                                   label=m.label, labelOpts={"position": 0.85 - 0.18 * (i % 3),
                                                             "color": MARKER_COLOR,
                                                             "fill": (255, 255, 255, 200)})
            self.timeline.addItem(line)
            self._timeline_items.append(line)

    # ---------------------------------------------------------------- markers
    def _fill_table(self, select: Marker | None = None) -> None:
        s = self.s
        self._filling_table = True
        s.markers.sort(key=lambda m: m.host_time)
        self.table.setRowCount(len(s.markers))
        for r, m in enumerate(s.markers):
            items = [QTableWidgetItem(format_time(s.marker_time(m))), QTableWidgetItem(_format_len(m.duration)),
                     QTableWidgetItem(m.label), QTableWidgetItem(m.note),
                     QTableWidgetItem("live" if m.source == "recorded" else m.source)]
            for c in (COL_TIME, COL_SOURCE):
                items[c].setFlags(items[c].flags() & ~Qt.ItemIsEditable)
            items[COL_SOURCE].setToolTip("live = added during recording; review = added here")
            for c, it in enumerate(items):
                self.table.setItem(r, c, it)
        if select is not None and select in s.markers:
            self.table.selectRow(s.markers.index(select))
        self._filling_table = False

    def _selected_marker(self) -> Marker | None:
        rows = self.table.selectionModel().selectedRows()
        return self.s.markers[rows[0].row()] if rows and self.s else None

    def _row_clicked(self, row: int, col: int) -> None:
        if self.s is not None and 0 <= row < len(self.s.markers):
            self.set_time(self.s.marker_time(self.s.markers[row]))

    def _cell_changed(self, row: int, col: int) -> None:
        if self._filling_table or self.s is None:
            return
        m = self.s.markers[row]
        text = self.table.item(row, col).text()
        if col == COL_LABEL:
            m.label = text.strip() or "marker"
        elif col == COL_NOTE:
            m.note = text
        elif col == COL_LEN:
            d = _parse_len(text)
            if d is None:
                self.statusBar().showMessage("Length: type seconds (e.g. 2.5), or leave it blank", 5000)
                QTimer.singleShot(0, lambda: self._fill_table(m))
                return
            m.duration = d
            QTimer.singleShot(0, lambda: self._fill_table(m))
        self._set_dirty(True)
        self._draw_timeline()
        self._draw_eeg()

    def add_marker(self) -> None:
        if self.s is None:
            return
        if self.selection is not None and self.selection[1] > self.selection[0]:
            m = self.s.new_marker(self.selection[0], end=self.selection[1])
            self.clear_selection()
        else:
            m = self.s.new_marker(self.t)
        self._marker_edited(m)
        row = self.s.markers.index(m)
        self.table.setCurrentCell(row, COL_LABEL)
        self.table.editItem(self.table.item(row, COL_LABEL))

    def move_marker(self) -> None:
        m = self._selected_marker()
        if m is None:
            self.statusBar().showMessage("Select a marker in the list first", 4000)
            return
        self.s.set_marker_span(m, self.t, self.t + m.duration if m.duration else None)
        self._marker_edited(m)

    def span_marker(self) -> None:
        m = self._selected_marker()
        if m is None or self.selection is None or self.selection[1] <= self.selection[0]:
            self.statusBar().showMessage(
                "Select a marker in the list and drag across the EEG to select a time span first", 5000)
            return
        self.s.set_marker_span(m, *self.selection)
        self.clear_selection()
        self._marker_edited(m)

    def _marker_dragged(self, m: Marker, start: float, end: float | None = None) -> None:
        if self.s is None or m not in self.s.markers:
            return
        start = float(np.clip(start, 0, self.s.duration))
        if end is not None:
            end = float(np.clip(end, 0, self.s.duration))
        self.s.set_marker_span(m, start, end)
        QTimer.singleShot(0, lambda: self._marker_edited(m))  # don't rebuild the item mid-signal

    def _selection_row_changed(self) -> None:
        if not self._filling_table:
            self._draw_eeg()

    def delete_marker(self) -> None:
        m = self._selected_marker()
        if m is None:
            self.statusBar().showMessage("Select a marker in the list first", 4000)
            return
        self.s.markers.remove(m)
        self._marker_edited(None)
        if m.source == "recorded":
            self.statusBar().showMessage(
                "Deleted. The live-recorded markers are kept in markers_recorded.csv when you save.", 6000)

    def _marker_edited(self, select: Marker | None) -> None:
        self._fill_table(select)
        self._draw_timeline()
        self._draw_eeg()
        self._set_dirty(True)

    def jump_marker(self, direction: int) -> None:
        if self.s is None or not self.s.markers:
            return
        times = np.array([self.s.marker_time(m) for m in self.s.markers])
        cand = times[times > self.t + 1e-3] if direction > 0 else times[times < self.t - 1e-3]
        if cand.size:
            self.set_time(cand.min() if direction > 0 else cand.max())

    def _notes_changed(self) -> None:
        self._set_dirty(True)

    # ---------------------------------------------------------------- zoom
    def zoom_time(self, delta: float) -> None:
        """Scroll up = zoom in (fewer seconds shown), around the playhead."""
        self.window.setValue(float(np.clip(self.window.value() * 1.0015 ** -delta, WINDOW_MIN_S, WINDOW_MAX_S)))

    def zoom_amplitude(self, delta: float) -> None:
        """Scroll up = bigger traces (fewer µV per division)."""
        self._scale_exact = float(np.clip(self._scale_exact * 1.0015 ** -delta, SCALE_MIN, SCALE_MAX))
        if round(self._scale_exact) != self.scale.value():
            self.scale.setValue(round(self._scale_exact))

    def scroll_time(self, delta: float) -> None:
        self.set_time(self.t - delta / 120 * self.window.value() * 0.1)
        if self._play_timer.isActive():
            self._restart_play_clock()

    def eventFilter(self, obj, ev) -> bool:  # noqa: N802 (Qt API) - trackpad pinch on the EEG
        if ev.type() == QEvent.NativeGesture and self.s is not None:
            if ev.gestureType() == Qt.ZoomNativeGesture:
                self.window.setValue(float(np.clip(self.window.value() / (1 + ev.value()),
                                                   WINDOW_MIN_S, WINDOW_MAX_S)))
                return True
        return super().eventFilter(obj, ev)

    # ---------------------------------------------------------------- time span selection
    def set_selection(self, a: float, b: float) -> None:
        if self.s is None:
            return
        a, b = sorted((float(np.clip(a, 0, self.s.duration)), float(np.clip(b, 0, self.s.duration))))
        self.selection = (a, b)
        self._show_selection()

    def set_selection_edge(self, start: bool) -> None:
        """I / O keys: start or end the selection at the playhead (works while playing)."""
        a, b = self.selection or (self.t, self.t)
        if start:
            a, b = self.t, max(b, self.t)
        else:
            a, b = min(a, self.t), self.t
        self.set_selection(a, b)

    def clear_selection(self) -> None:
        self.selection = None
        self._show_selection()

    def _sel_region_dragged(self) -> None:
        if self.selection is not None:
            a, b = self.sel_region.getRegion()
            self.selection = (max(0.0, a), min(self.s.duration, b))
            self._show_selection(move_items=False)

    def _show_selection(self, move_items: bool = True) -> None:
        sel = self.selection
        self.sel_region.setVisible(sel is not None)
        self.tl_sel_region.setVisible(sel is not None)
        self.loop_btn.setEnabled(sel is not None and self.s is not None)
        if sel is None:
            self.sel_lbl.setText("")
            self.add_btn.setText("+ Add at playhead")
            return
        if move_items:  # (not while the user is dragging the region itself)
            self.sel_region.blockSignals(True)
            self.sel_region.setRegion(sel)
            self.sel_region.blockSignals(False)
        self.tl_sel_region.setRegion(sel)
        a, b = sel
        self.sel_lbl.setText(f"Selected {format_time(a)} – {format_time(b)} ({_format_len(b - a) or '0 s'}). "
                             "Play loops it (⟲); M or '+ Label span' labels it; Esc clears.")
        self.add_btn.setText("+ Label span")

    # ---------------------------------------------------------------- channels
    def _channel_pool(self) -> list[int]:
        """BDF indices of the channels the user chose to look at (all EEG/EXG by default)."""
        s = self.s
        if self._chosen is None:
            return list(s.eeg_channels)
        want = set(self._chosen)
        return [i for i in s.eeg_channels if s.bdf.labels[i] in want]

    def pick_channels(self) -> None:
        if self.s is None:
            return
        if self._picker is None:
            labels = [self.s.bdf.labels[i] for i in self.s.eeg_channels]
            self._picker = _ChannelPicker(self, labels, self._chosen)
            self._picker.finished.connect(lambda *_: setattr(self, "_picker", None))
            self._picker.setAttribute(Qt.WA_DeleteOnClose)
        self._picker.show()
        self._picker.raise_()
        self._picker.activateWindow()

    def set_channels(self, chosen: list[str] | None) -> None:
        self._chosen = chosen
        self.settings.setValue("review/channels", "" if chosen is None else "\n".join(chosen))
        self.first_ch.setValue(1)
        self._update_ch_btn()
        self._draw_eeg()

    def _update_ch_btn(self) -> None:
        n_all = len(self.s.eeg_channels) if self.s else 0
        if self._chosen is None:
            self.ch_btn.setText(f"Channels: all {n_all}")
        elif len(self._chosen) <= 3:
            self.ch_btn.setText("Channels: " + ", ".join(self._chosen) if self._chosen else "Channels: none")
        else:
            self.ch_btn.setText(f"Channels: {len(self._chosen)} of {n_all}")

    # ---------------------------------------------------------------- saving
    def _set_dirty(self, dirty: bool) -> None:
        self.dirty = dirty
        self.save_btn.setText("Save*" if dirty else "Save")
        self.save_btn.setStyleSheet("font-weight: bold; color: #c0392b;" if dirty else "")
        if self.s is not None:
            self.setWindowTitle(f"Review – {self.s.folder.name}{' (unsaved changes)' if dirty else ''}")

    def save(self) -> bool:
        if self.s is None:
            return True
        try:
            self.s.save_markers()
            self.s.save_review_notes(self.notes.toPlainText())
        except OSError as e:
            QMessageBox.critical(self, "Could not save", str(e))
            return False
        self._set_dirty(False)
        self.statusBar().showMessage(f"Saved markers and notes to {self.s.folder}", 5000)
        return True

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        r = QMessageBox.question(self, "Unsaved changes",
                                 "Save your marker and note changes first?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            return self.save()
        return r == QMessageBox.Discard

    def closeEvent(self, event) -> None:
        if not self._confirm_discard():
            event.ignore()
            return
        self._stop_play()
        if self._picker is not None:
            self._picker.close()
        QApplication.instance().removeEventFilter(self._keys)
        if self._video is not None:
            self._video.close()
        super().closeEvent(event)
