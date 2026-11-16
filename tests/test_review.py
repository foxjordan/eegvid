import csv
import shutil
import time

import cv2
import numpy as np
import pytest

from eegvid import clock
from eegvid.analysis.align import align_session
from eegvid.biosemi import BioSemiDevice, SimulatedRingBuffer
from eegvid.biosemi.protocol import build_layout
from eegvid.recording import RecordingSession
from eegvid.review import BdfReader, ReviewSession, format_time
from eegvid.review.session_data import MARKERS_BACKUP
from eegvid.review.video_reader import VideoReader
from eegvid.video import open_video_writer

FPS = 29.97  # like real webcams; a non-integer rate is what exposed OpenCV's seek errors
BITS, BLOCK = 12, 20


def stamp(k: int) -> np.ndarray:
    """A 240x320 frame showing k as 12 black/white blocks (survives compression)."""
    img = np.full((240, 320, 3), 128, np.uint8)
    for b in range(BITS):
        img[100:100 + BLOCK, 20 + b * 23:20 + b * 23 + BLOCK] = 255 if (k >> b) & 1 else 0
    return img


def unstamp(img: np.ndarray) -> int:
    return sum(1 << b for b in range(BITS)
               if img[100 + BLOCK // 2, 20 + b * 23 + BLOCK // 2].mean() > 128)


@pytest.fixture(scope="module")
def session_dir(tmp_path_factory):
    """A real simulator session (2048 Hz) plus a synthetic video whose frame k shows
    the number k (see ``stamp``), timestamped on the shared clock while recording."""
    root = tmp_path_factory.mktemp("rec")
    dev = BioSemiDevice(SimulatedRingBuffer(speed_mode=4, trigger_every=0.5))
    holder = {}
    info = dev.start(lambda c: holder.get("s") and holder["s"].add_eeg_chunk(c))
    chans = build_layout(info.speed_mode, info.mk2, 32, include_exg=True)
    s = RecordingSession(root, "rev", info, chans)
    holder["s"] = s
    frames = []
    writer, _ = open_video_writer(s.folder / "video.mp4", FPS, (320, 240))  # same path as the recorder
    t_end = clock.now() + 2.4
    k = 0
    while clock.now() < t_end:
        writer.write(stamp(k))
        frames.append((k, clock.now()))
        k += 1
        if k == 20:
            s.add_marker("live one")
        time.sleep(1 / FPS)
    writer.release()
    holder.pop("s")
    s.close()
    dev.stop()
    with open(s.folder / "video_frames.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "host_time"])
        w.writerows([(i, f"{t:.6f}") for i, t in frames])
    return s.folder


def test_bdf_reader_matches_mne(session_dir):
    mne = pytest.importorskip("mne")
    r = BdfReader(session_dir / "eeg.bdf")
    raw = mne.io.read_raw_bdf(session_dir / "eeg.bdf", preload=True, verbose="error")
    assert r.labels == raw.ch_names and r.sample_rate == raw.info["sfreq"]
    assert r.n_samples == raw.n_times
    picks = [0, 5, r.labels.index("EXG3")]
    for start, stop in [(0, 100), (2000, 2100), (1, raw.n_times)]:  # 2000-2100 spans a record boundary
        mine = r.read_physical(picks, start, stop)
        theirs = raw.get_data(picks=picks, start=start, stop=stop).T * 1e6
        np.testing.assert_allclose(mine, theirs, rtol=0, atol=1e-4)  # 1 LSB = 0.03 uV
    onsets, values = r.triggers()
    ev = mne.find_events(raw, stim_channel="Status", shortest_event=1, verbose="error")
    np.testing.assert_array_equal(onsets, ev[:, 0])
    np.testing.assert_array_equal(values, ev[:, 2])


def test_bdf_reader_handles_unfinished_header(session_dir, tmp_path):
    p = tmp_path / "crashed.bdf"
    shutil.copy(session_dir / "eeg.bdf", p)
    with open(p, "r+b") as f:
        f.seek(236)
        f.write(b"-1      ")
    assert BdfReader(p).n_samples == BdfReader(session_dir / "eeg.bdf").n_samples


def test_session_timeline(session_dir):
    s = ReviewSession(session_dir)
    assert s.fs == 2048 and s.has_video and s.complete
    assert 2.0 < s.duration < 4.0  # 2.4 s of frames, plus start-up (slower under Rosetta)
    assert s.n_samples == s.meta["eeg"]["samples_recorded"]
    assert len(s.trigger_times) >= 4
    np.testing.assert_allclose(np.diff(s.trigger_times), 0.5, atol=1 / s.fs)
    assert [m.label for m in s.markers] == ["live one"]
    assert 0 < s.marker_time(s.markers[0]) < s.duration
    for t in (0.1, 1.234, 2.0):
        assert abs(s.host_to_t(s.t_to_host(t)) - t) < 1e-9


def test_frame_lookup_and_seek(session_dir):
    s = ReviewSession(session_dir)
    ft = s.frame_times
    video = VideoReader(s.video_path)
    assert video.n_frames == len(ft)
    for k in range(len(ft)):
        assert s.frame_at(ft[k] + 1e-4) == k
        assert unstamp(video.read(k)) == k  # the frame shown is the one timestamped
    video.close()
    assert s.frame_at(ft[0] - 2) is None and s.frame_at(ft[-1] + 2) is None
    # Camera latency (host-clock seconds) shifts video earlier on the timeline by that
    # amount, converted to EEG seconds by the fitted clock rate.
    s.camera_latency_s = 0.1
    shift = 0.1 / (s.fit.seconds_per_sample * s.fs)
    assert abs(shift - 0.1) < 0.1 * 1e-2
    np.testing.assert_allclose(s.frame_times, ft - shift, rtol=0, atol=1e-9)


def test_marker_editing_roundtrip(session_dir, tmp_path):
    folder = tmp_path / "copy"
    shutil.copytree(session_dir, folder)
    original = (folder / "markers.csv").read_text()
    s = ReviewSession(folder)
    live_t = s.marker_time(s.markers[0])  # depends on how fast the machine recorded the fixture
    s.markers[0].label = "eyes closed"
    s.markers[0].note = "participant yawned"
    artifact_t = (live_t + s.duration) / 2
    s.new_marker(artifact_t, "artifact")
    s.save_markers()
    s.save_review_notes("Good session.\nSome movement at the end.")
    s.new_marker(live_t / 2, "second edit")
    s.save_markers()

    assert (folder / MARKERS_BACKUP).read_text() == original  # first-save backup, never overwritten
    again = ReviewSession(folder)
    assert [(m.label, m.source) for m in again.markers] == [
        ("second edit", "review"), ("eyes closed", "recorded"), ("artifact", "review")]
    assert again.markers[1].note == "participant yawned"
    assert abs(again.marker_time(again.markers[2]) - artifact_t) < 1e-6
    assert again.review_notes.startswith("Good session.")
    assert again.meta["eeg"]["sample_rate"] == 2048  # rest of session.json intact
    align_session(folder)  # the offline alignment tool still reads the edited file
    rows = list(csv.DictReader(open(folder / "markers_aligned.csv")))
    assert [r["label"] for r in rows] == ["second edit", "eyes closed", "artifact"]


def test_format_time():
    assert format_time(83.25) == "01:23.250"
    assert format_time(3725.5, ms=False) == "1:02:05"


def test_video_reader_is_frame_accurate(tmp_path):
    """Sequential, random and backward access all return exactly frame k (OpenCV's
    own seeking fails this: off by one on some frames, and on Intel Macs it misreads
    29.97 fps as 29.001 and drifts)."""
    import random
    p = tmp_path / "v.mp4"
    writer, codec = open_video_writer(p, FPS, (320, 240))
    n = 900  # 30 s
    for k in range(n):
        writer.write(stamp(k))
    writer.release()
    v = VideoReader(p)
    assert v.n_frames == n
    assert [unstamp(v.read(k)) for k in range(n)] == list(range(n))
    ks = random.Random(0).sample(range(n), 150)
    assert [unstamp(v.read(k)) for k in ks] == ks
    back = list(range(n - 1, n - 100, -1))
    assert [unstamp(v.read(k)) for k in back] == back
    assert v.read(-1) is None and v.read(n) is None
    v.close()


def test_video_reader_rejects_unfinished_file(tmp_path, session_dir):
    """A recording that crashed leaves an mp4 without its index; the reader must
    raise (the review window then shows EEG with a message instead of video)."""
    data = (session_dir / "video.mp4").read_bytes()
    p = tmp_path / "cut.mp4"
    p.write_bytes(data[: len(data) // 2])
    with pytest.raises(Exception):
        VideoReader(p)


def test_time_window_markers_roundtrip(session_dir, tmp_path):
    folder = tmp_path / "windows"
    shutil.copytree(session_dir, folder)
    s = ReviewSession(folder)
    w = s.new_marker(1.6, "blink", end=0.4)          # reversed span is put in order
    assert abs(s.marker_time(w) - 0.4) < 1e-6 and abs(w.duration - 1.2) < 1e-9
    next(m for m in s.markers if m.label == "live one").label = "point"
    s.save_markers()
    again = ReviewSession(folder)
    blink = next(m for m in again.markers if m.label == "blink")
    assert abs(again.marker_time(blink) - 0.4) < 1e-6 and abs(again.marker_end(blink) - 1.6) < 1e-6
    assert next(m for m in again.markers if m.label == "point").duration == 0
    align_session(folder)
    rows = {r["label"]: r for r in csv.DictReader(open(folder / "markers_aligned.csv"))}
    assert abs(float(rows["blink"]["duration_s"]) - 1.2) < 1e-6
    assert int(rows["blink"]["end_sample"]) - int(rows["blink"]["eeg_sample"]) == round(1.2 * 2048)
    assert rows["point"]["duration_s"] == "" and rows["point"]["end_sample"] == ""


def test_review_window_channels_zoom_and_spans(session_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication, QSettings
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    QCoreApplication.setOrganizationName("eegvid-tests")
    QSettings().clear()
    from eegvid.ui.review_window import ReviewWindow

    folder = tmp_path / "gui"
    shutil.copytree(session_dir, folder)
    win = ReviewWindow(folder)
    try:
        win.set_time(1.0)
        # channels: show only two, in recording order, with any reference
        win.set_channels(["Fp1", "A2"] if "Fp1" in win.s.bdf.labels else ["A2", "A1"])
        win.reference.setCurrentIndex(1)  # average of all
        shown = [c for c in win.eeg_curves if c.xData is not None and len(c.xData)]
        assert len(shown) == 2 and win._channel_pool() == [0, 1]
        assert QSettings().value("review/channels") == "A2\nA1"
        # zoom survives clicks, playhead moves and redraws
        win.window.setValue(2.0)
        win.zoom_time(240)                 # scroll up = zoom in
        zoomed = win.window.value()
        assert zoomed < 2.0
        win.zoom_amplitude(-240)
        scale = win.scale.value()
        win.set_time(1.5)
        x0, x1 = win.eeg_plot.getPlotItem().vb.viewRange()[0]
        assert abs((x1 - x0) - zoomed) < 1e-6 and win.scale.value() == scale
        # label a span: select, add, then it is a window marker
        win.set_selection(1.4, 0.6)
        assert win.selection == (0.6, 1.4)
        n = len(win.s.markers)
        win.add_marker()
        app.processEvents()
        assert win.selection is None and len(win.s.markers) == n + 1
        span = next(m for m in win.s.markers if m.duration > 0)
        assert abs(win.s.marker_time(span) - 0.6) < 1e-6 and abs(span.duration - 0.8) < 1e-6
        # I / O keys build a selection from the playhead
        win.set_time(0.2)
        win.set_selection_edge(start=True)
        win.set_time(0.5)
        win.set_selection_edge(start=False)
        assert win.selection == (0.2, 0.5)
        win.save()
        assert any(m.duration > 0 for m in ReviewSession(folder).markers)
    finally:
        win.dirty = False
        win.close()


def test_review_window_keys_frames_and_loop(session_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication, QEvent, QSettings, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    QCoreApplication.setOrganizationName("eegvid-tests")
    QSettings().clear()
    from eegvid.ui.review_window import ReviewWindow

    folder = tmp_path / "keys"
    shutil.copytree(session_dir, folder)
    win = ReviewWindow(folder)
    win.show()
    win.activateWindow()
    app.processEvents()

    def press(key, mod=Qt.NoModifier):
        target = QApplication.focusWidget() or win
        QApplication.sendEvent(target, QKeyEvent(QEvent.KeyPress, key, mod))
        app.processEvents()

    try:
        if not win.isActiveWindow():
            pytest.skip("offscreen platform did not activate the window")
        ft = win.s.frame_times
        k = len(ft) // 2  # slow machines write fewer frames into the fixture
        assert k >= 2
        win.set_time(float(ft[k]))
        # a number box has focus (e.g. after changing the Window): keys still drive playback
        win.window.setFocus()
        app.processEvents()
        before = win.window.value()
        press(Qt.Key_Right)
        assert win._frame_idx == k + 1 and abs(win.t - ft[k + 1]) < 1e-9
        press(Qt.Key_Left)
        press(Qt.Key_Left)
        assert win._frame_idx == k - 1
        press(Qt.Key_Right, Qt.ShiftModifier)
        assert abs(win.t - min(ft[k - 1] + 1, win.s.duration)) < 1e-9  # clipped at the end
        assert win.window.value() == before
        press(Qt.Key_Space)
        assert win._play_timer.isActive()
        press(Qt.Key_Right)  # stepping a frame pauses
        assert not win._play_timer.isActive()

        # loop: playback wraps inside the selection until paused
        win.set_selection(0.5, 0.8)
        win.loop_btn.setChecked(True)
        win.set_time(1.5)
        win.toggle_play()
        assert abs(win.t - 0.5) < 1e-9  # starts at the selection
        win._play_wall0 -= 0.35        # pretend 0.35 s passed: past the end
        win._play_tick()
        assert abs(win.t - 0.5) < 1e-9 and win._play_timer.isActive()
        win._play_wall0 -= 0.1
        win._play_tick()
        assert 0.55 < win.t < 0.65
        press(Qt.Key_Space)
        assert not win._play_timer.isActive()
    finally:
        win.dirty = False
        win.close()
