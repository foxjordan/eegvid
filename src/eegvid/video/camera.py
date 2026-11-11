"""Webcam capture with per-frame timestamps on the shared clock.

Changes from the original video_recorder.py:
  * Capture, disk writing and preview run on separate threads, so a slow
    disk or the GUI can't stall capture and cause dropped frames.
  * Timestamps come from clock.now() (monotonic, same clock as the EEG),
    not time.time().
  * The real frame rate is measured. Webcams rarely deliver exactly the
    requested FPS, and frames_*.csv (not the mp4 frame rate) is the source of
    truth for timing.
"""

from __future__ import annotations

import csv
import queue
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

from .. import clock


def _backend() -> int:
    if sys.platform == "win32":
        return cv2.CAP_DSHOW  # MSMF can take many seconds to open some webcams
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


# Codecs to try, in order. macOS: H.264 via AVFoundation (OpenCV's Intel-Mac wheel has no
# FFmpeg, so mp4v cannot be written there at all). Windows: mp4v via FFmpeg (OpenCV's
# H.264 needs a separate openh264 DLL on Windows).
CODECS = ["avc1", "mp4v"] if sys.platform == "darwin" else ["mp4v", "avc1"]


def open_video_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, str]:
    """Open an .mp4 writer with the first codec that works here. Returns (writer, codec)."""
    for codec in CODECS:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError(f"Could not create video file {path} (tried codecs {', '.join(CODECS)})")


def list_cameras(max_index: int = 5) -> list[int]:
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, _backend())
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found


class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: float = 30.0):
        self.index, self.req_width, self.req_height, self.req_fps = index, width, height, fps
        self.width = self.height = 0
        self.measured_fps = 0.0
        self.frames_written = 0
        self.frames_dropped = 0
        self.codec = ""
        self._cap: cv2.VideoCapture | None = None
        self._latest: np.ndarray | None = None
        self._latest_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rec_queue: queue.Queue | None = None
        self._writer_thread: threading.Thread | None = None
        self._frame_id = 0

    # -- streaming -------------------------------------------------------------
    def start(self) -> None:
        cap = cv2.VideoCapture(self.index, _backend())
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.index}. On macOS, check "
                               "System Settings > Privacy > Camera.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_height)
        cap.set(cv2.CAP_PROP_FPS, self.req_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # smaller buffer = less, and steadier, latency
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cap = cap
        self._stop.clear()
        self._thread = threading.Thread(target=self._capture_loop, name="camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.stop_recording()
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
        self._cap = None

    def latest_frame(self) -> np.ndarray | None:
        with self._latest_lock:
            return self._latest

    def _capture_loop(self) -> None:
        recent: list[float] = []
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            t = clock.now()
            if not ok:
                continue
            recent.append(t)
            if len(recent) > 60:
                recent.pop(0)
            if len(recent) > 1:
                self.measured_fps = (len(recent) - 1) / (recent[-1] - recent[0])
            with self._latest_lock:
                self._latest = frame
            q = self._rec_queue
            if q is not None:
                try:
                    q.put_nowait((frame, t))
                except queue.Full:
                    self.frames_dropped += 1

    # -- recording -------------------------------------------------------------
    def start_recording(self, video_path: Path, timestamps_path: Path) -> None:
        fps = self.measured_fps if self.measured_fps > 1 else self.req_fps
        writer, self.codec = open_video_writer(video_path, round(fps, 2), (self.width, self.height))
        self.frames_written = self.frames_dropped = 0
        self._frame_id = 0
        q: queue.Queue = queue.Queue(maxsize=int(fps * 4))
        self._writer_thread = threading.Thread(
            target=self._writer_loop, args=(q, writer, timestamps_path), name="video-writer", daemon=True)
        self._writer_thread.start()
        self._rec_queue = q

    def stop_recording(self) -> None:
        q, self._rec_queue = self._rec_queue, None
        if q is not None:
            q.put(None)
            self._writer_thread.join(timeout=10)
            self._writer_thread = None

    def _writer_loop(self, q: queue.Queue, writer: cv2.VideoWriter, ts_path: Path) -> None:
        with open(ts_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_id", "host_time"])
            while True:
                item = q.get()
                if item is None:
                    break
                frame, t = item
                writer.write(frame)
                w.writerow([self._frame_id, f"{t:.6f}"])
                self._frame_id += 1
                self.frames_written += 1
        writer.release()
