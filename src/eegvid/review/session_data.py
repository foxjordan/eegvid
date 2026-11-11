"""Loading a recorded session for review (no GUI code here).

Timeline: everything is placed on *session time* = seconds since the first sample
in eeg.bdf. Video frames and markers are timestamped on the host clock; the
clock fit from eeg_chunks.csv (``analysis.align``) maps host time <-> BDF sample.
Video frames are additionally shifted by the camera latency, so a frame is shown
at the moment its image was captured, not when it reached the computer.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..analysis.align import ClockFit, load_session

MARKERS_FILE = "markers.csv"
MARKERS_BACKUP = "markers_recorded.csv"  # untouched copy of what was recorded live


class BdfReader:
    """Random-access reader for BioSemi BDF files (memory-mapped; reads only what is asked)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            h = f.read(256)
            if h[:8] != b"\xffBIOSEMI":
                raise ValueError(f"{self.path.name} is not a BDF file")
            self.header_bytes = int(h[184:192])
            n_records = int(h[236:244])
            self.record_seconds = float(h[244:252])
            ns = int(h[252:256])
            sh = f.read(256 * ns)

        def fields(width: int, offset: int) -> list[str]:
            return [sh[offset + i * width: offset + (i + 1) * width].decode("ascii", "replace").strip()
                    for i in range(ns)]

        off = 0
        cols = {}
        for name, w in [("label", 16), ("transducer", 80), ("dim", 8), ("pmin", 8), ("pmax", 8),
                        ("dmin", 8), ("dmax", 8), ("prefilter", 80), ("spr", 8), ("reserved", 32)]:
            cols[name] = fields(w, off)
            off += w * ns
        self.labels = cols["label"]
        spr = {int(s) for s in cols["spr"]}
        if len(spr) != 1:
            raise ValueError("BDF files with different rates per channel are not supported")
        self.spr = spr.pop()
        self.sample_rate = self.spr / self.record_seconds
        pmin, pmax = np.array(cols["pmin"], float), np.array(cols["pmax"], float)
        dmin, dmax = np.array(cols["dmin"], float), np.array(cols["dmax"], float)
        self.gain = (pmax - pmin) / (dmax - dmin)
        self.offset = pmin - dmin * self.gain
        self.units = cols["dim"]
        record_bytes = ns * self.spr * 3
        available = (os.path.getsize(self.path) - self.header_bytes) // record_bytes
        # -1 = header never patched (recording did not close cleanly): trust the file size.
        self.n_records = available if n_records < 0 else min(n_records, available)
        self.n_samples = self.n_records * self.spr
        self._mm = np.memmap(self.path, dtype=np.uint8, mode="r", offset=self.header_bytes,
                             shape=(self.n_records, ns, self.spr, 3))
        self.status_index = self.labels.index("Status") if "Status" in self.labels else None

    def read_counts(self, channels: list[int] | np.ndarray, start: int, stop: int) -> np.ndarray:
        """Signed 24-bit integer samples, shape (stop - start, len(channels))."""
        start, stop = max(0, start), min(self.n_samples, stop)
        if stop <= start:
            return np.empty((0, len(channels)), dtype=np.int32)
        r0, r1 = start // self.spr, (stop - 1) // self.spr + 1
        b = self._mm[r0:r1][:, channels].astype(np.int32)        # (rec, ch, spr, 3)
        v = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)
        v = (v << 8) >> 8                                          # sign-extend 24 -> 32 bit
        v = v.transpose(0, 2, 1).reshape(-1, len(channels))        # (samples, ch)
        s0 = start - r0 * self.spr
        return v[s0:s0 + (stop - start)]

    def read_physical(self, channels, start: int, stop: int) -> np.ndarray:
        """Samples in physical units (µV for EEG channels)."""
        ch = np.asarray(channels)
        return self.read_counts(ch, start, stop) * self.gain[ch] + self.offset[ch]

    def triggers(self) -> tuple[np.ndarray, np.ndarray]:
        """(sample, value) of every trigger onset on the Status channel."""
        if self.status_index is None or self.n_samples == 0:
            return np.empty(0, int), np.empty(0, int)
        trig = np.empty(self.n_samples, dtype=np.int32)
        step = 600  # records per block, bounds memory on long files
        for r in range(0, self.n_records, step):
            b = self._mm[r:r + step, self.status_index].astype(np.int32)
            trig[r * self.spr:(r + b.shape[0]) * self.spr] = (b[..., 0] | (b[..., 1] << 8)).ravel()
        onset = np.flatnonzero((trig != 0) & (trig != np.concatenate([[0], trig[:-1]])))
        return onset, trig[onset]


@dataclass
class Marker:
    host_time: float
    label: str
    note: str = ""
    source: str = "recorded"  # "recorded" live during the session, or "review"
    duration: float = 0.0     # seconds; 0 = a point marker, > 0 = a labelled time window


class ReviewSession:
    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        for name in ("session.json", "eeg.bdf", "eeg_chunks.csv"):
            if not (self.folder / name).exists():
                raise FileNotFoundError(f"{self.folder} is not a recording folder (missing {name})")
        self.meta, self.fit = load_session(self.folder)
        self.fit: ClockFit
        self.bdf = BdfReader(self.folder / "eeg.bdf")
        self.fs = float(self.bdf.sample_rate)
        recorded = self.meta.get("eeg", {}).get("samples_recorded")
        # The writer pads the last record; samples_recorded is the real length.
        self.n_samples = min(self.bdf.n_samples, recorded) if recorded else self.bdf.n_samples
        self.duration = self.n_samples / self.fs
        self.eeg_channels = [i for i, lab in enumerate(self.bdf.labels) if lab != "Status"]
        self.complete = bool(self.meta.get("complete", False))

        cal = self.folder / "calibration.json"
        self.calibrated_latency_ms = (json.loads(cal.read_text())["camera_latency_ms"]
                                      if cal.exists() else None)
        self.camera_latency_s = (self.calibrated_latency_ms or 0.0) / 1000
        self.video_path = self.folder / "video.mp4"
        self._frame_host = np.empty(0)
        frames_csv = self.folder / "video_frames.csv"
        if self.video_path.exists() and frames_csv.exists():
            rows = _read_rows(frames_csv)
            self._frame_host = np.array([float(r["host_time"]) for r in rows])
        self.has_video = self._frame_host.size > 0
        self.markers = self._load_markers()
        self.trigger_samples, self.trigger_values = self.bdf.triggers()
        keep = self.trigger_samples < self.n_samples
        self.trigger_samples, self.trigger_values = self.trigger_samples[keep], self.trigger_values[keep]

    # -- time conversions ---------------------------------------------------
    def host_to_t(self, host_time) -> np.ndarray | float:
        return self.fit.host_to_sample(host_time) / self.fs

    def t_to_host(self, t: float) -> float:
        return self.fit.intercept + self.fit.seconds_per_sample * t * self.fs

    @property
    def trigger_times(self) -> np.ndarray:
        return self.trigger_samples / self.fs

    # -- video ----------------------------------------------------------------
    @property
    def frame_times(self) -> np.ndarray:
        """Session time at which each video frame's image was captured."""
        return self.host_to_t(self._frame_host - self.camera_latency_s)

    def limit_video_frames(self, n: int) -> None:
        """Use only the first ``n`` timestamped frames (when video.mp4 has fewer)."""
        self._frame_host = self._frame_host[:n]
        self.has_video = self._frame_host.size > 0

    def frame_at(self, t: float, tolerance: float = 0.5) -> int | None:
        """Index of the frame showing session time ``t`` (None outside the video)."""
        ft = self.frame_times
        if not ft.size or t < ft[0] - tolerance or t > ft[-1] + tolerance:
            return None
        return int(np.clip(np.searchsorted(ft, t, side="right") - 1, 0, ft.size - 1))

    # -- markers ----------------------------------------------------------------
    def _load_markers(self) -> list[Marker]:
        path = self.folder / MARKERS_FILE
        if not path.exists():
            return []
        out = [Marker(float(r["host_time"]), r.get("label", ""), r.get("note", "") or "",
                      r.get("source", "") or "recorded", _float(r.get("duration")))
               for r in _read_rows(path)]
        return sorted(out, key=lambda m: m.host_time)

    def marker_time(self, m: Marker) -> float:
        return float(self.host_to_t(m.host_time))

    def marker_end(self, m: Marker) -> float:
        return self.marker_time(m) + m.duration

    def set_marker_span(self, m: Marker, start: float, end: float | None = None) -> None:
        """Place ``m`` at session time ``start`` (and make it a window up to ``end``)."""
        if end is not None and end < start:
            start, end = end, start
        m.host_time = self.t_to_host(start)
        m.duration = 0.0 if end is None else max(0.0, end - start)

    def new_marker(self, t: float, label: str = "marker", end: float | None = None) -> Marker:
        m = Marker(0.0, label, "", "review")
        self.set_marker_span(m, t, end)
        self.markers.append(m)
        self.markers.sort(key=lambda x: x.host_time)
        return m

    def save_markers(self) -> None:
        """Write markers.csv (host_time, label, note, source, duration).

        The first save keeps the live-recorded file as markers_recorded.csv, so
        review edits never destroy what was captured during the session.
        """
        path = self.folder / MARKERS_FILE
        backup = self.folder / MARKERS_BACKUP
        if path.exists() and not backup.exists():
            shutil.copy2(path, backup)
        rows = [["host_time", "label", "note", "source", "duration"]] + [
            [f"{m.host_time:.6f}", m.label, m.note, m.source, f"{m.duration:.6f}" if m.duration else ""]
            for m in sorted(self.markers, key=lambda x: x.host_time)]
        _atomic_write(path, rows=rows)

    # -- notes -------------------------------------------------------------------
    @property
    def review_notes(self) -> str:
        return self.meta.get("review_notes", "")

    def save_review_notes(self, text: str) -> None:
        meta = json.loads((self.folder / "session.json").read_text())
        meta["review_notes"] = text
        _atomic_write(self.folder / "session.json", text=json.dumps(meta, indent=2))
        self.meta = meta


def _float(text: str | None) -> float:
    try:
        return max(0.0, float(text)) if text else 0.0
    except ValueError:
        return 0.0


def _read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _atomic_write(path: Path, *, rows=None, text: str | None = None) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            if rows is not None:
                csv.writer(f).writerows(rows)
            else:
                f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def format_time(t: float, ms: bool = True) -> str:
    sign = "-" if t < 0 else ""
    t = abs(t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    sec = f"{s:06.3f}" if ms else f"{int(s):02d}"
    return f"{sign}{int(h)}:{int(m):02d}:{sec}" if h else f"{sign}{int(m):02d}:{sec}"
