"""A recording session: one folder per session, holding

    eeg.bdf              EEG + Status channel (opens in ActiView-compatible tools / MNE)
    eeg_chunks.csv       bdf_sample_end, host_time  (EEG <-> shared clock mapping;
                         bdf_sample_end is fractional when saving at a reduced rate)
    video.mp4            camera video (if a camera was used)
    video_frames.csv     frame_id, host_time
    markers.csv          host_time, label  (operator markers from the GUI)
    session.json         metadata
    channel_names.cfg    copy of the ActiView .cfg the channel names came from (if one was used)

Use ``eegvid-align <folder>`` to add EEG time to every video frame.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import re
import shutil
import threading
from pathlib import Path

import numpy as np

from .. import __version__, clock
from ..dsp import Decimator, minus3db_hz
from ..biosemi.device import Chunk, DeviceInfo
from ..biosemi.protocol import Channel
from ..video.camera import Camera
from .bdf_writer import BdfWriter, SignalSpec


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "subject"


class RecordingSession:
    def __init__(self, root: Path, subject: str, device: DeviceInfo, channels: list[Channel],
                 camera: Camera | None = None, notes: str = "", *, decimation: int = 1,
                 electrode_offsets_mv: dict[str, float] | None = None,
                 names_cfg: str | Path | None = None):
        self.started_wall, self.started_host = clock.wall_and_host()
        start = dt.datetime.fromtimestamp(self.started_wall)
        self.folder = Path(root) / f"{_safe(subject)}_{start:%Y%m%d_%H%M%S}"
        self.folder.mkdir(parents=True, exist_ok=False)
        self.subject, self.device, self.channels, self.camera, self.notes = subject, device, channels, camera, notes
        self._adc_idx = np.array([c.adc_index for c in channels], dtype=np.intp)
        if device.sample_rate % decimation:
            raise ValueError(f"{device.sample_rate} Hz cannot be divided by {decimation}")
        self.decimation = decimation
        self.sample_rate = device.sample_rate // decimation
        self._decim = Decimator(decimation, len(channels))
        # The AD-box's own digital filter has its -3 dB point at about fs/4.9.
        self.lowpass_hz = device.sample_rate / 4.9
        if decimation > 1:
            self.lowpass_hz = min(self.lowpass_hz, minus3db_hz(self._decim.h, device.sample_rate))
        lp = round(self.lowpass_hz)
        specs = [SignalSpec(c.label, prefilter=f"HP:DC; LP:{lp} Hz") for c in channels] + [SignalSpec.status()]
        self.offsets = {"start": electrode_offsets_mv, "end": None}
        self.names_cfg = Path(names_cfg).name if names_cfg else None
        if names_cfg:
            shutil.copy2(names_cfg, self.folder / "channel_names.cfg")
        self.bdf = BdfWriter(self.folder / "eeg.bdf", specs, self.sample_rate, start,
                             patient=_safe(subject), recording=f"eegvid {__version__}")
        self._chunks_f = open(self.folder / "eeg_chunks.csv", "w", newline="")
        self._chunks = csv.writer(self._chunks_f)
        self._chunks.writerow(["bdf_sample_end", "host_time"])
        self._markers_f = open(self.folder / "markers.csv", "w", newline="")
        self._markers = csv.writer(self._markers_f)
        self._markers.writerow(["host_time", "label"])
        self._lock = threading.Lock()
        self.first_device_sample: int | None = None
        self.closed = False
        if camera is not None:
            camera.start_recording(self.folder / "video.mp4", self.folder / "video_frames.csv")
        self._write_meta()

    # Called from the acquisition thread.
    def add_eeg_chunk(self, chunk: Chunk) -> None:
        with self._lock:
            if self.closed:
                return
            if self.first_device_sample is None:
                self.first_device_sample = chunk.sample_start
            adc = chunk.adc24()[:, self._adc_idx]
            adc, status = self._decim.process(adc, chunk.status24().astype(np.int64))
            if adc.shape[0]:
                self.bdf.write(np.column_stack([adc, status.astype(np.int32)]))
            # Raw sample k sits at saved sample k / decimation (the filter delay is
            # compensated), so map the chunk's last raw sample onto the saved timeline.
            end = chunk.sample_end - self.first_device_sample
            if self.decimation == 1:
                self._chunks.writerow([end, f"{chunk.host_time:.6f}"])
            else:
                self._chunks.writerow([f"{(end - 1) / self.decimation + 1:.4f}", f"{chunk.host_time:.6f}"])

    def add_marker(self, label: str) -> float:
        t = clock.now()
        with self._lock:
            self._markers.writerow([f"{t:.6f}", label])
            self._markers_f.flush()
        return t

    @property
    def elapsed(self) -> float:
        return clock.now() - self.started_host

    def close(self, electrode_offsets_mv: dict[str, float] | None = None) -> None:
        self.offsets["end"] = electrode_offsets_mv
        if self.camera is not None:
            self.camera.stop_recording()
        with self._lock:
            if self.closed:
                return
            self.closed = True
            adc, status = self._decim.flush()
            if adc.shape[0]:
                self.bdf.write(np.column_stack([adc, status.astype(np.int32)]))
            self.bdf.close()
            self._chunks_f.close()
            self._markers_f.close()
        self._write_meta(final=True)

    def _write_meta(self, final: bool = False) -> None:
        meta = {
            "software": f"eegvid {__version__}",
            "subject": self.subject,
            "notes": self.notes,
            "started_wall_clock": dt.datetime.fromtimestamp(self.started_wall).isoformat(),
            "started_host_time": self.started_host,
            "clock": "time.perf_counter (monotonic, shared by EEG and video)",
            "eeg": {
                "sample_rate": self.sample_rate,
                "device_sample_rate": self.device.sample_rate,
                "decimation": self.decimation,
                "lowpass_hz": round(self.lowpass_hz, 1),
                "speed_mode": self.device.speed_mode.mode,
                "mk2": self.device.mk2,
                "channels": [c.label for c in self.channels] + ["Status"],
                "channel_slots": [c.slot for c in self.channels],
                "channel_names_from": self.names_cfg,
                "samples_recorded": self.bdf.n_samples,
                "first_device_sample": self.first_device_sample,
                "electrode_offsets_mV": {
                    k: None if v is None else {lab: round(float(mv), 2) for lab, mv in v.items()}
                    for k, v in self.offsets.items()},
            },
            "video": None if self.camera is None else {
                "camera_index": self.camera.index,
                "codec": self.camera.codec,
                "width": self.camera.width,
                "height": self.camera.height,
                "measured_fps": round(self.camera.measured_fps, 3),
                "frames_written": self.camera.frames_written,
                "frames_dropped": self.camera.frames_dropped,
            },
            "complete": final,
        }
        (self.folder / "session.json").write_text(json.dumps(meta, indent=2))
