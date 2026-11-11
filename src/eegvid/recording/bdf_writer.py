"""Streaming BioSemi BDF writer (24-bit EDF variant, compatible with ActiView).

Writes data records as they fill, so a crash loses at most the last second.
The record count in the header is -1 until ``close()`` patches it in. MNE
and EDFbrowser can usually still open a file whose header was never patched.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DIG_MIN, DIG_MAX = -(2**23), 2**23 - 1


@dataclass(frozen=True)
class SignalSpec:
    label: str
    transducer: str = "Active Electrode"
    physical_dim: str = "uV"
    physical_min: float = -262144
    physical_max: float = 262143
    prefilter: str = "HP:DC"

    @staticmethod
    def status() -> "SignalSpec":
        return SignalSpec("Status", "Triggers and Status", "Boolean", DIG_MIN, DIG_MAX, "No filtering")


def _field(value, width: int) -> bytes:
    s = str(value)
    if isinstance(value, float) and value.is_integer():
        s = str(int(value))
    return s.encode("ascii", "replace")[:width].ljust(width, b" ")


class BdfWriter:
    def __init__(self, path: str | Path, signals: list[SignalSpec], sample_rate: int,
                 start: dt.datetime | None = None, patient: str = "X", recording: str = "",
                 record_seconds: int = 1):
        self.path = Path(path)
        self.signals = signals
        self.spr = int(sample_rate * record_seconds)
        self.record_seconds = record_seconds
        self.n_records = 0
        self.n_samples = 0
        self._pending = np.empty((0, len(signals)), dtype=np.int32)
        self._f = open(self.path, "wb")
        self._write_header(start or dt.datetime.now(), patient, recording)

    def _write_header(self, start: dt.datetime, patient: str, recording: str) -> None:
        ns = len(self.signals)
        h = bytearray()
        h += b"\xffBIOSEMI"
        h += _field(patient, 80) + _field(recording, 80)
        h += _field(start.strftime("%d.%m.%y"), 8) + _field(start.strftime("%H.%M.%S"), 8)
        h += _field(256 * (ns + 1), 8)
        h += _field("24BIT", 44)
        self._nrec_offset = len(h)
        h += _field(-1, 8)
        h += _field(self.record_seconds, 8)
        h += _field(ns, 4)
        for attr, w in [("label", 16), ("transducer", 80), ("physical_dim", 8),
                        ("physical_min", 8), ("physical_max", 8)]:
            for s in self.signals:
                h += _field(getattr(s, attr), w)
        for s in self.signals:
            h += _field(DIG_MIN, 8)
        for s in self.signals:
            h += _field(DIG_MAX, 8)
        for s in self.signals:
            h += _field(s.prefilter, 80)
        for s in self.signals:
            h += _field(self.spr, 8)
        for s in self.signals:
            h += _field("", 32)
        assert len(h) == 256 * (ns + 1)
        self._f.write(h)

    def write(self, samples: np.ndarray) -> None:
        """samples: (n, n_signals) 24-bit integer values."""
        if samples.ndim != 2 or samples.shape[1] != len(self.signals):
            raise ValueError(f"expected (n, {len(self.signals)}) array, got {samples.shape}")
        self._pending = np.concatenate([self._pending, samples.astype(np.int32, copy=False)])
        self.n_samples += samples.shape[0]
        while self._pending.shape[0] >= self.spr:
            self._write_record(self._pending[: self.spr])
            self._pending = self._pending[self.spr:]

    def _write_record(self, rec: np.ndarray) -> None:
        # BDF: per record, each signal's samples contiguous, 3 bytes little-endian each.
        data = np.ascontiguousarray(rec.T, dtype="<i4").view(np.uint8).reshape(-1, 4)[:, :3]
        self._f.write(data.tobytes())
        self.n_records += 1

    def close(self) -> None:
        if self._f.closed:
            return
        if self._pending.shape[0]:
            # Pad the final partial record by repeating the last sample. The
            # true sample count is in the session metadata.
            pad = np.repeat(self._pending[-1:], self.spr - self._pending.shape[0], axis=0)
            self._write_record(np.concatenate([self._pending, pad]))
            self._pending = self._pending[:0]
        self._f.seek(self._nrec_offset)
        self._f.write(_field(self.n_records, 8))
        self._f.close()
