"""Electrode offsets: BioSemi's electrode-contact check.

The ActiveTwo has no impedance measurement. Instead ActiView's Offset tab shows
each electrode's DC voltage relative to CMS; a well-gelled electrode has a small,
stable offset. Lab guidance commonly keeps offsets within +-40 mV (stricter labs
use +-20 to +-25 mV). Readings near the +-262 mV input range mean no contact.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import numpy as np

from .protocol import UV_PER_LSB_24

DEFAULT_LIMIT_MV = 40.0
RAIL_MV = 250.0  # the ADC saturates at +-262 mV; beyond this the electrode is off/open


class OffsetMonitor:
    """Rolling mean of every ADC word over the last ``window_s`` seconds.

    ``push`` is called from the acquisition thread with each chunk's 24-bit ADC
    counts; ``offsets_mv`` can be read from any thread.
    """

    def __init__(self, sample_rate: int, window_s: float = 0.5):
        self.window = max(1, int(sample_rate * window_s))
        self._parts: deque[tuple[np.ndarray, int]] = deque()
        self._sum: np.ndarray | None = None
        self._count = 0
        self._lock = threading.Lock()

    def push(self, adc24: np.ndarray) -> None:
        s = adc24.sum(axis=0, dtype=np.int64)
        n = adc24.shape[0]
        with self._lock:
            if self._sum is None or self._sum.shape != s.shape:
                self._sum, self._count, self._parts = np.zeros_like(s), 0, deque()
            self._parts.append((s, n))
            self._sum += s
            self._count += n
            while self._parts and self._count - self._parts[0][1] >= self.window:
                old, k = self._parts.popleft()
                self._sum -= old
                self._count -= k

    def offsets_mv(self, adc_index: np.ndarray) -> np.ndarray | None:
        with self._lock:
            if self._sum is None or self._count == 0:
                return None
            return self._sum[adc_index] / self._count * UV_PER_LSB_24 / 1000.0


@dataclass(frozen=True)
class OffsetSummary:
    n_total: int
    n_ok: int
    bad: list[tuple[str, float]]  # (label, mV), worst first

    @property
    def all_ok(self) -> bool:
        return not self.bad


def summarize(labels: list[str], mv: np.ndarray, limit_mv: float) -> OffsetSummary:
    bad = [(lab, float(v)) for lab, v in zip(labels, mv) if abs(v) > limit_mv]
    bad.sort(key=lambda b: -abs(b[1]))
    return OffsetSummary(len(labels), len(labels) - len(bad), bad)


def describe(label: str, mv: float) -> str:
    return f"{label} (no contact)" if abs(mv) >= RAIL_MV else f"{label} ({mv:+.0f} mV)"
