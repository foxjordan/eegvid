"""Colour-coded tile grid of electrode offsets (one tile per recorded channel)."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..biosemi.offsets import RAIL_MV

# (background, text) per state
COLORS = {
    "ok": ("#2e7d32", "#ffffff"),
    "warn": ("#f9a825", "#000000"),
    "bad": ("#c62828", "#ffffff"),
    "open": ("#37474f", "#ffffff"),
    "none": ("#cfd8dc", "#455a64"),
}


def state(mv: float | None, limit: float) -> str:
    if mv is None or not np.isfinite(mv):
        return "none"
    a = abs(mv)
    if a >= RAIL_MV:
        return "open"
    if a > 2 * limit:
        return "bad"
    if a > limit:
        return "warn"
    return "ok"


class OffsetGrid(QWidget):
    """EEG electrodes in rows of 8 (16 above 64 channels, so each 32-electrode
    cap bank A, B, ... fills whole rows); EXG channels on their own row."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 200)
        self.labels: list[str] = []
        self.values: np.ndarray | None = None
        self.limit = 40.0
        self.valid = True
        self._pos: list[tuple[int, int]] = []
        self._cols = 8
        self._rows = 0

    def set_channels(self, labels: list[str], exg: list[bool] | None = None) -> None:
        self.labels = list(labels)
        self.values = None
        if exg is None:
            exg = [lab.startswith("EXG") for lab in labels]
        eeg_idx = [i for i, x in enumerate(exg) if not x]
        exg_idx = [i for i, x in enumerate(exg) if x]
        eeg, exg = eeg_idx, exg_idx
        self._cols = 8 if len(eeg) <= 64 else 16
        self._pos = [(0, 0)] * len(labels)
        for k, i in enumerate(eeg):
            self._pos[i] = divmod(k, self._cols)
        r0 = math.ceil(len(eeg) / self._cols)
        for k, i in enumerate(exg):
            r, c = divmod(k, self._cols)
            self._pos[i] = (r0 + r, c)
        self._rows = r0 + math.ceil(len(exg) / self._cols)
        self.update()

    def set_values(self, values: np.ndarray | None, limit: float, valid: bool = True) -> None:
        """``valid=False`` (CMS/DRL out of range) greys every tile: the numbers are
        still shown, but they don't describe electrode contact."""
        self.values = values
        self.limit = limit
        self.valid = valid
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if not self.labels or not self._rows:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        gap = 3.0
        cell = min(self.width() / self._cols, self.height() / self._rows)
        x0 = (self.width() - cell * self._cols) / 2
        big = QFont(self.font())
        big.setPixelSize(max(8, int(cell * 0.24)))
        big.setBold(True)
        small = QFont(self.font())
        small.setPixelSize(max(7, int(cell * 0.2)))
        for i, lab in enumerate(self.labels):
            r, c = self._pos[i]
            rect = QRectF(x0 + c * cell + gap / 2, r * cell + gap / 2, cell - gap, cell - gap)
            v = None if self.values is None else float(self.values[i])
            bg, fg = COLORS[state(v, self.limit) if self.valid else "none"]
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(bg))
            p.drawRoundedRect(rect, 4, 4)
            p.setPen(QColor(fg))
            top = QRectF(rect.x(), rect.y() + rect.height() * 0.1, rect.width(), rect.height() * 0.45)
            bot = QRectF(rect.x(), rect.y() + rect.height() * 0.5, rect.width(), rect.height() * 0.4)
            p.setFont(big)
            p.drawText(top, Qt.AlignCenter, lab)
            if v is not None and cell >= 34:
                p.setFont(small)
                p.drawText(bot, Qt.AlignCenter, "open" if abs(v) >= RAIL_MV else f"{v:+.0f}")
        p.end()
