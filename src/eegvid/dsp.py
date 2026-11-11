"""Signal processing helpers (numpy only; scipy is not bundled into the app).

``Decimator``      streaming anti-aliased downsampling, used when saving at a
                   lower rate than the AD-box runs at (ActiView's "Decimation"),
                   and for the on-screen display.
``display_filter`` display-only high-pass / low-pass / notch filtering.
"""

from __future__ import annotations

import numpy as np

TRIGGER_MASK = 0xFFFF


def lowpass_fir(factor: int, taps_per_phase: int = 16) -> np.ndarray:
    """Linear-phase low-pass FIR for decimating by ``factor``.

    Blackman-windowed sinc with 2*D+1 taps, D = taps_per_phase * factor (so the
    group delay D is a whole number of output samples). The cutoff sits at the
    output Nyquist frequency. The passband is flat to ~0.4 x output rate, and
    anything that could alias into it is attenuated by more than 70 dB.
    """
    d = taps_per_phase * factor
    n = np.arange(-d, d + 1)
    h = np.sinc(n / factor) * np.blackman(2 * d + 1)
    return h / h.sum()  # exact unity gain at DC


def minus3db_hz(h: np.ndarray, fs: float) -> float:
    """Frequency where |H| first drops below 1/sqrt(2), for the BDF header."""
    nfft = 1 << 16
    mag = np.abs(np.fft.rfft(h, nfft))
    i = int(np.argmax(mag < 2 ** -0.5))
    return i * fs / nfft


class Decimator:
    """Streaming low-pass + downsample of (n, channels) blocks.

    Output sample k represents input sample k*factor exactly: the filter delay is
    compensated, so the decimated stream stays on the same timeline as the raw one.
    Output appears D input samples late (D/fs seconds; 62 ms for 2048 Hz / 4).
    The first sample is repeated backwards before the start, so the large DC offsets
    of an ActiveTwo don't produce a start-up transient.

    ``status`` (optional, 24-bit status words) is decimated without filtering: each
    output takes the flag bits of its own input sample and the first non-zero
    trigger value within its block of ``factor`` input samples, so trigger pulses
    shorter than the output sample period are not lost.
    """

    def __init__(self, factor: int, n_channels: int, *, integer: bool = True,
                 taps_per_phase: int = 16):
        if factor < 1:
            raise ValueError("factor must be >= 1")
        self.factor = factor
        self.n_channels = n_channels
        self.integer = integer
        self.h = lowpass_fir(factor, taps_per_phase) if factor > 1 else np.ones(1)
        self.delay = (len(self.h) - 1) // 2
        self._buf: np.ndarray | None = None      # input samples, abs index _b0..
        self._sbuf: np.ndarray | None = None     # status words, same indexing
        self._b0 = 0
        self._has_status = False
        self._n_in = 0       # real input samples received
        self._next = 0       # next output index to produce

    @property
    def n_out(self) -> int:
        return self._next

    def process(self, x: np.ndarray, status: np.ndarray | None = None):
        x = np.asarray(x, dtype=np.float64)
        self._has_status = status is not None
        if self.factor == 1:
            self._n_in += x.shape[0]
            self._next = self._n_in
            return self._finish(x, status)
        if x.shape[0] == 0:
            return self._empty()
        if self._buf is None:
            d = self.delay
            self._buf = np.repeat(x[:1], d, axis=0)
            self._b0 = -d
            if status is not None:
                self._sbuf = np.repeat(np.asarray(status[:1]) & ~np.int64(TRIGGER_MASK), d)
        self._buf = np.concatenate([self._buf, x])
        if status is not None:
            self._sbuf = np.concatenate([self._sbuf, np.asarray(status, dtype=np.int64)])
        self._n_in += x.shape[0]
        return self._emit(limit=None)

    def flush(self):
        """Emit the remaining outputs at end of stream (pads with the last sample)."""
        if self.factor == 1 or self._buf is None:
            return self._empty()
        pad = self.delay + self.factor
        self._buf = np.concatenate([self._buf, np.repeat(self._buf[-1:], pad, axis=0)])
        if self._sbuf is not None:
            self._sbuf = np.concatenate([self._sbuf, np.repeat(self._sbuf[-1:] & ~np.int64(TRIGGER_MASK), pad)])
        n_total = -(-self._n_in // self.factor)  # ceil
        return self._emit(limit=n_total)

    def _emit(self, limit: int | None):
        f, d = self.factor, self.delay
        avail_end = self._b0 + self._buf.shape[0]  # exclusive abs index of buffered input
        last = (avail_end - 1 - d) // f            # last output whose window is complete
        if limit is not None:
            last = min(last, limit - 1)
        else:
            last = min(last, (self._n_in - 1) // f)
        ks = np.arange(self._next, last + 1)
        if ks.size == 0:
            return self._empty()
        centers = ks * f - self._b0                  # buffer index of each output's input sample
        win = np.lib.stride_tricks.sliding_window_view(self._buf, len(self.h), axis=0)
        y = win[centers - d] @ self.h                # (k, channels)
        st = None
        if self._sbuf is not None:
            blocks = self._sbuf[centers[:, None] + np.arange(f)[None, :]]
            trig = blocks & TRIGGER_MASK
            first_nz = np.where(trig.any(axis=1), trig[np.arange(len(ks)), np.argmax(trig != 0, axis=1)], 0)
            st = (blocks[:, 0] & ~np.int64(TRIGGER_MASK)) | first_nz
        self._next = int(ks[-1]) + 1
        drop = self._next * f - d - self._b0         # keep what the next output still needs
        if drop > 0:
            self._buf = self._buf[drop:]
            if self._sbuf is not None:
                self._sbuf = self._sbuf[drop:]
            self._b0 += drop
        return self._finish(y, st)

    def _empty(self):
        return self._finish(np.empty((0, self.n_channels)),
                            np.empty(0, np.int64) if self._has_status else None)

    def _finish(self, y: np.ndarray, st):
        if self.integer:
            y = np.clip(np.rint(y), -(2**23), 2**23 - 1).astype(np.int32)
        if st is None:
            return y
        return y, np.asarray(st, dtype=np.int64)


def display_filter(data: np.ndarray, fs: float, highpass: float = 0.0, lowpass: float = 0.0,
                   notch: float = 0.0) -> np.ndarray:
    """Zero-phase filtering of a short display window (n, channels), display only.

    Removes a linear trend first (BioSemi is DC-coupled), mirror-pads to tame edge
    effects, then applies Butterworth-shaped magnitude responses in the frequency
    domain: 2nd-order high-pass, 4th-order low-pass, and a notch at ``notch`` Hz
    and its harmonics. A value of 0 switches that filter off.
    """
    n = data.shape[0]
    if n < 4:
        return data
    t = np.arange(n, dtype=float) - (n - 1) / 2   # centred, so the fit is two dot products
    mean = data.mean(axis=0)
    slope = (t @ data) / (t @ t)
    data = data - mean - np.outer(t, slope)
    if not (highpass or lowpass or notch):
        return data
    p = min(n - 1, int(fs))
    padded = np.concatenate([data[p:0:-1], data, data[-2:-p - 2:-1]])
    f = np.fft.rfftfreq(padded.shape[0], 1 / fs)
    g = np.ones_like(f)
    if highpass:
        with np.errstate(divide="ignore"):
            g *= 1 / np.sqrt(1 + (highpass / np.where(f == 0, 1e-12, f)) ** 4)
    if lowpass:
        g *= 1 / np.sqrt(1 + (f / lowpass) ** 8)
    if notch:
        for f0 in np.arange(notch, fs / 2, notch):
            g *= 1 - np.exp(-0.5 * ((f - f0) / 1.0) ** 2)
    out = np.fft.irfft(np.fft.rfft(padded, axis=0) * g[:, None], n=padded.shape[0], axis=0)
    return out[p:p + n]
