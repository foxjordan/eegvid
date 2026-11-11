"""Acquisition thread: polls the ring buffer and emits decoded sample chunks."""

from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .. import clock
from .protocol import (
    CMD_INIT_OR_STOP,
    CMD_START,
    RING_BUFFER_BYTES,
    RING_BUFFER_WORDS,
    SPEED_MODES,
    SYNC_WORD,
    SpeedMode,
    decode_status,
)


class AcquisitionError(RuntimeError):
    pass


class Driver(Protocol):
    def open(self) -> None: ...
    def usb_write(self, data: bytes) -> None: ...
    def read_multiple_sweeps(self, ring: ctypes.Array) -> None: ...
    def read_pointer(self) -> int: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class DeviceInfo:
    speed_mode: SpeedMode
    mk2: bool
    words_per_sample: int

    @property
    def sample_rate(self) -> int:
        return self.speed_mode.sample_rate

    @property
    def n_adc(self) -> int:
        return self.words_per_sample - 2


@dataclass
class Chunk:
    """A block of consecutive samples.

    raw:          (n, words_per_sample) int32, exactly as received
    sample_start: device-global index of raw[0] (counted from connect)
    host_time:    shared-clock time at which these samples were first seen.
                  Applies to the *last* sample (sample_start + n - 1).
    """
    raw: np.ndarray
    sample_start: int
    host_time: float

    @property
    def n(self) -> int:
        return self.raw.shape[0]

    @property
    def sample_end(self) -> int:
        return self.sample_start + self.n

    def adc24(self) -> np.ndarray:
        """ADC values as 24-bit signed counts (LSB = 1/32 uV). Shape (n, n_adc)."""
        return self.raw[:, 2:] >> 8

    def status24(self) -> np.ndarray:
        return self.raw[:, 1].view(np.uint32) >> 8


class BioSemiDevice:
    """Runs the BioSemi handshake on a background thread.

    ``on_chunk(chunk)`` is called from that thread. Keep it fast (write to a
    file, copy into a display buffer). ``on_error(exc)`` runs if acquisition dies.
    """

    def __init__(self, driver: Driver, poll_interval: float = 0.01, start_timeout: float = 3.0):
        self.driver = driver
        self.poll_interval = poll_interval
        self.start_timeout = start_timeout
        self.info: DeviceInfo | None = None
        self.last_status = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._on_chunk: Callable[[Chunk], None] = lambda c: None
        self._on_error: Callable[[BaseException], None] = lambda e: None

    # -- public API --------------------------------------------------------
    def start(self, on_chunk: Callable[[Chunk], None],
              on_error: Callable[[BaseException], None] | None = None) -> DeviceInfo:
        """Start streaming. Blocks until the first sample has been decoded."""
        self._on_chunk = on_chunk
        if on_error:
            self._on_error = on_error
        self._stop.clear()
        self._ready.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, name="biosemi-acq", daemon=True)
        self._thread.start()
        if not self._ready.wait(self.start_timeout + 2.0) or self.info is None:
            self.stop()
            raise self._error or AcquisitionError("Timed out waiting for the receiver")
        return self.info

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- acquisition loop ----------------------------------------------------
    def _run(self) -> None:
        drv = self.driver
        ring = ctypes.create_string_buffer(RING_BUFFER_BYTES)
        words = np.frombuffer(ring, dtype="<u4")
        try:
            drv.open()
            drv.usb_write(CMD_INIT_OR_STOP)
            drv.read_multiple_sweeps(ring)
            drv.usb_write(CMD_START)
            self._loop(words)
        except BaseException as e:  # noqa: BLE001 - report everything to the UI
            started = self._ready.is_set()
            self._error = e
            self._ready.set()
            if started:  # startup errors are raised from start() instead
                self._on_error(e)
        finally:
            try:
                drv.usb_write(CMD_INIT_OR_STOP)
            except Exception:
                pass
            try:
                drv.close()
            except Exception:
                pass

    def _loop(self, words: np.ndarray) -> None:
        last_ptr = 0
        written = 0      # total words written by the receiver (unwrapped)
        consumed = 0     # total words handed to on_chunk
        sample_index = 0
        wps = None
        t_start = clock.now()

        while not self._stop.is_set():
            ptr = (self.driver.read_pointer() // 4) % RING_BUFFER_WORDS
            t = clock.now()
            written += (ptr - last_ptr) % RING_BUFFER_WORDS
            last_ptr = ptr

            backlog = written - consumed
            if backlog > RING_BUFFER_WORDS:
                raise AcquisitionError("Ring buffer overflow: the computer could not keep up")

            if wps is None:
                if backlog >= 2:
                    wps = self._identify(words)
                elif t - t_start > self.start_timeout:
                    raise AcquisitionError(
                        "No data from the receiver. Check that the AD-box is on, the "
                        "battery is charged and the optical fibre is connected.")

            if wps is not None:
                n = backlog // wps
                if n > 0:
                    block = self._take(words, consumed, n * wps).reshape(n, wps)
                    if not np.all(block[:, 0] == SYNC_WORD):
                        raise AcquisitionError(
                            "Lost sync with the receiver (sync word missing). "
                            "Was the speed mode switch changed during acquisition?")
                    consumed += n * wps
                    self.last_status = decode_status(block[-1, 1])
                    chunk = Chunk(block.view(np.int32), sample_index, t)
                    sample_index += n
                    self._on_chunk(chunk)

            time.sleep(self.poll_interval)

    def _identify(self, words: np.ndarray) -> int:
        if int(words[0]) != SYNC_WORD:
            raise AcquisitionError(f"First word is 0x{int(words[0]):08X}, expected sync word 0x{SYNC_WORD:08X}")
        st = decode_status(words[1])
        mode = SPEED_MODES.get(st.speed_mode)
        if mode is None:
            raise AcquisitionError(
                f"Speed mode {st.speed_mode} is not supported yet (supported: "
                f"{sorted(SPEED_MODES)}). Change the speed switch on the AD-box.")
        wps = mode.words_per_sample(st.mk2)
        self.info = DeviceInfo(mode, st.mk2, wps)
        self.last_status = st
        self._ready.set()
        return wps

    @staticmethod
    def _take(words: np.ndarray, start: int, count: int) -> np.ndarray:
        """Copy ``count`` words beginning at unwrapped position ``start``."""
        s = start % RING_BUFFER_WORDS
        e = s + count
        if e <= RING_BUFFER_WORDS:
            return words[s:e].copy()
        return np.concatenate([words[s:], words[: e - RING_BUFFER_WORDS]])
