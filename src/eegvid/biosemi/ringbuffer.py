"""Low-level access to the BioSemi USB receiver.

``RingBufferLibrary`` wraps BioSemi's RingBuffer library (from their free SDK,
or shipped with ActiView) via ctypes. ``SimulatedRingBuffer`` has the same
interface and fills the ring buffer with synthetic data in the exact wire
format, so the rest of the app runs unchanged without hardware.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

from .. import clock as clock_source
from .protocol import (
    CMD_START,
    RING_BUFFER_BYTES,
    RING_BUFFER_WORDS,
    SPEED_MODES,
    SYNC_WORD,
    decode_status,
    encode_status,
)

LIB_ENV_VAR = "EEGVID_BIOSEMI_LIB"

if sys.platform == "win32":
    _LIB_NAMES = ["RingBuffer.dll", "Labview_DLL.dll", "labview_dll.dll"]
elif sys.platform == "darwin":
    # The framework (also inside ActiView-Mac) is preferred: it carries its own libusb.
    # The bare dylib links libusb from an absolute path on BioSemi's build machine.
    _LIB_NAMES = ["BioSemiRingBuffer64.framework/BioSemiRingBuffer64",
                  "libRingBuffer.dylib", "libRingBuffer.0.dylib", "libRingBuffer.0.0.1.dylib",
                  "liblabview_dll.dylib", "labview_dll.dylib"]
else:
    _LIB_NAMES = ["libRingBuffer.so", "libRingBuffer.so.0", "liblabview_dll.so", "labview_dll.so"]


def _search_dirs() -> list[Path]:
    dirs = []
    if getattr(sys, "frozen", False):  # PyInstaller bundle
        exe_dir = Path(sys.executable).resolve().parent
        dirs.append(Path(getattr(sys, "_MEIPASS", exe_dir)) / "drivers")
        dirs.append(exe_dir / "drivers")
        if sys.platform == "darwin" and exe_dir.parent.parent.suffix == ".app":
            # .app/Contents/MacOS/<exe>: also look beside the .app itself
            dirs.append(exe_dir.parent.parent.parent / "drivers")
    if sys.platform == "darwin":
        dirs.append(Path.home() / "Library" / "Application Support" / "EEGVid" / "drivers")
    elif sys.platform == "win32" and os.environ.get("APPDATA"):
        dirs.append(Path(os.environ["APPDATA"]) / "EEGVid" / "drivers")
    else:
        dirs.append(Path.home() / ".local" / "share" / "eegvid" / "drivers")
    dirs.append(Path(__file__).resolve().parents[3] / "packaging" / "drivers")  # source checkout
    dirs.append(Path.cwd() / "drivers")
    return dirs


def find_libraries(explicit: str | None = None) -> list[Path]:
    """All existing candidate library files, most preferred first."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get(LIB_ENV_VAR):
        candidates.append(Path(os.environ[LIB_ENV_VAR]))
    dirs = _search_dirs()
    for d in dirs:
        candidates += [d / n for n in _LIB_NAMES]
    found = [c for c in candidates if c.is_file()]
    if found:
        return found
    searched = "\n  ".join(str(d) for d in dirs)
    raise FileNotFoundError(
        "BioSemi RingBuffer library not found. Download the SDK for your OS from "
        "https://www.biosemi.com/faq/make_own_acquisition_software.htm and put the "
        f"library in one of these folders, choose it under 'Library', or set {LIB_ENV_VAR}.\n"
        f"Looked for: {', '.join(_LIB_NAMES)}\nIn:\n  {searched}"
    )


def find_library(explicit: str | None = None) -> Path:
    return find_libraries(explicit)[0]


def _load(paths: list[Path]) -> tuple[Path, ctypes.CDLL]:
    """Load the first candidate that this process can actually load."""
    errors = []
    for p in paths:
        try:
            return p, ctypes.CDLL(str(p))
        except OSError as e:
            errors.append(f"{p}: {e}")
    hint = ""
    if any("incompatible architecture" in e for e in errors):
        import platform
        hint = (f"\nThis app is running as {platform.machine()}, but the BioSemi library is "
                "built for a different processor. On Apple Silicon Macs use the Intel (x86_64) "
                "build of this app, which runs under Rosetta like ActiView-Mac does.")
    raise OSError("Could not load the BioSemi RingBuffer library." + hint
                  + "\nTried:\n  " + "\n  ".join(errors))


class RingBufferLibrary:
    """ctypes binding for OPEN_DRIVER_ASYNC / USB_WRITE / READ_MULTIPLE_SWEEPS /
    READ_POINTER / CLOSE_DRIVER_ASYNC."""

    def __init__(self, path: str | None = None):
        self.path, self._lib = _load(find_libraries(path))
        self._handle = None
        L = self._lib
        # The 2015 RingBuffer header declares nBytesToRead as INT_PTR (pointer-sized);
        # older Labview_DLL builds used a 32-bit DWORD. A pointer-sized argument is
        # correct for INT_PTR and safe for DWORD on x86-64/arm64, where 32-bit
        # arguments are read from the low half of the same register.
        dword = ctypes.c_ssize_t
        L.OPEN_DRIVER_ASYNC.restype = ctypes.c_void_p
        L.OPEN_DRIVER_ASYNC.argtypes = []
        L.USB_WRITE.restype = ctypes.c_int
        L.USB_WRITE.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.READ_MULTIPLE_SWEEPS.restype = ctypes.c_int
        L.READ_MULTIPLE_SWEEPS.argtypes = [ctypes.c_void_p, ctypes.c_void_p, dword]
        L.READ_POINTER.restype = ctypes.c_int
        L.READ_POINTER.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        L.CLOSE_DRIVER_ASYNC.restype = ctypes.c_int
        L.CLOSE_DRIVER_ASYNC.argtypes = [ctypes.c_void_p]
        # Always an 8-byte zeroed slot, so a 32- or 64-bit write both read back correctly.
        self._ptr_slot = ctypes.c_uint64(0)

    def open(self) -> None:
        h = self._lib.OPEN_DRIVER_ASYNC()
        if not h or h in (0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF):
            raise OSError("OPEN_DRIVER_ASYNC failed: is the USB receiver plugged in "
                          "and the BioSemi driver installed?")
        self._handle = h

    def usb_write(self, data: bytes) -> None:
        buf = ctypes.create_string_buffer(bytes(data), 64)
        if not self._lib.USB_WRITE(self._handle, buf):
            raise OSError("USB_WRITE failed")

    def read_multiple_sweeps(self, ring: ctypes.Array) -> None:
        if not self._lib.READ_MULTIPLE_SWEEPS(self._handle, ring, ctypes.sizeof(ring)):
            raise OSError("READ_MULTIPLE_SWEEPS failed")

    def read_pointer(self) -> int:
        """Byte offset of the receiver's write position in the ring.
        VERIFY: community code treats it as bytes. device.py checks the sync
        word on every sample, so a wrong unit fails loudly instead of silently."""
        self._ptr_slot.value = 0
        if not self._lib.READ_POINTER(self._handle, ctypes.byref(self._ptr_slot)):
            raise OSError("READ_POINTER failed")
        return int(self._ptr_slot.value & 0xFFFFFFFF)

    def close(self) -> None:
        if self._handle:
            self._lib.CLOSE_DRIVER_ASYNC(self._handle)
            self._handle = None


class SimulatedRingBuffer:
    """Emulates the receiver: writes sync/status/ADC words into the caller's ring
    at real-time speed.

    Signals: DC offsets like a prepared ActiveTwo cap (about -20 mV, with A5,
    A21 and B10 deliberately out of range for the offset check), a
    10 Hz "alpha" rhythm, noise, and a trigger pulse every ``trigger_every`` s
    whose value cycles 1..4. Every ``tap_every`` s a large artifact is added to
    every channel, simulating a physical tap for testing calibration.
    """

    def __init__(self, speed_mode: int = 4, mk2: bool = True, *, trigger_every: float = 2.0,
                 tap_every: float = 10.0, seed: int = 0):
        self.mode = SPEED_MODES[speed_mode]
        self.mk2 = mk2
        self.wps = self.mode.words_per_sample(mk2)
        self.rate = self.mode.sample_rate
        self.trigger_every = trigger_every
        self.tap_every = tap_every
        self._rng = np.random.default_rng(seed)
        n_adc = self.wps - 2
        self._offsets_uv = -20000.0 + self._rng.normal(0, 3000, n_adc)
        # A few poorly prepared electrodes, so the offset check has something to flag:
        # A5 and A21 outside +-40 mV, B10 with no contact (pinned at the input range).
        for i, uv in {4: -55000.0, 20: 90000.0, 41: -300000.0}.items():
            if i < n_adc:
                self._offsets_uv[i] = uv
        self._ring: np.ndarray | None = None
        self._t0: float | None = None
        self._n_generated = 0
        self._speed_mode = speed_mode
        self._status_base = encode_status(0, speed_mode=speed_mode, mk2=mk2)

    @property
    def cms_in_range(self) -> bool:
        return decode_status(self._status_base).cms_in_range

    @cms_in_range.setter
    def cms_in_range(self, ok: bool) -> None:
        """Simulate CMS/DRL losing (or regaining) contact."""
        self._status_base = encode_status(0, speed_mode=self._speed_mode, mk2=self.mk2, cms_in_range=ok)

    def open(self) -> None:
        pass

    def usb_write(self, data: bytes) -> None:
        if bytes(data[:1]) == CMD_START[:1]:
            self._t0 = clock_source.now()
            self._n_generated = 0
        else:
            self._t0 = None

    def read_multiple_sweeps(self, ring: ctypes.Array) -> None:
        assert ctypes.sizeof(ring) == RING_BUFFER_BYTES
        self._ring = np.frombuffer(ring, dtype="<u4")

    def read_pointer(self) -> int:
        if self._t0 is not None and self._ring is not None:
            target = int((clock_source.now() - self._t0) * self.rate)
            if target > self._n_generated:
                self._generate(self._n_generated, target)
                self._n_generated = target
        return (self._n_generated * self.wps * 4) % RING_BUFFER_BYTES

    def close(self) -> None:
        self._t0 = None

    def _generate(self, k0: int, k1: int) -> None:
        n = k1 - k0
        t = np.arange(k0, k1) / self.rate
        n_adc = self.wps - 2
        uv = (self._offsets_uv[None, :]
              + 10.0 * np.sin(2 * np.pi * 10.0 * t)[:, None]
              + self._rng.normal(0, 3.0, (n, n_adc)))
        if self.tap_every:
            phase = np.mod(t, self.tap_every) - self.tap_every / 2
            uv += (800.0 * np.exp(-(phase / 0.01) ** 2))[:, None]
        counts24 = np.clip(np.round(uv * 32), -(2**23), 2**23 - 1).astype(np.int64)
        block = np.empty((n, self.wps), dtype=np.uint32)
        block[:, 0] = SYNC_WORD
        trig = np.zeros(n, dtype=np.uint32)
        if self.trigger_every:
            k_per = int(self.trigger_every * self.rate)
            k = np.arange(k0, k1)
            on = (k % k_per) < int(0.01 * self.rate)
            trig[on] = (k[on] // k_per) % 4 + 1
        block[:, 1] = self._status_base | (trig << 8)
        block[:, 2:] = ((counts24 << 8) & 0xFFFFFFFF).astype(np.uint32)
        flat = block.ravel()
        start = (k0 * self.wps) % RING_BUFFER_WORDS
        end = start + flat.size
        if end <= RING_BUFFER_WORDS:
            self._ring[start:end] = flat
        else:
            first = RING_BUFFER_WORDS - start
            self._ring[start:] = flat[:first]
            self._ring[: end - RING_BUFFER_WORDS] = flat[first:]
