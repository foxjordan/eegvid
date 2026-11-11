"""The single clock shared by every stream in the app.

The EEG and the video are both timestamped with this clock in the same
process, so they can be aligned without a tap or clap. ``time.time()`` (the
wall clock) is a poor choice for this: NTP can step or slew it in the middle
of a recording. ``perf_counter`` is monotonic and high-resolution on Windows
(QPC), macOS (mach_absolute_time) and Linux.
"""

import time


def now() -> float:
    """Seconds on the shared monotonic host clock."""
    return time.perf_counter()


def wall_and_host() -> tuple[float, float]:
    """A (wall-clock epoch seconds, host clock) pair, saved in session metadata."""
    return time.time(), time.perf_counter()
