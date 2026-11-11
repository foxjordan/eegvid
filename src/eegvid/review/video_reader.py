"""Frame-accurate video access for review (PyAV / FFmpeg).

OpenCV's seeking is not frame-accurate: depending on backend and codec it lands
one frame early for some frames, and on macOS-Intel (AVFoundation) it misreads
the frame rate (e.g. 29.001 for 29.97 fps), which drifts by minutes over an hour.
Here every frame is identified by its presentation timestamp (pts) from the
container, so frame k is exactly the k-th frame written, matching row k of
video_frames.csv.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np

FORWARD_DECODE_LIMIT = 45  # frames; a short forward jump decodes instead of seeking


class VideoReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        # Index: the pts of every frame, in presentation order, read from the packet
        # headers only (no decoding), so it is fast even for long recordings.
        with av.open(str(self.path)) as c:
            vs = c.streams.video[0]
            pts = [p.pts for p in c.demux(vs) if p.pts is not None]
        self.pts = np.sort(np.asarray(pts, dtype=np.int64))
        self.n_frames = int(self.pts.size)
        self._container = av.open(str(self.path))
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        self._frames = None   # active decode iterator
        self._next = None     # index the iterator will produce next (None = unknown)

    def _index_of(self, pts: int) -> int:
        return int(np.searchsorted(self.pts, pts))

    def read(self, k: int) -> np.ndarray | None:
        """Frame ``k`` as an RGB array (h, w, 3), or None if out of range."""
        if not 0 <= k < self.n_frames:
            return None
        if self._frames is None or self._next is None or not 0 <= k - self._next <= FORWARD_DECODE_LIMIT:
            self._container.seek(int(self.pts[k]), stream=self._stream, backward=True, any_frame=False)
            self._frames = self._container.decode(self._stream)
            self._next = None
        for frame in self._frames:
            if frame.pts is None:
                continue
            idx = self._index_of(frame.pts)
            self._next = idx + 1
            if idx >= k:
                return frame.to_ndarray(format="rgb24")
        self._frames = None
        return None

    def close(self) -> None:
        self._container.close()
