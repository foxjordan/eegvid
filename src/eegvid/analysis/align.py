"""Map video frames (and operator markers) onto the EEG timeline.

    eegvid-align SESSION_DIR [--camera-latency-ms 80]
        -> video_frames_aligned.csv, markers_aligned.csv

    eegvid-align calibrate SESSION_DIR --frame 788 (--trigger | --channel Fz)
        Measures the fixed camera latency from one session where a tap/flash
        is visible on video AND appears in the EEG (as a trigger, or as an
        artifact on a channel). Do this once per camera/computer setup and
        reuse the number.

How it works: the EEG and video are timestamped on the same host clock, so
no sync event is needed for routine sessions. eeg_chunks.csv pairs BDF sample
indices with the host time at which they arrived. A straight-line fit of
sample vs time absorbs any drift between the BioSemi crystal and the computer
clock, which a single tap/clap point cannot do. A webcam frame reaches the
computer a roughly constant 30-150 ms after the light hit the sensor; that is
the camera latency, measured once with ``calibrate``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ClockFit:
    """host_time = intercept + seconds_per_sample * bdf_sample"""
    intercept: float
    seconds_per_sample: float
    residual_ms_p95: float

    def host_to_sample(self, host_time):
        return (np.asarray(host_time) - self.intercept) / self.seconds_per_sample


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    cols = list(zip(*body)) if body else [[] for _ in header]
    out = {}
    for name, col in zip(header, cols):
        try:
            out[name] = np.asarray(col, dtype=float)
        except ValueError:
            out[name] = np.asarray(col, dtype=object)
    return out


def fit_sample_clock(sample_end: np.ndarray, host_time: np.ndarray) -> ClockFit:
    """Fit EEG sample index to host time.

    A chunk's host_time is when the computer *saw* the samples, which is always
    after they were acquired, by a variable USB/polling delay. So we fit a line,
    then shift it down to the lower envelope (5th percentile of the residuals),
    which estimates the minimum transport delay rather than the average one.
    """
    x = np.asarray(sample_end, dtype=float) - 1  # host_time refers to the last sample of the chunk
    y = np.asarray(host_time, dtype=float)
    if x.size < 2:
        raise ValueError("Need at least two EEG chunks to fit the clock")
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    intercept += np.percentile(resid, 5)
    resid = y - (intercept + slope * x)
    return ClockFit(float(intercept), float(slope), float(np.percentile(np.abs(resid), 95) * 1e3))


def load_session(folder: Path) -> tuple[dict, ClockFit]:
    folder = Path(folder)
    meta = json.loads((folder / "session.json").read_text())
    ch = _read_csv(folder / "eeg_chunks.csv")
    return meta, fit_sample_clock(ch["bdf_sample_end"], ch["host_time"])


def align_session(folder: Path, camera_latency_s: float = 0.0) -> Path | None:
    folder = Path(folder)
    meta, fit = load_session(folder)
    fs = meta["eeg"]["sample_rate"]
    print(f"EEG clock fit: {1 / fit.seconds_per_sample:.4f} samples per host-second "
          f"(nominal {fs}); 95% of chunk timestamps within {fit.residual_ms_p95:.2f} ms")

    out = None
    frames_csv = folder / "video_frames.csv"
    if frames_csv.exists():
        fr = _read_csv(frames_csv)
        samples = fit.host_to_sample(fr["host_time"] - camera_latency_s)
        out = folder / "video_frames_aligned.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_id", "host_time", "eeg_sample", "eeg_time_s"])
            for fid, t, s in zip(fr["frame_id"].astype(int), fr["host_time"], samples):
                w.writerow([fid, f"{t:.6f}", int(round(s)), f"{s / fs:.6f}"])
        print(f"Wrote {out} ({len(samples)} frames, camera latency {camera_latency_s * 1e3:.1f} ms)")

    mk_csv = folder / "markers.csv"
    if mk_csv.exists():
        mk = _read_csv(mk_csv)
        if len(mk.get("host_time", [])):
            s = fit.host_to_sample(mk["host_time"])
            with open(folder / "markers_aligned.csv", "w", newline="") as f:
                w = csv.writer(f)
                # duration_s > 0: a labelled time window (from the review window), else a point
                w.writerow(["label", "eeg_sample", "eeg_time_s", "duration_s", "end_sample"])
                durations = mk.get("duration", [""] * len(s))
                for label, si, d in zip(mk["label"], s, durations):
                    d = float(d) if str(d).strip() not in ("", "nan") else 0.0
                    end = int(round(si + d * fs)) if d > 0 else ""
                    w.writerow([label, int(round(si)), f"{si / fs:.6f}", f"{d:.6f}" if d > 0 else "", end])
    return out


def calibrate(folder: Path, frame_id: int, *, use_trigger: bool, channel: str | None,
              search_window_s: float = 1.0) -> float:
    """Return camera latency in seconds (video lags reality by this much)."""
    try:
        import mne
    except ImportError:
        sys.exit("calibrate needs MNE: pip install 'eegvid[analysis]'")
    folder = Path(folder)
    meta, fit = load_session(folder)
    fs = meta["eeg"]["sample_rate"]
    fr = _read_csv(folder / "video_frames.csv")
    idx = np.flatnonzero(fr["frame_id"].astype(int) == frame_id)
    if not idx.size:
        sys.exit(f"Frame {frame_id} not in video_frames.csv")
    predicted = float(fit.host_to_sample(fr["host_time"][idx[0]]))  # with zero latency

    raw = mne.io.read_raw_bdf(folder / "eeg.bdf", preload=True, verbose="error")
    lo = max(0, int(predicted - search_window_s * fs * 2))  # the video lags, so the event is earlier
    hi = min(raw.n_times, int(predicted + search_window_s * fs * 0.5))
    if use_trigger:
        events = mne.find_events(raw, stim_channel="Status", shortest_event=1, verbose="error")
        cand = events[(events[:, 0] >= lo) & (events[:, 0] < hi)]
        if not len(cand):
            sys.exit("No Status trigger near that frame")
        event_sample = int(cand[np.argmin(np.abs(cand[:, 0] - predicted)), 0])
    else:
        # BioSemi data is DC-coupled with large offsets: high-pass before looking
        # for the artifact, and only search near the predicted time. (A global
        # argmax over the raw signal tends to find drift, not the tap.)
        seg = raw.copy().pick([channel]).filter(1.0, None, verbose="error").get_data()[0, lo:hi]
        event_sample = lo + int(np.argmax(np.abs(seg)))
    latency = (predicted - event_sample) / fs
    print(f"Frame {frame_id}: predicted EEG sample {predicted:.0f}, event at {event_sample}. "
          f"Camera latency = {latency * 1e3:.1f} ms")
    (folder / "calibration.json").write_text(json.dumps(
        {"frame_id": frame_id, "event_sample": event_sample, "camera_latency_ms": latency * 1e3}, indent=2))
    return latency


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "calibrate":
        p = argparse.ArgumentParser(prog="eegvid-align calibrate")
        p.add_argument("session")
        p.add_argument("--frame", type=int, required=True, help="frame where the tap/flash is visible")
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--trigger", action="store_true", help="event is a trigger on the Status channel")
        g.add_argument("--channel", help="event is an artifact on this EEG channel (e.g. Fz or A1)")
        a = p.parse_args(argv[1:])
        calibrate(Path(a.session), a.frame, use_trigger=a.trigger, channel=a.channel)
        return
    p = argparse.ArgumentParser(prog="eegvid-align", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session")
    p.add_argument("--camera-latency-ms", type=float, default=None,
                   help="defaults to calibration.json in the session folder if present, else 0")
    a = p.parse_args(argv)
    lat = a.camera_latency_ms
    cal = Path(a.session) / "calibration.json"
    if lat is None and cal.exists():
        lat = json.loads(cal.read_text())["camera_latency_ms"]
    align_session(Path(a.session), (lat or 0.0) / 1e3)


if __name__ == "__main__":
    main()
