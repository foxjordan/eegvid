# EEG Video Recorder (eegvid)

Records BioSemi ActiveTwo EEG and webcam video together, aligned to one
shared clock. Lab members download one app, click **Connect → Record →
Stop**, and get one folder per session.

## Quick start (developers)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                      # runs entirely on the built-in simulator
python -m eegvid            # launch the GUI; pick "Simulator" to try it without hardware
```

## What a session folder contains

| file | contents |
|---|---|
| `eeg.bdf` | EEG + `Status` (trigger) channel, same format as ActiView, opens in MNE / EEGLAB / EDFbrowser |
| `eeg_chunks.csv` | BDF sample index ↔ host-clock time pairs |
| `video.mp4` | webcam video |
| `video_frames.csv` | frame number ↔ host-clock time |
| `markers.csv` | markers: `host_time, label` (+ `note, source, duration` once edited in the review window; `duration` in seconds, blank for a point marker) |
| `markers_recorded.csv` | untouched copy of the live markers, made the first time the review window saves |
| `session.json` | subject, settings, channel list, electrode offsets, dropped-frame count, review notes |
| `channel_names.cfg` | copy of the ActiView `.cfg` the electrode names came from (if one was loaded) |

Then:

```bash
eegvid-align "EEG Recordings/P001_20261110_093000"
# -> video_frames_aligned.csv: frame_id, eeg_sample, eeg_time_s
```

## Electrode names

By default channels use BioSemi's input names (A1…A32, B1…, EXG1–8). **Channel names → Load .cfg…**
takes names from the `[Labels]` section of an ActiView configuration file (the same files ActiView
loads, e.g. `Configuring/10-20system32+8.cfg`), so `A1` is saved as `Fp1`. The names go into the BDF,
the offsets, the display and the review window; `session.json` also lists each channel's input slot.
Pick the file that matches the cap and the *EEG channels* setting (a 32-channel layout names only
the A bank). Names used twice are made unique, e.g. bank C's third input becomes `C3 [C3]` when A8 is
already `C3`.

## Reviewing a session

**Review a session…** (or File → Review a session, ⌘O / Ctrl+O) opens a recording
folder with the video, the EEG and a timeline of the whole session in sync.
Drag the red playhead or click anywhere on the timeline or EEG to jump; markers
(orange) and Status triggers (blue) are drawn on both.

| key | action |
|---|---|
| Space | play / pause |
| ← / → | previous / next video frame (pauses; hold to keep stepping). Shift: 1 s |
| [ / ] | previous / next marker |
| M | add a marker at the playhead (or label the selected time span) |
| I / O | start / end a time-span selection at the playhead (works while playing) |
| L | loop selection on/off: while a span is selected, Play repeats it until you pause |
| Esc | clear the selection |
| ⌘S / Ctrl+S | save markers and notes |

On the EEG: **click** moves the playhead, **drag** selects a time span, **scroll or
pinch** zooms time, and **Shift+scroll** (or scrolling over the channel names) zooms
amplitude. The zoom is kept in the *Window* and *µV/div* boxes, so clicking around
never resets it. **Channels…** picks which channels to show: tick them, or type names
such as `Fp1, Fp2` and press Enter. The choice is remembered for the next session.

A marker can be a point or a **labelled time window**: select a span and press M,
type a length in the marker list, or select a marker and drag its edges on the EEG.
Markers can be relabelled, annotated, moved to the playhead, added or deleted;
**Save** writes `markers.csv` and the review notes in `session.json`. The video
is placed using the camera latency (from the session's `calibration.json` if
present, otherwise the value in the top-right box).

## How syncing works (and why there's no clap)

The EEG and the video are captured **in the same process** and timestamped
with the **same monotonic clock** (`eegvid/clock.py`). A line fitted to
`eeg_chunks.csv` maps any host time to an EEG sample. This corrects for drift
between the BioSemi crystal and the PC clock, which a single tap or clap cannot.

The one thing software can't see is the **camera's own latency** (light hits
the sensor → the frame reaches Python), usually 30–150 ms and fairly constant
for a given camera, resolution and computer. Measure it once per setup:

1. Record a short session in which something is visible on camera *and*
   shows up in the EEG at the same moment. Best: an LED driven by the same
   TTL pulse sent to the BioSemi trigger port. Acceptable: tapping an electrode.
2. Find the frame number where it happens (scrub `video.mp4`).
3. `eegvid-align calibrate SESSION --frame 788 --trigger` (or `--channel A1`)
4. Use that latency for later sessions: `eegvid-align SESSION --camera-latency-ms 85`

## Layout

```
src/eegvid/
  clock.py               shared clock
  biosemi/protocol.py    USB data format (sync word, status bits, speed modes)
  biosemi/ringbuffer.py  ctypes binding to BioSemi's RingBuffer library + simulator
  biosemi/device.py      acquisition thread → Chunk objects
  video/camera.py        threaded capture + writer
  recording/bdf_writer.py streaming BDF writer
  recording/session.py   one session folder
  analysis/align.py      clock fit, frame alignment, latency calibration (CLI)
  ui/                    PySide6 + pyqtgraph GUI
packaging/               PyInstaller spec, drivers/ folder
```

## Building the downloadable app

```bash
# put the BioSemi library in packaging/drivers/ first (see the README there)
pyinstaller packaging/eegvid.spec --noconfirm
```

Build on each target OS (PyInstaller doesn't cross-compile). The GitHub
Actions workflow in `.github/workflows/build.yml` builds Windows and macOS
zips on every `v*` tag. Unsigned apps trigger a SmartScreen / Gatekeeper
warning the first time; code signing removes that.

## Before using on real hardware (checklist)

The USB protocol follows BioSemi's
[acquisition FAQ](https://www.biosemi.com/faq/make_own_acquisition_software.htm).
A few details come from community code and are marked `VERIFY` in
`biosemi/protocol.py` and `ringbuffer.py`:

- [ ] `READ_POINTER` returns a byte offset (a wrong unit fails loudly with "lost sync")
- [ ] Speed-mode bits and sample rates in the status word (compare with the ActiView display)
- [ ] EXG1–8 positions for your speed mode (record the same signal in ActiView and here, then compare)
- [ ] Record 1 h side by side with ActiView; check for no overflow and identical triggers
- [ ] Run camera latency calibration on the lab PC + webcam
