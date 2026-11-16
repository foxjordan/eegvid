# eegvid — project notes for Claude

Desktop app (PySide6 + pyqtgraph, packaged with PyInstaller) for a university EEG lab:
records BioSemi ActiveTwo EEG and webcam video together. End users are lab
assistants/students: it must stay simple (Connect → Record → Stop) and installable
without a Python setup.

## Commands
- Setup: `python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"`
- Tests: `.venv/bin/python -m pytest -q` (all run on the built-in simulator, no hardware)
- Run GUI: `.venv/bin/python -m eegvid` (choose "Simulator" without hardware)
- Build app: `.venv/bin/pyinstaller packaging/eegvid.spec --noconfirm`
- Build Intel Mac app (the one that talks to BioSemi hardware on a Mac): one-time
  `arch -x86_64 /Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10 -m venv .venv-x86_64 && arch -x86_64 .venv-x86_64/bin/pip install -e ".[dev]"`,
  then `arch -x86_64 .venv-x86_64/bin/pyinstaller packaging/eegvid.spec --noconfirm --distpath dist/x86_64 --workpath build/x86_64`
- Mac installer for the lab: `packaging/macos/make_dmg.sh` (builds the Intel app, then
  dist/EEGVideoRecorder-<version>-mac.dmg with packaging/INSTALL.txt as "Read Me First"). If a
  "Developer ID Application" cert is in the keychain it signs (`packaging/macos/sign_app.sh`: hardened
  runtime, inside-out by path depth, `entitlements.plist` incl. camera) and, with notarytool profile
  `eegvid-notary`, notarizes + staples. Both are set up on the dev Mac (Developer ID Application:
  Darkfox Development, LLC); v0.1.0 notarized and accepted 2026-11-11. Only the dmg is stapled (the
  app inside is notarized but its first launch needs internet for the ticket check).
- Windows installer: Inno Setup `packaging/windows/eegvid.iss`, built on Windows or by
  `.github/workflows/build.yml` (windows + macos-15-intel jobs; tests pass on both). Not yet run on a
  real Windows PC. Releases: GitHub pre-release v0.1.0 (private repo) holds the notarized dmg + exe.
- Align a session: `eegvid-align SESSION_DIR [--camera-latency-ms N]`; calibrate: `eegvid-align calibrate SESSION_DIR --frame N (--trigger | --channel A1)`

## Key design decisions
- Sync: EEG and video share one monotonic clock (`clock.py`, perf_counter) in one
  process. `eeg_chunks.csv` maps BDF sample ↔ host time; `analysis/align.py` fits a
  line (lower envelope) to correct crystal drift. Camera latency is measured once
  per setup with `calibrate`, not per session. No clap/tap is needed in routine sessions.
- BioSemi USB protocol follows https://www.biosemi.com/faq/make_own_acquisition_software.htm
  via ctypes to BioSemi's RingBuffer library (`biosemi/ringbuffer.py`). The library is not
  committed except RingBuffer.dll (Win64) and BioSemiRingBuffer64.framework (Mac) in
  `packaging/drivers/`, which the installer builds need; the GitHub repo must stay private. Items marked `VERIFY` (READ_POINTER units,
  speed-mode bits/rates, EXG positions) are not yet confirmed on real hardware.
- Sample rate is set by the AD-box speed-mode switch, never by software. "Save sample rate"
  = ActiView-style decimation (`dsp.Decimator`): zero-delay-compensated FIR, saved sample k ==
  raw sample k*factor, triggers kept (first non-zero per block); eeg_chunks.csv
  bdf_sample_end becomes fractional. Display filters/reference never touch saved data.
- Electrode check = DC offset vs CMS (`biosemi/offsets.py`; ActiveTwo has no impedance
  measurement). Default limit ±40 mV, adjustable; offsets saved in session.json at start/end.
- Review window (`ui/review_window.py`, data in `review/session_data.py`): own memory-mapped
  BDF reader (MNE isn't bundled). Timeline = seconds since first BDF sample; video/markers
  mapped via the eeg_chunks clock fit; camera latency shifts frames. Saving markers backs up
  the live file once to markers_recorded.csv; review notes go in session.json. Markers may have a
  `duration` (s; time windows). EEG zoom state lives in the Window/µV-per-div boxes (custom
  `_EegViewBox`: scroll = time, Shift+scroll = amplitude, drag = select span), never in pyqtgraph's view.
- Video codec: OpenCV 5's Intel-Mac wheel has no FFmpeg (mp4v can't be written), so Macs
  record H.264 `avc1` via AVFoundation; Windows tries mp4v first (`video.open_video_writer`).
- Review video is read with PyAV (`review/video_reader.py`), indexed by packet pts. NEVER use
  OpenCV seeking for review: CAP_PROP_POS_FRAMES lands 1 frame early on some frames (both archs)
  and the Intel/AVFoundation backend misreads 29.97 fps as 29.001 (drifts minutes per hour).
  `av` is imported lazily (cv2 and av both bundle libavdevice; macOS warns if both load).
- CMS/DRL have no offset (all offsets are vs CMS); only status bit 20 'CMS in range'. On the lab's
  ActiveTwo the bit is CLEAR when in range (opposite of BioSemi's FAQ; observed 2026-11-11), set by
  `protocol.CMS_BIT_SET_MEANS_IN_RANGE`. When it is
  false the offsets tab greys the grid and shows a banner. Simulator: `sim.cms_in_range = False`.
- Channel names: `Channel.slot` is the hardware input (A1.., EXG1..), `label` the name, optionally from
  an ActiView .cfg `[Labels]` (`biosemi/channel_names.py`; ActiView's own set is in
  ActiView902-Mac/Configuring/). Duplicates become `name [slot]`. Use `is_exg`, never the label, to
  detect EXG channels.
- BDF writer mimics ActiView (physical range -262144..262143 µV → ~0.5 µV offset, expected).
- MNE is analysis-only (optional extra) and is excluded from the packaged recorder.

## Status (2026-11-10)
- Tests pass; GUI smoke-tested with simulator; macOS bundle builds and launches.
- Not yet tested: real ActiveTwo hardware, real webcam capture (camera permission).
- Likely target acquisition OS: Windows. BioSemi's macOS SDK is i386/x86_64 only, with
  precompiled .o files and no source, so it can't load on Apple Silicon (confirmed).
- BUT ActiView902-Mac (Intel-only LabVIEW 2021 app, runs under Rosetta on the lab's
  Apple Silicon Mac) ships the same x86_64 BioSemiRingBuffer64.framework. So an x86_64
  build of eegvid (Rosetta) bundling that framework is a viable Mac route. eegvid itself
  never needs the LabVIEW Runtime.
- Intel build verified 2026-11-10: library + bundled libusb load, app runs translated.
  Connect with real hardware not yet tried. Rosetta ends with macOS 28.
- The lab moved to a Mac because ActiView + hardware misbehaved on Windows 11 (symptoms
  not yet recorded). The Windows RingBuffer.dll uses the same WinUSB driver as ActiView,
  so driver-level problems carry over; LabVIEW/ActiView-app problems don't. Diagnose with
  the SDK's Windows64/RingBuffer_SyncTest.exe first.
- The user's original scripts (tap-artifact sync with MNE, OpenCV recorder) were
  replaced by `analysis/align.py` and `video/camera.py`.
