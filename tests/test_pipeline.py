"""End to end without hardware: simulator -> device thread -> session -> BDF -> MNE."""

import time

import numpy as np
import pytest

from eegvid.analysis.align import align_session, fit_sample_clock, load_session
from eegvid.biosemi import BioSemiDevice, SimulatedRingBuffer
from eegvid.biosemi.protocol import build_layout
from eegvid.recording import BdfWriter, RecordingSession, SignalSpec


def test_device_streams_simulator():
    chunks = []
    dev = BioSemiDevice(SimulatedRingBuffer(speed_mode=4))
    info = dev.start(chunks.append)
    time.sleep(1.0)
    dev.stop()
    assert info.sample_rate == 2048 and info.words_per_sample == 282
    n = sum(c.n for c in chunks)
    assert 1500 < n < 2600
    # contiguous indices
    for a, b in zip(chunks, chunks[1:]):
        assert b.sample_start == a.sample_end


def test_session_writes_readable_bdf(tmp_path):
    mne = pytest.importorskip("mne")
    dev = BioSemiDevice(SimulatedRingBuffer(speed_mode=4, trigger_every=0.5))
    session = {}
    info = dev.start(lambda c: session.get("s") and session["s"].add_eeg_chunk(c))
    chans = build_layout(info.speed_mode, info.mk2, 32, include_exg=True)
    session["s"] = RecordingSession(tmp_path, "test subj", info, chans)
    time.sleep(2.3)
    s = session.pop("s")
    s.close()
    dev.stop()

    raw = mne.io.read_raw_bdf(s.folder / "eeg.bdf", preload=True, verbose="error")
    assert raw.info["sfreq"] == 2048
    assert raw.ch_names[:2] == ["A1", "A2"] and "EXG8" in raw.ch_names
    assert raw.n_times >= s.bdf.n_samples  # padded up to a whole record
    a1 = raw.get_data(picks="A1")[0] * 1e6
    assert -40000 < a1.mean() < 0  # simulated DC offset, in uV
    events = mne.find_events(raw, stim_channel="Status", shortest_event=1, verbose="error")
    assert len(events) >= 3 and set(events[:, 2]) <= {1, 2, 3, 4}

    meta, fit = load_session(s.folder)
    assert abs(1 / fit.seconds_per_sample - 2048) < 20
    assert align_session(s.folder) is None  # no video in this test


def test_clock_fit_recovers_drift():
    rng = np.random.default_rng(1)
    k = np.arange(1, 3600) * 20  # chunk ends
    true_spp = 1 / 2048 * (1 + 30e-6)  # 30 ppm crystal mismatch
    host = 100.0 + (k - 1) * true_spp + rng.exponential(0.002, k.size)  # positive transport delay
    fit = fit_sample_clock(k, host)
    true_t = 100.0 + (k - 1) * true_spp
    pred_t = fit.intercept + (k - 1) * fit.seconds_per_sample
    assert np.max(np.abs(pred_t - true_t)) < 0.5e-3  # sub-millisecond over the whole span


def test_bdf_values_roundtrip(tmp_path):
    mne = pytest.importorskip("mne")
    data = np.column_stack([np.arange(-1000, 1000) * 37, np.zeros(2000, dtype=int)])
    w = BdfWriter(tmp_path / "x.bdf", [SignalSpec("A1"), SignalSpec.status()], 1000)
    w.write(data[:700])
    w.write(data[700:])
    w.close()
    raw = mne.io.read_raw_bdf(tmp_path / "x.bdf", preload=True, verbose="error")
    np.testing.assert_allclose(raw.get_data(picks="A1")[0] * 1e6, data[:, 0] / 32, atol=0.6)  # ActiView-standard header range: ~0.5 uV offset
