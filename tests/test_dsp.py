import time

import numpy as np
import pytest

from eegvid.analysis.align import load_session
from eegvid.biosemi import BioSemiDevice, SimulatedRingBuffer
from eegvid.biosemi.offsets import OffsetMonitor, summarize
from eegvid.biosemi.protocol import build_layout
from eegvid.dsp import Decimator, display_filter, minus3db_hz
from eegvid.recording import RecordingSession

FS = 2048


def _run_chunked(dec, x, status=None, chunk=37):
    ys, ss = [], []
    for i in range(0, len(x), chunk):
        out = dec.process(x[i:i + chunk], None if status is None else status[i:i + chunk])
        y, s = out if status is not None else (out, None)
        ys.append(y)
        ss.append(s)
    out = dec.flush()
    y, s = out if status is not None else (out, None)
    ys.append(y)
    ss.append(s)
    return np.concatenate(ys), (None if status is None else np.concatenate(ss))


@pytest.mark.parametrize("factor", [2, 4, 8])
def test_decimator_length_dc_and_passband(factor):
    n = FS * 3 + 5  # not a multiple of the factor
    t = np.arange(n) / FS
    fs_out = FS / factor
    f_pass = 0.3 * fs_out
    x = (-20000 * 32 + 3200 * np.sin(2 * np.pi * f_pass * t))[:, None]  # -20 mV DC + 100 uV sine
    y, _ = _run_chunked(Decimator(factor, 1, integer=False), x)
    assert len(y) == -(-n // factor)
    sl = slice(len(y) // 4, -len(y) // 4)
    mid = y[sl, 0]
    tk = np.arange(len(y))[sl] * factor / FS
    basis = np.column_stack([np.ones_like(tk), np.sin(2 * np.pi * f_pass * tk), np.cos(2 * np.pi * f_pass * tk)])
    dc, a, b = np.linalg.lstsq(basis, mid, rcond=None)[0]
    assert abs(dc - (-20000 * 32)) < 5                     # DC preserved
    assert abs(np.hypot(a, b) / 3200 - 1) < 0.005          # flat passband (< 0.05 dB)
    # no start-up transient from the large DC offset
    assert np.max(np.abs(y[:20, 0] - (-20000 * 32))) < 3300


@pytest.mark.parametrize("factor", [2, 4, 8])
def test_decimator_rejects_aliases(factor):
    fs_out = FS / factor
    t = np.arange(FS * 2) / FS
    x = np.sin(2 * np.pi * 0.7 * fs_out * t)[:, None] * 1e4  # would alias to 0.3 x fs_out
    y, _ = _run_chunked(Decimator(factor, 1, integer=False), x)
    assert np.max(np.abs(y[50:-50])) < 1e4 * 10 ** (-60 / 20)  # > 60 dB down


@pytest.mark.parametrize("factor", [2, 4, 8])
def test_decimator_keeps_timing(factor):
    """Saved sample k must equal the raw signal at raw sample k*factor: any leftover
    filter delay would show up as a phase error on this sine."""
    f = 0.1 * FS / factor
    t = np.arange(FS * 2) / FS
    x = np.sin(2 * np.pi * f * t)[:, None] * 1000
    y, _ = _run_chunked(Decimator(factor, 1, integer=False), x)
    k = np.arange(len(y))
    expected = x[k * factor, 0]
    # 1 raw sample of delay would give an error of 2*pi*f/FS*1000 = 2.4..16 units.
    assert np.max(np.abs(y[20:-20, 0] - expected[20:-20])) < 1.0


def test_decimator_keeps_short_triggers_and_flags():
    factor = 8
    n = FS
    status = np.full(n, 1 << 20, dtype=np.int64)  # CMS-in-range flag set throughout
    pulses = {101: 7, 503: 9, 1203: 3}            # 1-sample pulses at odd positions
    for i, v in pulses.items():
        status[i] |= v
    y, st = _run_chunked(Decimator(factor, 2), np.zeros((n, 2)), status)
    assert len(st) == len(y) == n // factor
    nz = np.flatnonzero(st & 0xFFFF)
    assert [(int(k), int(st[k] & 0xFFFF)) for k in nz] == [(i // factor, v) for i, v in pulses.items()]
    assert np.all(st & (1 << 20))


def test_decimator_factor_one_is_passthrough():
    x = np.arange(30).reshape(15, 2)
    st = np.arange(15)
    y, s = _run_chunked(Decimator(1, 2), x, st, chunk=4)
    np.testing.assert_array_equal(y, x)
    np.testing.assert_array_equal(s, st)


def test_lowpass_cutoff_reported():
    lp = minus3db_hz(Decimator(4, 1).h, FS)
    assert 0.4 * 512 < lp < 0.5 * 512


def test_offset_monitor_and_summary():
    sim = SimulatedRingBuffer(speed_mode=4)
    dev = BioSemiDevice(sim)
    mon = OffsetMonitor(FS)
    info = dev.start(lambda c: mon.push(c.adc24()))
    time.sleep(0.8)
    dev.stop()
    chans = build_layout(info.speed_mode, info.mk2, 64, include_exg=True)
    idx = np.array([c.adc_index for c in chans])
    mv = mon.offsets_mv(idx)
    expected = np.clip(sim._offsets_uv[idx], -262144, 262143) / 1000
    np.testing.assert_allclose(mv, expected, atol=0.5)
    s = summarize([c.label for c in chans], mv, 40.0)
    assert [b[0] for b in s.bad] == ["B10", "A21", "A5"]  # worst first
    assert s.n_ok == len(chans) - 3


def test_display_filter_notch_and_highpass():
    fs = 256
    t = np.arange(fs * 5) / fs
    alpha = np.sin(2 * np.pi * 10 * t)
    x = np.column_stack([alpha + np.sin(2 * np.pi * 50 * t) + 0.5 * t])  # alpha + mains + drift
    y = display_filter(x, fs, highpass=0.5, notch=50)
    err = y[fs:-fs, 0] - alpha[fs:-fs]
    assert np.std(err) < 0.1


def test_session_saves_at_reduced_rate(tmp_path):
    mne = pytest.importorskip("mne")
    dev = BioSemiDevice(SimulatedRingBuffer(speed_mode=4, trigger_every=0.5))
    holder = {}
    info = dev.start(lambda c: holder.get("s") and holder["s"].add_eeg_chunk(c))
    chans = build_layout(info.speed_mode, info.mk2, 32, include_exg=False)
    offs = {"A1": -21.5, "A5": -55.0}
    holder["s"] = RecordingSession(tmp_path, "dec", info, chans, decimation=4, electrode_offsets_mv=offs)
    time.sleep(2.3)
    s = holder.pop("s")
    s.close(electrode_offsets_mv={"A1": -22.0})
    dev.stop()

    raw = mne.io.read_raw_bdf(s.folder / "eeg.bdf", preload=True, verbose="error")
    assert raw.info["sfreq"] == 512
    events = mne.find_events(raw, stim_channel="Status", shortest_event=1, verbose="error")
    assert len(events) >= 3 and set(events[:, 2]) <= {1, 2, 3, 4}
    # Trigger every 0.5 s: consecutive onsets exactly 256 saved samples apart.
    assert set(np.diff(events[:, 0])) == {256}
    a1 = raw.get_data(picks="A1")[0] * 1e6
    assert -40000 < a1.mean() < 0

    meta, fit = load_session(s.folder)
    assert meta["eeg"]["sample_rate"] == 512 and meta["eeg"]["device_sample_rate"] == 2048
    assert meta["eeg"]["electrode_offsets_mV"] == {"start": offs, "end": {"A1": -22.0}}
    assert abs(1 / fit.seconds_per_sample - 512) < 5


def test_cms_status_reaches_device():
    sim = SimulatedRingBuffer(speed_mode=4)
    dev = BioSemiDevice(sim)
    dev.start(lambda c: None)
    time.sleep(0.2)
    assert dev.last_status.cms_in_range
    sim.cms_in_range = False
    time.sleep(0.2)
    assert not dev.last_status.cms_in_range
    sim.cms_in_range = True
    time.sleep(0.2)
    dev.stop()
    assert dev.last_status.cms_in_range
