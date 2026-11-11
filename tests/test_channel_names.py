import json
import time

import pytest

from eegvid.biosemi import BioSemiDevice, SimulatedRingBuffer
from eegvid.biosemi.channel_names import apply_names, parse_actiview_cfg
from eegvid.biosemi.protocol import SPEED_MODES, build_layout, electrode_label
from eegvid.recording import RecordingSession

TEN_TWENTY_32 = ["Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3", "CP1", "CP5", "P7", "P3", "Pz",
                 "PO3", "O1", "Oz", "O2", "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8", "FC6", "FC2",
                 "F4", "F8", "AF4", "Fp2", "Fz", "Cz"]


@pytest.fixture
def cfg(tmp_path):
    """Same layout as BioSemi's 10-20system32+8.cfg: A bank named, other banks keep slot names."""
    lines = ["[System]", "Warning = \"0\"", "ElecGain = 0\t//0: 31.25nV/bit, 1: 125nV/bit", "", "[Labels]"]
    lines += [f"Chan{i + 1} = {n}" for i, n in enumerate(TEN_TWENTY_32)]
    lines += [f"Chan{i + 1} = {electrode_label(i)}" for i in range(32, 256)]
    lines += [f"Tou{k} = EXG{k}" for k in range(1, 9)]
    lines += ["Aux1 = GSR1", "Jazz1 = EyeX", "Box1 = Ana1", "Trig = Status", "",
              "[Selectors]", "Chan1 = NotALabel", "Decimation = 0"]
    p = tmp_path / "10-20system32+8.cfg"
    p.write_text("\n".join(lines))
    return p


def test_parse(cfg):
    names = parse_actiview_cfg(cfg)
    assert names["A1"] == "Fp1" and names["A32"] == "Cz" and names["A8"] == "C3"
    assert names["B1"] == "B1" and names["C3"] == "C3" and names["H32"] == "H32"
    assert names["EXG1"] == "EXG1" and len(names) == 256 + 8
    assert "Aux1" not in names and "GSR1" not in names.values()  # sensors aren't channels here


def test_parse_rejects_files_without_labels(tmp_path):
    p = tmp_path / "x.cfg"
    p.write_text("[System]\nWarning = 0\n")
    with pytest.raises(ValueError):
        parse_actiview_cfg(p)


def test_apply_names_64_and_renamed_exg(cfg):
    names = parse_actiview_cfg(cfg)
    names["EXG1"] = "VEOG"  # labs often rename the touchproof inputs
    chans = build_layout(SPEED_MODES[4], True, 64, include_exg=True)
    named, notes = apply_names(chans, names)
    assert not notes
    assert [c.label for c in named[:32]] == TEN_TWENTY_32
    assert named[32].label == "B1" and named[64].label == "VEOG"
    assert named[64].slot == "EXG1" and named[64].is_exg and not named[0].is_exg
    assert [c.adc_index for c in named] == [c.adc_index for c in chans]  # data mapping unchanged


def test_apply_names_resolves_duplicates(cfg):
    chans = build_layout(SPEED_MODES[4], True, 128, include_exg=False)
    named, notes = apply_names(chans, parse_actiview_cfg(cfg))
    labels = [c.label for c in named]
    assert len(set(labels)) == len(labels)
    assert labels[7] == "C3" and labels[66] == "C3 [C3]"   # A8 is the 10-20 C3; bank C's 3rd input
    assert labels[22] == "C4" and labels[67] == "C4 [C4]"
    assert len(notes) == 2


def test_session_uses_names(cfg, tmp_path):
    mne = pytest.importorskip("mne")
    dev = BioSemiDevice(SimulatedRingBuffer(speed_mode=4))
    holder = {}
    info = dev.start(lambda c: holder.get("s") and holder["s"].add_eeg_chunk(c))
    chans, _ = apply_names(build_layout(info.speed_mode, info.mk2, 32, include_exg=True),
                           parse_actiview_cfg(cfg))
    holder["s"] = RecordingSession(tmp_path, "names", info, chans, names_cfg=cfg)
    time.sleep(1.2)
    s = holder.pop("s")
    s.close()
    dev.stop()
    raw = mne.io.read_raw_bdf(s.folder / "eeg.bdf", verbose="error")
    assert raw.ch_names[:3] == ["Fp1", "AF3", "F7"] and "Cz" in raw.ch_names and "EXG8" in raw.ch_names
    meta = json.loads((s.folder / "session.json").read_text())["eeg"]
    assert meta["channel_slots"][:2] == ["A1", "A2"] and meta["channel_names_from"] == cfg.name
    assert (s.folder / "channel_names.cfg").read_text() == cfg.read_text()
