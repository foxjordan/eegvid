"""ActiveTwo USB data format.

Source: https://www.biosemi.com/faq/make_own_acquisition_software.htm

Each sample is a run of 32-bit little-endian words:
    word 0      sync word, always 0xFFFFFF00
    word 1      status / trigger channel
    word 2..    ADC channels (24-bit value << 8, so the LSB byte is 0)

24-bit LSB = 1/32 uV; 32-bit LSB = 1/8192 uV.

Items marked VERIFY come from community implementations (e.g. the LSL BioSemi
app) rather than the FAQ page. Check them against ActiView on the real hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

RING_BUFFER_BYTES = 2**25  # 32 MiB, as recommended by BioSemi
RING_BUFFER_WORDS = RING_BUFFER_BYTES // 4

CMD_INIT_OR_STOP = bytes(64)            # 64 zero bytes
CMD_START = bytes([0xFF]) + bytes(63)   # enable handshake

SYNC_WORD = 0xFFFFFF00

UV_PER_LSB_24 = 1.0 / 32.0
UV_PER_LSB_32 = 1.0 / 8192.0


@dataclass(frozen=True)
class SpeedMode:
    mode: int
    sample_rate: int
    words_mk1: int
    words_mk2: int
    eeg_slots: int  # number of cap-electrode slots (A1..)

    def words_per_sample(self, mk2: bool) -> int:
        return self.words_mk2 if mk2 else self.words_mk1


# Word counts from the BioSemi FAQ. Sample rates and slot counts: VERIFY.
# Modes 0-3 (daisy-chained / Mk2 multi-rate) are not supported yet.
SPEED_MODES: dict[int, SpeedMode] = {
    4: SpeedMode(4, 2048, 258, 282, 256),
    5: SpeedMode(5, 4096, 130, 154, 128),
    6: SpeedMode(6, 8192, 66, 90, 64),
    7: SpeedMode(7, 16384, 34, 58, 32),
    8: SpeedMode(8, 2048, 290, 314, 256),  # with AIB
}

# Mk2 layout after the cap slots: 8 EXG words, then 16 sensor/aux words
# (282 = 2 + 256 + 8 + 16). VERIFY on hardware.
MK2_EXG_COUNT = 8


# Status bit 20 and CMS/DRL. BioSemi's trigger FAQ says the bit is "high when CMS is
# within range", but on the lab's ActiveTwo (2026-11-11) it read the other way round:
# the bit was clear with CMS and DRL both on the skin, and set with neither or only one
# of them connected. Treat a CLEAR bit as "in range". If a unit ever shows the
# opposite (compare with the blue CMS/DRL light on the AD-box), flip this constant.
CMS_BIT_SET_MEANS_IN_RANGE = False


@dataclass(frozen=True)
class Status:
    trigger: int        # bits 0-15
    epoch_start: bool   # bit 16
    cms_in_range: bool  # bit 20 (see CMS_BIT_SET_MEANS_IN_RANGE)
    battery_low: bool   # bit 22
    mk2: bool           # bit 23
    speed_mode: int     # bits 17,18,19 + bit 21 as the high bit


def status24(word32: int) -> int:
    """Strip the zero LSB byte from a 32-bit status word."""
    return (int(word32) & 0xFFFFFFFF) >> 8


def decode_status(word32: int) -> Status:
    s = status24(word32)
    speed = ((s >> 17) & 0b111) | (((s >> 21) & 1) << 3)
    return Status(
        trigger=s & 0xFFFF,
        epoch_start=bool(s & (1 << 16)),
        cms_in_range=bool(s & (1 << 20)) == CMS_BIT_SET_MEANS_IN_RANGE,
        battery_low=bool(s & (1 << 22)),
        mk2=bool(s & (1 << 23)),
        speed_mode=speed,
    )


def encode_status(trigger: int, *, speed_mode: int, mk2: bool = True,
                  cms_in_range: bool = True, battery_low: bool = False,
                  epoch_start: bool = False) -> int:
    """Inverse of decode_status (used by the simulator and tests). Returns a 32-bit word."""
    s = trigger & 0xFFFF
    s |= (speed_mode & 0b111) << 17
    s |= ((speed_mode >> 3) & 1) << 21
    s |= int(epoch_start) << 16
    s |= int(cms_in_range == CMS_BIT_SET_MEANS_IN_RANGE) << 20
    s |= int(battery_low) << 22
    s |= int(mk2) << 23
    return (s << 8) & 0xFFFFFFFF


def electrode_label(i: int) -> str:
    """0 -> A1, 31 -> A32, 32 -> B1, ... (BioSemi cap naming)."""
    return f"{chr(ord('A') + i // 32)}{i % 32 + 1}"


@dataclass(frozen=True)
class Channel:
    label: str      # display / BDF name (from an ActiView .cfg if loaded, else the slot)
    adc_index: int  # index into the ADC words (sample word 2 + adc_index)
    slot: str = ""  # hardware input: A1..H32 or EXG1..8 (never renamed)

    def __post_init__(self):
        if not self.slot:
            object.__setattr__(self, "slot", self.label)

    @property
    def is_exg(self) -> bool:
        return self.slot.startswith("EXG")


def build_layout(mode: SpeedMode, mk2: bool, n_eeg: int, include_exg: bool) -> list[Channel]:
    """Choose which ADC words are recorded and how they are labelled."""
    n_adc = mode.words_per_sample(mk2) - 2
    if n_eeg > mode.eeg_slots:
        raise ValueError(f"Speed mode {mode.mode} has only {mode.eeg_slots} electrode slots")
    chans = [Channel(electrode_label(i), i) for i in range(n_eeg)]
    if include_exg and mk2:
        base = mode.eeg_slots
        chans += [Channel(f"EXG{k + 1}", base + k) for k in range(MK2_EXG_COUNT)
                  if base + k < n_adc]
    return chans
