from eegvid.biosemi.protocol import SPEED_MODES, build_layout, decode_status, electrode_label, encode_status


def test_status_roundtrip():
    for mode in SPEED_MODES:
        w = encode_status(0x1234, speed_mode=mode, mk2=True, cms_in_range=False, battery_low=True)
        st = decode_status(w)
        assert st.trigger == 0x1234
        assert st.speed_mode == mode
        assert st.mk2 and st.battery_low and not st.cms_in_range
        assert w & 0xFF == 0  # zero LSB byte, like the receiver


def test_word_counts_match_faq():
    assert SPEED_MODES[4].words_per_sample(mk2=True) == 282
    assert SPEED_MODES[8].words_per_sample(mk2=False) == 290


def test_layout():
    assert electrode_label(0) == "A1" and electrode_label(33) == "B2"
    ch = build_layout(SPEED_MODES[4], True, 64, include_exg=True)
    assert [c.label for c in ch[-8:]] == [f"EXG{i}" for i in range(1, 9)]
    assert ch[64].adc_index == 256


def test_cms_bit_polarity_matches_hardware():
    # Observed on the lab's ActiveTwo: bit 20 is CLEAR while CMS/DRL are in range.
    from eegvid.biosemi.protocol import decode_status
    assert decode_status(0).cms_in_range
    assert not decode_status((1 << 20) << 8).cms_in_range
    assert not (encode_status(0, speed_mode=4, cms_in_range=True) >> 8) & (1 << 20)
