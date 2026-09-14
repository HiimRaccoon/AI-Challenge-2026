from wp04.api import _asr_canonical_anchor


def test_asr_anchor_uses_nearest_explicit_canonical_frame():
    index = {"L21_V001": ([879000, 900000], [(879000, 26370), (900000, 27000)])}
    row = {"video_id": "L21_V001", "start_ms": 890880, "end_ms": 898560}
    # Midpoint 894720 is closer to the supplied canonical frame at 900000.
    assert _asr_canonical_anchor(row, index) == (27000, 900000)


def test_asr_anchor_fails_closed_when_the_canonical_mapping_is_missing():
    assert _asr_canonical_anchor({"video_id": "L21_V001", "start_ms": 1, "end_ms": 2}, {}) is None
