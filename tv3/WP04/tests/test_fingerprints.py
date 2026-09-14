from wp04.fingerprints import build_input_fingerprint


def test_fingerprint_changes_when_audio_checksum_or_vad_config_changes():
    baseline = build_input_fingerprint({"frames": ["f1"], "audio": "a1"}, {"vad_threshold": 0.5}, {"asr": "v1"})
    changed_audio = build_input_fingerprint({"frames": ["f1"], "audio": "a2"}, {"vad_threshold": 0.5}, {"asr": "v1"})
    changed_vad = build_input_fingerprint({"audio": "a1", "frames": ["f1"]}, {"vad_threshold": 0.6}, {"asr": "v1"})
    assert baseline != changed_audio
    assert baseline != changed_vad


def test_fingerprint_changes_when_keyframe_checksum_changes():
    before = build_input_fingerprint({"frames": [{"path": "a.jpg", "sha256": "one"}]}, {}, {})
    after = build_input_fingerprint({"frames": [{"path": "a.jpg", "sha256": "two"}]}, {}, {})
    assert before != after
