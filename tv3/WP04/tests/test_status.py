from wp04.status import ModalityStatus, should_skip


def test_no_audio_is_valid_but_missing_expected_audio_is_failure():
    assert ModalityStatus.no_audio("v", "asr", "f").state == "no_audio"
    assert ModalityStatus.failed("v", "asr", "f", "missing audio").state == "failed"


def test_resume_skips_only_matching_ready_or_no_audio_fingerprint():
    assert should_skip(ModalityStatus.ready("v", "ocr", "abc"), "abc")
    assert not should_skip(ModalityStatus.ready("v", "ocr", "abc"), "changed")


def test_status_can_carry_both_run_ids_for_auditable_artifacts():
    status = ModalityStatus.ready("v", "ocr", "fp", preprocess_run_id="tv1", wp04_artifact_set_id="wp04")
    assert status.to_dict()["wp04_artifact_set_id"] == "wp04"
