from wp04.contracts import AudioRecord, FrameRecord
from wp04.adapters import OCRRawDetection
from wp04.pipeline import Pipeline, FakeAdapter, WP04Pipeline
from wp04.contracts import WP04RunIdentity
from wp04.storage import ArtifactStore

def test_pipeline_isolates_one_adapter_failure():
    result = Pipeline({"ocr": FakeAdapter(error="load failed"), "object": FakeAdapter()}).run(["L01_V001"], "fp")
    assert result[0].state == "failed"
    assert result[1].state == "ready"


class FakeOCR:
    def detect(self, frame: FrameRecord):
        return []


class FailingOCR:
    def detect(self, frame: FrameRecord):
        raise RuntimeError("model checkpoint missing")


class FakeASR:
    def transcribe(self, audio: AudioRecord):
        return []


class FakeObjects:
    def detect(self, frame: FrameRecord):
        return []


class BatchOCR:
    model_name = "craft-parseq-vn"
    model_version = "v1"

    def __init__(self):
        self.batches = []

    def detect_batch(self, frames):
        self.batches.append([frame.frame_id for frame in frames])
        return [[OCRRawDetection("BÃNH MÃŒ", (0.1, 0.2, 0.3, 0.4), 0.9)] for _ in frames]


class BatchObjects:
    def __init__(self):
        self.batches = []

    def detect_batch(self, frames):
        self.batches.append([frame.frame_id for frame in frames])
        return [[] for _ in frames]


class UnexpectedASR:
    def transcribe(self, audio: AudioRecord):
        raise AssertionError("disabled ASR must not run")


class UnexpectedObjects:
    def detect(self, frame: FrameRecord):
        raise AssertionError("disabled object detection must not run")


def test_asr_without_declared_audio_is_no_audio_and_does_not_stop_ocr():
    frame = FrameRecord("tv1", "L01_V001", 42, 1, 1400)
    audio = AudioRecord("tv1", "L01_V001", None, None, False, None)
    result = WP04Pipeline(FakeOCR(), FakeASR(), FakeObjects()).run([frame], {frame.video_id: audio}, "fp")
    assert result.status_for("L01_V001", "asr").state == "no_audio"
    assert result.status_for("L01_V001", "ocr").state == "ready"


def test_failed_ocr_is_visible_while_object_pipeline_completes():
    frame = FrameRecord("tv1", "L01_V001", 42, 1, 1400)
    audio = AudioRecord("tv1", "L01_V001", None, None, False, None)
    result = WP04Pipeline(FailingOCR(), FakeASR(), FakeObjects()).run([frame], {frame.video_id: audio}, "fp")
    assert result.status_for("L01_V001", "ocr").state == "failed"
    assert result.status_for("L01_V001", "object").state == "ready"


def test_expected_audio_missing_is_failed_not_no_audio():
    frame = FrameRecord("tv1", "L01_V001", 42, 1, 1400)
    result = WP04Pipeline(FakeOCR(), FakeASR(), FakeObjects()).run([frame], {}, "fp")
    assert result.status_for("L01_V001", "asr").state == "failed"


class DetectingOCR:
    model_name = "craft-parseq-vn"
    model_version = "v1"

    def detect(self, frame: FrameRecord):
        return [OCRRawDetection("BÁNH MÌ", (0.1, 0.2, 0.3, 0.4), 0.9)]


def test_run_and_store_persists_normalized_ocr_before_ready_status(tmp_path):
    frame = FrameRecord("tv1", "L01_V001", 42, 1, 1400)
    audio = AudioRecord("tv1", "L01_V001", None, None, False, None)
    store = ArtifactStore(tmp_path, WP04RunIdentity("tv1", "wp04", "inputs", "config"))
    result = WP04Pipeline(DetectingOCR(), FakeASR(), FakeObjects()).run_and_store([frame], {frame.video_id: audio}, "fp", store)
    assert result.status_for("L01_V001", "ocr").state == "ready"
    row = store.read_records("ocr", "L01_V001")[0]
    assert row["normalized_text"] == "bánh mì"
    assert row["evidence_id"] == "ocr:L01_V001:42:0"


class WrongShapeOCR:
    def detect(self, frame: FrameRecord):
        return [{"text": "not a raw detection"}]


def test_wrong_raw_adapter_output_is_failed_not_ready(tmp_path):
    frame = FrameRecord("tv1", "L01_V001", 42, 1, 1400)
    audio = AudioRecord("tv1", "L01_V001", None, None, False, None)
    store = ArtifactStore(tmp_path, WP04RunIdentity("tv1", "wp04", "inputs", "config"))
    result = WP04Pipeline(WrongShapeOCR(), FakeASR(), FakeObjects()).run_and_store([frame], {frame.video_id: audio}, "fp", store)
    assert result.status_for("L01_V001", "ocr").state == "failed"


def test_disabled_modalities_are_not_called_and_metadata_is_stored(tmp_path):
    frame = FrameRecord("tv1", "L01_V001", 42, 1, 1400)
    audio = AudioRecord("tv1", "L01_V001", None, None, False, None)
    store = ArtifactStore(tmp_path, WP04RunIdentity("tv1", "wp04", "inputs", "config"))
    result = WP04Pipeline(
        DetectingOCR(), UnexpectedASR(), UnexpectedObjects(),
        enabled={"ocr": True, "asr": False, "object": False, "metadata": True},
    ).run_and_store([frame], {frame.video_id: audio}, "fp", store)
    assert {status.modality: status.state for status in result.statuses} == {
        "ocr": "ready", "asr": "disabled", "object": "disabled", "metadata": "ready",
    }
    assert store.read_records("metadata", "L01_V001")
    assert not store.read_records("object", "L01_V001")


def test_run_and_store_uses_configured_frame_batches(tmp_path):
    frames = [
        FrameRecord("tv1", "L01_V001", frame_id, frame_id, frame_id * 1000)
        for frame_id in (1, 2, 3)
    ]
    ocr, objects = BatchOCR(), BatchObjects()
    store = ArtifactStore(tmp_path, WP04RunIdentity("tv1", "wp04", "inputs", "config"))

    WP04Pipeline(
        ocr, FakeASR(), objects,
        enabled={"ocr": True, "asr": False, "object": True, "metadata": False},
        ocr_batch_size=2, object_batch_size=2,
    ).run_and_store(frames, {}, "fp", store)

    assert ocr.batches == [[1, 2], [3]]
    assert objects.batches == [[1, 2], [3]]
    assert len(store.read_records("ocr", "L01_V001")) == 3


def test_temporal_sampling_keeps_boundaries_and_respects_interval(tmp_path):
    frames = [
        FrameRecord("tv1", "L01_V001", frame_id, frame_id, timestamp_ms, shot_id=shot_id)
        for frame_id, (timestamp_ms, shot_id) in enumerate(
            ((0, "s1"), (100, "s1"), (200, "s1"), (300, "s2"), (400, "s2"), (450, "s2")),
            start=1,
        )
    ]
    ocr, objects = BatchOCR(), BatchObjects()
    store = ArtifactStore(tmp_path, WP04RunIdentity("tv1", "wp04", "inputs", "config"))

    WP04Pipeline(
        ocr, FakeASR(), objects,
        enabled={"ocr": True, "asr": False, "object": True, "metadata": False},
        ocr_batch_size=8, object_batch_size=8,
        ocr_sample_interval_ms=200, object_sample_interval_ms=200,
    ).run_and_store(frames, {}, "fp", store)

    assert ocr.batches == [[1, 3, 4, 5, 6]]
    assert objects.batches == [[1, 3, 4, 5, 6]]
    assert [row["frame_id"] for row in store.read_records("ocr", "L01_V001")] == [1, 3, 4, 5, 6]


def test_run_and_store_resumes_only_when_entire_video_is_complete(tmp_path):
    frame = FrameRecord("tv1", "L01_V001", 42, 1, 1400)
    audio = AudioRecord("tv1", "L01_V001", None, None, False, None)
    store = ArtifactStore(tmp_path, WP04RunIdentity("tv1", "wp04", "inputs", "config"))
    first_ocr, first_objects = BatchOCR(), BatchObjects()
    pipeline = WP04Pipeline(first_ocr, FakeASR(), first_objects)
    pipeline.run_and_store([frame], {frame.video_id: audio}, "fp", store)

    second_ocr, second_objects = BatchOCR(), BatchObjects()
    resumed = WP04Pipeline(second_ocr, FakeASR(), second_objects).run_and_store(
        [frame], {frame.video_id: audio}, "fp", store,
    )

    assert second_ocr.batches == []
    assert second_objects.batches == []
    assert {status.state for status in resumed.statuses} == {"ready", "no_audio"}
