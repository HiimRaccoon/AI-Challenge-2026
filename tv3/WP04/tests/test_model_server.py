import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


def _load_model_server():
    source = Path(__file__).parents[1] / "deploy" / "model-server" / "model_server.py"
    spec = importlib.util.spec_from_file_location("wp04_model_server_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rfdetr_parser_maps_class_ids_to_semantic_labels(tmp_path: Path):
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 50)).save(image_path)

    class FakeOutput:
        xyxy = [[10, 5, 80, 45]]
        confidence = [0.9]
        class_id = [0]

    module = _load_model_server()
    rows = module._parse_rfdetr_output(FakeOutput(), image_path, class_names=["person"])

    assert rows == [{
        "label": "person",
        "bbox_xyxy_norm": [0.1, 0.1, 0.8, 0.9],
        "confidence": 0.9,
    }]


def test_rfdetr_engine_predicts_a_list_of_keyframes_as_one_batch(tmp_path: Path):
    image_paths = [tmp_path / "frame-1.jpg", tmp_path / "frame-2.jpg"]
    for image_path in image_paths:
        Image.new("RGB", (100, 50)).save(image_path)

    class FakeOutput:
        xyxy = [[10, 5, 80, 45]]
        confidence = [0.9]
        class_id = [0]

    class FakeModel:
        class_names = ["person"]

        def __init__(self):
            self.calls = []

        def predict(self, paths, **kwargs):
            self.calls.append(paths)
            return [FakeOutput() for _ in paths]

    module = _load_model_server()
    engine = module.ObjectEngine()
    engine._model = FakeModel()

    rows = engine.infer_many([str(path) for path in image_paths])

    assert engine._model.calls == [[str(path) for path in image_paths]]
    assert [row[0]["label"] for row in rows] == ["person", "person"]


def test_craft_detector_batches_same_sized_images():
    class FakeDetector:
        def __init__(self):
            self.calls = []

        def detect(self, images, reformat=True):
            self.calls.append((images.shape, reformat))
            return [
                [[1, 4, 2, 8]]
                for _ in images
            ], [
                []
                for _ in images
            ]

    module = _load_model_server()
    engine = module.OCREngine()
    detector = FakeDetector()
    engine._detector = detector

    boxes = engine._detect_boxes_many([
        np.zeros((32, 64, 3), dtype=np.uint8),
        np.zeros((32, 64, 3), dtype=np.uint8),
    ])

    assert detector.calls == [((2, 32, 64, 3), False)]
    assert boxes == [[(1.0, 2.0, 4.0, 8.0)], [(1.0, 2.0, 4.0, 8.0)]]
