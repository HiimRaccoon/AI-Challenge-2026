"""Dependency-light T014 media-route regressions for constrained environments."""
from __future__ import annotations

import asyncio
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from starlette.requests import Request

from tv4 import api
from tv4.kis_pipeline import KisServices
from tv4.media_identity import MediaRecord


class _Noop:
    pass


class _TV1Frames:
    def __init__(self, records: list[dict]) -> None:
        self.records = records

    def frames(self, video_id: str) -> list[dict]:
        return [record for record in self.records if record.get("video_id") == video_id]


class _Authority:
    certification_id = "e4-1"
    certification_report_sha256 = "c" * 64


class _ProofRefiner:
    exact_proof_authority = _Authority()

    def __init__(self, *, frame: dict | None = None) -> None:
        self.frame = frame

    def neighbors(self, request):
        offset = request["offsets"][0]
        frame = self.frame if self.frame is not None else {
            "video_id": "VID1", "frame_id": 1 + offset, "timestamp_ms": 1 + offset, "pts": 100 + offset,
            "time_base": "1/1000", "preprocess_run_id": "run", "mapping_guaranteed": True,
            "submission_selectable": True, "identity_source": "certified_run_consecutive_original_decode",
            "media_identity_verified": True, "producer_compatibility_verified": True,
            "certification_id": "e4-1", "certification_report_sha256": "c" * 64, "source_sha256": "a" * 64,
        }
        return {"provenance_mode": "live", "video_id": "VID1", "anchor_frame_id": 1,
                "steps": [{"offset": offset, "frame": frame}]}


class _FakeImage:
    def save(self, output, format: str) -> None:
        if format != "JPEG":
            raise ValueError("wrong image format")
        output.write(b"jpeg")


class _FakeFrame:
    def __init__(self, pts: int) -> None:
        self.pts = pts

    def to_image(self) -> _FakeImage:
        return _FakeImage()


class _FakeContainer:
    def __init__(self, frames: list[_FakeFrame], time_base=Fraction(1, 1000)) -> None:
        self.streams = type("Streams", (), {"video": [type("Stream", (), {"time_base": time_base})()]})()
        self.frames = frames

    def __enter__(self): return self
    def __exit__(self, *_args): return False
    def seek(self, *_args, **_kwargs): pass
    def decode(self, **_kwargs): return iter(self.frames)


class _FakeAv:
    def __init__(self, container: _FakeContainer) -> None:
        self.container = container

    def open(self, _path: str) -> _FakeContainer:
        return self.container


async def _stream_bytes(response) -> bytes:
    return b"".join([chunk async for chunk in response.body_iterator])


class T014MediaRoutesDirectTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name) / "raw"; self.root.mkdir()
        (self.root / "VID1.mp4").write_bytes(b"0123456789")
        self.assets = Path(self.tmp.name) / "assets"; self.assets.mkdir()
        self.key = self.assets / "key.jpg"; self.key.write_bytes(b"key")
        self.thumb = self.assets / "thumb.jpg"; self.thumb.write_bytes(b"thumb")
        self.record = {
            "video_id": "VID1", "frame_id": 7, "timestamp_ms": 71, "pts": 70,
            "preprocess_run_id": "run", "keyframe_path": str(self.key), "thumbnail_path": str(self.thumb),
        }
        registry = {"VID1": MediaRecord("VID1", "VID1.mp4", "a" * 64, "run", "1/1000", "media/VID1", "frames/VID1")}
        self.services = KisServices(
            tv1=_TV1Frames([self.record]), tv3=_Noop(), visual=_Noop(), preprocess_run_id="run",
            original_media_root=self.root, media_registry=registry, allowed_media_extensions=frozenset({".mp4"}),
            derivative_asset_root=self.assets, allowed_image_extensions=frozenset({".jpg"}),
        )
        self.services_patch = patch.object(api, "_get_services", return_value=self.services)
        self.fixture_patch = patch.object(api, "FIXTURE_MODE", False)
        self.services_patch.start(); self.fixture_patch.start()

    def tearDown(self) -> None:
        self.fixture_patch.stop(); self.services_patch.stop(); self.tmp.cleanup()

    def test_stream_uses_only_the_registry_owned_original(self) -> None:
        request = Request({"type": "http", "method": "GET", "headers": [(b"range", b"bytes=2-5")]})
        response = api.stream_original_video("VID1", request)
        self.assertEqual(response.status_code, 206)
        self.assertEqual(asyncio.run(_stream_bytes(response)), b"2345")

    def test_selected_images_use_tv1_metadata_with_no_media_service(self) -> None:
        keyframe = api.keyframe_image("VID1", 7)
        thumbnail = api.thumbnail_image("VID1", 7)
        selected_original = api.original_frame_image("VID1", 7)
        self.assertEqual(Path(keyframe.path).read_bytes(), b"key")
        self.assertEqual(Path(thumbnail.path).read_bytes(), b"thumb")
        self.assertEqual(Path(selected_original.path).read_bytes(), b"key")
        self.assertEqual(selected_original.headers["x-original-frame-id"], "7")
        self.assertEqual(selected_original.headers["x-inspection-derivative"], "selected-keyframe")
        self.assertFalse(hasattr(self.services, "media_api_base_url"))
        self.assertFalse(hasattr(api, "_upstream_media_get"))

    def test_arbitrary_timestamp_remains_noncanonical(self) -> None:
        self.assertEqual(api.resolve_original_timestamp("VID1", 71), {
            "video_id": "VID1", "requested_timestamp_ms": 71, "mode": "nearest",
            "canonical_frame_id": None, "canonical_timestamp_ms": None,
            "inspection_only": True, "submission_selectable": False,
            "state": "canonical_frame_unresolved_from_arbitrary_playback_timestamp",
        })

    def test_selected_assets_fail_closed_on_identity_or_path_drift(self) -> None:
        with self.assertRaises(HTTPException) as unknown:
            api.keyframe_image("VID1", 999)
        self.assertEqual(unknown.exception.status_code, 404)

        bad_run = dict(self.record); bad_run["preprocess_run_id"] = "other"
        self.services.tv1 = _TV1Frames([bad_run])
        with self.assertRaises(HTTPException) as wrong_run:
            api.keyframe_image("VID1", 7)
        self.assertEqual(wrong_run.exception.status_code, 502)

        outside = Path(self.tmp.name) / "outside.jpg"; outside.write_bytes(b"outside")
        escaped = dict(self.record); escaped["keyframe_path"] = str(outside)
        self.services.tv1 = _TV1Frames([escaped])
        with self.assertRaises(HTTPException) as escaped_path:
            api.keyframe_image("VID1", 7)
        self.assertEqual(escaped_path.exception.status_code, 404)

        text_asset = self.assets / "not-image.txt"; text_asset.write_text("no", encoding="utf-8")
        disallowed = dict(self.record); disallowed["thumbnail_path"] = str(text_asset)
        self.services.tv1 = _TV1Frames([disallowed])
        with self.assertRaises(HTTPException) as wrong_extension:
            api.thumbnail_image("VID1", 7)
        self.assertEqual(wrong_extension.exception.status_code, 404)

    def test_exact_neighbor_identity_remains_wp09_proven(self) -> None:
        self.services.refine = _ProofRefiner()
        result = api.exact_frame_neighbors(api.ExactNeighborRequest(video_id="VID1", frame_id=1, timestamp_ms=1, offsets=[1]))
        self.assertEqual(result["steps"][0]["frame"]["frame_id"], 2)

    def test_exact_image_revalidates_plus_one_proof_and_headers_without_wp04(self) -> None:
        self.services.refine = _ProofRefiner()
        request = api.ExactNeighborRequest(video_id="VID1", frame_id=1, timestamp_ms=1, offsets=[1])
        with patch.object(api, "_decode_proven_frame_jpeg", return_value=b"jpeg") as decoder:
            response = api.exact_frame_image(request)
        self.assertEqual(response.body, b"jpeg")
        self.assertEqual(response.headers["x-original-video-id"], "VID1")
        self.assertEqual(response.headers["x-original-frame-id"], "2")
        self.assertEqual(response.headers["x-pts"], "101")
        self.assertEqual(response.headers["x-time-base"], "1/1000")
        self.assertEqual(response.headers["x-timestamp-ms"], "2")
        decoder.assert_called_once()
        self.assertFalse(hasattr(self.services, "media_api_base_url"))
        self.assertFalse(hasattr(api, "_upstream_media_get"))

    def test_exact_image_accepts_valid_certified_anchor(self) -> None:
        self.services.refine = _ProofRefiner()
        request = api.ExactNeighborRequest(video_id="VID1", frame_id=1, timestamp_ms=1, offsets=[0])
        with patch.object(api, "_decode_proven_frame_jpeg", return_value=b"jpeg"):
            response = api.exact_frame_image(request)
        self.assertEqual(response.headers["x-original-frame-id"], "1")

    def test_exact_image_preserves_certified_anchor_and_cumulative_offset(self) -> None:
        self.services.refine = _ProofRefiner()
        request = api.ExactNeighborRequest(
            video_id="VID1", frame_id=2, timestamp_ms=2, offsets=[1],
            certified_anchor_frame_id=1, certified_anchor_timestamp_ms=1, cumulative_offset=1,
        )
        with patch.object(api, "_decode_proven_frame_jpeg", return_value=b"jpeg"):
            response = api.exact_frame_image(request)
        self.assertEqual(response.headers["x-original-frame-id"], "3")

    def test_exact_image_accepts_valid_negative_neighbor(self) -> None:
        self.services.refine = _ProofRefiner()
        request = api.ExactNeighborRequest(video_id="VID1", frame_id=1, timestamp_ms=1, offsets=[-1])
        with patch.object(api, "_decode_proven_frame_jpeg", return_value=b"jpeg"):
            response = api.exact_frame_image(request)
        self.assertEqual(response.headers["x-original-frame-id"], "0")

    def test_exact_image_rejects_forged_stale_and_boundary_proofs(self) -> None:
        request = api.ExactNeighborRequest(video_id="VID1", frame_id=1, timestamp_ms=1, offsets=[1])
        forged = _ProofRefiner(frame={**_ProofRefiner().neighbors({"offsets": [1]})["steps"][0]["frame"], "certification_id": "forged"})
        stale = _ProofRefiner(frame={**_ProofRefiner().neighbors({"offsets": [1]})["steps"][0]["frame"], "certification_report_sha256": "d" * 64})
        boundary = _ProofRefiner(frame=None)
        boundary.neighbors = lambda request: {"provenance_mode": "live", "video_id": "VID1", "anchor_frame_id": 1, "steps": [{"offset": request["offsets"][0], "frame": None}]}
        for refiner in (forged, stale, boundary):
            self.services.refine = refiner
            with self.assertRaises(HTTPException) as rejected:
                api.exact_frame_image(request)
            self.assertEqual(rejected.exception.status_code, 409)

    def test_exact_image_decoder_requires_exact_pts_time_base_and_timestamp(self) -> None:
        proof = {**_ProofRefiner().neighbors({"offsets": [1]})["steps"][0]["frame"], "timestamp_ms": 101}
        valid = api._decode_proven_frame_jpeg(self.root / "VID1.mp4", proof, av_module=_FakeAv(_FakeContainer([_FakeFrame(101)])))
        self.assertEqual(valid, b"jpeg")
        for bad_container, bad_proof in (
            (_FakeContainer([_FakeFrame(102)]), proof),
            (_FakeContainer([_FakeFrame(101)], time_base=Fraction(1, 900)), proof),
            (_FakeContainer([_FakeFrame(101)]), {**proof, "timestamp_ms": 999}),
            (_FakeContainer([_FakeFrame(101), _FakeFrame(101)]), proof),
        ):
            with self.assertRaises(ValueError):
                api._decode_proven_frame_jpeg(self.root / "VID1.mp4", bad_proof, av_module=_FakeAv(bad_container))


if __name__ == "__main__":
    unittest.main()
