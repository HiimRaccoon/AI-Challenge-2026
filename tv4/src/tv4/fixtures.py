"""Canned, contract-accurate responses for TV4_FIXTURE_MODE.

Every field/shape here matches exactly what the live endpoints in `api.py`
return -- TV5 can build against these on day one and swap to the live API
later with zero UI code changes, only an env var flip.
"""
from __future__ import annotations

from copy import deepcopy

KIS_RESPONSE = {
    "provenance_mode": "fixture",
    "query_id": "kis-fixture-001",
    "candidates": [
        {
            "query_id": "kis-fixture-001", "video_id": "L21_V001", "frame_id": 10690,
            "timestamp_ms": 356333, "source": "fusion", "rank": 1, "schema_version": "1.0.0",
            "event_index": None, "representative_frame_id": None,
            "window_start_ms": None, "window_end_ms": None,
            "raw_score": None, "score": 0.0405,
            "model_scores": {"bge_vl": 0.109, "metaclip2": 0.199, "perception": 0.224},
            "model_ranks": {"bge_vl": 45, "metaclip2": 17, "perception": 44},
            "matched_filters": [], "evidence_refs": ["keyframe:keyframes/L21_V001/83.jpg"],
            "provenance_sources": ["visual"], "provenance": {}, "confidence": None,
            "preprocess_run_id": "run_v1_batch1", "created_at_utc": "2026-08-14T05:36:45Z",
        },
        {
            "query_id": "kis-fixture-001", "video_id": "L21_V002", "frame_id": 23940,
            "timestamp_ms": 798000, "source": "fusion", "rank": 2, "schema_version": "1.0.0",
            "event_index": None, "representative_frame_id": None,
            "window_start_ms": None, "window_end_ms": None,
            "raw_score": None, "score": 0.0371,
            "model_scores": {"bge_vl": 0.116, "metaclip2": 0.187, "perception": 0.213},
            "model_ranks": {"bge_vl": 12, "metaclip2": 37, "perception": 55},
            "matched_filters": [], "evidence_refs": ["keyframe:keyframes/L21_V002/214.jpg"],
            "provenance_sources": ["visual"], "provenance": {}, "confidence": None,
            "preprocess_run_id": "run_v1_batch1", "created_at_utc": "2026-08-14T05:36:45Z",
        },
    ],
}

VQA_RESPONSE = {
    "provenance_mode": "fixture",
    "query_id": "qa-fixture-001",
    "results": [
        {
            "rank": 1, "video_id": "L05_V005", "frame_id": 888, "timestamp_ms": 29600,
            "confidence": None, "answer": "màu xanh", "verified": True, "manual_review": False,
            "evidence": {
                "query_id": "qa-fixture-001", "video_id": "L05_V005", "frame_id": 888,
                "timestamp_ms": 29600, "keyframe_path": "keyframes/L05_V005/0888.jpg",
                "ocr_texts": [], "asr_texts": ["...cô ấy đang cầm ly màu xanh..."],
                "object_labels": ["person", "cup"], "neighbor_frame_ids": [880, 896],
                "provenance": {"candidate_source": ["visual", "asr"]},
            },
        },
    ],
}

# Fixture VQA is advisory and deliberately non-selectable. Keep the legacy
# fields above for old consumers while exercising the rich T016 contract.
VQA_RESPONSE["results"][0].update({
    "verified": False,
    "manual_review": True,
    "proposal": VQA_RESPONSE["results"][0]["answer"],
    "approved": False,
    "verifier_status": "fixture_unverified",
    "retry_count": 0,
    "manual_required": True,
    "status": "manual_required",
    "degraded_reasons": ["fixture_non_authoritative"],
})
VQA_RESPONSE["results"][0]["evidence"].update({
    "query_text": "a person holding a blue cup",
    "question": "What color is the cup?",
    "selected_frames": [
        {"video_id": "L05_V005", "frame_id": 880, "timestamp_ms": 29333, "keyframe_path": "keyframes/L05_V005/0880.jpg", "preprocess_run_id": "fixture-run", "provenance": {"source": "fixture"}, "submission_selectable": False},
        {"video_id": "L05_V005", "frame_id": 888, "timestamp_ms": 29600, "keyframe_path": "keyframes/L05_V005/0888.jpg", "preprocess_run_id": "fixture-run", "provenance": {"source": "fixture"}, "submission_selectable": False},
        {"video_id": "L05_V005", "frame_id": 896, "timestamp_ms": 29867, "keyframe_path": "keyframes/L05_V005/0896.jpg", "preprocess_run_id": "fixture-run", "provenance": {"source": "fixture"}, "submission_selectable": False},
    ],
    "ocr_evidence": [{"detection_id": "fixture-ocr-888", "video_id": "L05_V005", "frame_id": 888, "timestamp_ms": 29600, "raw_text": "AWARD", "normalized_text": "award", "bbox_xyxy_norm": [0.1, 0.2, 0.6, 0.3], "polygon_norm": None, "confidence": 0.91, "crop_evidence_path": None, "crop_sha256": None, "source_keyframe_sha256": "fixture", "preprocess_run_id": "fixture-run", "model_name": "fixture-ocr", "model_version": "1", "provenance": {"branch": "ocr"}}],
    "asr_evidence": [{"segment_id": "fixture-asr-1", "video_id": "L05_V005", "start_ms": 29000, "end_ms": 30200, "text": "person holding a blue cup", "normalized_text": "person holding a blue cup", "words": [{"word": "blue", "start_ms": 29900, "end_ms": 30100, "probability": 0.9}], "context": [{"segment_id": "fixture-asr-before", "text": "host speaks"}], "confidence": 0.88, "language": "en", "preprocess_run_id": "fixture-run", "model_name": "fixture-asr", "model_version": "1", "provenance": {"branch": "asr"}}],
    "object_evidence": [{"detection_id": "fixture-object-888", "video_id": "L05_V005", "frame_id": 888, "timestamp_ms": 29600, "label": "cup", "canonical_label": "cup", "bbox_xyxy_norm": [0.55, 0.4, 0.7, 0.8], "confidence": 0.87, "source_keyframe_path": "keyframes/L05_V005/0888.jpg", "source_keyframe_sha256": "fixture", "preprocess_run_id": "fixture-run", "model_name": "fixture-object", "model_version": "1", "provenance": {"branch": "object"}}],
    "metadata_evidence": [{"metadata_id": "fixture-meta-1", "video_id": "L05_V005", "source": "fixture", "values": {"title": "award ceremony"}, "window_start_ms": None, "window_end_ms": None, "confidence": None, "preprocess_run_id": "fixture-run", "model_name": None, "model_version": None, "source_record_sha256": "fixture", "provenance": {"branch": "metadata"}}],
    "availability": {"frames": "available", "ocr": "available", "asr": "available", "object": "available", "metadata": "available"},
})
VQA_RESPONSE["results"][0]["evidence"]["provenance"].update({"fixture": True, "submission_selectable": False})
VQA_RESPONSE["results"][0].update({
    "answerability_score": 0.82,
    "answerability_components": {
        "retrieval_relevance": 0.8,
        "modality_match": 1.0,
        "transcript_answer_cue": 0.55,
        "visual_object_answer_cue": 1.0,
        "temporal_completeness": 0.67,
        "total": 0.82,
    },
    "answer_confidence": 0.82,
    "verifier_score": 0.7,
    "independently_supported": True,
    "contradiction": False,
    "verification": {
        "supported": True,
        "visual_support": 0.15,
        "audio_text_support": 1.0,
        "ocr_support": 0.0,
        "producer_support": 0.0,
        "contradiction": False,
        "final_support_score": 0.7,
        "manual_required": False,
        "visual_verifier_available": False,
        "visual_frame_ids": [],
        "visual_verifier_reason": "fixture_non_authoritative",
        "reasons": ["fixture_non_authoritative"],
    },
    "support_start_ms": 29000,
    "support_end_ms": 30200,
    "frame_candidates": [
        {"frame_id": 888, "timestamp_ms": 29600, "score": 0.9, "support_modalities": ["asr", "object"]},
        {"frame_id": 880, "timestamp_ms": 29333, "score": 0.7, "support_modalities": ["asr"]},
    ],
})

VQA_EMPTY_EVIDENCE_RESPONSE = {
    "provenance_mode": "fixture", "query_id": "qa-fixture-empty", "results": [{
        "rank": 1, "video_id": "L05_V005", "frame_id": 888, "timestamp_ms": 29600,
        "confidence": None, "answer": "", "verified": False, "manual_review": True,
        "proposal": "", "approved": False, "verifier_status": "insufficient_evidence",
        "retry_count": 0, "manual_required": True, "status": "abstained",
        "degraded_reasons": ["empty_evidence"],
        "evidence": {"query_id": "qa-fixture-empty", "query_text": "fixture empty", "question": "fixture question", "video_id": "L05_V005", "frame_id": 888, "timestamp_ms": 29600, "keyframe_path": None, "selected_frames": [], "ocr_evidence": [], "asr_evidence": [], "object_evidence": [], "metadata_evidence": [], "availability": {"frames": "empty", "ocr": "empty", "asr": "empty", "object": "empty", "metadata": "empty"}, "ocr_texts": [], "asr_texts": [], "object_labels": [], "neighbor_frame_ids": [], "provenance": {"fixture": True, "submission_selectable": False}},
    }],
}

VQA_RETRY_EXHAUSTED_RESPONSE = deepcopy(VQA_RESPONSE)
VQA_RETRY_EXHAUSTED_RESPONSE["query_id"] = "qa-fixture-retry-exhausted"
VQA_RETRY_EXHAUSTED_RESPONSE["results"][0].update({
    "retry_count": 1,
    "verifier_status": "rejected",
    "manual_required": True,
    "manual_review": True,
    "status": "manual_required",
    "degraded_reasons": ["fixture_retry_exhausted"],
})
VQA_RETRY_EXHAUSTED_RESPONSE["results"][0]["evidence"]["query_id"] = "qa-fixture-retry-exhausted"


def vqa_fixture_response(query_id: str | None) -> dict:
    """Deterministic fixture cases; none represents an approved submission."""
    if query_id == "qa-fixture-empty":
        return VQA_EMPTY_EVIDENCE_RESPONSE
    if query_id == "qa-fixture-retry-exhausted":
        return VQA_RETRY_EXHAUSTED_RESPONSE
    return VQA_RESPONSE


TRAKE_RESPONSE = {
    "provenance_mode": "fixture",
    "query_id": "trake-fixture-001",
    "result": {
        "video_id": "L10_V010",
        "frame_ids": [101, 156, 203, 251],
        "timestamps_ms": [4040, 6240, 8120, 10040],
        "event_scores": [0.9, 0.7, 0.85, 0.6],
        "aggregate_score": 3.05,
        "preprocess_run_id": "run_v1_batch1",
        "candidates": [
            {
                "query_id": "trake-fixture-001-ev0",
                "video_id": "L10_V010",
                "frame_id": 101,
                "timestamp_ms": 4040,
                "source": "fusion",
                "rank": 1,
                "score": 0.9,
                "event_index": 0,
                "certified_anchor_frame_id": 101,
                "certified_anchor_timestamp_ms": 4040,
                "anchor_offset": 0,
                "preprocess_run_id": "run_v1_batch1",
            },
            {
                "query_id": "trake-fixture-001-ev1",
                "video_id": "L10_V010",
                "frame_id": 156,
                "timestamp_ms": 6240,
                "source": "fusion",
                "rank": 1,
                "score": 0.7,
                "event_index": 1,
                "certified_anchor_frame_id": 156,
                "certified_anchor_timestamp_ms": 6240,
                "anchor_offset": 0,
                "preprocess_run_id": "run_v1_batch1",
            },
            {
                "query_id": "trake-fixture-001-ev2",
                "video_id": "L10_V010",
                "frame_id": 203,
                "timestamp_ms": 8120,
                "source": "fusion",
                "rank": 1,
                "score": 0.85,
                "event_index": 2,
                "certified_anchor_frame_id": 203,
                "certified_anchor_timestamp_ms": 8120,
                "anchor_offset": 0,
                "preprocess_run_id": "run_v1_batch1",
            },
            {
                "query_id": "trake-fixture-001-ev3",
                "video_id": "L10_V010",
                "frame_id": 251,
                "timestamp_ms": 10040,
                "source": "fusion",
                "rank": 1,
                "score": 0.6,
                "event_index": 3,
                "certified_anchor_frame_id": 251,
                "certified_anchor_timestamp_ms": 10040,
                "anchor_offset": 0,
                "preprocess_run_id": "run_v1_batch1",
            },
        ],
    },
}

# Backward-compatible Top-K envelope. Fixture alternatives remain explicitly
# non-authoritative candidates and are useful only for UI contract exercises.
TRAKE_RESPONSE["result"]["rank"] = 1
TRAKE_RESPONSE["result"].update({
    "verifier_score": 0.88,
    "entity_continuity": 0.9,
    "state_consistency": 0.87,
    "order_consistent": True,
    "story_consistency": 0.89,
    "verification": {
        "event_scores": [0.9, 0.7, 0.85, 0.6],
        "entity_continuity": 0.9,
        "state_consistency": 0.87,
        "order_consistent": True,
        "story_consistency": 0.89,
        "modality_support": 0.8,
        "micro_clip_support": 0.6,
        "clip_manual_required": False,
        "whole_video_prior": 1.0,
        "implausible_gap_penalty": 0.0,
        "contradiction_penalty": 0.0,
        "final_sequence_score": 0.88,
        "visual_verifier_available": False,
        "visual_frame_ids": [[], [], [], []],
        "visual_verifier_reasons": ["fixture_non_authoritative", "fixture_non_authoritative", "fixture_non_authoritative", "fixture_non_authoritative"],
    },
})
TRAKE_RESPONSE["result"]["events"] = [
    {
        "event_index": index,
        "frame_id": frame_id,
        "timestamp_ms": TRAKE_RESPONSE["result"]["timestamps_ms"][index],
        "start_ms": None,
        "end_ms": None,
        "score": TRAKE_RESPONSE["result"]["event_scores"][index],
        "alternatives": [],
    }
    for index, frame_id in enumerate(TRAKE_RESPONSE["result"]["frame_ids"])
]
TRAKE_RESPONSE["results"] = [deepcopy(TRAKE_RESPONSE["result"])]

EXACT_NEIGHBOR_RESPONSE = {
    "provenance_mode": "fixture",
    "video_id": "L21_V001",
    "anchor_frame_id": 10690,
    "degraded_reason": None,
    "steps": [
        {
            "offset": -1,
            "degraded_reason": None,
            "frame": {
                "video_id": "L21_V001", "frame_id": 10689, "timestamp_ms": 356300,
                "pts": 10689000, "time_base": "1/30000", "preprocess_run_id": "run_v1_batch1",
                "media_record_ref": "L21_V001", "mapping_ref": "tv1-frames/L21_V001",
                "mapping_guaranteed": False, "submission_selectable": False,
                "identity_source": "fixture", "degraded_reason": "fixture_mode",
            },
        },
        {
            "offset": 0,
            "degraded_reason": None,
            "frame": {
                "video_id": "L21_V001", "frame_id": 10690, "timestamp_ms": 356333,
                "pts": 10690000, "time_base": "1/30000", "preprocess_run_id": "run_v1_batch1",
                "media_record_ref": "L21_V001", "mapping_ref": "tv1-frames/L21_V001",
                "mapping_guaranteed": False, "submission_selectable": False,
                "identity_source": "fixture", "degraded_reason": "fixture_mode",
            },
        },
    ],
}

FEEDBACK_RESPONSE = {
    "provenance_mode": "fixture",
    "status": "ok",
    "session_id": "feedback-fixture-001",
    "revision": 0,
    "wp03_run_id": "run_v1_batch1",
    "expires_at_utc": "2026-08-20T00:00:00Z",
    "candidates": [
        {
            "video_id": "L21_V001",
            "frame_id": 10690,
            "rank": 1,
            "timestamp_ms": 356333,
            "keyframe_path": "keyframes/L21_V001/83.jpg",
            "submission_selectable": False,
            "provenance_mode": "fixture",
            "source": "feedback",
            "preprocess_run_id": "run_v1_batch1",
        },
        {
            "video_id": "L21_V002",
            "frame_id": 23940,
            "rank": 2,
            "timestamp_ms": 798000,
            "keyframe_path": "keyframes/L21_V002/214.jpg",
            "submission_selectable": False,
            "provenance_mode": "fixture",
            "source": "feedback",
            "preprocess_run_id": "run_v1_batch1",
        },
    ],
}
