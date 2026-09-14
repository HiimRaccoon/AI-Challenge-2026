import { describe, it, expect, vi, beforeEach } from 'vitest'
import { fetchVqaAnswer, Tv4ApiError } from '../../src/api/tv4Client'
import { VqaRequest, VqaResponse } from '../../src/types/contracts'

describe('T030 — VQA API Client & Data Contract Characterization', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('preserves distinct query_text and question in request payload', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        query_id: 'qa-001',
        results: [],
        provenance_mode: 'live',
      }),
    } as Response)
    global.fetch = fetchMock

    const req: VqaRequest = {
      query_text: 'người phụ nữ mặc áo xanh lá cây',
      question: 'Cô ấy đang cầm vật gì trên tay?',
      query_id: 'qa-001',
      top_k: 50,
      top_k_answers: 3,
    }

    await fetchVqaAnswer(req)

    expect(fetchMock).toHaveBeenCalledWith(
      '/vqa/answer',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query_text: 'người phụ nữ mặc áo xanh lá cây',
          question: 'Cô ấy đang cầm vật gì trên tay?',
          query_id: 'qa-001',
          top_k: 50,
          top_k_answers: 3,
        }),
      })
    )
  })

  it('bounds top_k to [1, 100] and top_k_answers to [1, 20]', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ query_id: 'qa-bound', results: [] }),
    } as Response)
    global.fetch = fetchMock

    // Upper bounds clamping
    await fetchVqaAnswer({
      query_text: 'test query',
      question: 'test question',
      top_k: 250,
      top_k_answers: 50,
    })

    expect(fetchMock).toHaveBeenCalledWith(
      '/vqa/answer',
      expect.objectContaining({
        body: JSON.stringify({
          query_text: 'test query',
          question: 'test question',
          query_id: null,
          top_k: 100,
          top_k_answers: 20,
        }),
      })
    )
  })

  it('validates client-side constraints before network call', async () => {
    // Missing query_text
    await expect(
      fetchVqaAnswer({ query_text: '', question: 'What is this?' })
    ).rejects.toThrow(/query_text and question are required/i)

    // Missing question
    await expect(
      fetchVqaAnswer({ query_text: 'A red car', question: '   ' })
    ).rejects.toThrow(/query_text and question are required/i)
  })

  it('parses rich, provenance-preserving EvidencePack and VqaResult', async () => {
    const mockResponse: VqaResponse = {
      query_id: 'qa-rich-001',
      provenance_mode: 'live',
      results: [
        {
          rank: 1,
          video_id: 'L05_V005',
          frame_id: 888,
          timestamp_ms: 29600,
          confidence: 0.95,
          answer: 'màu xanh',
          verified: true,
          manual_review: false,
          proposal: 'màu xanh',
          approved: false, // Invariant: Backend NEVER creates an operator-approved answer!
          verifier_status: 'verified',
          retry_count: 0,
          manual_required: false,
          status: 'verified',
          degraded_reasons: [],
          evidence: {
            query_id: 'qa-rich-001',
            query_text: 'a person holding a blue cup',
            question: 'What color is the cup?',
            video_id: 'L05_V005',
            frame_id: 888,
            timestamp_ms: 29600,
            keyframe_path: 'keyframes/L05_V005/0888.jpg',
            selected_frames: [
              {
                video_id: 'L05_V005',
                frame_id: 880,
                timestamp_ms: 29333,
                keyframe_path: 'keyframes/L05_V005/0880.jpg',
                preprocess_run_id: 'run_v1_batch1',
                provenance: { source: 'visual' },
                submission_selectable: true,
              },
              {
                video_id: 'L05_V005',
                frame_id: 888,
                timestamp_ms: 29600,
                keyframe_path: 'keyframes/L05_V005/0888.jpg',
                preprocess_run_id: 'run_v1_batch1',
                provenance: { source: 'visual' },
                submission_selectable: true,
              },
            ],
            ocr_evidence: [
              {
                detection_id: 'ocr-det-1',
                video_id: 'L05_V005',
                frame_id: 888,
                timestamp_ms: 29600,
                raw_text: 'DANH HIỆU',
                normalized_text: 'danh hiệu',
                bbox_xyxy_norm: [0.12, 0.34, 0.56, 0.78],
                polygon_norm: [[0.12, 0.34], [0.56, 0.34], [0.56, 0.78], [0.12, 0.78]],
                confidence: 0.94,
                crop_evidence_path: 'crops/ocr/ocr-det-1.jpg',
                crop_sha256: 'sha-crop-01',
                source_keyframe_sha256: 'sha-kf-01',
                preprocess_run_id: 'run_v1_batch1',
                model_name: 'paddleocr',
                model_version: 'v4',
                provenance: { branch: 'ocr' },
                source_record: {},
              },
            ],
            asr_evidence: [
              {
                segment_id: 'asr-seg-1',
                video_id: 'L05_V005',
                start_ms: 28500,
                end_ms: 30500,
                text: 'chiếc cúp màu xanh',
                normalized_text: 'chiếc cúp màu xanh',
                words: [
                  { word: 'chiếc', start_ms: 28500, end_ms: 28900, probability: 0.98 },
                  { word: 'cúp', start_ms: 28900, end_ms: 29300, probability: 0.97 },
                  { word: 'màu', start_ms: 29300, end_ms: 29700, probability: 0.99 },
                  { word: 'xanh', start_ms: 29700, end_ms: 30500, probability: 0.99 },
                ],
                context: [{ segment_id: 'asr-seg-0', text: 'người dẫn chương trình nói' }],
                confidence: 0.96,
                language: 'vi',
                preprocess_run_id: 'run_v1_batch1',
                model_name: 'whisper-large-v3',
                model_version: 'v3',
                provenance: { branch: 'asr' },
                source_record: {},
              },
            ],
            object_evidence: [
              {
                detection_id: 'obj-det-1',
                video_id: 'L05_V005',
                frame_id: 888,
                timestamp_ms: 29600,
                label: 'cup',
                canonical_label: 'trophy',
                bbox_xyxy_norm: [0.45, 0.5, 0.65, 0.85],
                confidence: 0.89,
                source_keyframe_path: 'keyframes/L05_V005/0888.jpg',
                source_keyframe_sha256: 'sha-kf-01',
                preprocess_run_id: 'run_v1_batch1',
                model_name: 'yolov8x-world',
                model_version: 'v8',
                provenance: { branch: 'object' },
                source_record: {},
              },
            ],
            metadata_evidence: [
              {
                metadata_id: 'meta-1',
                video_id: 'L05_V005',
                source: 'youtube_title',
                values: { title: 'Lễ Trao Giải 2026' },
                window_start_ms: 0,
                window_end_ms: 60000,
                confidence: 1.0,
                preprocess_run_id: 'run_v1_batch1',
                model_name: null,
                model_version: null,
                source_record_sha256: 'sha-meta-01',
                provenance: { branch: 'metadata' },
                source_record: {},
              },
            ],
            availability: {
              frames: 'available',
              ocr: 'available',
              asr: 'available',
              object: 'available',
              metadata: 'available',
            },
            ocr_texts: ['DANH HIỆU'],
            asr_texts: ['chiếc cúp màu xanh'],
            object_labels: ['cup'],
            neighbor_frame_ids: [880, 888],
            provenance: { source: 'multimodal_pipeline' },
          },
        },
      ],
    }

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    } as Response)

    const res = await fetchVqaAnswer({
      query_text: 'a person holding a blue cup',
      question: 'What color is the cup?',
    })

    expect(res.results.length).toBe(1)
    const r = res.results[0]

    // Canonical identity
    expect(r.video_id).toBe('L05_V005')
    expect(r.frame_id).toBe(888)
    expect(r.timestamp_ms).toBe(29600)

    // Advisory proposal != approval
    expect(r.proposal).toBe('màu xanh')
    expect(r.approved).toBe(false)
    expect(r.verified).toBe(true)

    // Evidence modalities
    expect(r.evidence.ocr_evidence[0].raw_text).toBe('DANH HIỆU')
    expect(r.evidence.ocr_evidence[0].bbox_xyxy_norm).toEqual([0.12, 0.34, 0.56, 0.78])
    expect(r.evidence.asr_evidence[0].words?.length).toBe(4)
    expect(r.evidence.object_evidence[0].label).toBe('cup')
    expect(r.evidence.selected_frames.length).toBe(2)
  })

  it('handles empty evidence response truthfully without fabricating candidates', async () => {
    const emptyResponse: VqaResponse = {
      query_id: 'qa-empty-001',
      provenance_mode: 'live',
      results: [
        {
          rank: 1,
          video_id: 'L05_V005',
          frame_id: 888,
          timestamp_ms: 29600,
          confidence: null,
          answer: '',
          verified: false,
          manual_review: true,
          proposal: '',
          approved: false,
          verifier_status: 'insufficient_evidence',
          retry_count: 0,
          manual_required: true,
          status: 'abstained',
          degraded_reasons: ['empty_evidence'],
          evidence: {
            query_id: 'qa-empty-001',
            query_text: 'obscure scene',
            question: 'What is written?',
            video_id: 'L05_V005',
            frame_id: 888,
            timestamp_ms: 29600,
            keyframe_path: null,
            selected_frames: [],
            ocr_evidence: [],
            asr_evidence: [],
            object_evidence: [],
            metadata_evidence: [],
            availability: {
              frames: 'empty',
              ocr: 'empty',
              asr: 'empty',
              object: 'empty',
              metadata: 'empty',
            },
            ocr_texts: [],
            asr_texts: [],
            object_labels: [],
            neighbor_frame_ids: [],
            provenance: { mode: 'live' },
          },
        },
      ],
    }

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => emptyResponse,
    } as Response)

    const res = await fetchVqaAnswer({
      query_text: 'obscure scene',
      question: 'What is written?',
    })

    const r = res.results[0]
    expect(r.proposal).toBe('')
    expect(r.verified).toBe(false)
    expect(r.manual_required).toBe(true)
    expect(r.status).toBe('abstained')
    expect(r.degraded_reasons).toContain('empty_evidence')
    expect(r.evidence.ocr_evidence).toEqual([])
  })

  it('handles retry exhausted response (retry_count = 1)', async () => {
    const retryExhaustedResponse: VqaResponse = {
      query_id: 'qa-retry-001',
      provenance_mode: 'live',
      results: [
        {
          rank: 1,
          video_id: 'L05_V005',
          frame_id: 888,
          timestamp_ms: 29600,
          confidence: null,
          answer: 'màu đỏ',
          verified: false,
          manual_review: true,
          proposal: 'màu đỏ',
          approved: false,
          verifier_status: 'rejected',
          retry_count: 1, // Maximum 1 controlled retry
          manual_required: true,
          status: 'manual_required',
          degraded_reasons: ['verifier_rejected_retry_exhausted'],
          evidence: {
            query_id: 'qa-retry-001',
            video_id: 'L05_V005',
            frame_id: 888,
            timestamp_ms: 29600,
            keyframe_path: 'keyframes/L05_V005/0888.jpg',
            selected_frames: [],
            ocr_evidence: [],
            asr_evidence: [],
            object_evidence: [],
            metadata_evidence: [],
            availability: { frames: 'available', ocr: 'available' },
            ocr_texts: [],
            asr_texts: [],
            object_labels: [],
            neighbor_frame_ids: [],
            provenance: {},
          },
        },
      ],
    }

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => retryExhaustedResponse,
    } as Response)

    const res = await fetchVqaAnswer({
      query_text: 'car on highway',
      question: 'What is the color?',
    })

    const r = res.results[0]
    expect(r.retry_count).toBe(1)
    expect(r.verifier_status).toBe('rejected')
    expect(r.manual_required).toBe(true)
    expect(r.status).toBe('manual_required')
  })

  it('maps HTTP errors into structured Tv4ApiError', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 502,
      statusText: 'Bad Gateway',
      json: async () => ({ detail: 'KIS pipeline failed: upstream timeout' }),
    } as Response)

    await expect(
      fetchVqaAnswer({ query_text: 'failing query', question: 'failing question' })
    ).rejects.toThrow(Tv4ApiError)
  })
})
