import {
  ExactImageHeaders,
  ExactImageResult,
  ExactNeighborRequest,
  ExactNeighborResponse,
  FeedbackRefineRequest,
  FeedbackResetRequest,
  FeedbackResponse,
  FeedbackStartRequest,
  FeedbackUndoRequest,
  KisRequest,
  KisResponse,
  SearchCandidate,
  TV4HealthResponse,
  TrakeRequest,
  TrakeRerunRequest,
  TrakeRerunResponse,
  TrakeResponse,
  VqaRequest,
  VqaResponse,
} from '../types/contracts'
import {
  isFixtureModeActive,
  generateFixtureCandidates,
  generateFixtureNeighbors,
  getFixturePreviewDataUri,
} from '../fixtures/fixtureData'

const DEFAULT_BASE_URL = '' // Uses Vite proxy in development or relative path

export class Tv4ApiError extends Error {
  constructor(
    public status: number,
    public statusText: string,
    public detail?: string
  ) {
    super(`TV4 API Error ${status}: ${detail || statusText}`)
    this.name = 'Tv4ApiError'
  }
}

/** Helper to resolve fixture video IDs (FIXTURE_V001 -> L21_V001) to existing preprocessed media */
function resolveMediaVideoId(videoId: string): string {
  if (videoId.startsWith('FIXTURE_V')) {
    return videoId.replace('FIXTURE_V', 'L21_V')
  }
  return videoId
}

export async function parseErrorDetail(res: Response): Promise<string | undefined> {
  try {
    let rawText: string | undefined
    if (typeof res.text === 'function') {
      try {
        rawText = await res.text()
      } catch {}
    }
    if (!rawText || !rawText.trim()) {
      if (typeof res.json === 'function') {
        try {
          const json = await res.json()
          return json.detail || JSON.stringify(json)
        } catch {}
      }
      return undefined
    }
    try {
      const json = JSON.parse(rawText)
      return json.detail || JSON.stringify(json)
    } catch {
      return rawText
    }
  } catch {
    return undefined
  }
}

export async function fetchHealth(baseUrl = DEFAULT_BASE_URL): Promise<TV4HealthResponse> {
  // Deterministic Fixture Mode Check
  if (isFixtureModeActive()) {
    return {
      status: 'ok',
      mode: 'fixture',
      preprocess_run_id: 'fixture_preview_run_v1',
    }
  }

  const res = await fetch(`${baseUrl}/health`)
  if (!res.ok) {
    const detail = await parseErrorDetail(res)
    throw new Tv4ApiError(res.status, res.statusText, detail)
  }
  return res.json()
}

export async function searchKis(
  req: KisRequest,
  baseUrl = DEFAULT_BASE_URL
): Promise<KisResponse> {
  const boundedTopK = Math.max(1, Math.min(100, req.top_k || 100))

  // Deterministic Fixture Mode Check
  if (isFixtureModeActive()) {
    const allFixture = generateFixtureCandidates(req.query_text, boundedTopK)
    return {
      query_id: `fixture-query-${Date.now().toString(36)}`,
      candidates: allFixture,
      provenance_mode: 'fixture',
    }
  }

  const res = await fetch(`${baseUrl}/kis/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query_text: req.query_text,
      query_id: req.query_id ?? null,
      top_k: boundedTopK,
      weights: req.weights ?? null,
    }),
  })
  if (!res.ok) {
    const detail = await parseErrorDetail(res)
    throw new Tv4ApiError(res.status, res.statusText, detail)
  }
  return res.json()
}

export async function fetchExactNeighbors(
  req: ExactNeighborRequest,
  baseUrl = DEFAULT_BASE_URL
): Promise<ExactNeighborResponse> {
  // Deterministic Fixture Mode Check
  if (isFixtureModeActive()) {
    return generateFixtureNeighbors(
      {
        query_id: 'fixture',
        video_id: req.video_id,
        frame_id: req.frame_id,
        timestamp_ms: req.timestamp_ms,
        source: 'fixture',
        rank: 1,
      },
      req.cumulative_offset
    )
  }

  const res = await fetch(`${baseUrl}/exact-frame/neighbors`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      video_id: req.video_id,
      frame_id: req.frame_id,
      timestamp_ms: req.timestamp_ms,
      offsets: req.offsets,
      certified_anchor_frame_id: req.certified_anchor_frame_id ?? null,
      certified_anchor_timestamp_ms: req.certified_anchor_timestamp_ms ?? null,
      cumulative_offset: req.cumulative_offset,
    }),
  })
  if (!res.ok) {
    const detail = await parseErrorDetail(res)
    throw new Tv4ApiError(res.status, res.statusText, detail)
  }
  return res.json()
}

export async function fetchExactImageBlob(
  req: ExactNeighborRequest,
  baseUrl = DEFAULT_BASE_URL
): Promise<ExactImageResult> {
  if (req.offsets.length !== 1) {
    throw new Error('fetchExactImageBlob requires exactly one offset')
  }

  // Deterministic Fixture Mode Check
  if (isFixtureModeActive()) {
    const inspectedFrameId = req.frame_id + req.cumulative_offset
    const headers: ExactImageHeaders = {
      video_id: req.video_id,
      frame_id: inspectedFrameId,
      pts: inspectedFrameId * 512,
      time_base: '1/12800',
      timestamp_ms: req.timestamp_ms,
      preprocess_run_id: 'fixture_preview_run_v1',
      certification_id: 'fixture-preview-non-canonical',
      submission_selectable: false, // Invariant: Fixtures are NEVER submission selectable!
    }

    const previewUrl = getFixturePreviewDataUri(req.video_id, inspectedFrameId)
    return { blobUrl: previewUrl, headers }
  }

  const res = await fetch(`${baseUrl}/exact-frame/image`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      video_id: req.video_id,
      frame_id: req.frame_id,
      timestamp_ms: req.timestamp_ms,
      offsets: req.offsets,
      certified_anchor_frame_id: req.certified_anchor_frame_id ?? null,
      certified_anchor_timestamp_ms: req.certified_anchor_timestamp_ms ?? null,
      cumulative_offset: req.cumulative_offset,
    }),
  })

  if (!res.ok) {
    const detail = await parseErrorDetail(res)
    throw new Tv4ApiError(res.status, res.statusText, detail)
  }

  const blob = await res.blob()
  const blobUrl = URL.createObjectURL(blob)

  const headers: ExactImageHeaders = {
    video_id: res.headers.get('x-original-video-id') || undefined,
    frame_id: res.headers.get('x-original-frame-id')
      ? parseInt(res.headers.get('x-original-frame-id')!, 10)
      : undefined,
    pts: res.headers.get('x-pts') ? parseInt(res.headers.get('x-pts')!, 10) : undefined,
    time_base: res.headers.get('x-time-base') || undefined,
    timestamp_ms: res.headers.get('x-timestamp-ms')
      ? parseInt(res.headers.get('x-timestamp-ms')!, 10)
      : undefined,
    preprocess_run_id: res.headers.get('x-preprocess-run-id') || undefined,
    certification_id: res.headers.get('x-exact-certification-id') || undefined,
    certification_report_sha256:
      res.headers.get('x-exact-certification-report-sha256') || undefined,
    source_sha256: res.headers.get('x-exact-source-sha256') || undefined,
    submission_selectable: res.headers.get('x-submission-selectable') === 'true',
  }

  return { blobUrl, headers }
}

export function getVideoStreamUrl(videoId: string, baseUrl = DEFAULT_BASE_URL): string {
  const realVid = resolveMediaVideoId(videoId)
  return `${baseUrl}/videos/${encodeURIComponent(realVid)}/stream`
}

export function getThumbnailUrl(
  videoId: string,
  frameId: number,
  baseUrl = DEFAULT_BASE_URL
): string {
  if (isFixtureModeActive()) {
    return getFixturePreviewDataUri(videoId, frameId)
  }
  const realVid = resolveMediaVideoId(videoId)
  return `${baseUrl}/videos/${encodeURIComponent(realVid)}/thumbnails/${frameId}.jpg`
}

export function getKeyframeUrl(
  videoId: string,
  frameId: number,
  baseUrl = DEFAULT_BASE_URL
): string {
  if (isFixtureModeActive()) {
    return getFixturePreviewDataUri(videoId, frameId)
  }
  const realVid = resolveMediaVideoId(videoId)
  return `${baseUrl}/videos/${encodeURIComponent(realVid)}/keyframes/${frameId}.jpg`
}

export function getSelectedFrameUrl(
  videoId: string,
  frameId: number,
  baseUrl = DEFAULT_BASE_URL
): string {
  if (isFixtureModeActive()) {
    return getFixturePreviewDataUri(videoId, frameId)
  }
  const realVid = resolveMediaVideoId(videoId)
  return `${baseUrl}/videos/${encodeURIComponent(realVid)}/frames/${frameId}.jpg`
}

// ---------------------------------------------------------------------------
// Feedback API Surface (WP13 T020 / T028)
// ---------------------------------------------------------------------------

let fixtureFeedbackActiveEvents = 0

export async function startFeedback(
  req: FeedbackStartRequest,
  baseUrl = DEFAULT_BASE_URL
): Promise<FeedbackResponse> {
  if (!req.session_id || !req.original_query) {
    throw new Error('session_id and original_query are required')
  }

  if (isFixtureModeActive()) {
    fixtureFeedbackActiveEvents = 0
    const fixtureCands = generateFixtureCandidates(req.original_query, 4)
    return {
      session_id: req.session_id,
      revision: 0,
      candidates: fixtureCands,
      active_feedback_count: 0,
      max_active_feedback_events: 5,
      status: 'ok',
      provenance_mode: 'fixture',
      expires_at_utc: new Date(Date.now() + 86400000).toISOString(),
    }
  }

  const res = await fetch(`${baseUrl}/feedback/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: req.session_id,
      original_query: req.original_query,
    }),
  })

  if (!res.ok) {
    const detail = await parseErrorDetail(res)
    throw new Tv4ApiError(res.status, res.statusText, detail)
  }

  return res.json()
}

export async function refineFeedback(
  req: FeedbackRefineRequest,
  baseUrl = DEFAULT_BASE_URL
): Promise<FeedbackResponse> {
  if (!req.session_id || !req.video_id || req.frame_id < 0 || !req.feedback_text?.trim() || req.expected_revision < 0) {
    throw new Error('Valid session_id, video_id, non-negative frame_id, feedback_text and expected_revision are required')
  }

  if (isFixtureModeActive()) {
    if (fixtureFeedbackActiveEvents >= 5) {
      throw new Tv4ApiError(400, 'Bad Request', 'session permits at most five active feedback events')
    }
    fixtureFeedbackActiveEvents += 1
    const fixtureCands = generateFixtureCandidates('refined', 4)
    // Promote selected candidate to #1
    const sourceFrameId = req.source_candidate_frame_id ?? req.frame_id
    const selected = fixtureCands.find(c => c.video_id === req.video_id && (c.frame_id === req.frame_id || c.frame_id === sourceFrameId)) || {
      query_id: 'fixture',
      video_id: req.video_id,
      frame_id: req.frame_id,
      timestamp_ms: 0,
      source: 'feedback',
      rank: 1,
      submission_selectable: false,
    }
    const selectedExact: SearchCandidate = {
      ...selected,
      frame_id: req.frame_id,
      certified_anchor_frame_id: sourceFrameId,
      anchor_offset: req.source_candidate_frame_id ? (req.frame_id - req.source_candidate_frame_id) : ((selected as any).anchor_offset ?? 0),
    }
    const remainder = fixtureCands.filter(c => !(c.video_id === req.video_id && (c.frame_id === req.frame_id || c.frame_id === sourceFrameId)))
    const reordered = [selectedExact, ...remainder].map((c, i) => ({ ...c, rank: i + 1 }))

    return {
      session_id: req.session_id,
      revision: req.expected_revision + 1,
      candidates: reordered,
      active_feedback_count: fixtureFeedbackActiveEvents,
      max_active_feedback_events: 5,
      status: 'ok',
      provenance_mode: 'fixture',
    }
  }

  const res = await fetch(`${baseUrl}/feedback/refine`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: req.session_id,
      video_id: req.video_id,
      frame_id: req.frame_id,
      source_candidate_frame_id: req.source_candidate_frame_id ?? null,
      feedback_text: req.feedback_text,
      expected_revision: req.expected_revision,
    }),
  })

  if (!res.ok) {
    const detail = await parseErrorDetail(res)
    throw new Tv4ApiError(res.status, res.statusText, detail)
  }

  return res.json()
}

export async function undoFeedback(
  req: FeedbackUndoRequest,
  baseUrl = DEFAULT_BASE_URL
): Promise<FeedbackResponse> {
  if (!req.session_id || req.expected_revision < 0) {
    throw new Error('Valid session_id and expected_revision are required')
  }

  if (isFixtureModeActive()) {
    if (fixtureFeedbackActiveEvents > 0) {
      fixtureFeedbackActiveEvents -= 1
    }
    const fixtureCands = generateFixtureCandidates('undo', 4)
    return {
      session_id: req.session_id,
      revision: req.expected_revision + 1,
      candidates: fixtureCands,
      active_feedback_count: fixtureFeedbackActiveEvents,
      max_active_feedback_events: 5,
      status: 'ok',
      provenance_mode: 'fixture',
    }
  }

  const res = await fetch(`${baseUrl}/feedback/undo`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: req.session_id,
      expected_revision: req.expected_revision,
    }),
  })

  if (!res.ok) {
    const detail = await parseErrorDetail(res)
    throw new Tv4ApiError(res.status, res.statusText, detail)
  }

  return res.json()
}

export async function resetFeedback(
  req: FeedbackResetRequest,
  baseUrl = DEFAULT_BASE_URL
): Promise<FeedbackResponse> {
  if (!req.session_id || req.expected_revision < 0) {
    throw new Error('Valid session_id and expected_revision are required')
  }

  if (isFixtureModeActive()) {
    fixtureFeedbackActiveEvents = 0
    const fixtureCands = generateFixtureCandidates('reset', 4)
    return {
      session_id: req.session_id,
      revision: req.expected_revision + 1,
      candidates: fixtureCands,
      active_feedback_count: 0,
      max_active_feedback_events: 5,
      status: 'ok',
      provenance_mode: 'fixture',
    }
  }

  const res = await fetch(`${baseUrl}/feedback/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: req.session_id,
      expected_revision: req.expected_revision,
    }),
  })

  if (!res.ok) {
    const detail = await parseErrorDetail(res)
    throw new Tv4ApiError(res.status, res.statusText, detail)
  }

  return res.json()
}

export async function getFeedbackSession(
  sessionId: string,
  baseUrl = DEFAULT_BASE_URL
): Promise<FeedbackResponse> {
  if (!sessionId?.trim()) {
    throw new Error('sessionId is required')
  }

  if (isFixtureModeActive()) {
    const fixtureCands = generateFixtureCandidates('view', 4)
    return {
      session_id: sessionId,
      revision: 0,
      candidates: fixtureCands,
      status: 'ok',
      provenance_mode: 'fixture',
    }
  }

  const res = await fetch(`${baseUrl}/feedback/session/${encodeURIComponent(sessionId)}`)

  if (!res.ok) {
    const detail = await parseErrorDetail(res)
    throw new Tv4ApiError(res.status, res.statusText, detail)
  }

  return res.json()
}

// ---------------------------------------------------------------------------
// VQA API Surface (WP11 / T016 / T030)
// ---------------------------------------------------------------------------

export async function fetchVqaAnswer(
  req: VqaRequest,
  baseUrl = DEFAULT_BASE_URL
): Promise<VqaResponse> {
  if (!req.query_text?.trim() || !req.question?.trim()) {
    throw new Error('query_text and question are required for VQA')
  }

  const boundedTopK = Math.max(1, Math.min(100, req.top_k || 100))
  const boundedTopKAnswers = Math.max(1, Math.min(20, req.top_k_answers || 5))

  if (isFixtureModeActive()) {
    return {
      query_id: req.query_id || 'qa-fixture-001',
      provenance_mode: 'fixture',
      results: [
        {
          rank: 1,
          video_id: 'L05_V005',
          frame_id: 888,
          timestamp_ms: 29600,
          confidence: null,
          answer: 'màu xanh',
          verified: false,
          manual_review: true,
          proposal: 'màu xanh',
          approved: false,
          verifier_status: 'fixture_unverified',
          retry_count: 0,
          manual_required: true,
          status: 'manual_required',
          degraded_reasons: ['fixture_non_authoritative'],
          evidence: {
            query_id: req.query_id || 'qa-fixture-001',
            query_text: req.query_text,
            question: req.question,
            video_id: 'L05_V005',
            frame_id: 888,
            timestamp_ms: 29600,
            keyframe_path: 'keyframes/L05_V005/0888.jpg',
            selected_frames: [
              {
                video_id: 'L05_V005',
                frame_id: 888,
                timestamp_ms: 29600,
                keyframe_path: 'keyframes/L05_V005/0888.jpg',
                preprocess_run_id: 'fixture-run',
                provenance: { source: 'fixture' },
                submission_selectable: false,
              },
            ],
            ocr_evidence: [
              {
                detection_id: 'fixture-ocr-888',
                video_id: 'L05_V005',
                frame_id: 888,
                timestamp_ms: 29600,
                raw_text: 'AWARD',
                normalized_text: 'award',
                bbox_xyxy_norm: [0.1, 0.2, 0.6, 0.3],
                polygon_norm: null,
                confidence: 0.91,
                crop_evidence_path: null,
                crop_sha256: null,
                source_keyframe_sha256: 'fixture',
                preprocess_run_id: 'fixture-run',
                model_name: 'fixture-ocr',
                model_version: '1',
                provenance: { branch: 'ocr' },
                source_record: {},
              },
            ],
            asr_evidence: [
              {
                segment_id: 'fixture-asr-1',
                video_id: 'L05_V005',
                start_ms: 29000,
                end_ms: 30200,
                text: 'person holding a blue cup',
                normalized_text: 'person holding a blue cup',
                words: [{ word: 'blue', start_ms: 29900, end_ms: 30100, probability: 0.9 }],
                context: [{ segment_id: 'fixture-asr-before', text: 'host speaks' }],
                confidence: 0.88,
                language: 'en',
                preprocess_run_id: 'fixture-run',
                model_name: 'fixture-asr',
                model_version: '1',
                provenance: { branch: 'asr' },
                source_record: {},
              },
            ],
            object_evidence: [
              {
                detection_id: 'fixture-object-888',
                video_id: 'L05_V005',
                frame_id: 888,
                timestamp_ms: 29600,
                label: 'cup',
                canonical_label: 'cup',
                bbox_xyxy_norm: [0.55, 0.4, 0.7, 0.8],
                confidence: 0.87,
                source_keyframe_path: 'keyframes/L05_V005/0888.jpg',
                source_keyframe_sha256: 'fixture',
                preprocess_run_id: 'fixture-run',
                model_name: 'fixture-object',
                model_version: '1',
                provenance: { branch: 'object' },
                source_record: {},
              },
            ],
            metadata_evidence: [],
            availability: {
              frames: 'available',
              ocr: 'available',
              asr: 'available',
              object: 'available',
              metadata: 'empty',
            },
            ocr_texts: ['AWARD'],
            asr_texts: ['person holding a blue cup'],
            object_labels: ['cup'],
            neighbor_frame_ids: [880, 896],
            provenance: { fixture: true, submission_selectable: false },
          },
        },
      ],
    }
  }

  const payload: Record<string, unknown> = {
    query_text: req.query_text,
    question: req.question,
    query_id: req.query_id ?? null,
    top_k: boundedTopK,
    top_k_answers: boundedTopKAnswers,
  }
  if (req.weights !== undefined) {
    payload.weights = req.weights
  }

  const res = await fetch(`${baseUrl}/vqa/answer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  if (!res.ok) {
    const detail = await parseErrorDetail(res)
    throw new Tv4ApiError(res.status, res.statusText, detail)
  }

  return res.json()
}

export async function fetchTrakeAlign(
  req: TrakeRequest,
  baseUrl = DEFAULT_BASE_URL
): Promise<TrakeResponse> {
  if (!req.query_text || !req.query_text.trim()) {
    throw new Error('fetchTrakeAlign requires non-empty query_text')
  }

  if (isFixtureModeActive()) {
    const fixtureResult = {
      rank: 1,
      video_id: 'L10_V010',
      frame_ids: [101, 156, 203, 251],
      timestamps_ms: [4040, 6240, 8120, 10040],
      event_scores: [0.9, 0.7, 0.85, 0.6],
      aggregate_score: 3.05,
      preprocess_run_id: 'run_v1_batch1',
      events: [101, 156, 203, 251].map((frameId, index) => ({
        event_index: index,
        frame_id: frameId,
        timestamp_ms: [4040, 6240, 8120, 10040][index],
        score: [0.9, 0.7, 0.85, 0.6][index],
        alternatives: [],
      })),
    }
    return {
      query_id: req.query_id || 'trake-fixture-001',
      provenance_mode: 'fixture',
      result: fixtureResult,
      results: [fixtureResult],
    }
  }

  const payload: Record<string, unknown> = {
    query_text: req.query_text.trim(),
    strategy: req.strategy || 'dp',
  }
  if (req.events && req.events.length > 0) {
    payload.events = req.events
  }
  if (req.query_id) {
    payload.query_id = req.query_id
  }
  if (req.top_k_sequences !== undefined) {
    payload.top_k_sequences = Math.max(1, Math.min(5, req.top_k_sequences))
  }

  const res = await fetch(`${baseUrl}/trake/align`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  if (!res.ok) {
    const detail = await parseErrorDetail(res)
    throw new Tv4ApiError(res.status, res.statusText, detail)
  }

  return res.json()
}

export async function rerunTrakeEvent(
  req: TrakeRerunRequest,
  baseUrl = DEFAULT_BASE_URL
): Promise<TrakeRerunResponse> {
  if (!req.query_text.trim() || req.events.length < 2) {
    throw new Error('rerunTrakeEvent requires a query and at least two events')
  }
  if (req.event_index < 0 || req.event_index >= req.events.length) {
    throw new Error('rerunTrakeEvent event_index is outside events')
  }

  if (isFixtureModeActive()) {
    const lower = req.locked_events
      .filter((item) => item.event_index < req.event_index)
      .map((item) => item.timestamp_ms)
    const upper = req.locked_events
      .filter((item) => item.event_index > req.event_index)
      .map((item) => item.timestamp_ms)
    return {
      query_id: req.query_id || 'trake-fixture-rerun',
      event_index: req.event_index,
      video_id: req.video_id,
      lower_bound_ms: lower.length > 0 ? Math.max(...lower) : null,
      upper_bound_ms: upper.length > 0 ? Math.min(...upper) : null,
      alternatives: [],
      provenance_mode: 'fixture',
    }
  }

  const payload = {
    query_text: req.query_text.trim(),
    events: req.events,
    event_index: req.event_index,
    video_id: req.video_id,
    locked_events: req.locked_events,
    query_id: req.query_id ?? null,
    exclude_frame_id: req.exclude_frame_id ?? null,
    top_k: Math.max(1, Math.min(5, req.top_k ?? 5)),
  }
  const res = await fetch(`${baseUrl}/trake/rerun-event`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const detail = await parseErrorDetail(res)
    throw new Tv4ApiError(res.status, res.statusText, detail)
  }
  return res.json()
}

export interface VideoMeta {
  video_id: string
  fps: number
  fps_nominal?: number
  fps_average?: number
  duration_ms?: number
  width_px?: number
  height_px?: number
  frame_count?: number
  codec?: string
  time_base?: string
}

export async function fetchVideoMeta(
  videoId: string,
  baseUrl = DEFAULT_BASE_URL
): Promise<VideoMeta> {
  const realVid = resolveMediaVideoId(videoId)
  if (isFixtureModeActive()) {
    return {
      video_id: realVid,
      fps: 25.0,
      fps_nominal: 25.0,
      fps_average: 25.0,
      time_base: '1/12800',
    }
  }
  try {
    const res = await fetch(`${baseUrl}/videos/${encodeURIComponent(realVid)}/meta`)
    if (res.ok) {
      return await res.json()
    }
  } catch {}
  return { video_id: realVid, fps: 25.0 }
}

export interface VideoKeyframe {
  video_id: string
  frame_id: number
  timestamp_ms: number
  pts?: number
  keyframe_path?: string
}

export async function fetchVideoKeyframes(
  videoId: string,
  baseUrl = DEFAULT_BASE_URL
): Promise<VideoKeyframe[]> {
  const realVid = resolveMediaVideoId(videoId)
  if (isFixtureModeActive()) {
    return Array.from({ length: 15 }, (_, i) => ({
      video_id: realVid,
      frame_id: (i + 1) * 100,
      timestamp_ms: (i + 1) * 4000,
      pts: (i + 1) * 512,
    }))
  }
  // 1. Try TV1 direct (Port 8000) or proxy (/frames)
  try {
    const res = await fetch(`/frames/${encodeURIComponent(realVid)}`)
    if (res.ok) {
      const data = await res.json()
      if (Array.isArray(data) && data.length > 0) {
        return data
      }
    }
  } catch {}

  try {
    const res = await fetch(`http://127.0.0.1:8000/frames/${encodeURIComponent(realVid)}`)
    if (res.ok) {
      const data = await res.json()
      if (Array.isArray(data) && data.length > 0) {
        return data
      }
    }
  } catch {}

  // 2. Fallback to TV4
  try {
    const res = await fetch(`${baseUrl}/videos/${encodeURIComponent(realVid)}/keyframes`)
    if (res.ok) {
      const data = await res.json()
      if (Array.isArray(data) && data.length > 0) {
        return data
      }
    }
  } catch {}

  return []
}

const keyframeCacheMap = new Map<string, Promise<VideoKeyframe[]>>()

export async function getBestKeyframeUrl(
  videoId: string,
  frameId: number,
  timestampMs?: number
): Promise<string> {
  const realVid = resolveMediaVideoId(videoId)
  if (isFixtureModeActive()) {
    return getFixturePreviewDataUri(videoId, frameId)
  }

  let p = keyframeCacheMap.get(realVid)
  if (!p) {
    p = fetchVideoKeyframes(realVid)
    keyframeCacheMap.set(realVid, p)
  }

  try {
    const kfs = await p
    if (kfs && kfs.length > 0) {
      let best = kfs[0]
      const hasTimestamp = Number.isFinite(timestampMs)
      let minDiff = hasTimestamp
        ? Math.abs(best.timestamp_ms - (timestampMs as number))
        : Math.abs(best.frame_id - frameId)
      for (const kf of kfs) {
        const diff = hasTimestamp
          ? Math.abs(kf.timestamp_ms - (timestampMs as number))
          : Math.abs(kf.frame_id - frameId)
        if (diff < minDiff) {
          minDiff = diff
          best = kf
        }
      }
      if (best.keyframe_path) {
        const filename = best.keyframe_path.split('/').pop() || `${best.frame_id}.jpg`
        return `/keyframe-image/${encodeURIComponent(realVid)}/${filename}`
      }
      return `/videos/${encodeURIComponent(realVid)}/thumbnails/${best.frame_id}.jpg`
    }
  } catch {}

  return `/videos/${encodeURIComponent(realVid)}/thumbnails/${frameId}.jpg`
}
