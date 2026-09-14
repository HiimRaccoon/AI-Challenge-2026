/**
 * TV4 & WP13 Shared Contracts and Types
 * Strict typing reflecting actual TV4 routes and payloads.
 */

export type SystemMode = 'fixture' | 'live'

export type ReadinessStatus =
  | 'READY'
  | 'PARTIAL'
  | 'HANDOVER_PENDING'
  | 'CODE_GAP'
  | 'INCOMPATIBLE'
  | 'ACTUALLY_MISSING'
  | 'DEGRADED'
  | 'OFFLINE'

export interface TV4HealthResponse {
  status: 'ok' | 'degraded'
  mode: 'live' | 'fixture'
  preprocess_run_id?: string
  error?: string
}

export interface KisRequest {
  query_text: string
  query_id?: string | null
  top_k: number
  weights?: Record<string, number>
}

export interface SearchCandidate {
  query_id: string
  video_id: string
  frame_id: number
  timestamp_ms: number
  source: string
  rank: number
  schema_version?: string
  event_index?: number | null
  representative_frame_id?: number | null
  window_start_ms?: number | null
  window_end_ms?: number | null
  raw_score?: number | null
  score?: number | null
  model_scores?: Record<string, number>
  model_ranks?: Record<string, number>
  matched_filters?: string[]
  evidence_refs?: string[]
  provenance_sources?: string[]
  provenance?: Record<string, unknown>
  confidence?: number | null
  preprocess_run_id?: string
  created_at_utc?: string
  certified_anchor_frame_id?: number | null
  certified_anchor_timestamp_ms?: number | null
  cumulative_offset?: number | null
  anchor_offset?: number | null
}

export interface KisResponse {
  query_id: string
  candidates: SearchCandidate[]
  provenance_mode?: string
}

export interface ExactNeighborRequest {
  video_id: string
  frame_id: number
  timestamp_ms: number
  offsets: number[]
  certified_anchor_frame_id?: number | null
  certified_anchor_timestamp_ms?: number | null
  cumulative_offset: number
}

export interface ExactFrameProof {
  video_id: string
  frame_id: number
  timestamp_ms: number
  pts: number
  time_base: string
  preprocess_run_id: string
  media_record_ref?: string
  mapping_ref?: string
  mapping_guaranteed: boolean
  submission_selectable: boolean
  identity_source: string
  degraded_reason?: string | null
  media_identity_verified?: boolean
  producer_compatibility_verified?: boolean
  certification_id?: string
  certification_report_sha256?: string
  source_sha256?: string
}

export interface ExactStep {
  offset: number
  degraded_reason?: string | null
  frame: ExactFrameProof | null
}

export interface ExactNeighborResponse {
  video_id: string
  anchor_frame_id: number
  degraded_reason?: string | null
  steps: ExactStep[]
  provenance_mode?: string
  preprocess_run_id?: string
}

export interface ExactImageHeaders {
  video_id?: string
  frame_id?: number
  pts?: number
  time_base?: string
  timestamp_ms?: number
  preprocess_run_id?: string
  certification_id?: string
  certification_report_sha256?: string
  source_sha256?: string
  submission_selectable?: boolean
}

export interface ExactImageResult {
  blobUrl: string
  headers: ExactImageHeaders
}

export interface FeedbackStartRequest {
  session_id: string
  original_query: string
}

export interface FeedbackRefineRequest {
  session_id: string
  video_id: string
  frame_id: number
  source_candidate_frame_id?: number | null
  feedback_text: string
  expected_revision: number
}

export interface FeedbackUndoRequest {
  session_id: string
  expected_revision: number
}

export interface FeedbackResetRequest {
  session_id: string
  expected_revision: number
}

export interface FeedbackResponse {
  session_id: string
  revision: number
  candidates: SearchCandidate[]
  active_feedback_count?: number
  max_active_feedback_events?: number
  expires_at_utc?: string | null
  wp03_run_id?: string | null
  status: string
  provenance_mode?: string
}

export interface BasketItem {
  query_id: string
  video_id: string
  frame_id: number
  timestamp_ms?: number
  added_at_utc: string
  task?: 'KIS' | 'VQA' | 'TRAKE'
  answer?: string
  frame_ids?: number[]
  event_labels?: string[]
}

// ---------------------------------------------------------------------------
// VQA / Evidence Contracts (WP11 / T016 / T030)
// ---------------------------------------------------------------------------

export interface CanonicalFrameReference {
  video_id: string
  frame_id: number
  timestamp_ms: number
  keyframe_path?: string | null
  preprocess_run_id?: string | null
  provenance?: Record<string, unknown>
  submission_selectable?: boolean
}

export interface OcrEvidence {
  detection_id: string
  video_id: string
  frame_id: number
  timestamp_ms: number
  raw_text: string
  normalized_text: string
  bbox_xyxy_norm: [number, number, number, number]
  polygon_norm?: number[][] | null
  confidence?: number | null
  crop_evidence_path?: string | null
  crop_sha256?: string | null
  source_keyframe_sha256?: string | null
  preprocess_run_id?: string | null
  model_name?: string | null
  model_version?: string | null
  provenance?: Record<string, unknown>
  source_record?: Record<string, unknown>
}

export interface AsrEvidence {
  segment_id: string
  video_id: string
  start_ms: number
  end_ms: number
  text: string
  normalized_text?: string | null
  words?: Array<{ word: string; start_ms?: number; end_ms?: number; probability?: number }>
  context?: Array<{ segment_id: string; text: string }>
  confidence?: number | null
  language?: string | null
  preprocess_run_id?: string | null
  model_name?: string | null
  model_version?: string | null
  provenance?: Record<string, unknown>
  source_record?: Record<string, unknown>
}

export interface ObjectEvidence {
  detection_id: string
  video_id: string
  frame_id: number
  timestamp_ms: number
  label: string
  canonical_label?: string | null
  bbox_xyxy_norm: [number, number, number, number]
  confidence?: number | null
  source_keyframe_path?: string | null
  source_keyframe_sha256?: string | null
  preprocess_run_id?: string | null
  model_name?: string | null
  model_version?: string | null
  provenance?: Record<string, unknown>
  source_record?: Record<string, unknown>
}

export interface MetadataEvidence {
  metadata_id: string
  video_id: string
  source: string
  values: Record<string, unknown>
  window_start_ms?: number | null
  window_end_ms?: number | null
  confidence?: number | null
  preprocess_run_id?: string | null
  model_name?: string | null
  model_version?: string | null
  source_record_sha256?: string | null
  provenance?: Record<string, unknown>
  source_record?: Record<string, unknown>
}

export interface EvidencePack {
  query_id: string
  video_id: string
  frame_id: number
  timestamp_ms: number
  keyframe_path?: string | null
  query_text?: string | null
  question?: string | null
  selected_frames: CanonicalFrameReference[]
  ocr_evidence: OcrEvidence[]
  asr_evidence: AsrEvidence[]
  object_evidence: ObjectEvidence[]
  metadata_evidence: MetadataEvidence[]
  availability: Record<string, string>
  ocr_texts: string[]
  asr_texts: string[]
  object_labels: string[]
  neighbor_frame_ids: number[]
  provenance: Record<string, unknown>
}

export interface VqaResult {
  rank: number
  video_id: string
  frame_id: number
  timestamp_ms: number
  certified_anchor_frame_id?: number
  certified_anchor_timestamp_ms?: number
  anchor_offset?: number
  confidence?: number | null
  answer: string
  verified: boolean
  manual_review: boolean
  proposal: string
  approved: boolean
  verifier_status: string
  retry_count: number
  manual_required: boolean
  status: string
  degraded_reasons: string[]
  answerability_score?: number
  answerability_components?: {
    retrieval_relevance: number
    modality_match: number
    transcript_answer_cue: number
    visual_object_answer_cue: number
    temporal_completeness: number
    total: number
  }
  answer_confidence?: number
  verifier_score?: number
  independently_supported?: boolean
  contradiction?: boolean
  verification?: QaVerification
  support_start_ms?: number | null
  support_end_ms?: number | null
  frame_candidates?: VqaFrameCandidate[]
  evidence: EvidencePack
}

export interface QaVerification {
  supported: boolean
  visual_support: number
  audio_text_support: number
  ocr_support: number
  producer_support: number
  contradiction: boolean
  final_support_score: number
  manual_required: boolean
  visual_verifier_available?: boolean
  visual_frame_ids?: number[]
  visual_verifier_reason?: string | null
  reasons: string[]
  scene_supported?: boolean | null
  scene_score?: number | null
  scene_verifier_reason?: string | null
}

export interface VqaFrameCandidate {
  frame_id: number
  timestamp_ms: number
  score: number
  support_modalities: string[]
}

export interface VqaRequest {
  query_text: string
  question: string
  query_id?: string | null
  top_k?: number
  top_k_answers?: number
  weights?: Record<string, number>
}

export interface VqaResponse {
  query_id: string
  results: VqaResult[]
  retrieval_candidates?: SearchCandidate[]
  verified_candidates?: Array<{
    rank: number
    video_id: string
    frame_id: number
    timestamp_ms: number
    answerability_score: number
    relation_verification?: Record<string, unknown> | null
    dense_ocr?: Record<string, unknown> | null
  }>
  query_plan?: {
    answer_source: string
    device?: string | null
    value_type?: string | null
    temporal_rule?: string | null
    requires_dense_read: boolean
    relation_query?: string | null
  }
  provenance_mode?: string
}

export interface TrakeRequest {
  query_text: string
  events?: string[] | null
  query_id?: string | null
  strategy?: 'dp' | 'greedy'
  top_k_sequences?: number
}

export interface TrakeEventAlternative {
  frame_id: number
  timestamp_ms: number
  score?: number | null
  candidate?: SearchCandidate
}

export interface TrakeEventResult {
  event_index: number
  frame_id: number
  timestamp_ms: number
  start_ms?: number | null
  end_ms?: number | null
  score: number
  alternatives: TrakeEventAlternative[]
}

export interface TrakeHypothesisResult {
  rank?: number
  video_id: string
  frame_ids: number[]
  timestamps_ms?: number[]
  event_scores: number[]
  aggregate_score: number
  preprocess_run_id: string
  candidates?: SearchCandidate[]
  events?: TrakeEventResult[]
  verifier_score?: number | null
  entity_continuity?: number | null
  state_consistency?: number | null
  order_consistent?: boolean | null
  story_consistency?: number | null
  verification?: TrakeVerification
}

export interface TrakeVerification {
  event_scores: number[]
  entity_continuity: number
  state_consistency: number
  order_consistent: boolean
  story_consistency: number
  modality_support: number
  micro_clip_support: number
  clip_manual_required: boolean
  whole_video_prior: number
  implausible_gap_penalty: number
  contradiction_penalty: number
  final_sequence_score: number
  visual_verifier_available?: boolean
  visual_frame_ids?: number[][]
  visual_verifier_reasons?: Array<string | null>
}

export interface TrakeLockedEvent {
  event_index: number
  frame_id: number
  timestamp_ms: number
}

export interface TrakeRerunRequest {
  query_text: string
  events: string[]
  event_index: number
  video_id: string
  locked_events: TrakeLockedEvent[]
  query_id?: string | null
  exclude_frame_id?: number | null
  top_k?: number
}

export interface TrakeRerunResponse {
  query_id: string
  event_index: number
  video_id: string
  lower_bound_ms: number | null
  upper_bound_ms: number | null
  alternatives: TrakeEventAlternative[]
  provenance_mode?: string
}

export interface TrakeResponse {
  query_id: string
  result: TrakeHypothesisResult | null
  results?: TrakeHypothesisResult[]
  message?: string
  provenance_mode?: string
}

export interface TrakeEventSlot {
  event_index: number
  event_label: string
  video_id: string | null
  frame_id: number | null
  timestamp_ms?: number | null
  score?: number | null
  locked: boolean
  exact_proof?: ExactFrameProof | null
  validation_status: 'valid' | 'missing' | 'incompatible_video' | 'unverified'
  certified_anchor_frame_id?: number | null
  certified_anchor_timestamp_ms?: number | null
  anchor_offset?: number | null
  alternatives?: TrakeEventAlternative[]
}
