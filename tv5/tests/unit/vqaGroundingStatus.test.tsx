import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { RetrievalWorkspace } from '../../src/components/RetrievalWorkspace'
import { AppProvider } from '../../src/state/AppContext'
import { initialAppState } from '../../src/state/appState'
import type { VqaResult } from '../../src/types/contracts'

describe('VQA scene grounding status', () => {
  it.each(['scene_mismatch', 'scene_unverified'])('labels an answer proposal with %s', (reason) => {
    const result: VqaResult = {
      rank: 1, video_id: 'L21_V001', frame_id: 270, timestamp_ms: 10800,
      answer: '', proposal: '2M', verified: false, manual_review: true,
      approved: false, verifier_status: 'insufficient_evidence', retry_count: 0,
      manual_required: true, status: 'manual_required', degraded_reasons: [],
      independently_supported: false,
      verification: {
        supported: false, visual_support: 0, audio_text_support: 0, ocr_support: 1,
        producer_support: 1, contradiction: reason === 'scene_mismatch',
        final_support_score: 0.2, manual_required: true,
        scene_supported: false, reasons: [reason],
      },
      evidence: {
        query_id: 'qa-test', video_id: 'L21_V001', frame_id: 270, timestamp_ms: 10800,
        selected_frames: [], ocr_evidence: [], asr_evidence: [], object_evidence: [],
        metadata_evidence: [], availability: {}, ocr_texts: [], asr_texts: [],
        object_labels: [], neighbor_frame_ids: [], provenance: {},
      },
    }
    render(<AppProvider initialState={{ ...initialAppState, activeTab: 'retrieval',
      taskMode: 'VQA', mode: 'fixture', readiness: 'READY', vqaResults: [result], vqaHasSearched: true,
    }}><RetrievalWorkspace /></AppProvider>)
    expect(screen.getByText('VQA Answer Proposals')).toBeInTheDocument()
    expect(screen.getByTestId('vqa-grounding-status-1')).toHaveTextContent(
      reason === 'scene_mismatch' ? 'Scene mismatch' : 'Scene unverified',
    )
    expect(screen.queryByText('Machine verified')).not.toBeInTheDocument()
  })
})
