import { describe, it, expect } from 'vitest'
import { appReducer } from '../../src/state/appReducer'
import { initialAppState } from '../../src/state/appState'
import { VqaResponse, VqaResult } from '../../src/types/contracts'

const mockVqaResult1: VqaResult = {
  rank: 1,
  video_id: 'L05_V005',
  frame_id: 888,
  timestamp_ms: 29600,
  confidence: 0.95,
  answer: 'màu xanh lá',
  verified: true,
  manual_review: false,
  proposal: 'màu xanh lá',
  approved: false, // Backend never auto-approves
  verifier_status: 'verified',
  retry_count: 0,
  manual_required: false,
  status: 'verified',
  degraded_reasons: [],
  evidence: {
    query_id: 'qa-001',
    query_text: 'người phụ nữ mặc áo xanh lá',
    question: 'Chiếc áo màu gì?',
    video_id: 'L05_V005',
    frame_id: 888,
    timestamp_ms: 29600,
    keyframe_path: 'keyframes/L05_V005/0888.jpg',
    selected_frames: [],
    ocr_evidence: [],
    asr_evidence: [],
    object_evidence: [],
    metadata_evidence: [],
    availability: { frames: 'available' },
    ocr_texts: [],
    asr_texts: [],
    object_labels: [],
    neighbor_frame_ids: [],
    provenance: {},
  },
}

const mockVqaResult2: VqaResult = {
  rank: 2,
  video_id: 'L05_V008',
  frame_id: 1240,
  timestamp_ms: 41333,
  confidence: 0.82,
  answer: 'màu đỏ tươi',
  verified: false,
  manual_review: true,
  proposal: 'màu đỏ tươi',
  approved: false,
  verifier_status: 'unverified',
  retry_count: 0,
  manual_required: true,
  status: 'manual_required',
  degraded_reasons: [],
  evidence: {
    query_id: 'qa-001',
    query_text: 'người phụ nữ mặc áo xanh lá',
    question: 'Chiếc áo màu gì?',
    video_id: 'L05_V008',
    frame_id: 1240,
    timestamp_ms: 41333,
    keyframe_path: 'keyframes/L05_V008/1240.jpg',
    selected_frames: [],
    ocr_evidence: [],
    asr_evidence: [],
    object_evidence: [],
    metadata_evidence: [],
    availability: { frames: 'available' },
    ocr_texts: [],
    asr_texts: [],
    object_labels: [],
    neighbor_frame_ids: [],
    provenance: {},
  },
}

const mockVqaResponse: VqaResponse = {
  query_id: 'qa-001',
  results: [mockVqaResult1, mockVqaResult2],
  provenance_mode: 'live',
}

describe('T030 — VQA Frontend State Model & Invariants', () => {
  it('preserves distinct query_text and question in state', () => {
    let state = appReducer(initialAppState, {
      type: 'SET_QUERY_TEXT',
      payload: 'người cầm cúp',
    })
    state = appReducer(state, {
      type: 'SET_VQA_QUESTION',
      payload: 'Cúp có màu gì?',
    })

    expect(state.queryText).toBe('người cầm cúp')
    expect(state.vqaQuestion).toBe('Cúp có màu gì?')
  })

  // -------------------------------------------------------------------------
  // CRITICAL P0 INVARIANT: NO AUTO-CONFIRM
  // -------------------------------------------------------------------------

  it('CRITICAL P0: receiving verified proposal does NOT automatically approve or confirm', () => {
    let state = appReducer(initialAppState, {
      type: 'VQA_SEARCH_SUCCESS',
      payload: mockVqaResponse,
    })

    expect(state.vqaResults.length).toBe(2)
    expect(state.vqaActiveResult?.video_id).toBe('L05_V005')
    expect(state.vqaActiveResult?.proposal).toBe('màu xanh lá')
    expect(state.vqaActiveResult?.verified).toBe(true)

    // Draft is populated for convenience
    expect(state.vqaDraftAnswer).toBe('màu xanh lá')

    // INVARIANT: Approved answer MUST remain null!
    expect(state.vqaApprovedAnswer).toBeNull()

    // Basket MUST remain empty!
    expect(state.submissionBasket).toEqual([])
  })

  it('supports editing draft and explicit operator confirmation', () => {
    let state = appReducer(initialAppState, {
      type: 'VQA_SEARCH_SUCCESS',
      payload: mockVqaResponse,
    })

    // Operator edits the draft
    state = appReducer(state, {
      type: 'SET_VQA_DRAFT_ANSWER',
      payload: 'màu xanh lá cây đậm',
    })
    expect(state.vqaDraftAnswer).toBe('màu xanh lá cây đậm')
    expect(state.vqaApprovedAnswer).toBeNull() // Still unapproved

    // Operator clicks Confirm
    state = appReducer(state, { type: 'CONFIRM_VQA_ANSWER' })
    expect(state.vqaApprovedAnswer).toBe('màu xanh lá cây đậm')
  })

  it('preserves exact approved answer verbatim without silent modification', () => {
    let state = appReducer(initialAppState, {
      type: 'VQA_SEARCH_SUCCESS',
      payload: mockVqaResponse,
    })

    const exactAnswer = 'xe cứu hỏa 114 (biển số 51A-1234)'
    state = appReducer(state, {
      type: 'SET_VQA_DRAFT_ANSWER',
      payload: exactAnswer,
    })
    state = appReducer(state, { type: 'CONFIRM_VQA_ANSWER' })

    expect(state.vqaApprovedAnswer).toBe(exactAnswer)
  })

  // -------------------------------------------------------------------------
  // APPROVAL INVALIDATION
  // -------------------------------------------------------------------------

  it('invalidates approval when question is changed', () => {
    let state = appReducer(initialAppState, {
      type: 'VQA_SEARCH_SUCCESS',
      payload: mockVqaResponse,
    })
    state = appReducer(state, { type: 'CONFIRM_VQA_ANSWER' })
    expect(state.vqaApprovedAnswer).toBe('màu xanh lá')

    // Operator changes question
    state = appReducer(state, {
      type: 'SET_VQA_QUESTION',
      payload: 'Thời gian trong ngày là khi nào?',
    })

    expect(state.vqaApprovedAnswer).toBeNull()
  })

  it('invalidates approval when a different candidate is selected', () => {
    let state = appReducer(initialAppState, {
      type: 'VQA_SEARCH_SUCCESS',
      payload: mockVqaResponse,
    })
    state = appReducer(state, { type: 'CONFIRM_VQA_ANSWER' })
    expect(state.vqaApprovedAnswer).toBe('màu xanh lá')

    // Operator selects Candidate 2
    state = appReducer(state, {
      type: 'SELECT_VQA_RESULT',
      payload: mockVqaResult2,
    })

    expect(state.vqaActiveResult?.video_id).toBe('L05_V008')
    expect(state.vqaDraftAnswer).toBe('màu đỏ tươi')
    expect(state.vqaApprovedAnswer).toBeNull() // Invalidated!
  })

  it('invalidates approval when new search starts', () => {
    let state = appReducer(initialAppState, {
      type: 'VQA_SEARCH_SUCCESS',
      payload: mockVqaResponse,
    })
    state = appReducer(state, { type: 'CONFIRM_VQA_ANSWER' })
    expect(state.vqaApprovedAnswer).toBe('màu xanh lá')

    state = appReducer(state, { type: 'VQA_SEARCH_START' })
    expect(state.vqaApprovedAnswer).toBeNull()
    expect(state.isVqaSearching).toBe(true)
  })

  // -------------------------------------------------------------------------
  // BASKET ISOLATION & FIXTURE SAFETY
  // -------------------------------------------------------------------------

  it('VQA operations never mutate submission basket until explicit ADD_VQA_TO_BASKET of confirmed answer', () => {
    let state = appReducer(initialAppState, {
      type: 'VQA_SEARCH_SUCCESS',
      payload: mockVqaResponse,
    })

    // 1. Proposal reception does not add to basket
    expect(state.submissionBasket).toHaveLength(0)

    // 2. Draft edit does not add to basket
    state = appReducer(state, {
      type: 'SET_VQA_DRAFT_ANSWER',
      payload: 'màu xanh dương',
    })
    expect(state.submissionBasket).toHaveLength(0)

    // 3. Attempting to add unapproved draft does nothing
    state = appReducer(state, { type: 'ADD_VQA_TO_BASKET' })
    expect(state.submissionBasket).toHaveLength(0)

    // 4. Confirming answer alone does not add to basket
    state = appReducer(state, { type: 'CONFIRM_VQA_ANSWER' })
    expect(state.submissionBasket).toHaveLength(0)

    // 5. Explicit ADD_VQA_TO_BASKET now succeeds
    state = appReducer(state, { type: 'ADD_VQA_TO_BASKET' })
    expect(state.submissionBasket).toHaveLength(1)
    expect(state.submissionBasket[0]).toEqual(
      expect.objectContaining({
        video_id: 'L05_V005',
        frame_id: 888,
        task: 'VQA',
        answer: 'màu xanh dương',
      })
    )
  })

  it('fixture mode prevents adding VQA answers to real submission basket', () => {
    let state = appReducer(initialAppState, {
      type: 'SET_MODE',
      payload: 'fixture',
    })
    state = appReducer(state, {
      type: 'VQA_SEARCH_SUCCESS',
      payload: mockVqaResponse,
    })
    state = appReducer(state, { type: 'CONFIRM_VQA_ANSWER' })

    // In fixture mode, addition is blocked
    state = appReducer(state, { type: 'ADD_VQA_TO_BASKET' })
    expect(state.submissionBasket).toHaveLength(0)
  })

  it('handles taskMode transitions and synchronizes with search actions', () => {
    expect(initialAppState.taskMode).toBe('KIS')

    let state = appReducer(initialAppState, {
      type: 'SET_TASK_MODE',
      payload: 'VQA',
    })
    expect(state.taskMode).toBe('VQA')

    state = appReducer(state, {
      type: 'SET_TASK_MODE',
      payload: 'TRAKE',
    })
    expect(state.taskMode).toBe('TRAKE')

    state = appReducer(state, {
      type: 'KIS_SEARCH_START',
    })
    expect(state.taskMode).toBe('KIS')

    state = appReducer(state, {
      type: 'VQA_SEARCH_START',
    })
    expect(state.taskMode).toBe('VQA')
  })

  it('preserves certified root anchor lineage and accumulates offset upon COMMIT_VQA_FRAME', () => {
    let state = appReducer(initialAppState, {
      type: 'SET_TASK_MODE',
      payload: 'VQA',
    })
    state = appReducer(state, {
      type: 'VQA_SEARCH_SUCCESS',
      payload: mockVqaResponse,
    })

    expect(state.vqaActiveResult?.frame_id).toBe(888)
    expect(state.anchorCandidate?.certified_anchor_frame_id).toBe(888)
    expect(state.anchorCandidate?.anchor_offset).toBe(0)

    // Simulate stepping forward +3
    state = appReducer(state, {
      type: 'EXACT_STEP_START',
      payload: { offset: 3 },
    })
    expect(state.cumulativeOffset).toBe(3)

    // Commit answer frame 891
    state = appReducer(state, {
      type: 'COMMIT_VQA_FRAME',
      payload: {
        frame_id: 891,
        timestamp_ms: 29720,
      },
    })

    // Canonical answer frame updated to 891
    expect(state.vqaActiveResult?.frame_id).toBe(891)
    // Certified root anchor remains 888
    expect(state.vqaActiveResult?.certified_anchor_frame_id).toBe(888)
    expect(state.vqaActiveResult?.anchor_offset).toBe(3)
    expect(state.anchorCandidate?.certified_anchor_frame_id).toBe(888)
    expect(state.anchorCandidate?.anchor_offset).toBe(3)
    expect(state.anchorCandidate?.frame_id).toBe(891)
    // Cumulative offset reset to 0
    expect(state.cumulativeOffset).toBe(0)

    // Subsequent step +2
    state = appReducer(state, {
      type: 'EXACT_STEP_START',
      payload: { offset: 2 },
    })
    expect(state.cumulativeOffset).toBe(2)

    // Commit answer frame 893 (offset accumulates to +5)
    state = appReducer(state, {
      type: 'COMMIT_VQA_FRAME',
      payload: {
        frame_id: 893,
        timestamp_ms: 29800,
      },
    })
    expect(state.vqaActiveResult?.frame_id).toBe(893)
    expect(state.vqaActiveResult?.certified_anchor_frame_id).toBe(888)
    expect(state.vqaActiveResult?.anchor_offset).toBe(5)
    expect(state.anchorCandidate?.certified_anchor_frame_id).toBe(888)
    expect(state.anchorCandidate?.anchor_offset).toBe(5)
    expect(state.cumulativeOffset).toBe(0)
  })

  it('resets exact inspection transient state on candidate switch to isolate proofs', () => {
    let state = appReducer(initialAppState, {
      type: 'VQA_SEARCH_SUCCESS',
      payload: mockVqaResponse,
    })

    // Simulate active inspection on Candidate 1
    state = {
      ...state,
      exactNeighbors: {
        anchor_frame_id: 888,
        steps: [{ offset: 0, frame: { video_id: 'L05_V005', frame_id: 888, timestamp_ms: 29600 } as any, degraded_reason: null }],
        provenance_mode: 'live',
      },
      currentStep: { offset: 0, frame: { video_id: 'L05_V005', frame_id: 888, timestamp_ms: 29600 } as any, degraded_reason: null },
      exactImageBlobUrl: 'blob:test-888',
      exactImageHeaders: { video_id: 'L05_V005', frame_id: 888 } as any,
    }

    // Switch to Candidate 2
    state = appReducer(state, {
      type: 'SELECT_VQA_RESULT',
      payload: mockVqaResult2,
    })

    expect(state.vqaActiveResult?.video_id).toBe('L05_V008')
    expect(state.vqaActiveResult?.frame_id).toBe(1240)
    expect(state.activeCandidate?.video_id).toBe('L05_V008')
    expect(state.activeCandidate?.frame_id).toBe(1240)
    // Transient exact state is cleared
    expect(state.exactNeighbors).toBeNull()
    expect(state.currentStep).toBeNull()
    expect(state.exactImageBlobUrl).toBeNull()
    expect(state.exactImageHeaders).toBeNull()
  })
})
