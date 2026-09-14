import { describe, it, expect } from 'vitest'
import { appReducer } from '../../src/state/appReducer'
import { initialAppState } from '../../src/state/appState'
import { FeedbackResponse, SearchCandidate } from '../../src/types/contracts'

const mockCandidate1: SearchCandidate = {
  query_id: 'kis-001',
  video_id: 'L21_V001',
  frame_id: 10690,
  timestamp_ms: 356333,
  source: 'fusion',
  rank: 1,
  score: 0.85,
  preprocess_run_id: 'run_v1_batch1',
}

const mockCandidate2: SearchCandidate = {
  query_id: 'kis-001',
  video_id: 'L21_V002',
  frame_id: 23940,
  timestamp_ms: 798000,
  source: 'fusion',
  rank: 2,
  score: 0.72,
  preprocess_run_id: 'run_v1_batch1',
}

const mockCandidate3: SearchCandidate = {
  query_id: 'kis-001',
  video_id: 'L05_V005',
  frame_id: 888,
  timestamp_ms: 29600,
  source: 'fusion',
  rank: 3,
  score: 0.65,
  preprocess_run_id: 'run_v1_batch1',
}

describe('T028 — Feedback State Model & Reducer Transitions', () => {
  it('initializes feedback state cleanly and deterministically', () => {
    expect(initialAppState.feedbackSessionId).toBeNull()
    expect(initialAppState.feedbackOriginalQuery).toBeNull()
    expect(initialAppState.feedbackRevision).toBe(0)
    expect(initialAppState.feedbackReference).toBeNull()
    expect(initialAppState.feedbackDraftText).toBe('')
    expect(initialAppState.isFeedbackActive).toBe(false)
    expect(initialAppState.isFeedbackPending).toBe(false)
    expect(initialAppState.feedbackError).toBeNull()
    expect(initialAppState.originalCandidates).toEqual([])
    expect(initialAppState.submissionBasket).toEqual([])
  })

  it('preserves original KIS query immutability when starting feedback session', () => {
    // 1. KIS search populates query and original candidates
    let state = appReducer(initialAppState, { type: 'SET_QUERY_TEXT', payload: 'xe buýt màu đỏ' })
    state = appReducer(state, {
      type: 'KIS_SEARCH_SUCCESS',
      payload: {
        query_id: 'q-kis-1',
        candidates: [mockCandidate1, mockCandidate2, mockCandidate3],
      },
    })
    expect(state.queryText).toBe('xe buýt màu đỏ')
    expect(state.candidates.length).toBe(3)
    expect(state.originalCandidates.length).toBe(3)

    // 2. Start Feedback session
    const startResp: FeedbackResponse = {
      session_id: 'sess-fb-001',
      revision: 0,
      candidates: [mockCandidate1, mockCandidate2, mockCandidate3],
      status: 'ok',
      provenance_mode: 'live',
      expires_at_utc: '2026-08-20T00:00:00Z',
    }
    state = appReducer(state, { type: 'FEEDBACK_START_SUCCESS', payload: startResp })

    expect(state.isFeedbackActive).toBe(true)
    expect(state.feedbackSessionId).toBe('sess-fb-001')
    expect(state.feedbackOriginalQuery).toBe('xe buýt màu đỏ')
    expect(state.feedbackRevision).toBe(0)

    // 3. Modifying user draft text does NOT mutate original query
    state = appReducer(state, { type: 'SET_FEEDBACK_DRAFT', payload: 'ở ngã tư ban đêm' })
    expect(state.feedbackDraftText).toBe('ở ngã tư ban đêm')
    expect(state.feedbackOriginalQuery).toBe('xe buýt màu đỏ')
    expect(state.queryText).toBe('xe buýt màu đỏ')
  })

  it('enforces canonical reference candidate identity and revalidation', () => {
    let state = appReducer(initialAppState, {
      type: 'KIS_SEARCH_SUCCESS',
      payload: {
        query_id: 'q-kis-1',
        candidates: [mockCandidate1, mockCandidate2],
      },
    })

    // Setting a candidate from the pool succeeds
    state = appReducer(state, { type: 'SET_FEEDBACK_REFERENCE', payload: mockCandidate2 })
    expect(state.feedbackReference).toEqual(mockCandidate2)
    expect(state.feedbackReference?.video_id).toBe('L21_V002')
    expect(state.feedbackReference?.frame_id).toBe(23940)
    expect(state.feedbackError).toBeNull()

    // Setting an unrendered candidate not in pool fails gracefully with error
    const unrenderedCandidate: SearchCandidate = {
      query_id: 'q-unknown',
      video_id: 'UNKNOWN_VID',
      frame_id: 99999,
      timestamp_ms: 0,
      source: 'fusion',
      rank: 99,
    }
    state = appReducer(state, { type: 'SET_FEEDBACK_REFERENCE', payload: unrenderedCandidate })
    expect(state.feedbackReference).toBeNull()
    expect(state.feedbackError).toMatch(/not in the active candidate pool/i)
  })

  it('preserves separate original vs refined candidate snapshots and monotonic revisions', () => {
    // 1. Initial KIS candidates
    let state = appReducer(initialAppState, { type: 'SET_QUERY_TEXT', payload: 'người đi xe đạp' })
    state = appReducer(state, {
      type: 'KIS_SEARCH_SUCCESS',
      payload: {
        query_id: 'q-kis-1',
        candidates: [mockCandidate1, mockCandidate2, mockCandidate3],
      },
    })

    const initialOriginalCandidates = [...state.originalCandidates]
    expect(initialOriginalCandidates[0].video_id).toBe('L21_V001')

    // 2. Start Feedback (Revision 0)
    state = appReducer(state, {
      type: 'FEEDBACK_START_SUCCESS',
      payload: {
        session_id: 'sess-fb-mono',
        revision: 0,
        candidates: [mockCandidate1, mockCandidate2, mockCandidate3],
        status: 'ok',
      },
    })
    expect(state.feedbackRevision).toBe(0)

    // 3. Refine: Candidate 2 promoted to #1 (Revision 1)
    const refinedCandidates = [
      { ...mockCandidate2, rank: 1 },
      { ...mockCandidate1, rank: 2 },
      { ...mockCandidate3, rank: 3 },
    ]
    state = appReducer(state, {
      type: 'FEEDBACK_REFINE_SUCCESS',
      payload: {
        session_id: 'sess-fb-mono',
        revision: 1,
        candidates: refinedCandidates,
        status: 'ok',
      },
    })

    expect(state.feedbackRevision).toBe(1)
    // Refined view is updated
    expect(state.candidates[0].video_id).toBe('L21_V002')
    expect(state.candidates[0].rank).toBe(1)
    // Original snapshot remains completely intact!
    expect(state.originalCandidates[0].video_id).toBe('L21_V001')
    expect(state.originalCandidates[0].rank).toBe(1)
    expect(state.originalCandidates.length).toBe(3)

    // 4. Undo: (Revision advances to 2, ranking rolls back)
    const undoCandidates = [
      { ...mockCandidate1, rank: 1 },
      { ...mockCandidate2, rank: 2 },
      { ...mockCandidate3, rank: 3 },
    ]
    state = appReducer(state, {
      type: 'FEEDBACK_UNDO_SUCCESS',
      payload: {
        session_id: 'sess-fb-mono',
        revision: 2,
        candidates: undoCandidates,
        status: 'ok',
      },
    })
    expect(state.feedbackRevision).toBe(2)
    expect(state.candidates[0].video_id).toBe('L21_V001')

    // 5. Reset: (Revision advances to 3, restored to baseline)
    state = appReducer(state, {
      type: 'FEEDBACK_RESET_SUCCESS',
      payload: {
        session_id: 'sess-fb-mono',
        revision: 3,
        candidates: initialOriginalCandidates,
        status: 'ok',
      },
    })
    expect(state.feedbackRevision).toBe(3)
    expect(state.candidates[0].video_id).toBe('L21_V001')
    expect(state.feedbackDraftText).toBe('')
    expect(state.feedbackReference).toBeNull()
  })

  it('exiting feedback restores original KIS candidate list', () => {
    let state = appReducer(initialAppState, {
      type: 'KIS_SEARCH_SUCCESS',
      payload: {
        query_id: 'q-kis-1',
        candidates: [mockCandidate1, mockCandidate2],
      },
    })

    state = appReducer(state, {
      type: 'FEEDBACK_REFINE_SUCCESS',
      payload: {
        session_id: 'sess-fb-exit',
        revision: 1,
        candidates: [{ ...mockCandidate2, rank: 1 }, { ...mockCandidate1, rank: 2 }],
        status: 'ok',
      },
    })
    expect(state.candidates[0].video_id).toBe('L21_V002')

    // Clear feedback
    state = appReducer(state, { type: 'FEEDBACK_CLEAR' })
    expect(state.isFeedbackActive).toBe(false)
    expect(state.feedbackSessionId).toBeNull()
    expect(state.candidates[0].video_id).toBe('L21_V001')
  })

  it('maintains absolute Basket non-mutation during all feedback operations', () => {
    // 1. Add an item to basket
    let state = appReducer(initialAppState, {
      type: 'ADD_TO_BASKET',
      payload: {
        video_id: 'L21_V001',
        frame_id: 10690,
        timestamp_ms: 356333,
        added_at_utc: '2026-08-19T07:00:00Z',
      },
    })
    expect(state.submissionBasket.length).toBe(1)
    expect(state.submissionBasket[0].video_id).toBe('L21_V001')

    // 2. Execute Feedback Start
    state = appReducer(state, {
      type: 'FEEDBACK_START_SUCCESS',
      payload: {
        session_id: 'sess-basket-iso',
        revision: 0,
        candidates: [mockCandidate1, mockCandidate2],
        status: 'ok',
      },
    })
    expect(state.submissionBasket.length).toBe(1)

    // 3. Execute Feedback Refine
    state = appReducer(state, {
      type: 'FEEDBACK_REFINE_SUCCESS',
      payload: {
        session_id: 'sess-basket-iso',
        revision: 1,
        candidates: [mockCandidate2, mockCandidate1],
        status: 'ok',
      },
    })
    expect(state.submissionBasket.length).toBe(1)

    // 4. Execute Feedback Undo
    state = appReducer(state, {
      type: 'FEEDBACK_UNDO_SUCCESS',
      payload: {
        session_id: 'sess-basket-iso',
        revision: 2,
        candidates: [mockCandidate1, mockCandidate2],
        status: 'ok',
      },
    })
    expect(state.submissionBasket.length).toBe(1)

    // 5. Execute Feedback Reset
    state = appReducer(state, {
      type: 'FEEDBACK_RESET_SUCCESS',
      payload: {
        session_id: 'sess-basket-iso',
        revision: 3,
        candidates: [mockCandidate1, mockCandidate2],
        status: 'ok',
      },
    })
    expect(state.submissionBasket.length).toBe(1)

    // 6. Execute Feedback Clear
    state = appReducer(state, { type: 'FEEDBACK_CLEAR' })
    expect(state.submissionBasket.length).toBe(1)
    expect(state.submissionBasket[0].frame_id).toBe(10690)
  })

  it('ensures refined candidate is fully compatible with Inspection candidate selection', () => {
    let state = appReducer(initialAppState, {
      type: 'FEEDBACK_REFINE_SUCCESS',
      payload: {
        session_id: 'sess-inspect-comp',
        revision: 1,
        candidates: [mockCandidate2, mockCandidate1],
        status: 'ok',
      },
    })

    // Select refined candidate for Inspection
    const refinedCandidate = state.candidates[0]
    state = appReducer(state, { type: 'SELECT_CANDIDATE', payload: refinedCandidate })

    expect(state.activeCandidate).toEqual(refinedCandidate)
    expect(state.anchorCandidate).toEqual(refinedCandidate)
    expect(state.activeCandidate?.video_id).toBe('L21_V002')
    expect(state.activeCandidate?.frame_id).toBe(23940)
    // Anchor establishes cumulativeOffset = 0 without local frame arithmetic
    expect(state.cumulativeOffset).toBe(0)
  })

  it('handles feedback mutation failures without state corruption', () => {
    let state = appReducer(initialAppState, {
      type: 'FEEDBACK_START_SUCCESS',
      payload: {
        session_id: 'sess-err',
        revision: 0,
        candidates: [mockCandidate1],
        status: 'ok',
      },
    })

    // Refine failure
    state = appReducer(state, { type: 'FEEDBACK_REFINE_PENDING' })
    expect(state.isFeedbackPending).toBe(true)

    state = appReducer(state, {
      type: 'FEEDBACK_REFINE_FAILURE',
      payload: 'HTTP 409: RevisionConflict: session revision is stale',
    })
    expect(state.isFeedbackPending).toBe(false)
    expect(state.feedbackError).toBe('HTTP 409: RevisionConflict: session revision is stale')
    // Revision and existing candidate snapshot remain preserved
    expect(state.feedbackRevision).toBe(0)
    expect(state.candidates.length).toBe(1)
  })

  it('accurately tracks active feedback event count throughout refine, undo, reset lifecycle', () => {
    let state = appReducer(initialAppState, {
      type: 'FEEDBACK_START_SUCCESS',
      payload: {
        session_id: 'sess-count-test',
        revision: 0,
        active_feedback_count: 0,
        max_active_feedback_events: 5,
        candidates: [mockCandidate1],
        status: 'ok',
      },
    })
    expect(state.feedbackActiveCount).toBe(0)
    expect(state.feedbackMaxEvents).toBe(5)

    // 1..5 Refines
    for (let i = 1; i <= 5; i++) {
      state = appReducer(state, {
        type: 'FEEDBACK_REFINE_SUCCESS',
        payload: {
          session_id: 'sess-count-test',
          revision: i,
          active_feedback_count: i,
          max_active_feedback_events: 5,
          candidates: [mockCandidate1],
          status: 'ok',
        },
      })
      expect(state.feedbackActiveCount).toBe(i)
    }

    // 6th Refine Failure with limit message sets count to 5 and friendly error
    state = appReducer(state, {
      type: 'FEEDBACK_REFINE_FAILURE',
      payload: 'Maximum 5 active refinements reached. Undo or Reset to continue.',
    })
    expect(state.feedbackActiveCount).toBe(5)
    expect(state.feedbackError).toBe('Maximum 5 active refinements reached. Undo or Reset to continue.')

    // Undo reduces count to 4 and clears error
    state = appReducer(state, {
      type: 'FEEDBACK_UNDO_SUCCESS',
      payload: {
        session_id: 'sess-count-test',
        revision: 6,
        active_feedback_count: 4,
        max_active_feedback_events: 5,
        candidates: [mockCandidate1],
        status: 'ok',
      },
    })
    expect(state.feedbackActiveCount).toBe(4)
    expect(state.feedbackError).toBeNull()

    // Reset clears count to 0 and clears error
    state = appReducer(state, {
      type: 'FEEDBACK_RESET_SUCCESS',
      payload: {
        session_id: 'sess-count-test',
        revision: 7,
        active_feedback_count: 0,
        max_active_feedback_events: 5,
        candidates: [mockCandidate1],
        status: 'ok',
      },
    })
    expect(state.feedbackActiveCount).toBe(0)
    expect(state.feedbackError).toBeNull()
  })
})
