import { describe, it, expect } from 'vitest'
import { appReducer } from '../../src/state/appReducer'
import { initialAppState } from '../../src/state/appState'
import { TrakeResponse, ExactFrameProof } from '../../src/types/contracts'

const mockTrakeResponseSuccess: TrakeResponse = {
  query_id: 'trake-001',
  result: {
    video_id: 'L10_V010',
    frame_ids: [101, 156, 203, 251],
    event_scores: [0.9, 0.7, 0.85, 0.6],
    aggregate_score: 3.05,
    preprocess_run_id: 'run_v1_batch1',
  },
}

const mockProof158: ExactFrameProof = {
  video_id: 'L10_V010',
  frame_id: 158,
  timestamp_ms: 5266,
  pts: 158000,
  time_base: '1/30000',
  preprocess_run_id: 'run_v1_batch1',
  mapping_guaranteed: true,
  submission_selectable: true,
  identity_source: 'certified_frame',
}

const mockProofOtherVideo: ExactFrameProof = {
  video_id: 'L99_V999',
  frame_id: 500,
  timestamp_ms: 16666,
  pts: 500000,
  time_base: '1/30000',
  preprocess_run_id: 'run_v1_batch1',
  mapping_guaranteed: true,
  submission_selectable: true,
  identity_source: 'certified_frame',
}

describe('T032 — TRAKE Frontend State Model & Invariants', () => {
  it('preserves input semantic event order and initializes exactly N event slots', () => {
    const events = ['Giậm nhảy', 'Bay qua xà', 'Tiếp đất', 'Đứng dậy']

    const state = appReducer(initialAppState, {
      type: 'SET_TRAKE_EVENTS',
      payload: events,
    })

    expect(state.trakeEvents).toEqual(['Giậm nhảy', 'Bay qua xà', 'Tiếp đất', 'Đứng dậy'])
    expect(state.trakeSlots).toHaveLength(4)

    // Semantic order invariant: slot 0..3 matches events 0..3 exactly
    expect(state.trakeSlots[0]).toEqual(
      expect.objectContaining({ event_index: 0, event_label: 'Giậm nhảy', frame_id: null, locked: false })
    )
    expect(state.trakeSlots[1]).toEqual(
      expect.objectContaining({ event_index: 1, event_label: 'Bay qua xà', frame_id: null, locked: false })
    )
    expect(state.trakeSlots[2]).toEqual(
      expect.objectContaining({ event_index: 2, event_label: 'Tiếp đất', frame_id: null, locked: false })
    )
    expect(state.trakeSlots[3]).toEqual(
      expect.objectContaining({ event_index: 3, event_label: 'Đứng dậy', frame_id: null, locked: false })
    )
    expect(state.trakeValidationStatus).toBe('incomplete')
  })

  it('preserves distinct positions for duplicate semantic event labels without collapsing', () => {
    const events = ['Quay đầu', 'Tăng tốc', 'Quay đầu']

    const state = appReducer(initialAppState, {
      type: 'SET_TRAKE_EVENTS',
      payload: events,
    })

    expect(state.trakeSlots).toHaveLength(3)
    expect(state.trakeSlots[0].event_index).toBe(0)
    expect(state.trakeSlots[0].event_label).toBe('Quay đầu')
    expect(state.trakeSlots[2].event_index).toBe(2)
    expect(state.trakeSlots[2].event_label).toBe('Quay đầu')
  })

  it('populates single video hypothesis and aligns frames to semantic slots', () => {
    let state = appReducer(initialAppState, {
      type: 'SET_TRAKE_EVENTS',
      payload: ['Giậm nhảy', 'Bay qua xà', 'Tiếp đất', 'Đứng dậy'],
    })

    state = appReducer(state, {
      type: 'TRAKE_SEARCH_SUCCESS',
      payload: mockTrakeResponseSuccess,
    })

    expect(state.trakeVideoId).toBe('L10_V010')
    expect(state.trakeAggregateScore).toBe(3.05)
    expect(state.trakeSlots).toHaveLength(4)

    // Semantic alignment invariant: frame_ids mapped directly to event indices
    expect(state.trakeSlots[0].frame_id).toBe(101)
    expect(state.trakeSlots[0].video_id).toBe('L10_V010')
    expect(state.trakeSlots[1].frame_id).toBe(156)
    expect(state.trakeSlots[2].frame_id).toBe(203)
    expect(state.trakeSlots[3].frame_id).toBe(251)
    expect(state.trakeValidationStatus).toBe('valid')
  })

  it('lock protection: locked slots are not overwritten by subsequent hypothesis refresh', () => {
    let state = appReducer(initialAppState, {
      type: 'SET_TRAKE_EVENTS',
      payload: ['Giậm nhảy', 'Bay qua xà', 'Tiếp đất', 'Đứng dậy'],
    })

    state = appReducer(state, {
      type: 'TRAKE_SEARCH_SUCCESS',
      payload: mockTrakeResponseSuccess,
    })

    // Operator locks Slot 1 (Bay qua xà -> Frame 156)
    state = appReducer(state, {
      type: 'LOCK_TRAKE_SLOT',
      payload: { event_index: 1 },
    })
    expect(state.trakeSlots[1].locked).toBe(true)

    // New hypothesis response arrives with different frames for L10_V010
    const newHypothesis: TrakeResponse = {
      query_id: 'trake-002',
      result: {
        video_id: 'L10_V010',
        frame_ids: [110, 999, 220, 260],
        event_scores: [0.95, 0.95, 0.9, 0.85],
        aggregate_score: 3.6,
        preprocess_run_id: 'run_v1_batch1',
      },
    }

    state = appReducer(state, {
      type: 'TRAKE_SEARCH_SUCCESS',
      payload: newHypothesis,
    })

    // Unlocked slots updated to new hypothesis
    expect(state.trakeSlots[0].frame_id).toBe(110)
    expect(state.trakeSlots[2].frame_id).toBe(220)
    expect(state.trakeSlots[3].frame_id).toBe(260)

    // P0 INVARIANT: Locked slot 1 was NOT overwritten by frame 999; remains 156
    expect(state.trakeSlots[1].frame_id).toBe(156)
    expect(state.trakeSlots[1].locked).toBe(true)

    // Explicit unlock allows replacement
    state = appReducer(state, {
      type: 'UNLOCK_TRAKE_SLOT',
      payload: { event_index: 1 },
    })
    expect(state.trakeSlots[1].locked).toBe(false)

    state = appReducer(state, {
      type: 'TRAKE_SEARCH_SUCCESS',
      payload: newHypothesis,
    })
    expect(state.trakeSlots[1].frame_id).toBe(999)
  })

  it('manual correction: attaches authoritative exact-frame proof without altering slot position or reordering', () => {
    let state = appReducer(initialAppState, {
      type: 'SET_TRAKE_EVENTS',
      payload: ['Giậm nhảy', 'Bay qua xà', 'Tiếp đất', 'Đứng dậy'],
    })

    state = appReducer(state, {
      type: 'TRAKE_SEARCH_SUCCESS',
      payload: mockTrakeResponseSuccess,
    })

    // Operator inspects slot 1 ("Bay qua xà") and steps to frame 158 via exact neighbor service
    state = appReducer(state, {
      type: 'CORRECT_TRAKE_SLOT',
      payload: {
        event_index: 1,
        frame_id: 158,
        timestamp_ms: 5266,
        proof: mockProof158,
      },
    })

    // Slot 1 updated with authoritative proof
    expect(state.trakeSlots[1].frame_id).toBe(158)
    expect(state.trakeSlots[1].timestamp_ms).toBe(5266)
    expect(state.trakeSlots[1].exact_proof?.pts).toBe(158000)
    expect(state.trakeSlots[1].event_index).toBe(1)
    expect(state.trakeSlots[1].event_label).toBe('Bay qua xà')

    // Other slots completely intact
    expect(state.trakeSlots[0].frame_id).toBe(101)
    expect(state.trakeSlots[2].frame_id).toBe(203)
    expect(state.trakeSlots[3].frame_id).toBe(251)
    expect(state.trakeValidationStatus).toBe('valid')
  })

  it('single-video invariant: rejects mixed video identities across event slots', () => {
    let state = appReducer(initialAppState, {
      type: 'SET_TRAKE_EVENTS',
      payload: ['Giậm nhảy', 'Bay qua xà', 'Tiếp đất', 'Đứng dậy'],
    })

    state = appReducer(state, {
      type: 'TRAKE_SEARCH_SUCCESS',
      payload: mockTrakeResponseSuccess,
    })

    // Corrupting slot 2 with a frame from another video
    state = appReducer(state, {
      type: 'CORRECT_TRAKE_SLOT',
      payload: {
        event_index: 2,
        frame_id: 500,
        timestamp_ms: 16666,
        proof: mockProofOtherVideo,
      },
    })

    expect(state.trakeSlots[2].validation_status).toBe('incompatible_video')
    expect(state.trakeValidationStatus).toBe('mixed_video')

    // Attempting to add mixed-video chain to basket is REJECTED
    state = appReducer(state, { type: 'ADD_TRAKE_TO_BASKET' })
    expect(state.submissionBasket).toHaveLength(0)
  })

  it('pre-basket validation: only complete, valid N-event chains can be added to basket', () => {
    let state = appReducer(initialAppState, {
      type: 'SET_TRAKE_EVENTS',
      payload: ['Giậm nhảy', 'Bay qua xà', 'Tiếp đất', 'Đứng dậy'],
    })

    // 1. Incomplete chain (no search yet) -> REJECTED
    state = appReducer(state, { type: 'ADD_TRAKE_TO_BASKET' })
    expect(state.submissionBasket).toHaveLength(0)

    // 2. Search success -> complete valid chain, but unlocked (ALIGNED, not yet operator READY)
    state = appReducer(state, {
      type: 'TRAKE_SEARCH_SUCCESS',
      payload: mockTrakeResponseSuccess,
    })
    expect(state.trakeValidationStatus).toBe('valid')

    // 3. Unlocked valid chain cannot enter basket
    state = appReducer(state, { type: 'ADD_TRAKE_TO_BASKET' })
    expect(state.submissionBasket).toHaveLength(0)

    // 4. Partially locked valid chain cannot enter basket
    state = appReducer(state, { type: 'LOCK_TRAKE_SLOT', payload: { event_index: 0 } })
    state = appReducer(state, { type: 'LOCK_TRAKE_SLOT', payload: { event_index: 1 } })
    state = appReducer(state, { type: 'ADD_TRAKE_TO_BASKET' })
    expect(state.submissionBasket).toHaveLength(0)

    // 5. All slots locked (operator READY) -> SUCCESS
    state = appReducer(state, { type: 'LOCK_TRAKE_SLOT', payload: { event_index: 2 } })
    state = appReducer(state, { type: 'LOCK_TRAKE_SLOT', payload: { event_index: 3 } })
    state = appReducer(state, { type: 'ADD_TRAKE_TO_BASKET' })
    expect(state.submissionBasket).toHaveLength(1)
    expect(state.submissionBasket[0]).toEqual(
      expect.objectContaining({
        video_id: 'L10_V010',
        frame_id: 101,
        task: 'TRAKE',
        frame_ids: [101, 156, 203, 251],
        event_labels: ['Giậm nhảy', 'Bay qua xà', 'Tiếp đất', 'Đứng dậy'],
      })
    )
    expect(state.submissionBasket[0].answer).toBeUndefined()
  })

  it('fixture safety: prevents adding TRAKE chains to real submission basket in fixture mode', () => {
    let state = appReducer(initialAppState, {
      type: 'SET_MODE',
      payload: 'fixture',
    })

    state = appReducer(state, {
      type: 'SET_TRAKE_EVENTS',
      payload: ['Giậm nhảy', 'Bay qua xà', 'Tiếp đất', 'Đứng dậy'],
    })

    state = appReducer(state, {
      type: 'TRAKE_SEARCH_SUCCESS',
      payload: mockTrakeResponseSuccess,
    })

    // In fixture mode, addition is blocked
    state = appReducer(state, { type: 'ADD_TRAKE_TO_BASKET' })
    expect(state.submissionBasket).toHaveLength(0)
  })

  it('basket isolation: search, locks, selections, corrections do NOT mutate submission basket', () => {
    let state = appReducer(initialAppState, {
      type: 'SET_TRAKE_EVENTS',
      payload: ['Giậm nhảy', 'Bay qua xà', 'Tiếp đất', 'Đứng dậy'],
    })
    expect(state.submissionBasket).toHaveLength(0)

    state = appReducer(state, { type: 'TRAKE_SEARCH_START' })
    expect(state.submissionBasket).toHaveLength(0)

    state = appReducer(state, {
      type: 'TRAKE_SEARCH_SUCCESS',
      payload: mockTrakeResponseSuccess,
    })
    expect(state.submissionBasket).toHaveLength(0)

    state = appReducer(state, {
      type: 'LOCK_TRAKE_SLOT',
      payload: { event_index: 0 },
    })
    expect(state.submissionBasket).toHaveLength(0)

    state = appReducer(state, {
      type: 'CORRECT_TRAKE_SLOT',
      payload: { event_index: 1, frame_id: 158, proof: mockProof158 },
    })
    expect(state.submissionBasket).toHaveLength(0)

    state = appReducer(state, {
      type: 'SELECT_TRAKE_SLOT',
      payload: 2,
    })
    expect(state.submissionBasket).toHaveLength(0)
  })

  it('handles empty alignment result truthfully without fabricating frames', () => {
    let state = appReducer(initialAppState, {
      type: 'SET_TRAKE_EVENTS',
      payload: ['Event A', 'Event B'],
    })

    state = appReducer(state, {
      type: 'TRAKE_SEARCH_SUCCESS',
      payload: {
        query_id: 'trake-none',
        result: null,
        message: 'no monotonic alignment found',
      },
    })

    expect(state.trakeVideoId).toBeNull()
    expect(state.trakeSlots[0].frame_id).toBeNull()
    expect(state.trakeSlots[1].frame_id).toBeNull()
    expect(state.trakeValidationStatus).toBe('empty')
    expect(state.trakeError).toBe('no monotonic alignment found')

    state = appReducer(state, { type: 'ADD_TRAKE_TO_BASKET' })
    expect(state.submissionBasket).toHaveLength(0)
  })

  it('mode isolation: TRAKE error does not pollute KIS or Q&A error state on mode switch', () => {
    let state = appReducer(initialAppState, {
      type: 'SET_TASK_MODE',
      payload: 'TRAKE',
    })

    state = appReducer(state, {
      type: 'TRAKE_SEARCH_FAILURE',
      payload: 'no monotonic alignment found',
    })

    expect(state.trakeError).toBe('no monotonic alignment found')
    expect(state.searchError).toBeNull()
    expect(state.vqaError).toBeNull()

    // Switch to VQA
    state = appReducer(state, {
      type: 'SET_TASK_MODE',
      payload: 'VQA',
    })
    expect(state.taskMode).toBe('VQA')
    expect(state.trakeError).toBe('no monotonic alignment found') // Preserved in its own scope
    expect(state.vqaError).toBeNull() // Destination mode is clean

    // Switch to KIS
    state = appReducer(state, {
      type: 'SET_TASK_MODE',
      payload: 'KIS',
    })
    expect(state.taskMode).toBe('KIS')
    expect(state.searchError).toBeNull() // Destination mode is clean
  })

  it('in-flight response isolation: stale TRAKE response after mode switch does not overwrite target mode active candidate', () => {
    // 1. User is in KIS mode with a valid KIS candidate
    const mockKisCandidate = {
      query_id: 'kis-001',
      video_id: 'L21_V001',
      frame_id: 10690,
      timestamp_ms: 356333,
      source: 'fusion',
      rank: 1,
    }

    let state = appReducer(initialAppState, {
      type: 'KIS_SEARCH_SUCCESS',
      payload: {
        query_id: 'kis-001',
        candidates: [mockKisCandidate],
      },
    })
    expect(state.activeCandidate?.frame_id).toBe(10690)

    // 2. In-flight TRAKE response arrives while taskMode is still KIS
    state = appReducer(state, {
      type: 'TRAKE_SEARCH_SUCCESS',
      payload: mockTrakeResponseSuccess,
    })

    // Active candidate MUST remain KIS candidate (10690), NOT overwritten by TRAKE (101)
    expect(state.activeCandidate?.frame_id).toBe(10690)
    expect(state.activeCandidate?.video_id).toBe('L21_V001')

    // TRAKE slots are safely updated in their own state
    expect(state.trakeSlots[0].frame_id).toBe(101)
  })

  it('canonical identity & certified anchor: preserves authoritative timestamps_ms and root lineage', () => {
    let state = appReducer(initialAppState, {
      type: 'SET_TASK_MODE',
      payload: 'TRAKE',
    })

    state = appReducer(state, {
      type: 'SET_TRAKE_EVENTS',
      payload: ['Giậm nhảy', 'Bay qua xà'],
    })

    const trakeResWithTimestamps: TrakeResponse = {
      query_id: 'trake-ts-01',
      result: {
        video_id: 'L25_V084',
        frame_ids: [58779, 58900],
        timestamps_ms: [2351160, 2356000],
        event_scores: [0.92, 0.88],
        aggregate_score: 1.8,
        preprocess_run_id: 'run_v1_batch1',
      },
    }

    state = appReducer(state, {
      type: 'TRAKE_SEARCH_SUCCESS',
      payload: trakeResWithTimestamps,
    })

    // Assert canonical timestamp_ms is preserved on slot 0 and slot 1
    expect(state.trakeSlots[0].frame_id).toBe(58779)
    expect(state.trakeSlots[0].timestamp_ms).toBe(2351160)
    expect(state.trakeSlots[0].certified_anchor_frame_id).toBe(58779)
    expect(state.trakeSlots[0].certified_anchor_timestamp_ms).toBe(2351160)
    expect(state.trakeSlots[0].anchor_offset).toBe(0)

    expect(state.trakeSlots[1].frame_id).toBe(58900)
    expect(state.trakeSlots[1].timestamp_ms).toBe(2356000)

    // Active candidate should have authoritative timestamp_ms
    expect(state.activeCandidate?.timestamp_ms).toBe(2351160)
    expect(state.activeCandidate?.certified_anchor_frame_id).toBe(58779)

    // Select Slot 1
    state = appReducer(state, {
      type: 'SELECT_TRAKE_SLOT',
      payload: 1,
    })
    expect(state.activeCandidate?.frame_id).toBe(58900)
    expect(state.activeCandidate?.timestamp_ms).toBe(2356000)
    expect(state.activeCandidate?.certified_anchor_frame_id).toBe(58900)
    expect(state.activeCandidate?.certified_anchor_timestamp_ms).toBe(2356000)

    // Operator steps +2 from root anchor 58900 (frame 58902) and commits
    state = { ...state, cumulativeOffset: 2 }
    state = appReducer(state, {
      type: 'CORRECT_TRAKE_SLOT',
      payload: {
        event_index: 1,
        frame_id: 58902,
        timestamp_ms: 2356080,
      },
    })

    expect(state.trakeSlots[1].frame_id).toBe(58902)
    expect(state.trakeSlots[1].timestamp_ms).toBe(2356080)
    expect(state.trakeSlots[1].certified_anchor_frame_id).toBe(58900) // Certified root preserved
    expect(state.trakeSlots[1].anchor_offset).toBe(2) // Accumulated persistent offset
    expect(state.cumulativeOffset).toBe(0) // Reset for next stepping
  })

  it('stale response isolation: start TRAKE -> switch to Q&A with no Q&A candidate selected -> TRAKE success arrives -> Q&A active/inspection state remains untouched (null)', () => {
    // 1. Initial app state in TRAKE mode starts search
    let state = appReducer(initialAppState, {
      type: 'SET_TASK_MODE',
      payload: 'TRAKE',
    })
    state = appReducer(state, {
      type: 'TRAKE_SEARCH_START',
    })

    // 2. Operator switches to Q&A (VQA) before TRAKE finishes
    // Q&A has had no search yet, so activeCandidate and anchorCandidate are null
    state = appReducer(state, {
      type: 'SET_TASK_MODE',
      payload: 'VQA',
    })
    expect(state.taskMode).toBe('VQA')
    expect(state.activeCandidate).toBeNull()
    expect(state.anchorCandidate).toBeNull()

    // 3. Stale TRAKE response arrives while operator is in Q&A
    state = appReducer(state, {
      type: 'TRAKE_SEARCH_SUCCESS',
      payload: mockTrakeResponseSuccess,
    })

    // 4. CRITICAL INVARIANT: Destination mode (Q&A) active and anchor candidates MUST remain null!
    expect(state.activeCandidate).toBeNull()
    expect(state.anchorCandidate).toBeNull()
    expect(state.exactNeighbors).toBeNull()
    expect(state.currentStep).toBeNull()
    expect(state.exactImageBlobUrl).toBeNull()

    // 5. Stale TRAKE failure arrives -> destination Q&A error remains clean
    state = appReducer(state, {
      type: 'TRAKE_SEARCH_FAILURE',
      payload: 'TRAKE timeout error',
    })
    expect(state.vqaError).toBeNull()
    expect(state.searchError).toBeNull()
    expect(state.trakeError).toBe('TRAKE timeout error')
  })

  it('re-entry proof lineage: inspect event -> step +N -> commit -> leave -> re-enter same event preserves certified root anchor and accumulated offset', () => {
    let state = appReducer(initialAppState, {
      type: 'SET_TASK_MODE',
      payload: 'TRAKE',
    })
    state = appReducer(state, {
      type: 'SET_TRAKE_EVENTS',
      payload: ['Giậm nhảy', 'Bay qua xà'],
    })
    state = appReducer(state, {
      type: 'TRAKE_SEARCH_SUCCESS',
      payload: {
        query_id: 'trake-lineage-01',
        result: {
          video_id: 'L25_V084',
          frame_ids: [58779, 58900],
          timestamps_ms: [2351160, 2356000],
          event_scores: [0.95, 0.9],
          aggregate_score: 1.85,
          preprocess_run_id: 'run_v1_batch1',
        },
      },
    })

    // 1. Select and inspect Event #0 (coarse frame 58779)
    state = appReducer(state, { type: 'SELECT_TRAKE_SLOT', payload: 0 })
    expect(state.activeCandidate?.frame_id).toBe(58779)
    expect(state.anchorCandidate?.certified_anchor_frame_id).toBe(58779)
    expect(state.anchorCandidate?.anchor_offset).toBe(0)

    // 2. Operator steps +3 from coarse anchor (frame 58782, ts 2351280)
    state = { ...state, cumulativeOffset: 3 }
    state = appReducer(state, {
      type: 'CORRECT_TRAKE_SLOT',
      payload: {
        event_index: 0,
        frame_id: 58782,
        timestamp_ms: 2351280,
      },
    })

    // Assert slot 0 state after commit
    expect(state.trakeSlots[0].frame_id).toBe(58782)
    expect(state.trakeSlots[0].timestamp_ms).toBe(2351280)
    expect(state.trakeSlots[0].certified_anchor_frame_id).toBe(58779)
    expect(state.trakeSlots[0].anchor_offset).toBe(3)
    expect(state.cumulativeOffset).toBe(0)

    // 3. Operator switches to Event #1
    state = appReducer(state, { type: 'SELECT_TRAKE_SLOT', payload: 1 })
    expect(state.activeCandidate?.frame_id).toBe(58900)
    expect(state.anchorCandidate?.certified_anchor_frame_id).toBe(58900)
    expect(state.anchorCandidate?.anchor_offset).toBe(0)

    // 4. Operator returns and re-selects corrected Event #0
    state = appReducer(state, { type: 'SELECT_TRAKE_SLOT', payload: 0 })
    expect(state.activeCandidate?.frame_id).toBe(58782)
    expect(state.activeCandidate?.timestamp_ms).toBe(2351280)
    // CRITICAL INVARIANT: Re-entering MUST reconstruct root certified anchor 58779 with offset 3
    expect(state.anchorCandidate?.certified_anchor_frame_id).toBe(58779)
    expect(state.anchorCandidate?.certified_anchor_timestamp_ms).toBe(2351160)
    expect(state.anchorCandidate?.anchor_offset).toBe(3)
    expect(state.cumulativeOffset).toBe(0)

    // 5. Operator can step again (+2 more from 58782 -> offset +5 from root anchor 58779)
    state = { ...state, cumulativeOffset: 2 }
    state = appReducer(state, {
      type: 'CORRECT_TRAKE_SLOT',
      payload: {
        event_index: 0,
        frame_id: 58784,
        timestamp_ms: 2351360,
      },
    })
    expect(state.trakeSlots[0].frame_id).toBe(58784)
    expect(state.trakeSlots[0].certified_anchor_frame_id).toBe(58779)
    expect(state.trakeSlots[0].anchor_offset).toBe(5)
    expect(state.cumulativeOffset).toBe(0)
  })
})
