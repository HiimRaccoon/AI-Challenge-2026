import { describe, it, expect } from 'vitest'
import { appReducer } from '../../src/state/appReducer'
import { initialAppState } from '../../src/state/appState'
import { SearchCandidate } from '../../src/types/contracts'

const mockCandidate1: SearchCandidate = {
  query_id: 'kis-001',
  video_id: 'L21_V001',
  frame_id: 19220,
  timestamp_ms: 640666,
  source: 'fusion',
  rank: 1,
  score: 0.85,
  model_scores: { bge_vl: 0.8, metaclip2: 0.9 },
  model_ranks: { bge_vl: 1, metaclip2: 2 },
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

describe('appReducer state transitions', () => {
  it('initializes deterministically', () => {
    expect(initialAppState.cumulativeOffset).toBe(0)
    expect(initialAppState.candidates).toEqual([])
    expect(initialAppState.activeCandidate).toBeNull()
    expect(initialAppState.anchorCandidate).toBeNull()
  })

  it('updates health status and mode without assuming corpus ready', () => {
    const state = appReducer(initialAppState, {
      type: 'SET_HEALTH',
      payload: {
        health: { status: 'ok', mode: 'live', preprocess_run_id: 'run_v1_batch1' },
      },
    })
    expect(state.tv4Health?.status).toBe('ok')
    expect(state.mode).toBe('live')
    expect(state.preprocessRunId).toBe('run_v1_batch1')
    // Crucial: live process UP alone means PARTIAL readiness, not full corpus ready
    expect(state.readiness).toBe('PARTIAL')
  })

  it('sets offline readiness when health check fails', () => {
    const state = appReducer(initialAppState, {
      type: 'SET_HEALTH',
      payload: { health: null, error: 'Connection refused' },
    })
    expect(state.readiness).toBe('OFFLINE')
    expect(state.healthError).toBe('Connection refused')
  })

  it('bounds Top-K between 1 and 100', () => {
    let state = appReducer(initialAppState, { type: 'SET_TOP_K', payload: 150 })
    expect(state.topK).toBe(100)

    state = appReducer(initialAppState, { type: 'SET_TOP_K', payload: 0 })
    expect(state.topK).toBe(1)
  })

  it('handles KIS search lifecycle', () => {
    let state = appReducer(initialAppState, { type: 'KIS_SEARCH_START' })
    expect(state.isSearching).toBe(true)
    expect(state.searchError).toBeNull()

    state = appReducer(state, {
      type: 'KIS_SEARCH_SUCCESS',
      payload: {
        query_id: 'q-42',
        candidates: [mockCandidate1, mockCandidate2],
      },
    })
    expect(state.isSearching).toBe(false)
    expect(state.candidates.length).toBe(2)
    expect(state.queryId).toBe('q-42')
  })

  it('selecting candidate establishes anchor and resets cumulative offset to 0', () => {
    // Start with a state that had previous stepping
    const steppedState = {
      ...initialAppState,
      cumulativeOffset: 5,
      exactImageBlobUrl: 'blob:old',
    }

    const state = appReducer(steppedState, {
      type: 'SELECT_CANDIDATE',
      payload: mockCandidate1,
    })

    expect(state.activeCandidate).toEqual(mockCandidate1)
    expect(state.anchorCandidate).toEqual(mockCandidate1)
    // CRITICAL: cumulative offset MUST reset to 0
    expect(state.cumulativeOffset).toBe(0)
    expect(state.exactImageBlobUrl).toBeNull()
    expect(state.currentStep).toBeNull()
  })

  it('supports cumulative stepping beyond ±1 (e.g. +1, +2, +3)', () => {
    let state = appReducer(initialAppState, {
      type: 'SELECT_CANDIDATE',
      payload: mockCandidate1,
    })

    // Step to +1
    state = appReducer(state, { type: 'EXACT_STEP_START', payload: { offset: 1 } })
    expect(state.cumulativeOffset).toBe(1)
    expect(state.isStepping).toBe(true)

    state = appReducer(state, {
      type: 'EXACT_STEP_SUCCESS',
      payload: {
        video_id: 'L21_V001',
        anchor_frame_id: 19220,
        steps: [
          {
            offset: 1,
            frame: {
              video_id: 'L21_V001',
              frame_id: 19221,
              timestamp_ms: 640700,
              pts: 9841152,
              time_base: '1/15360',
              preprocess_run_id: 'run_v1_batch1',
              mapping_guaranteed: true,
              submission_selectable: true,
              identity_source: 'certified_run_consecutive_original_decode',
            },
          },
        ],
      },
    })
    expect(state.isStepping).toBe(false)
    expect(state.currentStep?.frame?.frame_id).toBe(19221)

    // Step further to +2
    state = appReducer(state, { type: 'EXACT_STEP_START', payload: { offset: 2 } })
    expect(state.cumulativeOffset).toBe(2)

    state = appReducer(state, {
      type: 'EXACT_STEP_SUCCESS',
      payload: {
        video_id: 'L21_V001',
        anchor_frame_id: 19220,
        steps: [
          {
            offset: 2,
            frame: {
              video_id: 'L21_V001',
              frame_id: 19222,
              timestamp_ms: 640733,
              pts: 9841664,
              time_base: '1/15360',
              preprocess_run_id: 'run_v1_batch1',
              mapping_guaranteed: true,
              submission_selectable: true,
              identity_source: 'certified_run_consecutive_original_decode',
            },
          },
        ],
      },
    })
    expect(state.currentStep?.frame?.frame_id).toBe(19222)
  })
})
