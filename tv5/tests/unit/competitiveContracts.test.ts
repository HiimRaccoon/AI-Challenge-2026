import { describe, expect, it } from 'vitest'
import { appReducer } from '../../src/state/appReducer'
import { initialAppState } from '../../src/state/appState'
import { TrakeResponse } from '../../src/types/contracts'


describe('competitive QA/TRAKE additive contracts', () => {
  it('stores Top-K TRAKE hypotheses and bounded event alternatives without breaking legacy result', () => {
    const first = {
      rank: 1,
      video_id: 'V1',
      frame_ids: [10, 20],
      timestamps_ms: [100, 200],
      event_scores: [0.9, 0.8],
      aggregate_score: 1.7,
      preprocess_run_id: 'run-1',
      events: [
        {
          event_index: 0,
          frame_id: 10,
          timestamp_ms: 100,
          score: 0.9,
          alternatives: [{ frame_id: 11, timestamp_ms: 120, score: 0.75 }],
        },
        {
          event_index: 1,
          frame_id: 20,
          timestamp_ms: 200,
          score: 0.8,
          alternatives: [],
        },
      ],
    }
    const second = {
      ...first,
      rank: 2,
      frame_ids: [11, 20],
      timestamps_ms: [120, 200],
      aggregate_score: 1.55,
      events: [
        { ...first.events[0], frame_id: 11, timestamp_ms: 120, alternatives: [] },
        first.events[1],
      ],
    }
    const response: TrakeResponse = {
      query_id: 'trake-top-k',
      result: first,
      results: [first, second],
    }

    let state = appReducer(initialAppState, {
      type: 'SET_TRAKE_EVENTS',
      payload: ['event 1', 'event 2'],
    })
    state = appReducer(state, { type: 'TRAKE_SEARCH_SUCCESS', payload: response })

    expect(state.trakeResults).toHaveLength(2)
    expect(state.trakeSlots[0].frame_id).toBe(10)
    expect(state.trakeSlots[0].alternatives?.[0].frame_id).toBe(11)

    state = appReducer(state, { type: 'SELECT_TRAKE_HYPOTHESIS', payload: 1 })
    expect(state.trakeActiveHypothesisIndex).toBe(1)
    expect(state.trakeSlots[0].frame_id).toBe(11)
    expect(state.trakeAggregateScore).toBe(1.55)
  })

  it('applies a valid event alternative and protects locked slots', () => {
    const result = {
      rank: 1,
      video_id: 'V1',
      frame_ids: [10, 20],
      timestamps_ms: [100, 200],
      event_scores: [0.9, 0.8],
      aggregate_score: 1.7,
      preprocess_run_id: 'run-1',
      events: [
        {
          event_index: 0,
          frame_id: 10,
          timestamp_ms: 100,
          score: 0.9,
          alternatives: [{ frame_id: 11, timestamp_ms: 120, score: 0.75 }],
        },
        {
          event_index: 1,
          frame_id: 20,
          timestamp_ms: 200,
          score: 0.8,
          alternatives: [],
        },
      ],
    }
    let state = appReducer(initialAppState, {
      type: 'SET_TRAKE_EVENTS',
      payload: ['event 1', 'event 2'],
    })
    state = appReducer(state, {
      type: 'TRAKE_SEARCH_SUCCESS',
      payload: { query_id: 'trake-alternative', result, results: [result] },
    })
    state = appReducer(state, {
      type: 'SELECT_TRAKE_EVENT_ALTERNATIVE',
      payload: { event_index: 0, alternative_index: 0 },
    })

    expect(state.trakeSlots[0].frame_id).toBe(11)
    expect(state.trakeSlots[0].alternatives?.[0].frame_id).toBe(10)
    expect(state.trakeActiveHypothesisIndex).toBeNull()

    state = appReducer(state, { type: 'LOCK_TRAKE_SLOT', payload: { event_index: 0 } })
    const locked = appReducer(state, {
      type: 'SELECT_TRAKE_EVENT_ALTERNATIVE',
      payload: { event_index: 0, alternative_index: 0 },
    })
    expect(locked).toBe(state)
  })

  it('stores rerun alternatives only on the requested unlocked event', () => {
    let state = appReducer(initialAppState, {
      type: 'SET_TRAKE_EVENTS',
      payload: ['event 1', 'event 2', 'event 3'],
    })
    state = {
      ...state,
      trakeVideoId: 'V1',
      trakeSlots: state.trakeSlots.map((slot, index) => ({
        ...slot,
        video_id: 'V1',
        frame_id: (index + 1) * 10,
        timestamp_ms: (index + 1) * 1000,
        validation_status: 'valid' as const,
        locked: index !== 1,
      })),
    }

    state = appReducer(state, {
      type: 'TRAKE_EVENT_RERUN_SUCCESS',
      payload: {
        query_id: 'rerun-one',
        event_index: 1,
        video_id: 'V1',
        lower_bound_ms: 1000,
        upper_bound_ms: 3000,
        alternatives: [{ frame_id: 21, timestamp_ms: 2200, score: 0.8 }],
      },
    })

    expect(state.trakeSlots[1].alternatives?.[0].frame_id).toBe(21)
    expect(state.trakeSlots[0].alternatives).toBeUndefined()
    expect(state.trakeSlots[2].locked).toBe(true)
  })
})
