import { describe, it, expect } from 'vitest'
import { render, screen, act } from '@testing-library/react'
import { AppProvider, useAppState, useAppDispatch } from '../../src/state/AppContext'
import { SearchCandidate } from '../../src/types/contracts'

const mockCandidates: SearchCandidate[] = [
  {
    query_id: 'kis-001',
    video_id: 'L21_V001',
    frame_id: 10690,
    timestamp_ms: 356333,
    source: 'fusion',
    rank: 1,
    score: 0.85,
    preprocess_run_id: 'run_v1_batch1',
  },
  {
    query_id: 'kis-001',
    video_id: 'L21_V002',
    frame_id: 23940,
    timestamp_ms: 798000,
    source: 'fusion',
    rank: 2,
    score: 0.72,
    preprocess_run_id: 'run_v1_batch1',
  },
]

function FeedbackWorkflowHarness() {
  const state = useAppState()
  const dispatch = useAppDispatch()

  return (
    <div>
      <div data-testid="is-feedback-active">{state.isFeedbackActive ? 'yes' : 'no'}</div>
      <div data-testid="original-query">{state.feedbackOriginalQuery || state.queryText}</div>
      <div data-testid="current-revision">{state.feedbackRevision}</div>
      <div data-testid="candidate-count">{state.candidates.length}</div>
      <div data-testid="top-candidate-video">{state.candidates[0]?.video_id || 'none'}</div>
      <div data-testid="top-candidate-frame">{state.candidates[0]?.frame_id ?? -1}</div>
      <div data-testid="original-candidate-video">{state.originalCandidates[0]?.video_id || 'none'}</div>
      <div data-testid="basket-count">{state.submissionBasket.length}</div>
      <div data-testid="active-candidate">{state.activeCandidate?.video_id || 'none'}</div>
      <div data-testid="reference-candidate">{state.feedbackReference?.video_id || 'none'}</div>

      <button
        data-testid="btn-search"
        onClick={() => {
          dispatch({ type: 'SET_QUERY_TEXT', payload: 'người lái thuyền' })
          dispatch({
            type: 'KIS_SEARCH_SUCCESS',
            payload: { query_id: 'kis-001', candidates: mockCandidates },
          })
        }}
      >
        Search
      </button>

      <button
        data-testid="btn-start-feedback"
        onClick={() => {
          dispatch({
            type: 'FEEDBACK_START_SUCCESS',
            payload: {
              session_id: 'sess-wf-01',
              revision: 0,
              candidates: mockCandidates,
              status: 'ok',
            },
          })
        }}
      >
        Start Feedback
      </button>

      <button
        data-testid="btn-select-ref"
        onClick={() => {
          dispatch({ type: 'SET_FEEDBACK_REFERENCE', payload: state.candidates[1] })
        }}
      >
        Select Reference
      </button>

      <button
        data-testid="btn-refine"
        onClick={() => {
          // Promote candidate 2 to rank 1
          const reordered = [
            { ...mockCandidates[1], rank: 1 },
            { ...mockCandidates[0], rank: 2 },
          ]
          dispatch({
            type: 'FEEDBACK_REFINE_SUCCESS',
            payload: {
              session_id: 'sess-wf-01',
              revision: 1,
              candidates: reordered,
              status: 'ok',
            },
          })
        }}
      >
        Refine
      </button>

      <button
        data-testid="btn-inspect-top"
        onClick={() => {
          dispatch({ type: 'SELECT_CANDIDATE', payload: state.candidates[0] })
        }}
      >
        Inspect Top
      </button>

      <button
        data-testid="btn-add-basket"
        onClick={() => {
          dispatch({
            type: 'ADD_TO_BASKET',
            payload: {
              video_id: state.candidates[0].video_id,
              frame_id: state.candidates[0].frame_id,
              timestamp_ms: state.candidates[0].timestamp_ms,
              added_at_utc: new Date().toISOString(),
            },
          })
        }}
      >
        Add to Basket
      </button>

      <button
        data-testid="btn-undo"
        onClick={() => {
          dispatch({
            type: 'FEEDBACK_UNDO_SUCCESS',
            payload: {
              session_id: 'sess-wf-01',
              revision: 2,
              candidates: mockCandidates,
              status: 'ok',
            },
          })
        }}
      >
        Undo
      </button>

      <button
        data-testid="btn-reset"
        onClick={() => {
          dispatch({
            type: 'FEEDBACK_RESET_SUCCESS',
            payload: {
              session_id: 'sess-wf-01',
              revision: 3,
              candidates: mockCandidates,
              status: 'ok',
            },
          })
        }}
      >
        Reset
      </button>

      <button
        data-testid="btn-clear"
        onClick={() => {
          dispatch({ type: 'FEEDBACK_CLEAR' })
        }}
      >
        Clear Feedback
      </button>
    </div>
  )
}

describe('T028 — Feedback Workflow Integration & Invariant Tests', () => {
  it('executes full operator lifecycle with query immutability, snapshot separation, and basket isolation', () => {
    render(
      <AppProvider>
        <FeedbackWorkflowHarness />
      </AppProvider>
    )

    // 1. Initial State
    expect(screen.getByTestId('is-feedback-active').textContent).toBe('no')
    expect(screen.getByTestId('basket-count').textContent).toBe('0')

    // 2. Perform KIS Search
    act(() => {
      screen.getByTestId('btn-search').click()
    })
    expect(screen.getByTestId('original-query').textContent).toBe('người lái thuyền')
    expect(screen.getByTestId('candidate-count').textContent).toBe('2')
    expect(screen.getByTestId('top-candidate-video').textContent).toBe('L21_V001')
    expect(screen.getByTestId('original-candidate-video').textContent).toBe('L21_V001')

    // 3. Start Feedback
    act(() => {
      screen.getByTestId('btn-start-feedback').click()
    })
    expect(screen.getByTestId('is-feedback-active').textContent).toBe('yes')
    expect(screen.getByTestId('current-revision').textContent).toBe('0')
    expect(screen.getByTestId('original-query').textContent).toBe('người lái thuyền')

    // 4. Select Reference Candidate (L21_V002)
    act(() => {
      screen.getByTestId('btn-select-ref').click()
    })
    expect(screen.getByTestId('reference-candidate').textContent).toBe('L21_V002')

    // 5. Trigger Refine
    act(() => {
      screen.getByTestId('btn-refine').click()
    })
    // Revision advances to 1
    expect(screen.getByTestId('current-revision').textContent).toBe('1')
    // Top candidate is now L21_V002 (frame 23940)
    expect(screen.getByTestId('top-candidate-video').textContent).toBe('L21_V002')
    expect(screen.getByTestId('top-candidate-frame').textContent).toBe('23940')
    // Original snapshot remains L21_V001
    expect(screen.getByTestId('original-candidate-video').textContent).toBe('L21_V001')
    // Original query remains immutable
    expect(screen.getByTestId('original-query').textContent).toBe('người lái thuyền')

    // 6. Inspect refined candidate -> establishes Inspection anchor with zero frame arithmetic
    act(() => {
      screen.getByTestId('btn-inspect-top').click()
    })
    expect(screen.getByTestId('active-candidate').textContent).toBe('L21_V002')

    // 7. Add candidate to Basket
    act(() => {
      screen.getByTestId('btn-add-basket').click()
    })
    expect(screen.getByTestId('basket-count').textContent).toBe('1')

    // 8. Undo Feedback -> rolls back candidate ranking, revision advances to 2, basket is untouched
    act(() => {
      screen.getByTestId('btn-undo').click()
    })
    expect(screen.getByTestId('current-revision').textContent).toBe('2')
    expect(screen.getByTestId('top-candidate-video').textContent).toBe('L21_V001')
    expect(screen.getByTestId('basket-count').textContent).toBe('1')

    // 9. Reset Feedback -> restores baseline, revision advances to 3, basket is untouched
    act(() => {
      screen.getByTestId('btn-reset').click()
    })
    expect(screen.getByTestId('current-revision').textContent).toBe('3')
    expect(screen.getByTestId('top-candidate-video').textContent).toBe('L21_V001')
    expect(screen.getByTestId('basket-count').textContent).toBe('1')

    // 10. Clear Feedback -> exits feedback mode, basket and original query remain intact
    act(() => {
      screen.getByTestId('btn-clear').click()
    })
    expect(screen.getByTestId('is-feedback-active').textContent).toBe('no')
    expect(screen.getByTestId('original-query').textContent).toBe('người lái thuyền')
    expect(screen.getByTestId('top-candidate-video').textContent).toBe('L21_V001')
    expect(screen.getByTestId('basket-count').textContent).toBe('1')
  })
})
