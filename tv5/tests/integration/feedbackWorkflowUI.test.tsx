import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { App } from '../../src/App'
import { AppProvider } from '../../src/state/AppContext'

const mockCandidates = [
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

describe('T029 — Feedback Workflow UI Integration Tests', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('executes full operator journey: Search -> Set Reference -> Refine -> Inspect -> Undo -> Reset -> Exit', async () => {
    const user = userEvent.setup()

    global.fetch = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (url.endsWith('/health')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ status: 'ok', mode: 'live', preprocess_run_id: 'run_v1_batch1' }),
        } as Response)
      }

      if (url.endsWith('/kis/search')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            query_id: 'kis-001',
            candidates: mockCandidates,
          }),
        } as Response)
      }

      if (url.endsWith('/feedback/start')) {
        const body = JSON.parse(init?.body as string)
        return Promise.resolve({
          ok: true,
          json: async () => ({
            session_id: body.session_id,
            revision: 0,
            candidates: mockCandidates,
            status: 'ok',
            provenance_mode: 'live',
          }),
        } as Response)
      }

      if (url.endsWith('/feedback/refine')) {
        const body = JSON.parse(init?.body as string)
        // Reorder candidates: promoted L21_V002 to rank 1
        const refined = [
          { ...mockCandidates[1], rank: 1 },
          { ...mockCandidates[0], rank: 2 },
        ]
        return Promise.resolve({
          ok: true,
          json: async () => ({
            session_id: body.session_id,
            revision: body.expected_revision + 1,
            candidates: refined,
            status: 'ok',
            provenance_mode: 'live',
          }),
        } as Response)
      }

      if (url.endsWith('/feedback/undo')) {
        const body = JSON.parse(init?.body as string)
        return Promise.resolve({
          ok: true,
          json: async () => ({
            session_id: body.session_id,
            revision: body.expected_revision + 1,
            candidates: mockCandidates,
            status: 'ok',
            provenance_mode: 'live',
          }),
        } as Response)
      }

      if (url.endsWith('/feedback/reset')) {
        const body = JSON.parse(init?.body as string)
        return Promise.resolve({
          ok: true,
          json: async () => ({
            session_id: body.session_id,
            revision: body.expected_revision + 1,
            candidates: mockCandidates,
            status: 'ok',
            provenance_mode: 'live',
          }),
        } as Response)
      }

      if (url.endsWith('/exact-frame/neighbors')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            video_id: 'L21_V002',
            anchor_frame_id: 23940,
            steps: [],
          }),
        } as Response)
      }

      return Promise.reject(new Error(`Unhandled URL: ${url}`))
    })

    render(
      <AppProvider>
        <App />
      </AppProvider>
    )

    // 1. Submit KIS Search
    const searchInput = screen.getByTestId('kis-query-input')
    await user.type(searchInput, 'người đi xe máy')
    await user.click(screen.getByTestId('kis-search-btn'))

    await waitFor(() => {
      expect(screen.getByTestId('candidate-card-1')).toBeInTheDocument()
      expect(screen.getByTestId('candidate-card-2')).toBeInTheDocument()
    })

    // Feedback panel is rendered
    expect(screen.getByTestId('feedback-panel')).toBeInTheDocument()
    expect(screen.getByTestId('feedback-refine-btn')).toBeDisabled()

    // 2. Set Candidate 2 as Reference explicitly
    const setRefBtn = screen.getByTestId('set-reference-btn-2')
    await user.click(setRefBtn)

    expect(screen.getByTestId('feedback-reference-display')).toHaveTextContent('L21_V002')
    expect(screen.getByTestId('feedback-reference-display')).toHaveTextContent('Frame 23940')

    // 3. Type feedback draft
    const feedbackInput = screen.getByTestId('feedback-text-input')
    await user.type(feedbackInput, 'mặc áo mưa màu vàng')
    expect(screen.getByTestId('feedback-refine-btn')).toBeEnabled()

    // 4. Click Refine
    await user.click(screen.getByTestId('feedback-refine-btn'))

    await waitFor(() => {
      expect(screen.getByTestId('feedback-revision-badge')).toHaveTextContent('Revision 1')
    })

    // Results now show L21_V002 at rank 1
    const card1 = screen.getByTestId('candidate-card-1')
    expect(card1).toHaveTextContent('L21_V002')
    expect(card1).toHaveTextContent('Frame 23940')

    // 5. Click refined candidate to inspect -> navigates to Inspection with canonical identity
    await user.click(card1)
    await waitFor(() => {
      expect(screen.getByTestId('inspected-video-id')).toHaveTextContent('L21_V002')
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('23940')
    })

    // Switch back to Retrieval tab
    const retrievalTab = screen.getByTestId('tab-retrieval')
    await user.click(retrievalTab)

    // 6. Test Undo
    const undoBtn = screen.getByTestId('feedback-undo-btn')
    await user.click(undoBtn)

    await waitFor(() => {
      expect(screen.getByTestId('feedback-revision-badge')).toHaveTextContent('Revision 2')
    })
    expect(screen.getByTestId('candidate-card-1')).toHaveTextContent('L21_V001')

    // 7. Test Reset
    const resetBtn = screen.getByTestId('feedback-reset-btn')
    await user.click(resetBtn)

    await waitFor(() => {
      expect(screen.getByTestId('feedback-revision-badge')).toHaveTextContent('Revision 3')
    })

    // 8. Test Exit Feedback
    const exitBtn = screen.getByTestId('feedback-exit-btn')
    await user.click(exitBtn)

    // Feedback session cleared, original KIS results restored
    expect(screen.queryByTestId('feedback-revision-badge')).not.toBeInTheDocument()
    expect(screen.getByTestId('candidate-card-1')).toHaveTextContent('L21_V001')
  })

  it('handles degraded zero-candidate live error gracefully without breaking Retrieval', async () => {
    const user = userEvent.setup()

    global.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.endsWith('/health')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ status: 'ok', mode: 'live', preprocess_run_id: 'run_v1_batch1' }),
        } as Response)
      }
      if (url.endsWith('/kis/search')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ query_id: 'kis-001', candidates: mockCandidates }),
        } as Response)
      }
      if (url.endsWith('/feedback/start')) {
        return Promise.resolve({
          ok: false,
          status: 500,
          statusText: 'Internal Server Error',
          json: async () => ({
            detail: 'feedback service error: visual search returned zero candidates for feedback session',
          }),
        } as Response)
      }
      return Promise.reject(new Error(`Unhandled URL: ${url}`))
    })

    render(
      <AppProvider>
        <App />
      </AppProvider>
    )

    // Search and set reference
    await user.type(screen.getByTestId('kis-query-input'), 'người đi bộ')
    await user.click(screen.getByTestId('kis-search-btn'))

    await waitFor(() => {
      expect(screen.getByTestId('candidate-card-1')).toBeInTheDocument()
    })

    await user.click(screen.getByTestId('set-reference-btn-1'))
    await user.type(screen.getByTestId('feedback-text-input'), 'gần cột đèn')
    await user.click(screen.getByTestId('feedback-refine-btn'))

    // Degradation banner appears truthfully
    await waitFor(() => {
      expect(screen.getByTestId('feedback-error-banner')).toBeInTheDocument()
      expect(screen.getByTestId('feedback-error-banner')).toHaveTextContent(
        'Feedback is unavailable for the current query (zero visual candidates)'
      )
    })

    // Original Retrieval results remain completely intact!
    expect(screen.getByTestId('candidate-card-1')).toHaveTextContent('L21_V001')
    expect(screen.getByTestId('candidate-card-2')).toHaveTextContent('L21_V002')
  })

  it('successfully refines feedback with an exact-corrected candidate reference', async () => {
    const user = userEvent.setup()
    const refineCalls: any[] = []

    global.fetch = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (url.endsWith('/health')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ status: 'ok', mode: 'live', preprocess_run_id: 'run_v1_batch1' }),
        } as Response)
      }
      if (url.endsWith('/kis/search')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ query_id: 'kis-001', candidates: mockCandidates }),
        } as Response)
      }
      if (url.endsWith('/exact-frame/neighbors')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            video_id: 'L21_V001',
            anchor_frame_id: 10690,
            provenance_mode: 'live',
            steps: [
              {
                offset: 1,
                frame: {
                  video_id: 'L21_V001',
                  frame_id: 10691,
                  timestamp_ms: 356366,
                  pts: 5473792,
                  time_base: '1/15360',
                  preprocess_run_id: 'run_v1_batch1',
                  mapping_guaranteed: true,
                  submission_selectable: true,
                },
              },
            ],
          }),
        } as Response)
      }
      if (url.endsWith('/exact-frame/image')) {
        return Promise.resolve({
          ok: true,
          blob: async () => new Blob(['fake-img'], { type: 'image/jpeg' }),
          headers: new Headers({
            'x-original-video-id': 'L21_V001',
            'x-original-frame-id': '10691',
            'x-submission-selectable': 'true',
          }),
        } as unknown as Response)
      }
      if (url.endsWith('/feedback/start')) {
        const body = JSON.parse(init?.body as string)
        return Promise.resolve({
          ok: true,
          json: async () => ({
            session_id: body.session_id,
            revision: 0,
            candidates: mockCandidates,
            status: 'ok',
            provenance_mode: 'live',
          }),
        } as Response)
      }
      if (url.endsWith('/feedback/refine')) {
        const body = JSON.parse(init?.body as string)
        refineCalls.push(body)
        const refined = [
          {
            video_id: 'L21_V001',
            frame_id: body.frame_id,
            certified_anchor_frame_id: body.source_candidate_frame_id,
            anchor_offset: body.frame_id - body.source_candidate_frame_id,
            rank: 1,
            timestamp_ms: 356366,
            source: 'feedback',
          },
          { ...mockCandidates[1], rank: 2 },
        ]
        return Promise.resolve({
          ok: true,
          json: async () => ({
            session_id: body.session_id,
            revision: body.expected_revision + 1,
            candidates: refined,
            status: 'ok',
            provenance_mode: 'live',
          }),
        } as Response)
      }
      return Promise.reject(new Error(`Unhandled URL: ${url}`))
    })

    render(
      <AppProvider>
        <App />
      </AppProvider>
    )

    // 1. Search KIS
    await user.type(screen.getByTestId('kis-query-input'), 'người chạy bộ')
    await user.click(screen.getByTestId('kis-search-btn'))

    await waitFor(() => {
      expect(screen.getByTestId('candidate-card-1')).toBeInTheDocument()
    })

    // 2. Inspect candidate 1, step to 10691, commit frame
    await user.click(screen.getByTestId('candidate-card-1'))
    await waitFor(() => {
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('10690')
    })
    await user.click(screen.getByTestId('btn-step-next'))
    await waitFor(() => {
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('10691')
    })
    await user.click(screen.getByTestId('kis-set-canonical-frame-btn'))

    // 3. Return to Retrieval tab and set exact-corrected candidate 1 as reference
    await user.click(screen.getByTestId('tab-retrieval'))
    expect(screen.getByTestId('candidate-card-1')).toHaveTextContent('10691')
    await user.click(screen.getByTestId('set-reference-btn-1'))

    expect(screen.getByTestId('feedback-reference-display')).toHaveTextContent('L21_V001')
    expect(screen.getByTestId('feedback-reference-display')).toHaveTextContent('Frame 10691')

    // 4. Enter feedback text and click Refine
    await user.type(screen.getByTestId('feedback-text-input'), 'mặc áo thun xanh')
    await user.click(screen.getByTestId('feedback-refine-btn'))

    await waitFor(() => {
      expect(screen.getByTestId('feedback-revision-badge')).toHaveTextContent('Revision 1')
    })

    // 5. Verify that source_candidate_frame_id = 10690 was sent in payload
    expect(refineCalls.length).toBe(1)
    expect(refineCalls[0].video_id).toBe('L21_V001')
    expect(refineCalls[0].frame_id).toBe(10691)
    expect(refineCalls[0].source_candidate_frame_id).toBe(10690)

    // 6. Verify result card #1 reflects exact frame 10691 without error
    expect(screen.getByTestId('candidate-card-1')).toHaveTextContent('10691')
    expect(screen.getByTestId('candidate-card-1')).not.toHaveTextContent('Preview unavailable')
  })

  it('enforces 5 active refinements limit in UI, disables 6th refine, and handles undo/reset/server-fallback', async () => {
    const user = userEvent.setup()
    let refineCallCount = 0
    let currentActiveEvents = 0
    let currentRev = 0

    global.fetch = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (url.endsWith('/health')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ status: 'ok', mode: 'live', preprocess_run_id: 'run_v1_batch1' }),
        } as Response)
      }
      if (url.endsWith('/kis/search')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ query_id: 'kis-001', candidates: mockCandidates }),
        } as Response)
      }
      if (url.endsWith('/feedback/start')) {
        const body = JSON.parse(init?.body as string)
        currentActiveEvents = 0
        currentRev = 0
        return Promise.resolve({
          ok: true,
          json: async () => ({
            session_id: body.session_id,
            revision: 0,
            active_feedback_count: 0,
            max_active_feedback_events: 5,
            candidates: mockCandidates,
            status: 'ok',
            provenance_mode: 'live',
          }),
        } as Response)
      }
      if (url.endsWith('/feedback/refine')) {
        refineCallCount++
        const body = JSON.parse(init?.body as string)
        if (currentActiveEvents >= 5) {
          return Promise.resolve({
            ok: false,
            status: 400,
            statusText: 'Bad Request',
            text: async () => JSON.stringify({ detail: 'session permits at most five active feedback events' }),
          } as Response)
        }
        currentActiveEvents++
        currentRev++
        return Promise.resolve({
          ok: true,
          json: async () => ({
            session_id: body.session_id,
            revision: currentRev,
            active_feedback_count: currentActiveEvents,
            max_active_feedback_events: 5,
            candidates: mockCandidates,
            status: 'ok',
            provenance_mode: 'live',
          }),
        } as Response)
      }
      if (url.endsWith('/feedback/undo')) {
        const body = JSON.parse(init?.body as string)
        if (currentActiveEvents > 0) currentActiveEvents--
        currentRev++
        return Promise.resolve({
          ok: true,
          json: async () => ({
            session_id: body.session_id,
            revision: currentRev,
            active_feedback_count: currentActiveEvents,
            max_active_feedback_events: 5,
            candidates: mockCandidates,
            status: 'ok',
            provenance_mode: 'live',
          }),
        } as Response)
      }
      if (url.endsWith('/feedback/reset')) {
        const body = JSON.parse(init?.body as string)
        currentActiveEvents = 0
        currentRev++
        return Promise.resolve({
          ok: true,
          json: async () => ({
            session_id: body.session_id,
            revision: currentRev,
            active_feedback_count: 0,
            max_active_feedback_events: 5,
            candidates: mockCandidates,
            status: 'ok',
            provenance_mode: 'live',
          }),
        } as Response)
      }
      return Promise.reject(new Error(`Unhandled URL: ${url}`))
    })

    render(
      <AppProvider>
        <App />
      </AppProvider>
    )

    // 1. Search KIS
    await user.type(screen.getByTestId('kis-query-input'), 'xe máy chạy trên phố')
    await user.click(screen.getByTestId('kis-search-btn'))

    await waitFor(() => {
      expect(screen.getByTestId('candidate-card-1')).toBeInTheDocument()
    })

    // Set reference candidate 1
    await user.click(screen.getByTestId('set-reference-btn-1'))

    // 2. Perform 5 Refinements
    const feedbackInput = screen.getByTestId('feedback-text-input')
    const refineBtn = screen.getByTestId('feedback-refine-btn')

    for (let i = 1; i <= 5; i++) {
      await user.type(feedbackInput, `refinement step ${i}`)
      expect(refineBtn).toBeEnabled()
      await user.click(refineBtn)

      await waitFor(() => {
        expect(screen.getByTestId('feedback-count-badge')).toHaveTextContent(`Refinements: ${i} / 5`)
      })
    }

    expect(refineCallCount).toBe(5)

    // 3. At 5/5, Refine is disabled and friendly limit banner appears
    expect(screen.getByTestId('feedback-count-badge')).toHaveTextContent('Refinements: 5 / 5')
    expect(screen.getByTestId('feedback-refine-btn')).toBeDisabled()
    expect(screen.getByTestId('feedback-limit-banner')).toBeInTheDocument()
    expect(screen.getByTestId('feedback-limit-banner')).toHaveTextContent(
      'Maximum 5 active refinements reached. Undo or Reset to continue.'
    )

    // 4. Proactive prevention: clicking refineBtn or typing Enter does NOT trigger 6th network request
    await user.click(refineBtn)
    expect(refineCallCount).toBe(5)

    // 5. Test Undo: reduces count to 4/5, clears limit banner, and re-enables Refine
    const undoBtn = screen.getByTestId('feedback-undo-btn')
    await user.click(undoBtn)

    await waitFor(() => {
      expect(screen.getByTestId('feedback-count-badge')).toHaveTextContent('Refinements: 4 / 5')
    })
    expect(screen.queryByTestId('feedback-limit-banner')).not.toBeInTheDocument()

    // 6. Test Refine 5th event again
    await user.type(feedbackInput, '5th refinement retry')
    expect(refineBtn).toBeEnabled()
    await user.click(refineBtn)

    await waitFor(() => {
      expect(screen.getByTestId('feedback-count-badge')).toHaveTextContent('Refinements: 5 / 5')
    })
    expect(screen.getByTestId('feedback-limit-banner')).toBeInTheDocument()
    expect(refineBtn).toBeDisabled()

    // 7. Test Reset: clears count to 0/5, clears limit banner, and re-enables Refine
    const resetBtn = screen.getByTestId('feedback-reset-btn')
    await user.click(resetBtn)

    await waitFor(() => {
      expect(screen.getByTestId('feedback-count-badge')).toHaveTextContent('Refinements: 0 / 5')
    })
    expect(screen.queryByTestId('feedback-limit-banner')).not.toBeInTheDocument()
  })

  it('handles server-side 400 5-event limit fallback gracefully without raw error string', async () => {
    const user = userEvent.setup()

    global.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.endsWith('/health')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ status: 'ok', mode: 'live', preprocess_run_id: 'run_v1_batch1' }),
        } as Response)
      }
      if (url.endsWith('/kis/search')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ query_id: 'kis-001', candidates: mockCandidates }),
        } as Response)
      }
      if (url.endsWith('/feedback/start')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            session_id: 's-fallback',
            revision: 0,
            active_feedback_count: 0,
            max_active_feedback_events: 5,
            candidates: mockCandidates,
            status: 'ok',
          }),
        } as Response)
      }
      if (url.endsWith('/feedback/refine')) {
        return Promise.resolve({
          ok: false,
          status: 400,
          statusText: 'Bad Request',
          text: async () => JSON.stringify({ detail: 'session permits at most five active feedback events' }),
        } as Response)
      }
      return Promise.reject(new Error(`Unhandled URL: ${url}`))
    })

    render(
      <AppProvider>
        <App />
      </AppProvider>
    )

    // Search and set reference
    await user.type(screen.getByTestId('kis-query-input'), 'ô tô đỏ')
    await user.click(screen.getByTestId('kis-search-btn'))
    await waitFor(() => {
      expect(screen.getByTestId('candidate-card-1')).toBeInTheDocument()
    })

    await user.click(screen.getByTestId('set-reference-btn-1'))
    await user.type(screen.getByTestId('feedback-text-input'), 'chạy nhanh')
    await user.click(screen.getByTestId('feedback-refine-btn'))

    // Friendly limit message appears instead of raw API error
    await waitFor(() => {
      expect(screen.getByTestId('feedback-limit-banner')).toBeInTheDocument()
      expect(screen.getByTestId('feedback-limit-banner')).toHaveTextContent(
        'Maximum 5 active refinements reached. Undo or Reset to continue.'
      )
    })
    expect(screen.queryByText(/session permits at most five active feedback events/i)).not.toBeInTheDocument()
    expect(screen.getByTestId('feedback-refine-btn')).toBeDisabled()
  })
})
