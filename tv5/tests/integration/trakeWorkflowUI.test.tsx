import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { App } from '../../src/App'
import { AppProvider } from '../../src/state/AppContext'

const mockTrakeSuccessResponse = {
  query_id: 'trake-int-001',
  provenance_mode: 'live',
  result: {
    video_id: 'L25_V084',
    frame_ids: [58779, 58900, 58950, 59000],
    timestamps_ms: [2351160, 2356000, 2358000, 2360000],
    event_scores: [0.92, 0.88, 0.85, 0.8],
    aggregate_score: 3.45,
    preprocess_run_id: 'run_v1_batch1',
    candidates: [
      {
        query_id: 'trake-int-001-ev0',
        video_id: 'L25_V084',
        frame_id: 58779,
        timestamp_ms: 2351160,
        source: 'fusion',
        rank: 1,
        score: 0.92,
        event_index: 0,
        certified_anchor_frame_id: 58779,
        certified_anchor_timestamp_ms: 2351160,
        anchor_offset: 0,
        preprocess_run_id: 'run_v1_batch1',
      },
      {
        query_id: 'trake-int-001-ev1',
        video_id: 'L25_V084',
        frame_id: 58900,
        timestamp_ms: 2356000,
        source: 'fusion',
        rank: 1,
        score: 0.88,
        event_index: 1,
        certified_anchor_frame_id: 58900,
        certified_anchor_timestamp_ms: 2356000,
        anchor_offset: 0,
        preprocess_run_id: 'run_v1_batch1',
      },
      {
        query_id: 'trake-int-001-ev2',
        video_id: 'L25_V084',
        frame_id: 58950,
        timestamp_ms: 2358000,
        source: 'fusion',
        rank: 1,
        score: 0.85,
        event_index: 2,
        certified_anchor_frame_id: 58950,
        certified_anchor_timestamp_ms: 2358000,
        anchor_offset: 0,
        preprocess_run_id: 'run_v1_batch1',
      },
      {
        query_id: 'trake-int-001-ev3',
        video_id: 'L25_V084',
        frame_id: 59000,
        timestamp_ms: 2360000,
        source: 'fusion',
        rank: 1,
        score: 0.8,
        event_index: 3,
        certified_anchor_frame_id: 59000,
        certified_anchor_timestamp_ms: 2360000,
        anchor_offset: 0,
        preprocess_run_id: 'run_v1_batch1',
      },
    ],
  },
}

const mockTrakeErrorResponse = {
  query_id: 'trake-err-001',
  provenance_mode: 'live',
  result: null,
  message: 'no monotonic alignment found',
}

const mockNeighborResponse58779 = {
  provenance_mode: 'live',
  video_id: 'L25_V084',
  anchor_frame_id: 58779,
  degraded_reason: null,
  steps: [
    {
      offset: -2,
      degraded_reason: null,
      frame: {
        video_id: 'L25_V084',
        frame_id: 58777,
        timestamp_ms: 2351080,
        pts: 587770,
        time_base: '1/25',
        preprocess_run_id: 'run_v1_batch1',
        mapping_guaranteed: true,
        submission_selectable: true,
        identity_source: 'certified_frame',
      },
    },
    {
      offset: -1,
      degraded_reason: null,
      frame: {
        video_id: 'L25_V084',
        frame_id: 58778,
        timestamp_ms: 2351120,
        pts: 587780,
        time_base: '1/25',
        preprocess_run_id: 'run_v1_batch1',
        mapping_guaranteed: true,
        submission_selectable: true,
        identity_source: 'certified_frame',
      },
    },
    {
      offset: 0,
      degraded_reason: null,
      frame: {
        video_id: 'L25_V084',
        frame_id: 58779,
        timestamp_ms: 2351160,
        pts: 587790,
        time_base: '1/25',
        preprocess_run_id: 'run_v1_batch1',
        mapping_guaranteed: true,
        submission_selectable: true,
        identity_source: 'certified_frame',
      },
    },
    {
      offset: 1,
      degraded_reason: null,
      frame: {
        video_id: 'L25_V084',
        frame_id: 58780,
        timestamp_ms: 2351200,
        pts: 587800,
        time_base: '1/25',
        preprocess_run_id: 'run_v1_batch1',
        mapping_guaranteed: true,
        submission_selectable: true,
        identity_source: 'certified_frame',
      },
    },
    {
      offset: 2,
      degraded_reason: null,
      frame: {
        video_id: 'L25_V084',
        frame_id: 58781,
        timestamp_ms: 2351240,
        pts: 587810,
        time_base: '1/25',
        preprocess_run_id: 'run_v1_batch1',
        mapping_guaranteed: true,
        submission_selectable: true,
        identity_source: 'certified_frame',
      },
    },
  ],
}

describe('T033 — TRAKE Workflow & Operator UI Integration Tests', () => {
  let user: ReturnType<typeof userEvent.setup>

  beforeEach(() => {
    user = userEvent.setup()
    vi.restoreAllMocks()

    globalThis.fetch = vi.fn((url: string | URL | Request, init?: RequestInit) => {
      const urlStr = typeof url === 'string' ? url : (url as URL).toString()

      if (urlStr.includes('/health')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({ status: 'ok', mode: 'live', preprocess_run_id: 'run_v1_batch1' }),
            { status: 200, headers: { 'Content-Type': 'application/json' } }
          )
        )
      }

      if (urlStr.includes('/trake/align')) {
        const reqBody = init?.body ? JSON.parse(init.body as string) : {}
        if (reqBody.query_text?.includes('unalignable')) {
          return Promise.resolve(
            new Response(JSON.stringify(mockTrakeErrorResponse), {
              status: 200,
              headers: { 'Content-Type': 'application/json' },
            })
          )
        }
        return Promise.resolve(
          new Response(JSON.stringify(mockTrakeSuccessResponse), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          })
        )
      }

      if (urlStr.includes('/exact-frame/neighbors')) {
        return Promise.resolve(
          new Response(JSON.stringify(mockNeighborResponse58779), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          })
        )
      }

      if (urlStr.includes('/exact-frame/image')) {
        return Promise.resolve(
          new Response(new Blob(['fake-jpg-binary'], { type: 'image/jpeg' }), {
            status: 200,
            headers: {
              'Content-Type': 'image/jpeg',
              'x-exact-frame-id': '58779',
              'x-exact-video-id': 'L25_V084',
              'x-exact-pts': '587790',
              'x-exact-time-base': '1/25',
              'x-exact-timestamp-ms': '2351160',
              'x-exact-preprocess-run-id': 'run_v1_batch1',
              'x-exact-submission-selectable': 'true',
            },
          })
        )
      }

      return Promise.resolve(
        new Response(JSON.stringify({ message: 'mock not found' }), {
          status: 404,
          headers: { 'Content-Type': 'application/json' },
        })
      )
    }) as any
  })

  it('BUG 1: TRAKE error does not leak to KIS or Q&A mode on switch', async () => {
    render(
      <AppProvider>
        <App />
      </AppProvider>
    )

    // 1. Switch to TRAKE mode
    const trakeToggleBtn = screen.getByTestId('task-mode-trake')
    await user.click(trakeToggleBtn)

    // 2. Click sample preset to load valid events, then type unalignable query
    const samplePresetBtn = screen.getByTestId('trake-sample-preset-btn')
    await user.click(samplePresetBtn)

    const queryInput = screen.getByTestId('kis-query-input')
    await user.clear(queryInput)
    await user.type(queryInput, 'unalignable sequence')

    const searchBtn = screen.getByTestId('kis-search-btn')
    await user.click(searchBtn)

    // 3. Error banner renders in TRAKE mode
    await waitFor(() => {
      const banner = screen.getByTestId('search-error-banner')
      expect(banner).toHaveTextContent('no monotonic alignment found')
    })

    // 4. Switch to Q&A (VQA) mode -> Error banner MUST disappear from Q&A
    const vqaToggleBtn = screen.getByTestId('task-mode-vqa')
    await user.click(vqaToggleBtn)

    expect(screen.queryByTestId('search-error-banner')).not.toBeInTheDocument()

    // 5. Switch to KIS mode -> Error banner MUST NOT appear in KIS
    const kisToggleBtn = screen.getByTestId('task-mode-kis')
    await user.click(kisToggleBtn)

    expect(screen.queryByTestId('search-error-banner')).not.toBeInTheDocument()

    // 6. Switch back to TRAKE -> Error banner remains in TRAKE
    await user.click(trakeToggleBtn)
    expect(screen.getByTestId('search-error-banner')).toHaveTextContent('no monotonic alignment found')
  })

  it('BUG 2 & 3: TRAKE initial exact inspection carries canonical timestamp, renders without 409, and UNLOCKED badge is aligned', async () => {
    render(
      <AppProvider>
        <App />
      </AppProvider>
    )

    // 1. Switch to TRAKE mode
    const trakeToggleBtn = screen.getByTestId('task-mode-trake')
    await user.click(trakeToggleBtn)

    // 2. Click sample preset to load valid events, then type query
    const samplePresetBtn = screen.getByTestId('trake-sample-preset-btn')
    await user.click(samplePresetBtn)

    const queryInput = screen.getByTestId('kis-query-input')
    await user.clear(queryInput)
    await user.type(queryInput, 'Vận động viên nhảy cao')

    const searchBtn = screen.getByTestId('kis-search-btn')
    await user.click(searchBtn)

    // 3. Wait for search to complete and timeline cards to appear
    await waitFor(() => {
      expect(screen.getByTestId('trake-slot-card-0')).toBeInTheDocument()
      expect(screen.getByTestId('trake-slot-card-1')).toBeInTheDocument()
    })

    // Assert timeline shows valid frames and timestamps
    expect(screen.getByTestId('trake-slot-card-0')).toHaveTextContent('58779')
    expect(screen.getByTestId('trake-slot-card-1')).toHaveTextContent('58900')

    // 4. Click Inspection tab to inspect Event 0
    const inspectionTab = screen.getByTestId('tab-inspection')
    await user.click(inspectionTab)

    // Wait for exact neighbor fetch and image load
    await waitFor(() => {
      expect(screen.getByTestId('inspection-workspace')).toBeInTheDocument()
    })

    // Assert /exact-frame/neighbors was called with true canonical timestamp 2351160 ms (NOT 0!)
    const neighborCalls = (globalThis.fetch as any).mock.calls.filter((call: any[]) =>
      call[0]?.includes('/exact-frame/neighbors')
    )
    expect(neighborCalls.length).toBeGreaterThan(0)
    const neighborBody = JSON.parse(neighborCalls[0][1].body)
    expect(neighborBody.video_id).toBe('L25_V084')
    expect(neighborBody.frame_id).toBe(58779)
    expect(neighborBody.timestamp_ms).toBe(2351160) // Authoritative timestamp!

    // Assert UNLOCKED badge renders cleanly with .slot-rail-lock-badge.is-unlocked
    const card0 = screen.getByTestId('trake-shortlist-card-0')
    const badge = within(card0).getByText('UNLOCKED')
    expect(badge).toHaveClass('slot-rail-lock-badge', 'is-unlocked')

    // 5. Click "Use for Event #1"
    const useForEventBtn = screen.getByTestId('trake-set-event-frame-btn')
    expect(useForEventBtn).toHaveTextContent('Use for Event #1')
    await user.click(useForEventBtn)

    // Verify slot remains valid and committed
    expect(within(card0).getByText(/Frame 58779/i)).toBeInTheDocument()
  })

  it('BUG 1, 2, 4: Exact-corrected preview, re-entry proof lineage, and structured basket display', async () => {
    render(
      <AppProvider>
        <App />
      </AppProvider>
    )

    // 1. Switch to TRAKE mode and search
    await user.click(screen.getByTestId('task-mode-trake'))
    await user.click(screen.getByTestId('trake-sample-preset-btn'))

    const queryInput = screen.getByTestId('kis-query-input')
    await user.clear(queryInput)
    await user.type(queryInput, 'Vận động viên nhảy cao')
    await user.click(screen.getByTestId('kis-search-btn'))

    await waitFor(() => {
      expect(screen.getByTestId('trake-slot-card-0')).toBeInTheDocument()
    })

    // 2. Go to Inspection
    await user.click(screen.getByTestId('tab-inspection'))
    await waitFor(() => {
      expect(screen.getByTestId('inspection-workspace')).toBeInTheDocument()
    })

    // 3. Step forward +1 (to frame 58780)
    const stepNextBtn = screen.getByTestId('btn-step-next')
    await user.click(stepNextBtn)

    // Commit frame 58780 for Event #1
    const useForEventBtn = screen.getByTestId('trake-set-event-frame-btn')
    await user.click(useForEventBtn)

    // 4. Click "Back to Results" -> check Retrieval timeline preview (BUG 1)
    const backBtn = screen.getByText('Back to Results')
    await user.click(backBtn)

    await waitFor(() => {
      expect(screen.getByTestId('trake-timeline-container')).toBeInTheDocument()
    })
    // Textual identity updated to 58780
    expect(screen.getByTestId('trake-slot-card-0')).toHaveTextContent('58780')

    // 5. Re-enter Inspection for Event #1 (BUG 2)
    const inspectBtn0 = screen.getByTestId('trake-slot-card-0')
    await user.dblClick(inspectBtn0)

    await waitFor(() => {
      expect(screen.getByTestId('inspection-workspace')).toBeInTheDocument()
    })

    // Assert re-entering preserved the certified root anchor 58779 and offset 1
    const calls = (globalThis.fetch as any).mock.calls.filter((c: any[]) =>
      c[0]?.includes('/exact-frame/neighbors')
    )
    const lastNeighborBody = JSON.parse(calls[calls.length - 1][1].body)
    expect(lastNeighborBody.certified_anchor_frame_id).toBe(58779)
    expect(lastNeighborBody.cumulative_offset).toBe(1)

    // 6. Lock all 4 event slots and add to Basket (BUG 4)
    await user.click(screen.getByText('Back to Results'))
    const lockBtn0 = within(screen.getByTestId('trake-slot-card-0')).getByRole('button', { name: /lock/i })
    await user.click(lockBtn0)
    const lockBtn1 = within(screen.getByTestId('trake-slot-card-1')).getByRole('button', { name: /lock/i })
    await user.click(lockBtn1)
    const lockBtn2 = within(screen.getByTestId('trake-slot-card-2')).getByRole('button', { name: /lock/i })
    await user.click(lockBtn2)
    const lockBtn3 = within(screen.getByTestId('trake-slot-card-3')).getByRole('button', { name: /lock/i })
    await user.click(lockBtn3)

    const addBasketBtn = screen.getByTestId('trake-add-basket-btn')
    expect(addBasketBtn).toBeEnabled()
    await user.click(addBasketBtn)

    // 7. Check Submission / Basket tab
    await user.click(screen.getByTestId('tab-evidence'))
    await waitFor(() => {
      expect(screen.getByTestId('basket-items-list')).toBeInTheDocument()
    })

    // Assert structured chain is rendered without fake Ans: "TRAKE[...]"
    const basketRow = screen.getByTestId('basket-item-0')
    expect(basketRow).toHaveTextContent('TRAKE')
    expect(basketRow).toHaveTextContent('L25_V084')
    expect(basketRow).toHaveTextContent('4 Events')
    expect(screen.getByTestId('basket-trake-chain-0')).toHaveTextContent('58780 → 58900 → 58950 → 59000')
    expect(screen.queryByTestId('basket-answer-0')).not.toBeInTheDocument()
  }, 15000)
})
