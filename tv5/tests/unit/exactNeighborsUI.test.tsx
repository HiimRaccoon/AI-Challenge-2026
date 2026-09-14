import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { App } from '../../src/App'
import { AppProvider } from '../../src/state/AppContext'
import { initialAppState } from '../../src/state/appState'
import { SearchCandidate } from '../../src/types/contracts'
import { resetExactImageCache } from '../../src/api/exactImageCache'

const mockCandA: SearchCandidate = {
  query_id: 'kis-test',
  video_id: 'L25_V004',
  frame_id: 21294,
  timestamp_ms: 710509,
  source: 'fusion',
  rank: 1,
  score: 0.95,
  preprocess_run_id: 'run_v1_batch1',
}

const mockCandB: SearchCandidate = {
  query_id: 'kis-test',
  video_id: 'L25_V005',
  frame_id: 33000,
  timestamp_ms: 880000,
  source: 'fusion',
  rank: 2,
  score: 0.88,
  preprocess_run_id: 'run_v1_batch1',
}

describe('P0 Exact-Neighbor & Exact-Frame Rendering Truthfulness', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    resetExactImageCache()
  })

  // TEST 1 — Distinct live neighbors
  it('TEST 1: renders distinct live neighbor frames and does NOT duplicate the anchor across tiles', async () => {
    const fetchMock = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (url.endsWith('/health')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ status: 'ok', mode: 'live', preprocess_run_id: 'run_v1_batch1' }),
        } as Response)
      }
      if (url.endsWith('/exact-frame/neighbors')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            video_id: 'L25_V004',
            anchor_frame_id: 21294,
            provenance_mode: 'live',
            steps: [
              {
                offset: -1,
                frame: {
                  video_id: 'L25_V004',
                  frame_id: 21293,
                  timestamp_ms: 710476,
                  pts: 21314782,
                  time_base: '1/12800',
                  preprocess_run_id: 'run_v1_batch1',
                  mapping_guaranteed: true,
                  submission_selectable: true,
                  identity_source: 'certified_run_consecutive_original_decode',
                },
              },
              {
                offset: 0,
                frame: {
                  video_id: 'L25_V004',
                  frame_id: 21294,
                  timestamp_ms: 710509,
                  pts: 21315294,
                  time_base: '1/12800',
                  preprocess_run_id: 'run_v1_batch1',
                  mapping_guaranteed: true,
                  submission_selectable: true,
                  identity_source: 'certified_run_consecutive_original_decode',
                },
              },
              {
                offset: 1,
                frame: {
                  video_id: 'L25_V004',
                  frame_id: 21295,
                  timestamp_ms: 710543,
                  pts: 21315806,
                  time_base: '1/12800',
                  preprocess_run_id: 'run_v1_batch1',
                  mapping_guaranteed: true,
                  submission_selectable: true,
                  identity_source: 'certified_run_consecutive_original_decode',
                },
              },
            ],
          }),
        } as Response)
      }
      if (url.endsWith('/exact-frame/image')) {
        const body = JSON.parse(init?.body as string)
        const relOff = body.offsets[0] ?? 0
        const frameId = relOff === -1 ? 21293 : relOff === 1 ? 21295 : 21294
        const headers = new Headers({
          'x-original-video-id': 'L25_V004',
          'x-original-frame-id': String(frameId),
          'x-pts': '21315294',
          'x-time-base': '1/12800',
          'x-timestamp-ms': '710509',
          'x-preprocess-run-id': 'run_v1_batch1',
          'x-submission-selectable': 'true',
        })
        return Promise.resolve({
          ok: true,
          blob: async () => new Blob(['jpeg-bytes'], { type: 'image/jpeg' }),
          headers,
        } as unknown as Response)
      }
      return Promise.reject(new Error(`Unhandled: ${url}`))
    })
    global.fetch = fetchMock

    render(
      <AppProvider
        initialState={{
          ...initialAppState,
          activeTab: 'inspection',
          mode: 'live',
          candidates: [mockCandA],
          activeCandidate: mockCandA,
          anchorCandidate: mockCandA,
          cumulativeOffset: 0,
        }}
      >
        <App />
      </AppProvider>
    )

    // Wait for exact neighbors to render
    await waitFor(() => {
      expect(screen.getByTestId('neighbor-card-0')).toHaveTextContent('Frame 21294')
      expect(screen.getByTestId('neighbor-card--1')).toHaveTextContent('Frame 21293')
      expect(screen.getByTestId('neighbor-card-1')).toHaveTextContent('Frame 21295')
    })

    // Verify it is NOT 3 copies of 21294
    const neighborStrip = screen.getByTestId('context-strip')
    const allText = neighborStrip.textContent || ''
    expect(allText).toContain('21293')
    expect(allText).toContain('21294')
    expect(allText).toContain('21295')
  })

  // TEST 2 — No local arithmetic: verifies UI displays exact non-consecutive mock IDs from backend
  it('TEST 2: displays exact backend-proven IDs without local frame math', async () => {
    global.fetch = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (url.endsWith('/health')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ status: 'ok', mode: 'live' }),
        } as Response)
      }
      if (url.endsWith('/exact-frame/neighbors')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            video_id: 'L25_V004',
            anchor_frame_id: 12345,
            provenance_mode: 'live',
            steps: [
              {
                offset: -1,
                frame: {
                  video_id: 'L25_V004',
                  frame_id: 9001, // Deliberately non-consecutive
                  timestamp_ms: 1000,
                  pts: 1000,
                  time_base: '1/1000',
                  preprocess_run_id: 'run_v1',
                  mapping_guaranteed: true,
                  submission_selectable: true,
                  identity_source: 'certified',
                },
              },
              {
                offset: 0,
                frame: {
                  video_id: 'L25_V004',
                  frame_id: 12345,
                  timestamp_ms: 2000,
                  pts: 2000,
                  time_base: '1/1000',
                  preprocess_run_id: 'run_v1',
                  mapping_guaranteed: true,
                  submission_selectable: true,
                  identity_source: 'certified',
                },
              },
              {
                offset: 1,
                frame: {
                  video_id: 'L25_V004',
                  frame_id: 700000, // Deliberately non-consecutive
                  timestamp_ms: 3000,
                  pts: 3000,
                  time_base: '1/1000',
                  preprocess_run_id: 'run_v1',
                  mapping_guaranteed: true,
                  submission_selectable: true,
                  identity_source: 'certified',
                },
              },
            ],
          }),
        } as Response)
      }
      if (url.endsWith('/exact-frame/image')) {
        const body = JSON.parse(init?.body as string)
        const relOff = body.offsets[0] ?? 0
        const frameId = relOff === -1 ? 9001 : relOff === 1 ? 700000 : 12345
        return Promise.resolve({
          ok: true,
          blob: async () => new Blob(['jpeg-bytes'], { type: 'image/jpeg' }),
          headers: new Headers({
            'x-original-video-id': 'L25_V004',
            'x-original-frame-id': String(frameId),
          }),
        } as unknown as Response)
      }
      return Promise.reject(new Error(`Unhandled: ${url}`))
    })

    render(
      <AppProvider
        initialState={{
          ...initialAppState,
          activeTab: 'inspection',
          mode: 'live',
          candidates: [mockCandA],
          activeCandidate: mockCandA,
          anchorCandidate: mockCandA,
          cumulativeOffset: 0,
        }}
      >
        <App />
      </AppProvider>
    )

    await waitFor(() => {
      expect(screen.getByTestId('neighbor-card--1')).toHaveTextContent('Frame 9001')
      expect(screen.getByTestId('neighbor-card-0')).toHaveTextContent('Frame 12345')
      expect(screen.getByTestId('neighbor-card-1')).toHaveTextContent('Frame 700000')
    })
  })

  // TEST 3 — Unavailable neighbor handling
  it('TEST 3: renders explicit boundary/unavailable state when neighbor frame is null', async () => {
    global.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.endsWith('/health')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ status: 'ok', mode: 'live' }),
        } as Response)
      }
      if (url.endsWith('/exact-frame/neighbors')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            video_id: 'L25_V004',
            anchor_frame_id: 21294,
            provenance_mode: 'live',
            steps: [
              {
                offset: -1,
                frame: null,
                degraded_reason: 'boundary',
              },
              {
                offset: 0,
                frame: {
                  video_id: 'L25_V004',
                  frame_id: 21294,
                  timestamp_ms: 710509,
                  pts: 21315294,
                  time_base: '1/12800',
                  preprocess_run_id: 'run_v1',
                  mapping_guaranteed: true,
                  submission_selectable: true,
                  identity_source: 'certified',
                },
              },
            ],
          }),
        } as Response)
      }
      if (url.endsWith('/exact-frame/image')) {
        return Promise.resolve({
          ok: true,
          blob: async () => new Blob(['jpeg-bytes'], { type: 'image/jpeg' }),
          headers: new Headers({
            'x-original-video-id': 'L25_V004',
            'x-original-frame-id': '21294',
          }),
        } as unknown as Response)
      }
      return Promise.reject(new Error(`Unhandled: ${url}`))
    })

    render(
      <AppProvider
        initialState={{
          ...initialAppState,
          activeTab: 'inspection',
          mode: 'live',
          candidates: [mockCandA],
          activeCandidate: mockCandA,
          anchorCandidate: mockCandA,
          cumulativeOffset: 0,
        }}
      >
        <App />
      </AppProvider>
    )

    await waitFor(() => {
      expect(screen.getByTestId('neighbor-card-0')).toHaveTextContent('Frame 21294')
      expect(screen.getByTestId('neighbor-card--1')).toHaveTextContent('Unavailable')
    })
    // Card -1 must NOT show anchor frame 21294
    expect(screen.getByTestId('neighbor-card--1')).not.toHaveTextContent('Frame 21294')
  })

  // TEST 4 — Anchor vs. Current inspected distinction
  it('TEST 4: stepping to +1 updates current inspected frame while anchor remains unchanged', async () => {
    const user = userEvent.setup()

    global.fetch = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (url.endsWith('/health')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ status: 'ok', mode: 'live' }),
        } as Response)
      }
      if (url.endsWith('/exact-frame/neighbors')) {
        const body = JSON.parse(init?.body as string)
        const cumOff = body.cumulative_offset
        return Promise.resolve({
          ok: true,
          json: async () => ({
            video_id: 'L25_V004',
            anchor_frame_id: 21294,
            provenance_mode: 'live',
            steps: [
              {
                offset: cumOff,
                frame: {
                  video_id: 'L25_V004',
                  frame_id: 21294 + cumOff,
                  timestamp_ms: 710509 + cumOff * 33,
                  pts: 21315294 + cumOff * 512,
                  time_base: '1/12800',
                  preprocess_run_id: 'run_v1',
                  mapping_guaranteed: true,
                  submission_selectable: true,
                  identity_source: 'certified',
                },
              },
            ],
          }),
        } as Response)
      }
      if (url.endsWith('/exact-frame/image')) {
        const body = JSON.parse(init?.body as string)
        const cumOff = body.cumulative_offset
        const frameId = 21294 + cumOff
        return Promise.resolve({
          ok: true,
          blob: async () => new Blob(['jpeg-bytes'], { type: 'image/jpeg' }),
          headers: new Headers({
            'x-original-video-id': 'L25_V004',
            'x-original-frame-id': String(frameId),
          }),
        } as unknown as Response)
      }
      return Promise.reject(new Error(`Unhandled: ${url}`))
    })

    render(
      <AppProvider
        initialState={{
          ...initialAppState,
          activeTab: 'inspection',
          mode: 'live',
          candidates: [mockCandA],
          activeCandidate: mockCandA,
          anchorCandidate: mockCandA,
          cumulativeOffset: 0,
        }}
      >
        <App />
      </AppProvider>
    )

    await waitFor(() => {
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('21294')
      expect(screen.getByTestId('cumulative-offset')).toHaveTextContent('0')
    })

    // Step Next (+1)
    await user.click(screen.getByTestId('btn-step-next'))

    await waitFor(() => {
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('21295')
      expect(screen.getByTestId('cumulative-offset')).toHaveTextContent('+1')
    })

    // Left shortlist card still displays anchor frame 21294
    expect(screen.getByTestId('shortlist-card-1')).toHaveTextContent('21294')
  })

  // TEST 5 — Crossing anchor: +1 -> 0 -> -1
  it('TEST 5: stepping across anchor +1 -> 0 -> -1 maintains correct signed cumulative state', async () => {
    const user = userEvent.setup()

    global.fetch = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (url.endsWith('/health')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ status: 'ok', mode: 'live' }),
        } as Response)
      }
      if (url.endsWith('/exact-frame/neighbors')) {
        const body = JSON.parse(init?.body as string)
        const cumOff = body.cumulative_offset
        return Promise.resolve({
          ok: true,
          json: async () => ({
            video_id: 'L25_V004',
            anchor_frame_id: 21294,
            provenance_mode: 'live',
            steps: [
              {
                offset: cumOff,
                frame: {
                  video_id: 'L25_V004',
                  frame_id: 21294 + cumOff,
                  timestamp_ms: 710509 + cumOff * 33,
                  pts: 21315294 + cumOff * 512,
                  time_base: '1/12800',
                  preprocess_run_id: 'run_v1',
                  mapping_guaranteed: true,
                  submission_selectable: true,
                  identity_source: 'certified',
                },
              },
            ],
          }),
        } as Response)
      }
      if (url.endsWith('/exact-frame/image')) {
        const body = JSON.parse(init?.body as string)
        const cumOff = body.cumulative_offset
        return Promise.resolve({
          ok: true,
          blob: async () => new Blob(['jpeg-bytes'], { type: 'image/jpeg' }),
          headers: new Headers({
            'x-original-video-id': 'L25_V004',
            'x-original-frame-id': String(21294 + cumOff),
          }),
        } as unknown as Response)
      }
      return Promise.reject(new Error(`Unhandled: ${url}`))
    })

    render(
      <AppProvider
        initialState={{
          ...initialAppState,
          activeTab: 'inspection',
          mode: 'live',
          candidates: [mockCandA],
          activeCandidate: mockCandA,
          anchorCandidate: mockCandA,
          cumulativeOffset: 0,
        }}
      >
        <App />
      </AppProvider>
    )

    // Step +1
    await user.click(screen.getByTestId('btn-step-next'))
    await waitFor(() => {
      expect(screen.getByTestId('cumulative-offset')).toHaveTextContent('+1')
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('21295')
    })

    // Step Anchor (0)
    await user.click(screen.getByTestId('btn-step-anchor'))
    await waitFor(() => {
      expect(screen.getByTestId('cumulative-offset')).toHaveTextContent('0')
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('21294')
    })

    // Step Prev (-1)
    await user.click(screen.getByTestId('btn-step-prev'))
    await waitFor(() => {
      expect(screen.getByTestId('cumulative-offset')).toHaveTextContent('-1')
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('21293')
    })
  })

  // TEST 6 — Stale request protection
  it('TEST 6: ignores late asynchronous response from previously selected candidate', async () => {
    const resolveRef = { current: null as ((val: any) => void) | null }

    global.fetch = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (url.endsWith('/health')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ status: 'ok', mode: 'live' }),
        } as Response)
      }
      if (url.endsWith('/exact-frame/neighbors')) {
        const body = JSON.parse(init?.body as string)
        if (body.video_id === 'L25_V004') {
          // Slow response for Candidate A
          return new Promise((resolve) => {
            resolveRef.current = () =>
              resolve({
                ok: true,
                json: async () => ({
                  video_id: 'L25_V004',
                  anchor_frame_id: 21294,
                  provenance_mode: 'live',
                  steps: [
                    {
                      offset: 0,
                      frame: {
                        video_id: 'L25_V004',
                        frame_id: 21294,
                        timestamp_ms: 710509,
                        pts: 21315294,
                        time_base: '1/12800',
                        preprocess_run_id: 'run_v1',
                        mapping_guaranteed: true,
                        submission_selectable: true,
                        identity_source: 'certified',
                      },
                    },
                  ],
                }),
              } as Response)
          })
        }
        // Fast response for Candidate B
        return Promise.resolve({
          ok: true,
          json: async () => ({
            video_id: 'L25_V005',
            anchor_frame_id: 33000,
            provenance_mode: 'live',
            steps: [
              {
                offset: 0,
                frame: {
                  video_id: 'L25_V005',
                  frame_id: 33000,
                  timestamp_ms: 880000,
                  pts: 33000000,
                  time_base: '1/12800',
                  preprocess_run_id: 'run_v1',
                  mapping_guaranteed: true,
                  submission_selectable: true,
                  identity_source: 'certified',
                },
              },
            ],
          }),
        } as Response)
      }
      if (url.endsWith('/exact-frame/image')) {
        const body = JSON.parse(init?.body as string)
        return Promise.resolve({
          ok: true,
          blob: async () => new Blob(['jpeg-bytes'], { type: 'image/jpeg' }),
          headers: new Headers({
            'x-original-video-id': body.video_id,
            'x-original-frame-id': String(body.frame_id),
          }),
        } as unknown as Response)
      }
      return Promise.reject(new Error(`Unhandled: ${url}`))
    })

    const user = userEvent.setup()

    render(
      <AppProvider
        initialState={{
          ...initialAppState,
          activeTab: 'inspection',
          mode: 'live',
          candidates: [mockCandA, mockCandB],
          activeCandidate: mockCandA,
          anchorCandidate: mockCandA,
          cumulativeOffset: 0,
        }}
      >
        <App />
      </AppProvider>
    )

    // Quickly switch to candidate B
    await user.click(screen.getByTestId('shortlist-card-2'))

    await waitFor(() => {
      expect(screen.getByTestId('inspected-video-id')).toHaveTextContent('L25_V005')
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('33000')
    })

    // Now resolve Candidate A's delayed response
    if (resolveRef.current) {
      resolveRef.current(null)
    }

    // Must remain Candidate B (L25_V005 / 33000)
    expect(screen.getByTestId('inspected-video-id')).toHaveTextContent('L25_V005')
    expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('33000')
  })

  // TEST 7 — Exact image usage for neighbor frames
  it('TEST 7: calls /exact-frame/image for neighbor tiles and does not call /thumbnails/', async () => {
    const exactImageCalls: string[] = []
    const thumbnailCalls: string[] = []

    global.fetch = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (url.endsWith('/health')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ status: 'ok', mode: 'live' }),
        } as Response)
      }
      if (url.endsWith('/exact-frame/neighbors')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            video_id: 'L25_V004',
            anchor_frame_id: 21294,
            provenance_mode: 'live',
            steps: [
              {
                offset: -1,
                frame: {
                  video_id: 'L25_V004',
                  frame_id: 21293,
                  timestamp_ms: 710476,
                  pts: 21314782,
                  time_base: '1/12800',
                  preprocess_run_id: 'run_v1',
                  mapping_guaranteed: true,
                  submission_selectable: true,
                  identity_source: 'certified',
                },
              },
              {
                offset: 0,
                frame: {
                  video_id: 'L25_V004',
                  frame_id: 21294,
                  timestamp_ms: 710509,
                  pts: 21315294,
                  time_base: '1/12800',
                  preprocess_run_id: 'run_v1',
                  mapping_guaranteed: true,
                  submission_selectable: true,
                  identity_source: 'certified',
                },
              },
              {
                offset: 1,
                frame: {
                  video_id: 'L25_V004',
                  frame_id: 21295,
                  timestamp_ms: 710543,
                  pts: 21315806,
                  time_base: '1/12800',
                  preprocess_run_id: 'run_v1',
                  mapping_guaranteed: true,
                  submission_selectable: true,
                  identity_source: 'certified',
                },
              },
            ],
          }),
        } as Response)
      }
      if (url.endsWith('/exact-frame/image')) {
        exactImageCalls.push(url)
        const body = JSON.parse(init?.body as string)
        const relOff = body.offsets[0] ?? 0
        const fid = relOff === -1 ? 21293 : relOff === 1 ? 21295 : 21294
        return Promise.resolve({
          ok: true,
          blob: async () => new Blob(['jpeg-bytes'], { type: 'image/jpeg' }),
          headers: new Headers({
            'x-original-video-id': 'L25_V004',
            'x-original-frame-id': String(fid),
          }),
        } as unknown as Response)
      }
      if (url.includes('/thumbnails/21293') || url.includes('/thumbnails/21295')) {
        thumbnailCalls.push(url)
      }
      return Promise.resolve({ ok: true, json: async () => ({}) } as Response)
    })

    render(
      <AppProvider
        initialState={{
          ...initialAppState,
          activeTab: 'inspection',
          mode: 'live',
          candidates: [mockCandA],
          activeCandidate: mockCandA,
          anchorCandidate: mockCandA,
          cumulativeOffset: 0,
        }}
      >
        <App />
      </AppProvider>
    )

    await waitFor(() => {
      expect(screen.getByTestId('neighbor-card--1')).toHaveTextContent('Frame 21293')
      expect(screen.getByTestId('neighbor-card-1')).toHaveTextContent('Frame 21295')
    })

    // /exact-frame/image MUST be called for exact frames
    expect(exactImageCalls.length).toBeGreaterThan(0)
    // Non-selected exact neighbors (21293, 21295) must NOT use thumbnail endpoint
    expect(thumbnailCalls).toHaveLength(0)
  })

  // TEST 8 — Fixture / Live isolation
  it('TEST 8: fixture preview synthetic behavior does not leak into live mode', async () => {
    global.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.endsWith('/health')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ status: 'ok', mode: 'live' }),
        } as Response)
      }
      if (url.endsWith('/exact-frame/neighbors')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            video_id: 'L25_V004',
            anchor_frame_id: 21294,
            provenance_mode: 'live',
            steps: [
              {
                offset: 0,
                frame: {
                  video_id: 'L25_V004',
                  frame_id: 21294,
                  timestamp_ms: 710509,
                  pts: 21315294,
                  time_base: '1/12800',
                  preprocess_run_id: 'run_v1',
                  mapping_guaranteed: true,
                  submission_selectable: true,
                  identity_source: 'certified_run_consecutive_original_decode',
                },
              },
            ],
          }),
        } as Response)
      }
      if (url.endsWith('/exact-frame/image')) {
        return Promise.resolve({
          ok: true,
          blob: async () => new Blob(['jpeg-bytes'], { type: 'image/jpeg' }),
          headers: new Headers({
            'x-original-video-id': 'L25_V004',
            'x-original-frame-id': '21294',
            'x-submission-selectable': 'true',
          }),
        } as unknown as Response)
      }
      return Promise.reject(new Error(`Unhandled: ${url}`))
    })

    render(
      <AppProvider
        initialState={{
          ...initialAppState,
          activeTab: 'inspection',
          mode: 'live',
          candidates: [mockCandA],
          activeCandidate: mockCandA,
          anchorCandidate: mockCandA,
          cumulativeOffset: 0,
        }}
      >
        <App />
      </AppProvider>
    )

    await waitFor(() => {
      // In live mode with certified proof, selectable badge must say "Selectable"
      expect(screen.getByTestId('submission-selectable-badge')).toHaveTextContent('Selectable')
      expect(screen.getByTestId('submission-selectable-badge')).not.toHaveTextContent('Preview')
    })
  })
})
