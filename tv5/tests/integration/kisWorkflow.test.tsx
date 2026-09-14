import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { App } from '../../src/App'
import { AppProvider } from '../../src/state/AppContext'

const mockCandidates = [
  {
    query_id: 'kis-001',
    video_id: 'L21_V001',
    frame_id: 19220,
    timestamp_ms: 640666,
    source: 'fusion',
    rank: 1,
    score: 0.0405,
    model_scores: { bge_vl: 0.109, metaclip2: 0.199 },
    model_ranks: { bge_vl: 45, metaclip2: 17 },
    preprocess_run_id: 'run_v1_batch1',
  },
  {
    query_id: 'kis-001',
    video_id: 'L21_V002',
    frame_id: 23940,
    timestamp_ms: 798000,
    source: 'fusion',
    rank: 2,
    score: 0.0371,
    preprocess_run_id: 'run_v1_batch1',
  },
]

describe('T026 / T027 KIS Workflow & Exact Stepping Integration', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('executes full operator journey: KIS search -> candidate selection -> video seek -> exact stepping', async () => {
    const user = userEvent.setup()

    global.fetch = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (url.endsWith('/health')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            status: 'ok',
            mode: 'live',
            preprocess_run_id: 'run_v1_batch1',
          }),
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

      if (url.endsWith('/exact-frame/neighbors')) {
        const body = JSON.parse(init?.body as string)
        const offset = body.cumulative_offset

        // Live certified response matching TV4 WP09 proof
        const provenFrameId = 19220 + offset // Backend generates this, not the frontend!
        return Promise.resolve({
          ok: true,
          json: async () => ({
            video_id: 'L21_V001',
            anchor_frame_id: 19220,
            provenance_mode: 'live',
            steps: [
              {
                offset: offset - 1,
                frame: {
                  video_id: 'L21_V001',
                  frame_id: provenFrameId - 1,
                  timestamp_ms: 640633,
                  pts: 9840128,
                  time_base: '1/15360',
                  preprocess_run_id: 'run_v1_batch1',
                  mapping_guaranteed: true,
                  submission_selectable: true,
                  identity_source: 'certified_run_consecutive_original_decode',
                  certification_id: 'e4-1b-run_v1_batch1-decoder-semantics',
                },
              },
              {
                offset: offset,
                frame: {
                  video_id: 'L21_V001',
                  frame_id: provenFrameId,
                  timestamp_ms: 640666,
                  pts: 9840640,
                  time_base: '1/15360',
                  preprocess_run_id: 'run_v1_batch1',
                  mapping_guaranteed: true,
                  submission_selectable: true,
                  identity_source: 'certified_run_consecutive_original_decode',
                  certification_id: 'e4-1b-run_v1_batch1-decoder-semantics',
                },
              },
              {
                offset: offset + 1,
                frame: {
                  video_id: 'L21_V001',
                  frame_id: provenFrameId + 1,
                  timestamp_ms: 640700,
                  pts: 9841152,
                  time_base: '1/15360',
                  preprocess_run_id: 'run_v1_batch1',
                  mapping_guaranteed: true,
                  submission_selectable: true,
                  identity_source: 'certified_run_consecutive_original_decode',
                  certification_id: 'e4-1b-run_v1_batch1-decoder-semantics',
                },
              },
            ],
          }),
        } as Response)
      }

      if (url.endsWith('/exact-frame/image')) {
        const body = JSON.parse(init?.body as string)
        const offset = body.cumulative_offset
        const provenFrameId = 19220 + offset

        const headers = new Headers({
          'x-original-video-id': 'L21_V001',
          'x-original-frame-id': String(provenFrameId),
          'x-pts': '9840640',
          'x-time-base': '1/15360',
          'x-timestamp-ms': '640666',
          'x-preprocess-run-id': 'run_v1_batch1',
          'x-exact-certification-id': 'e4-1b-run_v1_batch1-decoder-semantics',
          'x-submission-selectable': 'true',
        })
        return Promise.resolve({
          ok: true,
          blob: async () => new Blob(['fake-jpeg'], { type: 'image/jpeg' }),
          headers,
        } as unknown as Response)
      }

      return Promise.reject(new Error(`Unhandled URL: ${url}`))
    })

    render(
      <AppProvider>
        <App />
      </AppProvider>
    )

    // 1. Enter KIS query and submit
    const input = screen.getByTestId('kis-query-input')
    await user.type(input, 'người đi xe máy')

    const searchBtn = screen.getByTestId('kis-search-btn')
    await user.click(searchBtn)

    // 2. Candidate grid populates
    await waitFor(() => {
      expect(screen.getByTestId('candidate-card-1')).toBeInTheDocument()
      expect(screen.getByTestId('candidate-card-2')).toBeInTheDocument()
    })

    expect(screen.getByText('L21_V001')).toBeInTheDocument()
    expect(screen.getByText('F:19220')).toBeInTheDocument()
    expect(screen.getByText('SCORE: 0.0405')).toBeInTheDocument()

    // 3. Select Candidate 1 (L21_V001 / 19220)
    const card1 = screen.getByTestId('candidate-card-1')
    await user.click(card1)

    // 4. Inspection workspace opens with video player and anchor state
    await waitFor(() => {
      expect(screen.getByTestId('inspection-workspace')).toBeInTheDocument()
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('19220')
      expect(screen.getByTestId('inspected-video-id')).toHaveTextContent('L21_V001')
      expect(screen.getByTestId('cumulative-offset')).toHaveTextContent('0')
    })

    const videoPlayer = screen.getByTestId('original-video-player') as HTMLVideoElement
    expect(videoPlayer.src).toContain('/videos/L21_V001/stream')

    // 5. Step Next (+1)
    const btnNext = screen.getByTestId('btn-step-next')
    await user.click(btnNext)

    // 6. Verify cumulative offset and backend-proven frame ID updated
    await waitFor(() => {
      expect(screen.getByTestId('cumulative-offset')).toHaveTextContent('+1')
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('19221')
    })

    // 7. Step Next again (+2)
    await user.click(btnNext)

    await waitFor(() => {
      expect(screen.getByTestId('cumulative-offset')).toHaveTextContent('+2')
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('19222')
    })

    // 8. Step Previous back to +1
    const btnPrev = screen.getByTestId('btn-step-prev')
    await user.click(btnPrev)

    await waitFor(() => {
      expect(screen.getByTestId('cumulative-offset')).toHaveTextContent('+1')
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('19221')
    })

    // 9. Return to Anchor [0]
    const btnAnchor = screen.getByTestId('btn-step-anchor')
    await user.click(btnAnchor)

    await waitFor(() => {
      expect(screen.getByTestId('cumulative-offset')).toHaveTextContent('0')
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('19220')
    })
  })

  it('distinguishes exact stepping from explicit canonical commit (Use This Frame) and propagates canonical identity to left rail and Feedback Set Reference', async () => {
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
          json: async () => ({ query_id: 'kis-001', candidates: mockCandidates }),
        } as Response)
      }
      if (url.endsWith('/exact-frame/neighbors')) {
        const body = JSON.parse(init?.body as string)
        const anchorId = body.frame_id ?? 19220
        const offset = body.cumulative_offset
        const provenFrameId = anchorId + offset
        return Promise.resolve({
          ok: true,
          json: async () => ({
            video_id: 'L21_V001',
            anchor_frame_id: anchorId,
            provenance_mode: 'live',
            steps: [
              {
                offset: offset,
                frame: {
                  video_id: 'L21_V001',
                  frame_id: provenFrameId,
                  timestamp_ms: 640666 + offset * 33,
                  pts: 9840640 + offset * 512,
                  time_base: '1/15360',
                  preprocess_run_id: 'run_v1_batch1',
                  mapping_guaranteed: true,
                  submission_selectable: true,
                  identity_source: 'certified_run_consecutive_original_decode',
                  certification_id: 'e4-1b-run_v1_batch1-decoder-semantics',
                },
              },
            ],
          }),
        } as Response)
      }
      return Promise.resolve({ ok: true, json: async () => ({}) } as Response)
    })

    render(
      <AppProvider>
        <App />
      </AppProvider>
    )

    // Search KIS
    await user.type(screen.getByTestId('kis-query-input'), 'người chạy bộ')
    await user.click(screen.getByTestId('kis-search-btn'))

    await waitFor(() => {
      expect(screen.getByTestId('candidate-card-1')).toBeInTheDocument()
    })

    // Click candidate 1 to inspect
    await user.click(screen.getByTestId('candidate-card-1'))
    await waitFor(() => {
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('19220')
    })

    // Step Next (+1)
    await user.click(screen.getByTestId('btn-step-next'))
    await waitFor(() => {
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('19221')
    })

    // Assert: Before explicit commit, left shortlist rail card still shows original Frame 19220
    expect(screen.getByTestId('shortlist-card-1')).toHaveTextContent('19220')
    expect(screen.getByTestId('kis-set-canonical-frame-btn')).toBeInTheDocument()

    // Explicitly commit frame 19221
    await user.click(screen.getByTestId('kis-set-canonical-frame-btn'))

    // Assert: Left shortlist rail card and context now reflect committed Frame 19221
    await waitFor(() => {
      expect(screen.getByTestId('shortlist-card-1')).toHaveTextContent('19221')
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('19221')
      expect(screen.getByTestId('cumulative-offset')).toHaveTextContent('0')
    })

    // Return to Retrieval workspace
    await user.click(screen.getByTestId('tab-retrieval'))
    expect(screen.getByTestId('candidate-card-1')).toHaveTextContent('19221')

    // Set Reference on Candidate #1
    await user.click(screen.getByTestId('set-reference-btn-1'))
    expect(screen.getByTestId('feedback-reference-display')).toHaveTextContent('L21_V001')
    expect(screen.getByTestId('feedback-reference-display')).toHaveTextContent('19221')
  })

  it('P0 Regression: Exact Neighbor Re-Anchor preserves certified root anchor lineage, decodes exact image, and allows repeated stepping without anchor_not_found', async () => {
    const user = userEvent.setup()

    const backendRequests: Array<{ url: string; body: any }> = []

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
        const body = JSON.parse(init?.body as string)
        backendRequests.push({ url, body })

        // STRICT INVARIANT: WP09 only knows certified root anchor 19220
        const rootAnchorId = body.certified_anchor_frame_id ?? body.frame_id
        if (rootAnchorId !== 19220) {
          // If the frontend erroneously sends uncertified frame as anchor, fail like real WP09!
          return Promise.resolve({
            ok: true,
            json: async () => ({
              video_id: body.video_id,
              anchor_frame_id: rootAnchorId,
              degraded_reason: 'anchor_not_found',
              steps: [-2, -1, 0, 1, 2].map((off) => ({
                offset: body.cumulative_offset + off,
                degraded_reason: 'anchor_not_found',
                frame: null,
              })),
            }),
          } as Response)
        }

        const totalOffset = body.cumulative_offset
        return Promise.resolve({
          ok: true,
          json: async () => ({
            video_id: 'L21_V001',
            anchor_frame_id: 19220,
            provenance_mode: 'live',
            steps: [-2, -1, 0, 1, 2].map((rel) => {
              const eff = totalOffset + rel
              const provenFid = 19220 + eff
              return {
                offset: eff,
                degraded_reason: null,
                frame: {
                  video_id: 'L21_V001',
                  frame_id: provenFid,
                  timestamp_ms: 640666 + eff * 33,
                  pts: 9840640 + eff * 512,
                  time_base: '1/15360',
                  preprocess_run_id: 'run_v1_batch1',
                  mapping_guaranteed: true,
                  submission_selectable: true,
                  identity_source: 'certified_run_consecutive_original_decode',
                  certification_id: 'e4-1b-run_v1_batch1-decoder-semantics',
                },
              }
            }),
          }),
        } as Response)
      }

      if (url.endsWith('/exact-frame/image')) {
        const body = JSON.parse(init?.body as string)
        backendRequests.push({ url, body })

        const rootAnchorId = body.certified_anchor_frame_id ?? body.frame_id
        if (rootAnchorId !== 19220) {
          return Promise.resolve({
            ok: false,
            status: 409,
            statusText: 'Conflict',
            json: async () => ({ detail: 'exact frame image unavailable: proven step is unavailable' }),
          } as unknown as Response)
        }

        const effFid = 19220 + body.cumulative_offset + (body.offsets?.[0] ?? 0)
        const headers = new Headers({
          'x-original-video-id': 'L21_V001',
          'x-original-frame-id': String(effFid),
          'x-pts': String(9840640 + (effFid - 19220) * 512),
          'x-time-base': '1/15360',
          'x-timestamp-ms': String(640666 + (effFid - 19220) * 33),
          'x-preprocess-run-id': 'run_v1_batch1',
          'x-exact-certification-id': 'e4-1b-run_v1_batch1-decoder-semantics',
          'x-submission-selectable': 'true',
        })

        return Promise.resolve({
          ok: true,
          blob: async () => new Blob([`fake-jpeg-${effFid}`], { type: 'image/jpeg' }),
          headers,
        } as unknown as Response)
      }

      return Promise.reject(new Error(`Unhandled URL: ${url}`))
    })

    render(
      <AppProvider>
        <App />
      </AppProvider>
    )

    // 1. Search KIS
    await user.type(screen.getByTestId('kis-query-input'), 'người lái xe đạp')
    await user.click(screen.getByTestId('kis-search-btn'))

    await waitFor(() => {
      expect(screen.getByTestId('candidate-card-1')).toBeInTheDocument()
    })

    // 2. Select Candidate 1 (root anchor: Frame 19220)
    await user.click(screen.getByTestId('candidate-card-1'))
    await waitFor(() => {
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('19220')
      expect(screen.getByTestId('cumulative-offset')).toHaveTextContent('0')
    })

    // 3. Step forward to neighbor Frame 19221 (offset +1)
    await user.click(screen.getByTestId('btn-step-next'))
    await waitFor(() => {
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('19221')
      expect(screen.getByTestId('cumulative-offset')).toHaveTextContent('+1')
    })

    // 4. Commit Frame 19221 as canonical prediction ("Use This Frame")
    const commitBtn = screen.getByTestId('kis-set-canonical-frame-btn')
    expect(commitBtn).toBeInTheDocument()
    await user.click(commitBtn)

    // 5. Verify UI state immediately after commit:
    // - Inspected Frame remains 19221
    // - Cumulative offset resets to 0 (centered at 19221)
    // - Left shortlist rail reflects Frame 19221 and renders decoded exact preview
    // - Exact frame image is displayed without error
    await waitFor(() => {
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('19221')
      expect(screen.getByTestId('cumulative-offset')).toHaveTextContent('0')
      expect(screen.getByTestId('shortlist-card-1')).toHaveTextContent('19221')
      const shortlistImg = screen.getByTestId('shortlist-card-1').querySelector('img')
      expect(shortlistImg).toBeInTheDocument()
      expect(shortlistImg?.getAttribute('alt')).toContain('Rank 1')
    })

    // 6. Verify backend requests preserve certified_anchor_frame_id = 19220
    const lastNeighborReq = backendRequests.filter((r) => r.url.endsWith('/exact-frame/neighbors')).pop()
    expect(lastNeighborReq?.body.certified_anchor_frame_id).toBe(19220)
    expect(lastNeighborReq?.body.cumulative_offset).toBe(1)

    // 7. Step forward again to Frame 19222 (relative offset +1 from new center = cumulative_offset 2 from root)
    await user.click(screen.getByTestId('btn-step-next'))
    await waitFor(() => {
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('19222')
      expect(screen.getByTestId('cumulative-offset')).toHaveTextContent('+1')
    })

    const nextNeighborReq = backendRequests.filter((r) => r.url.endsWith('/exact-frame/neighbors')).pop()
    expect(nextNeighborReq?.body.certified_anchor_frame_id).toBe(19220)
    expect(nextNeighborReq?.body.cumulative_offset).toBe(2)

    // 8. Return to Retrieval tab and verify candidate #1 and Sequence Context preserve Frame 19221 and exact preview without degradation
    await user.click(screen.getByTestId('tab-retrieval'))
    expect(screen.getByTestId('candidate-card-1')).toHaveTextContent('19221')
    const retrievalImg = screen.getByTestId('candidate-card-1').querySelector('img')
    expect(retrievalImg).toBeInTheDocument()
    expect(screen.getByTestId('candidate-card-1')).not.toHaveTextContent('Preview unavailable')
    expect(screen.getByTestId('context-strip')).toBeInTheDocument()
    expect(screen.queryByText('anchor_not_found')).not.toBeInTheDocument()
  })
})
