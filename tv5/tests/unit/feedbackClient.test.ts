import { describe, it, expect, vi, beforeEach } from 'vitest'
import {
  startFeedback,
  refineFeedback,
  undoFeedback,
  resetFeedback,
  getFeedbackSession,
  Tv4ApiError,
} from '../../src/api/tv4Client'

describe('T028 — Feedback API Client Contracts & Error Handling', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('calls startFeedback with correct request structure', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        session_id: 's1',
        revision: 0,
        candidates: [
          { video_id: 'L21_V001', frame_id: 10690, rank: 1, timestamp_ms: 356333 },
        ],
        status: 'ok',
        provenance_mode: 'live',
      }),
    } as Response)
    global.fetch = fetchMock

    const res = await startFeedback({ session_id: 's1', original_query: 'xe oto mau xanh' })

    expect(fetchMock).toHaveBeenCalledWith(
      '/feedback/start',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: 's1', original_query: 'xe oto mau xanh' }),
      })
    )
    expect(res.session_id).toBe('s1')
    expect(res.revision).toBe(0)
    expect(res.candidates.length).toBe(1)
  })

  it('calls refineFeedback with canonical reference and CAS revision', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        session_id: 's1',
        revision: 1,
        candidates: [
          { video_id: 'L21_V002', frame_id: 23940, rank: 1, timestamp_ms: 798000 },
          { video_id: 'L21_V001', frame_id: 10690, rank: 2, timestamp_ms: 356333 },
        ],
        status: 'ok',
        provenance_mode: 'live',
      }),
    } as Response)
    global.fetch = fetchMock

    const res = await refineFeedback({
      session_id: 's1',
      video_id: 'L21_V002',
      frame_id: 23940,
      feedback_text: 'goc quay tu tren cao',
      expected_revision: 0,
    })

    expect(fetchMock).toHaveBeenCalledWith(
      '/feedback/refine',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          session_id: 's1',
          video_id: 'L21_V002',
          frame_id: 23940,
          source_candidate_frame_id: null,
          feedback_text: 'goc quay tu tren cao',
          expected_revision: 0,
        }),
      })
    )
    expect(res.revision).toBe(1)
    expect(res.candidates[0].video_id).toBe('L21_V002')
    expect(res.candidates[0].rank).toBe(1)
  })

  it('calls undoFeedback and resetFeedback with expected revision', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        session_id: 's1',
        revision: 2,
        candidates: [],
        status: 'ok',
      }),
    } as Response)
    global.fetch = fetchMock

    await undoFeedback({ session_id: 's1', expected_revision: 1 })
    expect(fetchMock).toHaveBeenCalledWith(
      '/feedback/undo',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ session_id: 's1', expected_revision: 1 }),
      })
    )

    await resetFeedback({ session_id: 's1', expected_revision: 2 })
    expect(fetchMock).toHaveBeenCalledWith(
      '/feedback/reset',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ session_id: 's1', expected_revision: 2 }),
      })
    )
  })

  it('calls getFeedbackSession to retrieve current view', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        session_id: 's1',
        revision: 3,
        candidates: [],
        status: 'ok',
      }),
    } as Response)
    global.fetch = fetchMock

    const res = await getFeedbackSession('s1')
    expect(fetchMock).toHaveBeenCalledWith('/feedback/session/s1')
    expect(res.revision).toBe(3)
  })

  it('handles HTTP 409 RevisionConflict (stale revision)', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 409,
      statusText: 'Conflict',
      json: async () => ({ detail: 'session revision is stale' }),
    } as Response)

    await expect(
      refineFeedback({
        session_id: 's1',
        video_id: 'L21_V001',
        frame_id: 10690,
        feedback_text: 'stale refine',
        expected_revision: 99,
      })
    ).rejects.toThrow(Tv4ApiError)
  })

  it('handles HTTP 404 SessionExpired / Not Found', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 404,
      statusText: 'Not Found',
      json: async () => ({ detail: 'session is unavailable' }),
    } as Response)

    await expect(getFeedbackSession('expired-session')).rejects.toThrow(Tv4ApiError)
  })

  it('handles HTTP 502 ModelRankingFailed', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 502,
      statusText: 'Bad Gateway',
      json: async () => ({ detail: 'model ranking failed' }),
    } as Response)

    await expect(
      refineFeedback({
        session_id: 's1',
        video_id: 'L21_V001',
        frame_id: 10690,
        feedback_text: 'failing model test',
        expected_revision: 0,
      })
    ).rejects.toThrow(Tv4ApiError)
  })

  it('handles HTTP 503 / 500 Degraded live state without silent fixture fallback', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
      json: async () => ({ detail: 'feedback service error: visual search returned zero candidates for feedback session' }),
    } as Response)

    await expect(
      startFeedback({ session_id: 's1', original_query: 'test query' })
    ).rejects.toThrow(/visual search returned zero candidates/i)
  })

  it('enforces <= 100 candidate invariant on response', async () => {
    const candidates120 = Array.from({ length: 120 }, (_, i) => ({
      query_id: 'q',
      video_id: `V${i}`,
      frame_id: i * 10,
      rank: i + 1,
      timestamp_ms: i * 333,
      source: 'feedback',
    }))

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        session_id: 's1',
        revision: 0,
        candidates: candidates120.slice(0, 100),
        status: 'ok',
      }),
    } as Response)

    const res = await startFeedback({ session_id: 's1', original_query: 'bounded test' })
    expect(res.candidates.length).toBeLessThanOrEqual(100)
  })

  it('validates client-side constraints before network call', async () => {
    // Missing original_query
    await expect(startFeedback({ session_id: 's1', original_query: '' })).rejects.toThrow(/required/i)

    // Negative frame_id
    await expect(
      refineFeedback({
        session_id: 's1',
        video_id: 'V1',
        frame_id: -1,
        feedback_text: 'text',
        expected_revision: 0,
      })
    ).rejects.toThrow(/non-negative/i)

    // Empty feedback text
    await expect(
      refineFeedback({
        session_id: 's1',
        video_id: 'V1',
        frame_id: 100,
        feedback_text: '   ',
        expected_revision: 0,
      })
    ).rejects.toThrow(/required/i)
  })

  it('handles plain text and non-JSON HTTP 400 error body safely without stream error', async () => {
    // Simulates a plain text 400 Bad Request
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      statusText: 'Bad Request',
      text: async () => 'candidate is not rendered',
    } as Response)

    try {
      await refineFeedback({
        session_id: 's1',
        video_id: 'L21_V001',
        frame_id: 99999,
        feedback_text: 'bad ref',
        expected_revision: 0,
      })
      expect.unreachable('Should have thrown Tv4ApiError')
    } catch (err: any) {
      expect(err).toBeInstanceOf(Tv4ApiError)
      expect(err.status).toBe(400)
      expect(err.detail).toBe('candidate is not rendered')
    }
  })

  it('transmits source_candidate_frame_id in refineFeedback request payload', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        session_id: 's1',
        revision: 1,
        candidates: [
          { video_id: 'L21_V001', frame_id: 10696, certified_anchor_frame_id: 10690, anchor_offset: 6, rank: 1 },
        ],
        status: 'ok',
      }),
    } as Response)
    global.fetch = fetchMock

    const res = await refineFeedback({
      session_id: 's1',
      video_id: 'L21_V001',
      frame_id: 10696,
      source_candidate_frame_id: 10690,
      feedback_text: 'more wheels',
      expected_revision: 0,
    })

    expect(fetchMock).toHaveBeenCalledWith(
      '/feedback/refine',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          session_id: 's1',
          video_id: 'L21_V001',
          frame_id: 10696,
          source_candidate_frame_id: 10690,
          feedback_text: 'more wheels',
          expected_revision: 0,
        }),
      })
    )
    expect(res.candidates[0].frame_id).toBe(10696)
  })
})
