import { describe, it, expect, vi, beforeEach } from 'vitest'
import { fetchTrakeAlign, rerunTrakeEvent, Tv4ApiError } from '../../src/api/tv4Client'
import { TrakeRequest, TrakeRerunRequest } from '../../src/types/contracts'

describe('T032 — TRAKE API Client & Data Contract Characterization', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('serializes ordered events, query_text, and strategy into /trake/align request', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        query_id: 'trake-001',
        result: {
          video_id: 'L10_V010',
          frame_ids: [101, 156, 203, 251],
          event_scores: [0.9, 0.7, 0.85, 0.6],
          aggregate_score: 3.05,
          preprocess_run_id: 'run_v1_batch1',
        },
      }),
    } as Response)
    global.fetch = fetchMock

    const req: TrakeRequest = {
      query_text: 'Vận động viên thực hiện cú nhảy cao',
      events: ['Giậm nhảy', 'Bay qua xà', 'Tiếp đất', 'Đứng dậy'],
      query_id: 'trake-001',
      strategy: 'dp',
    }

    const resp = await fetchTrakeAlign(req)

    expect(fetchMock).toHaveBeenCalledWith(
      '/trake/align',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query_text: 'Vận động viên thực hiện cú nhảy cao',
          strategy: 'dp',
          events: ['Giậm nhảy', 'Bay qua xà', 'Tiếp đất', 'Đứng dậy'],
          query_id: 'trake-001',
        }),
      })
    )

    expect(resp.query_id).toBe('trake-001')
    expect(resp.result).not.toBeNull()
    expect(resp.result?.video_id).toBe('L10_V010')
    expect(resp.result?.frame_ids).toEqual([101, 156, 203, 251])
    expect(resp.result?.event_scores).toEqual([0.9, 0.7, 0.85, 0.6])
    expect(resp.result?.aggregate_score).toBe(3.05)
    expect(resp.result?.preprocess_run_id).toBe('run_v1_batch1')
  })

  it('rejects empty or whitespace-only query_text without dispatching network request', async () => {
    const fetchMock = vi.fn()
    global.fetch = fetchMock

    await expect(fetchTrakeAlign({ query_text: '' })).rejects.toThrow(
      'fetchTrakeAlign requires non-empty query_text'
    )
    await expect(fetchTrakeAlign({ query_text: '   ' })).rejects.toThrow(
      'fetchTrakeAlign requires non-empty query_text'
    )

    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('handles truthful null alignment result when no monotonic path exists', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        query_id: 'trake-no-align',
        result: null,
        message: 'no monotonic alignment found',
      }),
    } as Response)
    global.fetch = fetchMock

    const resp = await fetchTrakeAlign({
      query_text: 'impossible event sequence',
      events: ['Event A', 'Event B'],
    })

    expect(resp.result).toBeNull()
    expect(resp.message).toBe('no monotonic alignment found')
  })

  it('supports greedy strategy specification according to WP12 contract', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        query_id: 'trake-greedy',
        result: {
          video_id: 'L05_V001',
          frame_ids: [50, 100],
          event_scores: [0.8, 0.75],
          aggregate_score: 1.55,
          preprocess_run_id: 'run_v1_batch1',
        },
      }),
    } as Response)
    global.fetch = fetchMock

    const resp = await fetchTrakeAlign({
      query_text: 'fast sequence',
      events: ['A', 'B'],
      strategy: 'greedy',
    })

    expect(fetchMock).toHaveBeenCalledWith(
      '/trake/align',
      expect.objectContaining({
        body: expect.stringContaining('"strategy":"greedy"'),
      })
    )
    expect(resp.result?.video_id).toBe('L05_V001')
  })

  it('reruns only one unlocked event with locked temporal anchors', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        query_id: 'trake-rerun',
        event_index: 1,
        video_id: 'V1',
        lower_bound_ms: 1000,
        upper_bound_ms: 3000,
        alternatives: [],
      }),
    } as Response)
    global.fetch = fetchMock
    const request: TrakeRerunRequest = {
      query_text: 'three ordered events',
      events: ['one', 'two', 'three'],
      event_index: 1,
      video_id: 'V1',
      query_id: 'trake-rerun',
      exclude_frame_id: 20,
      top_k: 5,
      locked_events: [
        { event_index: 0, frame_id: 10, timestamp_ms: 1000 },
        { event_index: 2, frame_id: 30, timestamp_ms: 3000 },
      ],
    }

    await rerunTrakeEvent(request)

    expect(fetchMock).toHaveBeenCalledWith(
      '/trake/rerun-event',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          query_text: 'three ordered events',
          events: ['one', 'two', 'three'],
          event_index: 1,
          video_id: 'V1',
          locked_events: request.locked_events,
          query_id: 'trake-rerun',
          exclude_frame_id: 20,
          top_k: 5,
        }),
      })
    )
  })

  it('surfaces TV4 API error status and detail on backend failure', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 502,
      statusText: 'Bad Gateway',
      json: async () => ({ detail: 'TRAKE pipeline failed: DP solver error' }),
    } as Response)
    global.fetch = fetchMock

    await expect(
      fetchTrakeAlign({
        query_text: 'failing query',
        events: ['E1', 'E2'],
      })
    ).rejects.toThrow(Tv4ApiError)
  })
})
