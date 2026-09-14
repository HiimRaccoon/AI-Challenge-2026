import { describe, it, expect, vi, beforeEach } from 'vitest'
import {
  fetchHealth,
  searchKis,
  fetchExactNeighbors,
  fetchExactImageBlob,
  getVideoStreamUrl,
  getThumbnailUrl,
  getKeyframeUrl,
  getBestKeyframeUrl,
} from '../../src/api/tv4Client'

describe('tv4Client API integration', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('fetches health status correctly', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        status: 'ok',
        mode: 'live',
        preprocess_run_id: 'run_v1_batch1',
      }),
    } as Response)

    const health = await fetchHealth()
    expect(health.status).toBe('ok')
    expect(health.mode).toBe('live')
    expect(health.preprocess_run_id).toBe('run_v1_batch1')
  })

  it('bounds top_k in searchKis request', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ query_id: 'q1', candidates: [] }),
    } as Response)
    global.fetch = fetchMock

    await searchKis({ query_text: 'xe may', top_k: 200 })

    expect(fetchMock).toHaveBeenCalledWith(
      '/kis/search',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          query_text: 'xe may',
          query_id: null,
          top_k: 100,
          weights: null,
        }),
      })
    )
  })

  it('calls exact-frame/neighbors with certified anchor parameters', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        video_id: 'L21_V001',
        anchor_frame_id: 19220,
        steps: [],
      }),
    } as Response)
    global.fetch = fetchMock

    await fetchExactNeighbors({
      video_id: 'L21_V001',
      frame_id: 19220,
      timestamp_ms: 640666,
      offsets: [-1, 0, 1],
      certified_anchor_frame_id: 19220,
      certified_anchor_timestamp_ms: 640666,
      cumulative_offset: 0,
    })

    expect(fetchMock).toHaveBeenCalledWith(
      '/exact-frame/neighbors',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          video_id: 'L21_V001',
          frame_id: 19220,
          timestamp_ms: 640666,
          offsets: [-1, 0, 1],
          certified_anchor_frame_id: 19220,
          certified_anchor_timestamp_ms: 640666,
          cumulative_offset: 0,
        }),
      })
    )
  })

  it('fetches exact image with parsed headers', async () => {
    const mockBlob = new Blob(['fake-image-bytes'], { type: 'image/jpeg' })
    const mockHeaders = new Headers({
      'x-original-video-id': 'L21_V001',
      'x-original-frame-id': '19220',
      'x-pts': '9840640',
      'x-time-base': '1/15360',
      'x-timestamp-ms': '640666',
      'x-preprocess-run-id': 'run_v1_batch1',
      'x-exact-certification-id': 'e4-1b-run_v1_batch1-decoder-semantics',
      'x-submission-selectable': 'true',
    })

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      blob: async () => mockBlob,
      headers: mockHeaders,
    } as unknown as Response)

    const result = await fetchExactImageBlob({
      video_id: 'L21_V001',
      frame_id: 19220,
      timestamp_ms: 640666,
      offsets: [0],
      certified_anchor_frame_id: 19220,
      certified_anchor_timestamp_ms: 640666,
      cumulative_offset: 0,
    })

    expect(result.blobUrl).toBe('blob:mock-blob-url')
    expect(result.headers.video_id).toBe('L21_V001')
    expect(result.headers.frame_id).toBe(19220)
    expect(result.headers.submission_selectable).toBe(true)
  })

  it('constructs correct media URLs', () => {
    expect(getVideoStreamUrl('L21_V001')).toBe('/videos/L21_V001/stream')
    expect(getThumbnailUrl('L21_V001', 19220)).toBe('/videos/L21_V001/thumbnails/19220.jpg')
    expect(getKeyframeUrl('L21_V001', 19220)).toBe('/videos/L21_V001/keyframes/19220.jpg')
  })

  it('resolves a stale frame to the nearest keyframe timestamp for preview', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ([
        {
          video_id: 'L23_V005', frame_id: 6062, timestamp_ms: 242480,
          keyframe_path: 'keyframes/L23_V005/367.jpg',
        },
        {
          video_id: 'L23_V005', frame_id: 5000, timestamp_ms: 242560,
          keyframe_path: 'keyframes/L23_V005/300.jpg',
        },
      ]),
    } as Response)

    const url = await getBestKeyframeUrl('L23_V005', 6064, 242560)

    expect(url).toBe('/keyframe-image/L23_V005/300.jpg')
  })
})
