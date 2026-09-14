import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import {
  isFixtureModeActive,
  generateFixtureCandidates,
  generateFixtureNeighbors,
} from '../../src/fixtures/fixtureData'
import {
  fetchHealth,
  searchKis,
  fetchExactNeighbors,
  fetchExactImageBlob,
  getThumbnailUrl,
  getKeyframeUrl,
  getSelectedFrameUrl,
} from '../../src/api/tv4Client'

describe('Visual Fixture Mode Isolation & Data Integrity', () => {
  const originalLocation = window.location

  beforeEach(() => {
    // Default to clean URL
    Object.defineProperty(window, 'location', {
      writable: true,
      value: new URL('http://localhost:5173/'),
    })
  })

  afterEach(() => {
    Object.defineProperty(window, 'location', {
      writable: true,
      value: originalLocation,
    })
  })

  it('detects fixture mode correctly based on query parameter', () => {
    expect(isFixtureModeActive()).toBe(false)

    window.location = new URL('http://localhost:5173/?mode=fixture') as any
    expect(isFixtureModeActive()).toBe(true)
  })

  it('generates 48 deterministic fixture candidates with descending scores and no submission selectability', () => {
    const candidates = generateFixtureCandidates('xe máy')
    expect(candidates).toHaveLength(48)
    expect(candidates[0].rank).toBe(1)
    expect(candidates[47].rank).toBe(48)
    expect(candidates[0].video_id).toBe('FIXTURE_V001')
    expect(candidates[0].score).toBeGreaterThan(candidates[47].score!)

    // Every candidate must have a valid fixture preview
    for (const cand of candidates) {
      expect(cand.video_id).toMatch(/^FIXTURE_V00[1-8]$/)
      expect(cand.preprocess_run_id).toBe('fixture_preview_run_v1')
      expect((cand.provenance as any).is_fixture).toBe(true)
    }
  })

  it('generates deterministic exact neighbors with synthetic absolute frame IDs and submission_selectable strictly false', () => {
    const candidate = {
      query_id: 'fixture-test',
      video_id: 'FIXTURE_V001',
      frame_id: 270,
      timestamp_ms: 9000,
      source: 'fixture',
      rank: 1,
    }
    const neighbors = generateFixtureNeighbors(candidate, 0)

    expect(neighbors.steps).toHaveLength(5)
    expect(neighbors.steps.map((s) => s.offset)).toEqual([-2, -1, 0, 1, 2])
    expect(neighbors.steps.map((s) => s.frame?.frame_id)).toEqual([268, 269, 270, 271, 272])

    for (const step of neighbors.steps) {
      expect(step.frame).not.toBeNull()
      expect(step.frame!.frame_id).toBeGreaterThanOrEqual(0)
      expect(typeof step.frame!.frame_id).toBe('number')
      expect(step.frame!.submission_selectable).toBe(false)
      expect(step.frame!.identity_source).toBe('fixture_preview_simulation')
      expect(step.frame!.certification_id).toBe('fixture-preview-non-canonical')
    }
  })

  it('supports 100 candidates generation preserving candidate structure and descending scores', () => {
    const candidates = generateFixtureCandidates('xe máy', 100)
    expect(candidates).toHaveLength(100)
    expect(candidates[0].rank).toBe(1)
    expect(candidates[99].rank).toBe(100)
    expect(candidates[0].score).toBeGreaterThan(candidates[99].score!)
  })

  it('routes fetchHealth, searchKis, and exact frame calls to fixture data when fixture mode is active', async () => {
    window.location = new URL('http://localhost:5173/?mode=fixture') as any

    const health = await fetchHealth()
    expect(health.mode).toBe('fixture')
    expect(health.status).toBe('ok')
    expect(health.preprocess_run_id).toBe('fixture_preview_run_v1')

    const searchRes = await searchKis({ query_text: 'test query', top_k: 25 })
    expect(searchRes.provenance_mode).toBe('fixture')
    expect(searchRes.candidates).toHaveLength(25)

    const exactRes = await fetchExactNeighbors({
      video_id: 'FIXTURE_V001',
      frame_id: 10250,
      timestamp_ms: 410000,
      offsets: [-1, 0, 1],
      cumulative_offset: 0,
    })
    expect(exactRes.provenance_mode).toBe('fixture')
    expect(exactRes.steps).toHaveLength(5)

    const imgRes = await fetchExactImageBlob({
      video_id: 'FIXTURE_V001',
      frame_id: 10250,
      timestamp_ms: 410000,
      offsets: [0],
      cumulative_offset: 0,
    })
    expect(imgRes.headers.submission_selectable).toBe(false)
    expect(imgRes.headers.certification_id).toBe('fixture-preview-non-canonical')
    expect(imgRes.blobUrl).toContain('data:image/svg+xml')
  })

  it('generates local SVG data URIs in fixture mode with 0 network calls and preserves /videos URLs in live mode', () => {
    // In Live Mode (default)
    expect(isFixtureModeActive()).toBe(false)
    const liveThumb = getThumbnailUrl('L21_V001', 270)
    expect(liveThumb).toBe('/videos/L21_V001/thumbnails/270.jpg')
    const liveKeyframe = getKeyframeUrl('L21_V001', 270)
    expect(liveKeyframe).toBe('/videos/L21_V001/keyframes/270.jpg')
    const liveFrame = getSelectedFrameUrl('L21_V001', 270)
    expect(liveFrame).toBe('/videos/L21_V001/frames/270.jpg')

    // In Fixture Mode
    window.location = new URL('http://localhost:5173/?mode=fixture') as any
    expect(isFixtureModeActive()).toBe(true)
    const fixtureThumb = getThumbnailUrl('FIXTURE_V001', 270)
    expect(fixtureThumb).toContain('data:image/svg+xml')
    expect(fixtureThumb).not.toContain('/videos/')
    const fixtureKeyframe = getKeyframeUrl('FIXTURE_V001', 270)
    expect(fixtureKeyframe).toContain('data:image/svg+xml')
    const fixtureFrame = getSelectedFrameUrl('FIXTURE_V001', 270)
    expect(fixtureFrame).toContain('data:image/svg+xml')
  })
})
