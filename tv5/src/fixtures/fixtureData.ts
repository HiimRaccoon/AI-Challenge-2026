import {
  ExactNeighborResponse,
  ExactStep,
  SearchCandidate,
} from '../types/contracts'

/**
 * Check if the frontend is currently running in development fixture mode
 * Triggered explicitly by `?mode=fixture` in the browser URL.
 */
export function isFixtureModeActive(): boolean {
  if (typeof window === 'undefined') return false
  const params = new URLSearchParams(window.location.search)
  return params.get('mode') === 'fixture'
}

/**
 * Deterministic local SVG Data URI for fixture preview media.
 * Completely frontend-local: 0 network requests, 0 latency, 0 503 errors.
 */
export function getFixturePreviewDataUri(videoId: string, frameId: number): string {
  const cleanVid = videoId.replace('FIXTURE_', '')
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 180" width="320" height="180">
  <rect width="320" height="180" fill="#090e17"/>
  <rect x="1" y="1" width="318" height="178" fill="none" stroke="#1e293b" stroke-width="1"/>
  <line x1="0" y1="90" x2="320" y2="90" stroke="#0f172a" stroke-width="1"/>
  <line x1="160" y1="0" x2="160" y2="180" stroke="#0f172a" stroke-width="1"/>
  <circle cx="160" cy="74" r="22" fill="#0f172a" stroke="#06b6d4" stroke-width="1.5"/>
  <polygon points="153,65 173,74 153,83" fill="#06b6d4"/>
  <text x="160" y="118" fill="#94a3b8" font-family="Segoe UI, -apple-system, sans-serif" font-size="12" font-weight="700" text-anchor="middle">${cleanVid}</text>
  <text x="160" y="136" fill="#64748b" font-family="Segoe UI, -apple-system, sans-serif" font-size="10.5" font-weight="600" text-anchor="middle">Frame ${frameId} · Synthetic Fixture</text>
</svg>`
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`
}

/**
 * 100 Deterministic verified video/frame pairs from TV1 corpus manifest
 * Every frame in this list corresponds to an existing valid thumbnail JPEG that returns HTTP 200.
 */
export const VERIFIED_FIXTURE_POOL: Array<{ video_id: string; frame_id: number; timestamp_ms: number }> = [
  // L21_V001 (15 frames)
  { video_id: 'L21_V001', frame_id: 0, timestamp_ms: 0 },
  { video_id: 'L21_V001', frame_id: 270, timestamp_ms: 9000 },
  { video_id: 'L21_V001', frame_id: 350, timestamp_ms: 11666 },
  { video_id: 'L21_V001', frame_id: 410, timestamp_ms: 13666 },
  { video_id: 'L21_V001', frame_id: 480, timestamp_ms: 16000 },
  { video_id: 'L21_V001', frame_id: 570, timestamp_ms: 19000 },
  { video_id: 'L21_V001', frame_id: 610, timestamp_ms: 20333 },
  { video_id: 'L21_V001', frame_id: 720, timestamp_ms: 24000 },
  { video_id: 'L21_V001', frame_id: 810, timestamp_ms: 27000 },
  { video_id: 'L21_V001', frame_id: 910, timestamp_ms: 30333 },
  { video_id: 'L21_V001', frame_id: 1140, timestamp_ms: 38000 },
  { video_id: 'L21_V001', frame_id: 1890, timestamp_ms: 63000 },
  { video_id: 'L21_V001', frame_id: 1990, timestamp_ms: 66333 },
  { video_id: 'L21_V001', frame_id: 2140, timestamp_ms: 71333 },
  { video_id: 'L21_V001', frame_id: 2230, timestamp_ms: 74333 },

  // L21_V002 (15 frames)
  { video_id: 'L21_V002', frame_id: 0, timestamp_ms: 0 },
  { video_id: 'L21_V002', frame_id: 10, timestamp_ms: 333 },
  { video_id: 'L21_V002', frame_id: 300, timestamp_ms: 10000 },
  { video_id: 'L21_V002', frame_id: 470, timestamp_ms: 15666 },
  { video_id: 'L21_V002', frame_id: 520, timestamp_ms: 17333 },
  { video_id: 'L21_V002', frame_id: 580, timestamp_ms: 19333 },
  { video_id: 'L21_V002', frame_id: 660, timestamp_ms: 22000 },
  { video_id: 'L21_V002', frame_id: 720, timestamp_ms: 24000 },
  { video_id: 'L21_V002', frame_id: 790, timestamp_ms: 26333 },
  { video_id: 'L21_V002', frame_id: 860, timestamp_ms: 28666 },
  { video_id: 'L21_V002', frame_id: 900, timestamp_ms: 30000 },
  { video_id: 'L21_V002', frame_id: 940, timestamp_ms: 31333 },
  { video_id: 'L21_V002', frame_id: 1150, timestamp_ms: 38333 },
  { video_id: 'L21_V002', frame_id: 1350, timestamp_ms: 45000 },
  { video_id: 'L21_V002', frame_id: 1370, timestamp_ms: 45666 },

  // L21_V003 (15 frames)
  { video_id: 'L21_V003', frame_id: 0, timestamp_ms: 0 },
  { video_id: 'L21_V003', frame_id: 224, timestamp_ms: 8960 },
  { video_id: 'L21_V003', frame_id: 312, timestamp_ms: 12480 },
  { video_id: 'L21_V003', frame_id: 360, timestamp_ms: 14400 },
  { video_id: 'L21_V003', frame_id: 416, timestamp_ms: 16640 },
  { video_id: 'L21_V003', frame_id: 432, timestamp_ms: 17280 },
  { video_id: 'L21_V003', frame_id: 464, timestamp_ms: 18560 },
  { video_id: 'L21_V003', frame_id: 512, timestamp_ms: 20480 },
  { video_id: 'L21_V003', frame_id: 680, timestamp_ms: 27200 },
  { video_id: 'L21_V003', frame_id: 736, timestamp_ms: 29440 },
  { video_id: 'L21_V003', frame_id: 776, timestamp_ms: 31040 },
  { video_id: 'L21_V003', frame_id: 968, timestamp_ms: 38720 },
  { video_id: 'L21_V003', frame_id: 1432, timestamp_ms: 57280 },
  { video_id: 'L21_V003', frame_id: 1600, timestamp_ms: 64000 },
  { video_id: 'L21_V003', frame_id: 1704, timestamp_ms: 68160 },

  // L21_V005 (15 frames)
  { video_id: 'L21_V005', frame_id: 0, timestamp_ms: 0 },
  { video_id: 'L21_V005', frame_id: 280, timestamp_ms: 9333 },
  { video_id: 'L21_V005', frame_id: 380, timestamp_ms: 12666 },
  { video_id: 'L21_V005', frame_id: 390, timestamp_ms: 13000 },
  { video_id: 'L21_V005', frame_id: 480, timestamp_ms: 16000 },
  { video_id: 'L21_V005', frame_id: 550, timestamp_ms: 18333 },
  { video_id: 'L21_V005', frame_id: 600, timestamp_ms: 20000 },
  { video_id: 'L21_V005', frame_id: 640, timestamp_ms: 21333 },
  { video_id: 'L21_V005', frame_id: 750, timestamp_ms: 25000 },
  { video_id: 'L21_V005', frame_id: 880, timestamp_ms: 29333 },
  { video_id: 'L21_V005', frame_id: 1070, timestamp_ms: 35666 },
  { video_id: 'L21_V005', frame_id: 1140, timestamp_ms: 38000 },
  { video_id: 'L21_V005', frame_id: 1360, timestamp_ms: 45333 },
  { video_id: 'L21_V005', frame_id: 1880, timestamp_ms: 62666 },
  { video_id: 'L21_V005', frame_id: 1950, timestamp_ms: 65000 },

  // L21_V006 (15 frames)
  { video_id: 'L21_V006', frame_id: 0, timestamp_ms: 0 },
  { video_id: 'L21_V006', frame_id: 290, timestamp_ms: 9666 },
  { video_id: 'L21_V006', frame_id: 500, timestamp_ms: 16666 },
  { video_id: 'L21_V006', frame_id: 510, timestamp_ms: 17000 },
  { video_id: 'L21_V006', frame_id: 620, timestamp_ms: 20666 },
  { video_id: 'L21_V006', frame_id: 670, timestamp_ms: 22333 },
  { video_id: 'L21_V006', frame_id: 830, timestamp_ms: 27666 },
  { video_id: 'L21_V006', frame_id: 1120, timestamp_ms: 37333 },
  { video_id: 'L21_V006', frame_id: 1420, timestamp_ms: 47333 },
  { video_id: 'L21_V006', frame_id: 1660, timestamp_ms: 55333 },
  { video_id: 'L21_V006', frame_id: 1850, timestamp_ms: 61666 },
  { video_id: 'L21_V006', frame_id: 1900, timestamp_ms: 63333 },
  { video_id: 'L21_V006', frame_id: 2170, timestamp_ms: 72333 },
  { video_id: 'L21_V006', frame_id: 2350, timestamp_ms: 78333 },
  { video_id: 'L21_V006', frame_id: 2410, timestamp_ms: 80333 },

  // L21_V007 (15 frames)
  { video_id: 'L21_V007', frame_id: 20, timestamp_ms: 666 },
  { video_id: 'L21_V007', frame_id: 280, timestamp_ms: 9333 },
  { video_id: 'L21_V007', frame_id: 320, timestamp_ms: 10666 },
  { video_id: 'L21_V007', frame_id: 330, timestamp_ms: 11000 },
  { video_id: 'L21_V007', frame_id: 460, timestamp_ms: 15333 },
  { video_id: 'L21_V007', frame_id: 670, timestamp_ms: 22333 },
  { video_id: 'L21_V007', frame_id: 730, timestamp_ms: 24333 },
  { video_id: 'L21_V007', frame_id: 960, timestamp_ms: 32000 },
  { video_id: 'L21_V007', frame_id: 1300, timestamp_ms: 43333 },
  { video_id: 'L21_V007', frame_id: 1370, timestamp_ms: 45666 },
  { video_id: 'L21_V007', frame_id: 1510, timestamp_ms: 50333 },
  { video_id: 'L21_V007', frame_id: 1610, timestamp_ms: 53666 },
  { video_id: 'L21_V007', frame_id: 1620, timestamp_ms: 54000 },
  { video_id: 'L21_V007', frame_id: 1750, timestamp_ms: 58333 },
  { video_id: 'L21_V007', frame_id: 1850, timestamp_ms: 61666 },

  // L21_V008 (10 frames)
  { video_id: 'L21_V008', frame_id: 8, timestamp_ms: 320 },
  { video_id: 'L21_V008', frame_id: 256, timestamp_ms: 10240 },
  { video_id: 'L21_V008', frame_id: 416, timestamp_ms: 16640 },
  { video_id: 'L21_V008', frame_id: 592, timestamp_ms: 23680 },
  { video_id: 'L21_V008', frame_id: 704, timestamp_ms: 28160 },
  { video_id: 'L21_V008', frame_id: 720, timestamp_ms: 28800 },
  { video_id: 'L21_V008', frame_id: 888, timestamp_ms: 35520 },
  { video_id: 'L21_V008', frame_id: 984, timestamp_ms: 39360 },
  { video_id: 'L21_V008', frame_id: 1144, timestamp_ms: 45760 },
  { video_id: 'L21_V008', frame_id: 1376, timestamp_ms: 55040 },
]

/**
 * Generates deterministic search candidates for fixture visual testing (up to 100 candidates)
 * All entries map to verified thumbnail images in the TV1 corpus.
 */
export function generateFixtureCandidates(query = 'xe máy', count = 48): SearchCandidate[] {
  const targetCount = Math.max(1, Math.min(100, count))
  const candidates: SearchCandidate[] = []

  for (let i = 0; i < targetCount; i++) {
    const item = VERIFIED_FIXTURE_POOL[i % VERIFIED_FIXTURE_POOL.length]
    const rank = i + 1
    const score = Number((0.089 - i * 0.00075).toFixed(4))
    const videoNum = (i % 8) + 1
    const fixtureVideoId = `FIXTURE_V00${videoNum}`

    candidates.push({
      query_id: `fixture-${query.slice(0, 10).replace(/\s+/g, '_')}-001`,
      video_id: fixtureVideoId,
      frame_id: item.frame_id,
      timestamp_ms: item.timestamp_ms,
      source: 'fixture_fusion',
      rank,
      score,
      model_scores: {
        siglip2: Number((score * 1.08).toFixed(3)),
        metaclip2: Number((score * 0.94).toFixed(3)),
      },
      preprocess_run_id: 'fixture_preview_run_v1',
      provenance: {
        is_fixture: true,
      },
    })
  }

  return candidates
}

/**
 * Generates deterministic synthetic neighbor response for fixture mode
 * Produces synthetic absolute frame IDs based on the anchor frame ID and offset.
 * Invariant: submission_selectable is FALSE on all fixture frames.
 */
export function generateFixtureNeighbors(
  anchorCandidate: SearchCandidate,
  targetOffset: number
): ExactNeighborResponse {
  const offsets = [targetOffset - 2, targetOffset - 1, targetOffset, targetOffset + 1, targetOffset + 2]

  const steps: ExactStep[] = offsets.map((off) => {
    // Synthetic absolute frame ID generation for fixture simulation
    const syntheticFrameId = Math.max(0, anchorCandidate.frame_id + off)
    const syntheticTimestampMs = Math.max(0, anchorCandidate.timestamp_ms + off * 40)
    const pts = syntheticFrameId * 512

    return {
      offset: off,
      degraded_reason: null,
      frame: {
        video_id: anchorCandidate.video_id,
        frame_id: syntheticFrameId,
        timestamp_ms: syntheticTimestampMs,
        pts,
        time_base: '1/12800',
        preprocess_run_id: 'fixture_preview_run_v1',
        mapping_guaranteed: false,
        submission_selectable: false, // Invariant: Fixtures are NEVER submission selectable!
        identity_source: 'fixture_preview_simulation',
        certification_id: 'fixture-preview-non-canonical',
      },
    }
  })

  return {
    video_id: anchorCandidate.video_id,
    anchor_frame_id: anchorCandidate.frame_id,
    degraded_reason: null,
    steps,
    provenance_mode: 'fixture',
    preprocess_run_id: 'fixture_preview_run_v1',
  }
}
