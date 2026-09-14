import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { App } from '../../src/App'
import { AppProvider } from '../../src/state/AppContext'
import { initialAppState } from '../../src/state/appState'
import { generateFixtureCandidates } from '../../src/fixtures/fixtureData'

describe('Retrieval Contact Sheet & Media Geometry Contracts', () => {
  it('renders all candidates with 16:9 media stages and lazy loading attributes', () => {
    const candidates = generateFixtureCandidates('xe máy', 48)

    render(
      <AppProvider
        initialState={{
          ...initialAppState,
          activeTab: 'retrieval',
          mode: 'fixture',
          readiness: 'READY',
          preprocessRunId: 'fixture_preview_run_v1',
          queryText: 'xe máy',
          queryId: 'fixture-query-1',
          topK: 48,
          candidates,
          originalCandidates: candidates,
        }}
      >
        <App />
      </AppProvider>
    )

    // Results container exists
    const grid = screen.getByTestId('candidate-grid')
    expect(grid).toBeInTheDocument()

    // 48 candidate cards rendered
    for (let i = 1; i <= 48; i++) {
      const card = screen.getByTestId(`candidate-card-${i}`)
      expect(card).toBeInTheDocument()
      const mediaStage = card.querySelector('.tile-image-stage')
      expect(mediaStage).not.toBeNull()
      const img = card.querySelector('img.tile-image-element') as HTMLImageElement
      expect(img).not.toBeNull()
      expect(img.getAttribute('loading')).toBe('lazy')
      expect(img.getAttribute('decoding')).toBe('async')
    }
  })

  it('preserves candidate tile structure and 16:9 media stages at 100 results without compact mode', () => {
    const candidates100 = generateFixtureCandidates('xe máy', 100)

    render(
      <AppProvider
        initialState={{
          ...initialAppState,
          activeTab: 'retrieval',
          mode: 'fixture',
          readiness: 'READY',
          preprocessRunId: 'fixture_preview_run_v1',
          queryText: 'xe máy',
          queryId: 'fixture-query-1',
          topK: 100,
          candidates: candidates100,
          originalCandidates: candidates100,
        }}
      >
        <App />
      </AppProvider>
    )

    const grid = screen.getByTestId('candidate-grid')
    expect(grid).toBeInTheDocument()

    // Verify first, 50th, and 100th candidate cards
    for (const i of [1, 50, 100]) {
      const card = screen.getByTestId(`candidate-card-${i}`)
      expect(card).toBeInTheDocument()
      const mediaStage = card.querySelector('.tile-image-stage')
      expect(mediaStage).not.toBeNull()
      // Card has rank chip, metadata pane
      expect(card.querySelector('.tile-rank-chip')).not.toBeNull()
      expect(card.querySelector('.tile-caption-pane')).not.toBeNull()
    }
  })

  it('Inspection Workspace renders video with preload="metadata" and stable controls', () => {
    const candidates = generateFixtureCandidates('xe máy', 20)
    const anchor = candidates[0]

    render(
      <AppProvider
        initialState={{
          ...initialAppState,
          activeTab: 'inspection',
          mode: 'fixture',
          readiness: 'READY',
          preprocessRunId: 'fixture_preview_run_v1',
          queryText: 'xe máy',
          queryId: 'fixture-query-1',
          topK: 20,
          candidates,
          originalCandidates: candidates,
          activeCandidate: anchor,
          anchorCandidate: anchor,
        }}
      >
        <App />
      </AppProvider>
    )

    const videoEl = screen.getByTestId('original-video-player') as HTMLVideoElement
    expect(videoEl).toBeInTheDocument()
    expect(videoEl.getAttribute('preload')).toBe('metadata')
    expect(videoEl.src).toContain('/videos/L21_V001/stream')
  })

  it('preserves Sequence Context 16:9 media stages and 5 distinct offset cells', () => {
    const candidates = generateFixtureCandidates('xe máy', 10)
    const anchor = candidates[0] // Frame 270

    render(
      <AppProvider
        initialState={{
          ...initialAppState,
          activeTab: 'retrieval',
          mode: 'fixture',
          readiness: 'READY',
          preprocessRunId: 'fixture_preview_run_v1',
          queryText: 'xe máy',
          queryId: 'fixture-query-1',
          topK: 10,
          candidates,
          originalCandidates: candidates,
          activeCandidate: anchor,
          anchorCandidate: anchor,
        }}
      >
        <App />
      </AppProvider>
    )

    const seqBar = screen.getByTestId('context-strip')
    expect(seqBar).toBeInTheDocument()
    const cells = seqBar.querySelectorAll('.sequence-cell')
    expect(cells.length).toBe(5)
    cells.forEach((cell) => {
      const stage = cell.querySelector('.sequence-thumb-stage')
      expect(stage).not.toBeNull()
    })
  })

  it('Inspection Workspace preserves Exact Frame stage and neighbor card 16:9 geometry', () => {
    const candidates = generateFixtureCandidates('xe máy', 10)
    const anchor = candidates[0]

    render(
      <AppProvider
        initialState={{
          ...initialAppState,
          activeTab: 'inspection',
          mode: 'fixture',
          readiness: 'READY',
          preprocessRunId: 'fixture_preview_run_v1',
          queryText: 'xe máy',
          queryId: 'fixture-query-1',
          topK: 10,
          candidates,
          originalCandidates: candidates,
          activeCandidate: anchor,
          anchorCandidate: anchor,
        }}
      >
        <App />
      </AppProvider>
    )

    const exactImg = screen.getByTestId('exact-frame-image')
    expect(exactImg).toBeInTheDocument()

    const contextStrip = screen.getByTestId('context-strip')
    expect(contextStrip).toBeInTheDocument()
    const neighborCards = contextStrip.querySelectorAll('.neighbor-card')
    expect(neighborCards.length).toBe(5)
    neighborCards.forEach((card) => {
      const thumbBox = card.querySelector('.neighbor-thumb-box')
      expect(thumbBox).not.toBeNull()
    })
  })
})
