import React from 'react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { AppProvider, useAppDispatch, useAppState } from '../../src/state/AppContext'
import { EvidenceSubmissionWorkspace } from '../../src/components/EvidenceSubmissionWorkspace'
import { SearchCandidate, BasketItem } from '../../src/types/contracts'
import * as exporter from '../../src/utils/submissionExporter'

const mockCandidates: SearchCandidate[] = [
  {
    rank: 1,
    video_id: 'L01_V001',
    frame_id: 100,
    timestamp_ms: 4000,
    score: 0.95,
  },
  {
    rank: 2,
    video_id: 'L01_V002',
    frame_id: 250,
    timestamp_ms: 10000,
    score: 0.88,
  },
]

const BasketTestHarness: React.FC = () => {
  const state = useAppState()
  const dispatch = useAppDispatch()

  return (
    <div>
      <div data-testid="basket-length">{state.submissionBasket.length}</div>
      <button
        data-testid="btn-init-candidates"
        onClick={() => {
          dispatch({ type: 'SEARCH_SUCCESS', payload: { candidates: mockCandidates, query_id: 'q1' } })
        }}
      >
        Init Candidates
      </button>
      <button
        data-testid="btn-add-kis-direct"
        onClick={() => {
          dispatch({
            type: 'ADD_KIS_TO_BASKET',
            payload: { candidate: mockCandidates[0] },
          })
        }}
      >
        Add KIS Direct
      </button>
      <button
        data-testid="btn-add-vqa-direct"
        onClick={() => {
          dispatch({
            type: 'ADD_TO_BASKET',
            payload: {
              video_id: 'L02_V005',
              frame_id: 500,
              task: 'VQA',
              answer: 'áo màu xanh',
              added_at_utc: new Date().toISOString(),
            },
          })
        }}
      >
        Add VQA Direct
      </button>
      <button
        data-testid="btn-add-trake-direct"
        onClick={() => {
          dispatch({
            type: 'ADD_TO_BASKET',
            payload: {
              video_id: 'L03_V010',
              frame_id: 1000,
              task: 'TRAKE',
              frame_ids: [1000, 1200],
              event_labels: ['bắt đầu', 'kết thúc'],
              added_at_utc: new Date().toISOString(),
            },
          })
        }}
      >
        Add TRAKE Direct
      </button>
    </div>
  )
}

describe('T038 / T039 — Unified Submission Basket & 1-Click Export UI', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('allows adding KIS, VQA, and TRAKE items into the unified submission basket', async () => {
    const user = userEvent.setup()

    render(
      <AppProvider>
        <BasketTestHarness />
      </AppProvider>
    )

    expect(screen.getByTestId('basket-length')).toHaveTextContent('0')

    // 1. Add KIS item
    await user.click(screen.getByTestId('btn-add-kis-direct'))
    expect(screen.getByTestId('basket-length')).toHaveTextContent('1')

    // 2. Add duplicate KIS item -> should be ignored (duplicate prevention)
    await user.click(screen.getByTestId('btn-add-kis-direct'))
    expect(screen.getByTestId('basket-length')).toHaveTextContent('1')

    // 3. Add VQA item
    await user.click(screen.getByTestId('btn-add-vqa-direct'))
    expect(screen.getByTestId('basket-length')).toHaveTextContent('2')

    // 4. Add TRAKE item
    await user.click(screen.getByTestId('btn-add-trake-direct'))
    expect(screen.getByTestId('basket-length')).toHaveTextContent('3')
  })

  it('renders submission basket items and executes task-filtered 1-click CSV download export', async () => {
    const user = userEvent.setup()
    const downloadSpy = vi.spyOn(exporter, 'triggerBrowserDownload').mockImplementation(() => {})

    render(
      <AppProvider>
        <BasketTestHarness />
        <EvidenceSubmissionWorkspace />
      </AppProvider>
    )

    // Initially empty
    expect(screen.getByTestId('basket-empty-state')).toBeInTheDocument()

    // Add items
    await user.click(screen.getByTestId('btn-add-kis-direct'))
    await user.click(screen.getByTestId('btn-add-vqa-direct'))

    // Basket rendered
    expect(screen.getByTestId('basket-count-badge')).toHaveTextContent('2 / 100')
    expect(screen.getByTestId('basket-item-0')).toBeInTheDocument()
    expect(screen.getByTestId('basket-item-1')).toBeInTheDocument()

    // Click Export KIS CSV button (taskMode is KIS by default) -> exports strictly KIS items without mixing VQA
    const exportBtn = screen.getByTestId('btn-export-csv')
    expect(exportBtn).toBeInTheDocument()
    await user.click(exportBtn)

    expect(downloadSpy).toHaveBeenCalledTimes(1)
    expect(downloadSpy.mock.calls[0][0]).toContain('L01_V001,100')
    expect(downloadSpy.mock.calls[0][0]).not.toContain('L02_V005,500,áo màu xanh')

    // Test Clear All
    const clearBtn = screen.getByTestId('btn-clear-basket')
    await user.click(clearBtn)
    expect(screen.getByTestId('basket-length')).toHaveTextContent('0')
  })

  it('strictly filters export by task type without mixing KIS, VQA, and TRAKE rows', () => {
    const basket: BasketItem[] = [
      { video_id: 'L01_V001.mp4', frame_id: 120, task: 'KIS', added_at_utc: '' },
      { video_id: 'L01_V002', frame_id: 450, task: 'VQA', answer: 'xe đạp màu đỏ', added_at_utc: '' },
      { video_id: 'L01_V003', frame_id: 600, task: 'TRAKE', frame_ids: [600, 750, 900], added_at_utc: '' },
    ]

    const kisCsv = exporter.exportBasketToCsvString(basket, 'KIS')
    expect(kisCsv).toBe('L01_V001,120')

    const vqaCsv = exporter.exportBasketToCsvString(basket, 'VQA')
    expect(vqaCsv).toBe('L01_V002,450,xe đạp màu đỏ')

    const trakeCsv = exporter.exportBasketToCsvString(basket, 'TRAKE')
    expect(trakeCsv).toBe('L01_V003,600,750,900')
  })
})
