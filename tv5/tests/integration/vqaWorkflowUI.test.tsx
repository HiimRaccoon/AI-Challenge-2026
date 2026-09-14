import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { App } from '../../src/App'
import { AppProvider } from '../../src/state/AppContext'

const mockVqaResponse = {
  query_id: 'qa-ui-001',
  provenance_mode: 'live',
  results: [
    {
      rank: 1,
      video_id: 'L05_V005',
      frame_id: 888,
      timestamp_ms: 29600,
      confidence: 0.95,
      answer: 'màu xanh dương',
      verified: true,
      manual_review: false,
      proposal: 'màu xanh dương',
      approved: false, // Backend never auto-approves
      verifier_status: 'verified',
      retry_count: 0,
      manual_required: false,
      status: 'verified',
      degraded_reasons: [],
      evidence: {
        query_id: 'qa-ui-001',
        query_text: 'người phụ nữ cầm chiếc cúp',
        question: 'Chiếc cúp có màu gì?',
        video_id: 'L05_V005',
        frame_id: 888,
        timestamp_ms: 29600,
        keyframe_path: 'keyframes/L05_V005/0888.jpg',
        selected_frames: [
          {
            video_id: 'L05_V005',
            frame_id: 888,
            timestamp_ms: 29600,
            keyframe_path: 'keyframes/L05_V005/0888.jpg',
            preprocess_run_id: 'run_v1_batch1',
            provenance: { source: 'visual' },
            submission_selectable: true,
          },
        ],
        ocr_evidence: [
          {
            detection_id: 'ocr-1',
            video_id: 'L05_V005',
            frame_id: 888,
            timestamp_ms: 29600,
            raw_text: 'CHAMPION 2026',
            normalized_text: 'champion 2026',
            bbox_xyxy_norm: [0.2, 0.3, 0.6, 0.4],
            confidence: 0.92,
            preprocess_run_id: 'run_v1_batch1',
            model_name: 'paddleocr',
          },
        ],
        asr_evidence: [
          {
            segment_id: 'asr-1',
            video_id: 'L05_V005',
            start_ms: 28000,
            end_ms: 31000,
            text: 'trao chiếc cúp màu xanh dương cho người chiến thắng',
            confidence: 0.96,
            language: 'vi',
            words: [
              { word: 'chiếc', probability: 0.99 },
              { word: 'cúp', probability: 0.98 },
              { word: 'màu', probability: 0.99 },
              { word: 'xanh', probability: 0.99 },
              { word: 'dương', probability: 0.97 },
            ],
          },
        ],
        object_evidence: [
          {
            detection_id: 'obj-1',
            video_id: 'L05_V005',
            frame_id: 888,
            timestamp_ms: 29600,
            label: 'cup',
            canonical_label: 'trophy',
            bbox_xyxy_norm: [0.4, 0.45, 0.65, 0.8],
            confidence: 0.89,
          },
        ],
        metadata_evidence: [
          {
            metadata_id: 'meta-1',
            video_id: 'L05_V005',
            source: 'title',
            values: { title: 'Chung Kết Vô Địch 2026' },
          },
        ],
        availability: {
          frames: 'available',
          ocr: 'available',
          asr: 'available',
          object: 'available',
          metadata: 'available',
        },
        ocr_texts: ['CHAMPION 2026'],
        asr_texts: ['trao chiếc cúp màu xanh dương cho người chiến thắng'],
        object_labels: ['cup'],
        neighbor_frame_ids: [880, 888, 896],
        provenance: { source: 'multimodal' },
      },
    },
    {
      rank: 2,
      video_id: 'L05_V008',
      frame_id: 1240,
      timestamp_ms: 41333,
      confidence: 0.75,
      answer: 'màu đỏ tươi',
      verified: false,
      manual_review: true,
      proposal: 'màu đỏ tươi',
      approved: false,
      verifier_status: 'unverified',
      retry_count: 0,
      manual_required: true,
      status: 'manual_required',
      degraded_reasons: [],
      evidence: {
        query_id: 'qa-ui-001',
        video_id: 'L05_V008',
        frame_id: 1240,
        timestamp_ms: 41333,
        keyframe_path: 'keyframes/L05_V008/1240.jpg',
        selected_frames: [],
        ocr_evidence: [],
        asr_evidence: [],
        object_evidence: [],
        metadata_evidence: [],
        availability: { frames: 'available' },
        ocr_texts: [],
        asr_texts: [],
        object_labels: [],
        neighbor_frame_ids: [],
        provenance: {},
      },
    },
  ],
}

describe('T031 — VQA Workflow & Operator UI Integration Tests', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('completes full operator journey: Query+Question -> VQA Answer -> Evidence Inspection -> Proposal -> Draft Edit -> Confirm -> Basket', async () => {
    const user = userEvent.setup()

    global.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.endsWith('/health')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ status: 'ok', mode: 'live', preprocess_run_id: 'run_v1_batch1' }),
        } as Response)
      }
      if (url.endsWith('/vqa/answer')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockVqaResponse,
        } as Response)
      }
      if (url.endsWith('/exact-frame/neighbors')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            video_id: 'L05_V005',
            anchor_frame_id: 888,
            steps: [
              { offset: -1, frame: { frame_id: 887, timestamp_ms: 29567, pts: '887', submission_selectable: true } },
              { offset: 0, frame: { frame_id: 888, timestamp_ms: 29600, pts: '888', submission_selectable: true } },
              { offset: 1, frame: { frame_id: 889, timestamp_ms: 29633, pts: '889', submission_selectable: true } },
            ],
          }),
        } as Response)
      }
      return Promise.reject(new Error(`Unhandled URL: ${url}`))
    })

    render(
      <AppProvider>
        <App />
      </AppProvider>
    )

    // 0. Verify initial KIS mode presentation
    expect(screen.getByTestId('header-task-badge')).toHaveTextContent('KIS')
    expect(screen.queryByTestId('vqa-question-input')).not.toBeInTheDocument()
    expect(screen.getByTestId('kis-search-btn')).toHaveTextContent('Search KIS')

    // 1. Switch to Q&A Task Mode
    const vqaModeBtn = screen.getByTestId('task-mode-vqa')
    await user.click(vqaModeBtn)

    expect(screen.getByTestId('header-task-badge')).toHaveTextContent('Q&A')
    expect(screen.getByTestId('kis-search-btn')).toHaveTextContent('Search Q&A')

    // 2. Enter Event Description and VQA Question in Retrieval workspace
    const queryInput = screen.getByTestId('kis-query-input')
    const questionInput = screen.getByTestId('vqa-question-input')

    await user.type(queryInput, 'người phụ nữ cầm chiếc cúp')
    await user.type(questionInput, 'Chiếc cúp có màu gì?')

    // 3. Click Search Q&A
    const searchBtn = screen.getByTestId('kis-search-btn')
    await user.click(searchBtn)

    // Verify candidate cards are displayed
    await waitFor(() => {
      expect(screen.getByTestId('candidate-card-1')).toBeInTheDocument()
      expect(screen.getByTestId('candidate-card-2')).toBeInTheDocument()
    })

    // 4. Click Candidate #1 to inspect in Inspection workspace
    await user.click(screen.getByTestId('candidate-card-1'))

    // Verify Inspection workspace is opened with target candidate
    await waitFor(() => {
      expect(screen.getByTestId('inspected-video-id')).toHaveTextContent('L05_V005')
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('888')
    })

    // 5. Verify Top Context Strip in Inspection Stage
    expect(screen.getByTestId('inspection-context-strip')).toBeInTheDocument()
    expect(screen.getByTestId('context-strip-event')).toHaveTextContent('người phụ nữ cầm chiếc cúp')
    expect(screen.getByTestId('context-strip-question')).toHaveTextContent('Chiếc cúp có màu gì?')

    // 6. Verify Main Q&A Region (Dedicated 2-Column Working Area Below Neighbors)
    expect(screen.getByTestId('inspection-main-qa-region')).toBeInTheDocument()
    expect(screen.getByTestId('vqa-answer-panel')).toBeInTheDocument()
    expect(screen.getByTestId('vqa-panel-question')).toHaveTextContent('Chiếc cúp có màu gì?')
    expect(screen.getByTestId('vqa-proposal-text')).toHaveTextContent('"màu xanh dương"')
    expect(screen.getByTestId('verifier-status-badge')).toHaveTextContent('Machine Verified')

    // P0 CRITICAL INVARIANT: Answer is initially unapproved
    expect(screen.getByTestId('vqa-unapproved-status')).toBeInTheDocument()
    expect(screen.queryByTestId('vqa-approved-card')).not.toBeInTheDocument()

    // 7. Verify Multimodal Evidence Inspector tabs inside Main Q&A Region
    expect(screen.getByTestId('evidence-inspector')).toBeInTheDocument()

    // Check OCR evidence tab
    expect(screen.getByTestId('pane-evidence-ocr')).toBeInTheDocument()
    expect(screen.getByText('"CHAMPION 2026"')).toBeInTheDocument()

    // Switch to ASR tab
    await user.click(screen.getByTestId('tab-evidence-asr'))
    expect(screen.getByTestId('pane-evidence-asr')).toBeInTheDocument()
    expect(
      screen.getByText('"trao chiếc cúp màu xanh dương cho người chiến thắng"')
    ).toBeInTheDocument()

    // Switch to Object tab
    await user.click(screen.getByTestId('tab-evidence-object'))
    expect(screen.getByTestId('pane-evidence-object')).toBeInTheDocument()
    expect(screen.getByText('cup')).toBeInTheDocument()
    expect(screen.getByText('(trophy)')).toBeInTheDocument()

    // 6. Test "Adopt as Draft"
    const adoptBtn = screen.getByTestId('btn-adopt-draft')
    await user.click(adoptBtn)

    const draftTextarea = screen.getByTestId('vqa-draft-input') as HTMLTextAreaElement
    expect(draftTextarea.value).toBe('màu xanh dương')

    // Operator edits draft with exact precision
    await user.clear(draftTextarea)
    await user.type(draftTextarea, 'màu xanh dương đậm (metallic blue)')

    // 7. Click "Confirm Answer"
    const confirmBtn = screen.getByTestId('btn-confirm-answer')
    await user.click(confirmBtn)

    // Verify Approved card appears with exact text
    await waitFor(() => {
      expect(screen.getByTestId('vqa-approved-card')).toBeInTheDocument()
      expect(screen.getByTestId('vqa-approved-text')).toHaveTextContent(
        'màu xanh dương đậm (metallic blue)'
      )
    })

    // 8. Click "Add to Submission Basket"
    const addBasketBtn = screen.getByTestId('btn-add-vqa-basket')
    await user.click(addBasketBtn)

    expect(addBasketBtn).toHaveTextContent('Added to Basket')

    // 9. Navigate to Evidence / Submission Tab and verify basket contents
    const evidenceTab = screen.getByTestId('tab-evidence')
    await user.click(evidenceTab)

    await waitFor(() => {
      expect(screen.getByTestId('submission-basket-card')).toBeInTheDocument()
      expect(screen.getByTestId('basket-count-badge')).toHaveTextContent('1 / 100 predictions')
      expect(screen.getByTestId('basket-answer-0')).toHaveTextContent(
        'màu xanh dương đậm (metallic blue)'
      )
    })

    // 10. Switch back to Inspection workspace and test invalidation on candidate change
    const inspectionTab = screen.getByTestId('tab-inspection')
    await user.click(inspectionTab)

    // Select Candidate #2 in shortlist
    const shortlistCards = screen.getAllByRole('button', { name: /L05_V008/i })
    if (shortlistCards.length > 0) {
      await user.click(shortlistCards[0])
    }

    // Invariant: approved state is now invalidated
    await waitFor(() => {
      expect(screen.getByTestId('vqa-unapproved-status')).toBeInTheDocument()
      expect(screen.queryByTestId('vqa-approved-card')).not.toBeInTheDocument()
    })
  }, 15000)

  it('handles empty evidence and verifier rejection without fabricating answers', async () => {
    const user = userEvent.setup()

    global.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.endsWith('/health')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ status: 'ok', mode: 'live', preprocess_run_id: 'run_v1_batch1' }),
        } as Response)
      }
      if (url.endsWith('/vqa/answer')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            query_id: 'qa-empty-001',
            provenance_mode: 'live',
            results: [
              {
                rank: 1,
                video_id: 'L05_V005',
                frame_id: 888,
                timestamp_ms: 29600,
                confidence: null,
                answer: '',
                verified: false,
                manual_review: true,
                proposal: '',
                approved: false,
                verifier_status: 'insufficient_evidence',
                retry_count: 0,
                manual_required: true,
                status: 'abstained',
                degraded_reasons: ['empty_evidence'],
                evidence: {
                  query_id: 'qa-empty-001',
                  video_id: 'L05_V005',
                  frame_id: 888,
                  timestamp_ms: 29600,
                  keyframe_path: null,
                  selected_frames: [],
                  ocr_evidence: [],
                  asr_evidence: [],
                  object_evidence: [],
                  metadata_evidence: [],
                  availability: { ocr: 'empty', asr: 'empty', object: 'empty' },
                  ocr_texts: [],
                  asr_texts: [],
                  object_labels: [],
                  neighbor_frame_ids: [],
                  provenance: {},
                },
              },
            ],
          }),
        } as Response)
      }
      return Promise.reject(new Error(`Unhandled URL: ${url}`))
    })

    render(
      <AppProvider>
        <App />
      </AppProvider>
    )

    // Switch to Q&A mode
    await user.click(screen.getByTestId('task-mode-vqa'))

    await user.type(screen.getByTestId('kis-query-input'), 'obscure dark video')
    await user.type(screen.getByTestId('vqa-question-input'), 'What is the sign?')
    await user.click(screen.getByTestId('kis-search-btn'))

    await waitFor(() => {
      expect(screen.getByTestId('candidate-card-1')).toBeInTheDocument()
    })

    await user.click(screen.getByTestId('candidate-card-1'))

    await waitFor(() => {
      expect(screen.getByTestId('vqa-proposal-text')).toHaveTextContent(
        'Empty proposal — insufficient machine evidence'
      )
      expect(screen.getByTestId('verifier-status-badge')).toHaveTextContent('Insufficient Evidence')
      expect(screen.getByText('empty_evidence')).toBeInTheDocument()
    })

    // Evidence tabs show truthful empty states
    expect(screen.getByText(/No OCR text detections found/i)).toBeInTheDocument()
  })

  it('clearly separates KIS and Q&A modes, hides question in KIS, and requires both in Q&A', async () => {
    const user = userEvent.setup()

    global.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.endsWith('/health')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ status: 'ok', mode: 'live', preprocess_run_id: 'run_v1_batch1' }),
        } as Response)
      }
      return Promise.reject(new Error(`Unhandled URL: ${url}`))
    })

    render(
      <AppProvider>
        <App />
      </AppProvider>
    )

    // 1. Initial KIS Mode Check
    expect(screen.getByTestId('header-task-badge')).toHaveTextContent('KIS')
    expect(screen.getByTestId('task-mode-kis')).toHaveClass('active-task')
    expect(screen.getByTestId('task-mode-vqa')).not.toHaveClass('active-task')
    expect(screen.queryByTestId('vqa-question-input')).not.toBeInTheDocument()
    expect(screen.queryByText(/VQA QUESTION \(OPTIONAL\)/i)).not.toBeInTheDocument()

    // TRAKE mode button is present in task selector
    const trakeBtn = screen.getByTestId('task-mode-trake')
    expect(trakeBtn).toBeInTheDocument()
    expect(trakeBtn).toHaveTextContent('TRAKE')

    // Search button in KIS mode
    const searchBtn = screen.getByTestId('kis-search-btn')
    expect(searchBtn).toHaveTextContent('Search KIS')
    expect(searchBtn).toBeDisabled() // Empty query

    // Type query in KIS mode -> search becomes enabled
    const queryInput = screen.getByTestId('kis-query-input')
    await user.type(queryInput, 'xe buýt màu đỏ')
    expect(searchBtn).toBeEnabled()

    // 2. Switch to Q&A Mode
    await user.click(screen.getByTestId('task-mode-vqa'))

    expect(screen.getByTestId('header-task-badge')).toHaveTextContent('Q&A')
    expect(screen.getByTestId('task-mode-vqa')).toHaveClass('active-task')
    expect(screen.getByTestId('task-mode-kis')).not.toHaveClass('active-task')

    // Question input is now visible and NOT labelled optional
    const questionInput = screen.getByTestId('vqa-question-input')
    expect(questionInput).toBeInTheDocument()
    expect(screen.queryByText(/optional/i)).not.toBeInTheDocument()

    // In Q&A mode, search button requires both Query AND Question
    expect(searchBtn).toHaveTextContent('Search Q&A')
    expect(searchBtn).toBeDisabled() // Question is still empty

    await user.type(questionInput, 'Biển số xe là bao nhiêu?')
    expect(searchBtn).toBeEnabled() // Both fields filled

    // 3. Switch back to KIS Mode
    await user.click(screen.getByTestId('task-mode-kis'))
    expect(screen.getByTestId('header-task-badge')).toHaveTextContent('KIS')
    expect(screen.queryByTestId('vqa-question-input')).not.toBeInTheDocument()
    expect(searchBtn).toHaveTextContent('Search KIS')
    expect(searchBtn).toBeEnabled() // Query still has text
  })

  it('P0 & P1: distinguishes VQA exact stepping from explicit answer frame commit (Use as Answer Frame), invalidates stale approvals, and ensures clean separated caption layout', async () => {
    const user = userEvent.setup()

    global.fetch = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (url.endsWith('/health')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ status: 'ok', mode: 'live', preprocess_run_id: 'run_v1_batch1' }),
        } as Response)
      }
      if (url.endsWith('/vqa/answer')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockVqaResponse,
        } as Response)
      }
      if (url.endsWith('/exact-frame/neighbors')) {
        const body = JSON.parse(init?.body as string)
        const anchorId = body.frame_id ?? 888
        const offset = body.cumulative_offset
        const provenFrameId = anchorId + offset
        return Promise.resolve({
          ok: true,
          json: async () => ({
            video_id: 'L05_V005',
            anchor_frame_id: anchorId,
            provenance_mode: 'live',
            steps: [
              {
                offset: offset,
                frame: {
                  video_id: 'L05_V005',
                  frame_id: provenFrameId,
                  timestamp_ms: 29600 + offset * 33,
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

    // Switch to Q&A mode
    await user.click(screen.getByTestId('task-mode-vqa'))

    // Type query + question and search
    await user.type(screen.getByTestId('kis-query-input'), 'người phụ nữ cầm chiếc cúp')
    await user.type(screen.getByTestId('vqa-question-input'), 'Chiếc cúp có màu gì?')
    await user.click(screen.getByTestId('kis-search-btn'))

    await waitFor(() => {
      expect(screen.getByTestId('candidate-card-1')).toBeInTheDocument()
    })

    // P1 Layout Check: Video ID and Frame text are rendered in distinct separated elements
    const card1 = screen.getByTestId('candidate-card-1')
    expect(within(card1).getByTestId('vqa-tile-vid')).toHaveTextContent('L05_V005')
    expect(within(card1).getByTestId('vqa-tile-fid')).toHaveTextContent('Frame 888')
    expect(within(card1).getByTestId('vqa-tile-caption-row')).toBeInTheDocument()

    // Click candidate 1 -> opens Inspection tab
    await user.click(card1)
    await waitFor(() => {
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('888')
    })

    // Draft answer & confirm
    await user.click(screen.getByTestId('btn-adopt-draft'))
    await user.click(screen.getByTestId('btn-confirm-answer'))
    expect(screen.getByTestId('vqa-approved-card')).toHaveTextContent('màu xanh dương')

    // Step Next (+1)
    await user.click(screen.getByTestId('btn-step-next'))
    await waitFor(() => {
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('889')
    })

    // Before explicit commit: shortlist rail card still shows original Frame 888
    expect(screen.getByTestId('vqa-shortlist-card-1')).toHaveTextContent('888')
    expect(screen.getByTestId('vqa-set-answer-frame-btn')).toBeInTheDocument()

    // Click explicit commit: Use as Answer Frame
    await user.click(screen.getByTestId('vqa-set-answer-frame-btn'))

    // After explicit commit:
    // 1) shortlist rail card updates to 889
    // 2) inspected frame remains 889 with cumulative offset reset to 0
    // 3) previous approved answer is invalidated
    await waitFor(() => {
      expect(screen.getByTestId('vqa-shortlist-card-1')).toHaveTextContent('889')
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('889')
      expect(screen.getByTestId('cumulative-offset')).toHaveTextContent('0')
    })
    expect(screen.queryByTestId('vqa-approved-card')).not.toBeInTheDocument()

    // Switch to Retrieval tab: VQA tile now displays Frame 889 with clean separated layout
    await user.click(screen.getByTestId('tab-retrieval'))
    const card1After = screen.getByTestId('candidate-card-1')
    expect(within(card1After).getByTestId('vqa-tile-fid')).toHaveTextContent('Frame 889')
    expect(within(card1After).getByTestId('vqa-tile-vid')).toHaveTextContent('L05_V005')
  })

  it('isolates proof between VQA candidate switches, renders initial exact image, preserves root lineage across repeated commits, and shows explicit evidence anchor distinction', async () => {
    const user = userEvent.setup()
    const neighborCalls: any[] = []

    global.fetch = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (url.endsWith('/health')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ status: 'ok', mode: 'live', preprocess_run_id: 'run_v1_batch1' }),
        } as Response)
      }
      if (url.endsWith('/vqa/answer')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockVqaResponse,
        } as Response)
      }
      if (url.endsWith('/exact-frame/neighbors')) {
        const body = JSON.parse(init?.body as string)
        neighborCalls.push(body)
        const anchorId = body.certified_anchor_frame_id ?? body.frame_id
        const offset = body.cumulative_offset
        const provenFrameId = anchorId + offset
        return Promise.resolve({
          ok: true,
          json: async () => ({
            video_id: body.video_id,
            anchor_frame_id: anchorId,
            provenance_mode: 'live',
            steps: [
              {
                offset: offset,
                frame: {
                  video_id: body.video_id,
                  frame_id: provenFrameId,
                  timestamp_ms: 29600 + offset * 33,
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
      if (url.endsWith('/exact-frame/image')) {
        const body = JSON.parse(init?.body as string)
        const anchorId = body.certified_anchor_frame_id ?? body.frame_id
        const targetFrameId = anchorId + body.cumulative_offset + (body.offsets?.[0] ?? 0)
        return Promise.resolve({
          ok: true,
          blob: async () => new Blob(['fake-jpg'], { type: 'image/jpeg' }),
          headers: new Headers({
            'x-video-id': body.video_id,
            'x-frame-id': String(targetFrameId),
            'x-pts': '1000',
            'x-time-base': '1/1000',
            'x-timestamp-ms': '1000',
            'x-preprocess-run-id': 'run_v1_batch1',
            'x-certification-id': 'cert-1',
            'x-submission-selectable': 'true',
          }),
        } as unknown as Response)
      }
      return Promise.resolve({ ok: true, json: async () => ({}) } as Response)
    })

    render(
      <AppProvider>
        <App />
      </AppProvider>
    )

    // 1. Switch to Q&A mode & Search
    await user.click(screen.getByTestId('task-mode-vqa'))
    await user.type(screen.getByTestId('kis-query-input'), 'người mặc áo')
    await user.type(screen.getByTestId('vqa-question-input'), 'màu gì?')
    await user.click(screen.getByTestId('kis-search-btn'))

    await waitFor(() => {
      expect(screen.getByTestId('candidate-card-1')).toBeInTheDocument()
      expect(screen.getByTestId('candidate-card-2')).toBeInTheDocument()
    })

    // 2. Select Candidate 1 (L05_V005 frame 888) and enter Inspection
    await user.click(screen.getByTestId('candidate-card-1'))
    await waitFor(() => {
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('888')
    })

    // 3. Immediately switch to Candidate 2 (L05_V008 frame 1240) in the left shortlist
    await user.click(screen.getByTestId('vqa-shortlist-card-2'))

    // Candidate 2's exact identity loads cleanly with NO proof mismatch error
    await waitFor(() => {
      expect(screen.getByTestId('inspected-video-id')).toHaveTextContent('L05_V008')
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('1240')
    })
    expect(screen.queryByText(/proof mismatch/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/Exact image proof mismatch/i)).not.toBeInTheDocument()

    // 4. Step +2 from Candidate 2 (frame 1240 -> 1242)
    await user.click(screen.getByTestId('btn-step-next'))
    await waitFor(() => {
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('1241')
    })
    await user.click(screen.getByTestId('btn-step-next'))
    await waitFor(() => {
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('1242')
    })

    // 5. Commit as Answer Frame (1242)
    await user.click(screen.getByTestId('vqa-set-answer-frame-btn'))

    await waitFor(() => {
      expect(screen.getByTestId('vqa-shortlist-card-2')).toHaveTextContent('1242')
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('1242')
      expect(screen.getByTestId('cumulative-offset')).toHaveTextContent('0')
    })

    // 6. Check Evidence vs Answer Frame explicit semantics
    expect(screen.getByTestId('evidence-source-tag')).toHaveTextContent('Evidence Source: L05_V008 · Frame 1240')
    expect(screen.getByTestId('vqa-target-badge')).toHaveTextContent('Answer Frame: L05_V008 · Frame 1242 (Rank #2) · Offset +2')

    // 7. Perform repeated exact stepping from committed frame 1242 (+1 -> 1243)
    await user.click(screen.getByTestId('btn-step-next'))
    await waitFor(() => {
      expect(screen.getByTestId('inspected-frame-id')).toHaveTextContent('1243')
    })

    // Verify neighbor call used certified root 1240 + cumulative offset 3
    const lastNeighborCall = neighborCalls[neighborCalls.length - 1]
    expect(lastNeighborCall.video_id).toBe('L05_V008')
    expect(lastNeighborCall.certified_anchor_frame_id).toBe(1240)
    expect(lastNeighborCall.cumulative_offset).toBe(3)

    // 8. Commit again at 1243
    await user.click(screen.getByTestId('vqa-set-answer-frame-btn'))
    await waitFor(() => {
      expect(screen.getByTestId('vqa-target-badge')).toHaveTextContent('Answer Frame: L05_V008 · Frame 1243 (Rank #2) · Offset +3')
    })

    // 9. Switch to Retrieval tab: verify exact-corrected candidate preview is rendered without black matte
    await user.click(screen.getByTestId('tab-retrieval'))
    const card2After = screen.getByTestId('candidate-card-2')
    expect(within(card2After).getByTestId('vqa-tile-fid')).toHaveTextContent('Frame 1243')
    expect(within(card2After).queryByText(/Preview unavailable/i)).not.toBeInTheDocument()
  })
})
