import React from 'react'
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { AppProvider, useAppDispatch, useAppState } from '../../src/state/AppContext'
import { VqaResponse } from '../../src/types/contracts'

const mockVqaResponse: VqaResponse = {
  query_id: 'qa-int-001',
  provenance_mode: 'live',
  results: [
    {
      rank: 1,
      video_id: 'L10_V010',
      frame_id: 5000,
      timestamp_ms: 166667,
      confidence: 0.92,
      answer: 'người đàn ông đội mũ bảo hiểm',
      verified: true,
      manual_review: false,
      proposal: 'người đàn ông đội mũ bảo hiểm',
      approved: false,
      verifier_status: 'verified',
      retry_count: 0,
      manual_required: false,
      status: 'verified',
      degraded_reasons: [],
      evidence: {
        query_id: 'qa-int-001',
        query_text: 'người đi xe máy dừng đèn đỏ',
        question: 'Ai đang điều khiển xe?',
        video_id: 'L10_V010',
        frame_id: 5000,
        timestamp_ms: 166667,
        keyframe_path: 'keyframes/L10_V010/5000.jpg',
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
    {
      rank: 2,
      video_id: 'L10_V012',
      frame_id: 8500,
      timestamp_ms: 283333,
      confidence: 0.78,
      answer: 'người phụ nữ mặc áo khoác đỏ',
      verified: false,
      manual_review: true,
      proposal: 'người phụ nữ mặc áo khoác đỏ',
      approved: false,
      verifier_status: 'unverified',
      retry_count: 0,
      manual_required: true,
      status: 'manual_required',
      degraded_reasons: [],
      evidence: {
        query_id: 'qa-int-001',
        query_text: 'người đi xe máy dừng đèn đỏ',
        question: 'Ai đang điều khiển xe?',
        video_id: 'L10_V012',
        frame_id: 8500,
        timestamp_ms: 283333,
        keyframe_path: 'keyframes/L10_V012/8500.jpg',
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

// Test harness exercising VQA state within AppContext
const VqaStateHarness: React.FC = () => {
  const state = useAppState()
  const dispatch = useAppDispatch()

  return (
    <div>
      <div data-testid="vqa-query">{state.queryText}</div>
      <div data-testid="vqa-question">{state.vqaQuestion}</div>
      <div data-testid="vqa-proposal">{state.vqaActiveResult?.proposal || 'none'}</div>
      <div data-testid="vqa-approved">{state.vqaApprovedAnswer || 'unapproved'}</div>
      <div data-testid="vqa-draft">{state.vqaDraftAnswer}</div>
      <div data-testid="basket-count">{state.submissionBasket.length}</div>

      <button
        data-testid="btn-init-vqa"
        onClick={() => {
          dispatch({ type: 'SET_QUERY_TEXT', payload: 'người đi xe máy dừng đèn đỏ' })
          dispatch({ type: 'SET_VQA_QUESTION', payload: 'Ai đang điều khiển xe?' })
          dispatch({ type: 'VQA_SEARCH_SUCCESS', payload: mockVqaResponse })
        }}
      >
        Init VQA
      </button>

      <button
        data-testid="btn-edit-draft"
        onClick={() => {
          dispatch({
            type: 'SET_VQA_DRAFT_ANSWER',
            payload: 'người đàn ông đội mũ bảo hiểm màu trắng',
          })
        }}
      >
        Edit Draft
      </button>

      <button
        data-testid="btn-confirm"
        onClick={() => {
          dispatch({ type: 'CONFIRM_VQA_ANSWER' })
        }}
      >
        Confirm
      </button>

      <button
        data-testid="btn-add-basket"
        onClick={() => {
          dispatch({ type: 'ADD_VQA_TO_BASKET' })
        }}
      >
        Add To Basket
      </button>

      <button
        data-testid="btn-select-cand-2"
        onClick={() => {
          dispatch({ type: 'SELECT_VQA_RESULT', payload: mockVqaResponse.results[1] })
        }}
      >
        Select Cand 2
      </button>
    </div>
  )
}

describe('T030 — VQA Workflow State Integration Tests', () => {
  it('manages complete operator lifecycle: Search -> Advisory Proposal -> Edit Draft -> Confirm -> Basket -> Invalidate on Switch', async () => {
    const user = userEvent.setup()

    render(
      <AppProvider>
        <VqaStateHarness />
      </AppProvider>
    )

    // 1. Initial VQA search success
    await user.click(screen.getByTestId('btn-init-vqa'))

    expect(screen.getByTestId('vqa-query')).toHaveTextContent('người đi xe máy dừng đèn đỏ')
    expect(screen.getByTestId('vqa-question')).toHaveTextContent('Ai đang điều khiển xe?')
    expect(screen.getByTestId('vqa-proposal')).toHaveTextContent('người đàn ông đội mũ bảo hiểm')
    expect(screen.getByTestId('vqa-draft')).toHaveTextContent('người đàn ông đội mũ bảo hiểm')

    // P0 INVARIANT: Unapproved immediately after receiving proposal
    expect(screen.getByTestId('vqa-approved')).toHaveTextContent('unapproved')
    expect(screen.getByTestId('basket-count')).toHaveTextContent('0')

    // 2. Operator edits draft answer
    await user.click(screen.getByTestId('btn-edit-draft'))
    expect(screen.getByTestId('vqa-draft')).toHaveTextContent(
      'người đàn ông đội mũ bảo hiểm màu trắng'
    )
    expect(screen.getByTestId('vqa-approved')).toHaveTextContent('unapproved')

    // 3. Operator explicitly confirms answer
    await user.click(screen.getByTestId('btn-confirm'))
    expect(screen.getByTestId('vqa-approved')).toHaveTextContent(
      'người đàn ông đội mũ bảo hiểm màu trắng'
    )

    // 4. Operator adds confirmed answer to basket
    await user.click(screen.getByTestId('btn-add-basket'))
    expect(screen.getByTestId('basket-count')).toHaveTextContent('1')

    // 5. Operator switches candidate -> approval is invalidated
    await user.click(screen.getByTestId('btn-select-cand-2'))
    expect(screen.getByTestId('vqa-proposal')).toHaveTextContent('người phụ nữ mặc áo khoác đỏ')
    expect(screen.getByTestId('vqa-approved')).toHaveTextContent('unapproved')

    // Basket retains previously confirmed item
    expect(screen.getByTestId('basket-count')).toHaveTextContent('1')
  })
})
