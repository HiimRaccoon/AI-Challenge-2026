import React from 'react'
import { useAppDispatch, useAppState } from '../state/AppContext'
import {
  AdoptIcon,
  CheckIcon,
  BasketIcon,
  WarningIcon,
  ClearIcon,
  QuestionIcon,
} from './Icons'
import { telemetry } from '../utils/telemetry'

export const VqaAnswerPanel: React.FC = () => {
  const {
    queryText,
    vqaQuestion,
    vqaActiveResult,
    vqaDraftAnswer,
    vqaApprovedAnswer,
    vqaError,
    mode,
    submissionBasket,
  } = useAppState()

  const dispatch = useAppDispatch()
  const isFixture = mode === 'fixture'

  if (!vqaActiveResult) {
    return (
      <div className="vqa-answer-panel-card empty-panel" data-testid="vqa-panel-empty">
        <span className="empty-panel-text">
          Perform a VQA search to generate machine proposals and verify answers.
        </span>
      </div>
    )
  }

  const proposal = vqaActiveResult.proposal
  const verifierStatus =
    vqaActiveResult.verifier_status ||
    vqaActiveResult.status ||
    (isFixture ? 'fixture_unverified' : 'unverified')
  const retryCount = vqaActiveResult.retry_count ?? 0
  const degradedReasons = vqaActiveResult.degraded_reasons || []
  const visualVerification = vqaActiveResult.verification

  const isApproved = vqaApprovedAnswer !== null && vqaApprovedAnswer.length > 0
  const canConfirm = Boolean(vqaDraftAnswer.trim())

  const isAlreadyInBasket =
    isApproved &&
    submissionBasket.some(
      (b) =>
        b.video_id === vqaActiveResult.video_id &&
        b.frame_id === vqaActiveResult.frame_id &&
        b.answer === vqaApprovedAnswer
    )

  const handleAdoptProposal = () => {
    if (proposal) {
      dispatch({ type: 'SET_VQA_DRAFT_ANSWER', payload: proposal })
    }
  }

  const handleDraftChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    dispatch({ type: 'SET_VQA_DRAFT_ANSWER', payload: e.target.value })
  }

  const handleConfirm = () => {
    if (canConfirm) {
      dispatch({ type: 'CONFIRM_VQA_ANSWER' })
    }
  }

  const handleClearApproval = () => {
    dispatch({ type: 'CLEAR_VQA_APPROVAL' })
  }

  const handleAddToBasket = () => {
    if (isApproved && !isFixture && vqaActiveResult) {
      dispatch({ type: 'ADD_VQA_TO_BASKET' })
      telemetry.record({
        action: 'ADD_VQA_TO_BASKET',
        taskMode: 'VQA',
        videoId: vqaActiveResult.video_id,
        frameId: vqaActiveResult.frame_id,
        details: {
          answer: vqaApprovedAnswer,
        },
      })
    }
  }

  return (
    <div className="vqa-answer-panel-card" data-testid="vqa-answer-panel">
      {/* Panel Header */}
      <div className="vqa-panel-header">
        <div className="vqa-header-title-group">
          <QuestionIcon size={16} className="text-cyan" />
          <span className="vqa-header-title">VQA Answer Verification</span>
        </div>
        <div className="vqa-target-candidate-badge tabular-nums" data-testid="vqa-target-badge">
          Answer Frame: {vqaActiveResult.video_id} · Frame {vqaActiveResult.frame_id} (Rank #{vqaActiveResult.rank})
          {vqaActiveResult.anchor_offset ? ` · Offset ${vqaActiveResult.anchor_offset > 0 ? `+${vqaActiveResult.anchor_offset}` : vqaActiveResult.anchor_offset}` : ''}
        </div>
      </div>

      {/* Query & Question Summary */}
      <div className="vqa-context-strip">
        <div className="vqa-context-line">
          <span className="context-label">Event:</span>
          <span className="context-value">{queryText || 'N/A'}</span>
        </div>
        <div className="vqa-context-line highlight-question">
          <span className="context-label">Question:</span>
          <span className="context-value" data-testid="vqa-panel-question">
            {vqaQuestion || vqaActiveResult.evidence.question || 'N/A'}
          </span>
        </div>
      </div>

      {/* ── 1. Advisory Machine Proposal Card ── */}
      <div className="vqa-proposal-card" data-testid="vqa-proposal-card">
        <div className="proposal-card-header">
          <div className="proposal-badge-group">
            <span className="advisory-badge">Advisory Proposal</span>
            {/* Verifier status badge */}
            <span
              className={`verifier-status-badge badge-${(verifierStatus || 'unverified').toLowerCase()}`}
              data-testid="verifier-status-badge"
            >
              {verifierStatus === 'verified'
                ? 'Machine Verified'
                : verifierStatus === 'rejected'
                ? 'Verifier Rejected'
                : verifierStatus === 'insufficient_evidence'
                ? 'Insufficient Evidence'
                : verifierStatus === 'fixture_unverified'
                ? 'Fixture Preview'
                : 'Unverified'}
            </span>

            {/* Retry counter */}
            {retryCount > 0 && (
              <span className="retry-counter-pill tabular-nums">
                Retry {retryCount}/1 {retryCount >= 1 ? '(Exhausted)' : ''}
              </span>
            )}
            {vqaActiveResult.answerability_score !== undefined && (
              <span className="retry-counter-pill tabular-nums" data-testid="vqa-answerability-score">
                Answerability {(vqaActiveResult.answerability_score * 100).toFixed(0)}%
              </span>
            )}
            {vqaActiveResult.verifier_score !== undefined && (
              <span className="retry-counter-pill tabular-nums" data-testid="vqa-verifier-score">
                Verifier {(vqaActiveResult.verifier_score * 100).toFixed(0)}%
              </span>
            )}
            {visualVerification && (
              <span
                className="retry-counter-pill tabular-nums"
                data-testid="vqa-visual-verifier-status"
                title={visualVerification.visual_verifier_reason || undefined}
              >
                {visualVerification.visual_verifier_available
                  ? `VLM ${(visualVerification.visual_support * 100).toFixed(0)}% · frames ${(visualVerification.visual_frame_ids || []).join(', ')}`
                  : `VLM fallback${visualVerification.visual_verifier_reason ? ` · ${visualVerification.visual_verifier_reason}` : ''}`}
              </span>
            )}
            {vqaActiveResult.contradiction && (
              <span className="retry-counter-pill" data-testid="vqa-contradiction-badge">
                Contradiction · manual review
              </span>
            )}
          </div>

          {proposal && (
            <button
              type="button"
              className="btn-adopt-draft"
              onClick={handleAdoptProposal}
              title="Copy machine proposal into editable draft"
              data-testid="btn-adopt-draft"
            >
              <AdoptIcon size={13} />
              <span>Adopt as Draft</span>
            </button>
          )}
        </div>

        <div className="proposal-text-body" data-testid="vqa-proposal-text">
          {proposal ? (
            <span className="proposal-content">"{proposal}"</span>
          ) : (
            <span className="proposal-empty-note">
              (Empty proposal — insufficient machine evidence)
            </span>
          )}
        </div>

        {vqaActiveResult.frame_candidates && vqaActiveResult.frame_candidates.length > 0 && (
          <div className="proposal-degraded-reasons" data-testid="vqa-frame-candidates">
            {vqaActiveResult.frame_candidates.map((frame) => (
              <span key={frame.frame_id} className="degraded-reason-pill tabular-nums">
                Frame {frame.frame_id} · {(frame.score * 100).toFixed(0)}%
                {frame.support_modalities.length > 0 ? ` · ${frame.support_modalities.join('+')}` : ''}
              </span>
            ))}
          </div>
        )}

        {/* Degraded reasons if present */}
        {degradedReasons.length > 0 && (
          <div className="proposal-degraded-reasons">
            {degradedReasons.map((r, i) => (
              <span key={i} className="degraded-reason-pill">
                {r}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* ── 2. Editable Answer Draft Area ── */}
      <div className="vqa-draft-section">
        <div className="draft-section-header">
          <span className="draft-section-title">Operator Answer Draft</span>
          <span className="draft-disclaimer-note">
            Draft is unapproved until confirmed.
          </span>
        </div>

        <textarea
          className="vqa-draft-textarea"
          rows={3}
          value={vqaDraftAnswer}
          onChange={handleDraftChange}
          placeholder="Type or adopt verified answer for competition submission..."
          data-testid="vqa-draft-input"
        />

        <div className="draft-action-row">
          <button
            type="button"
            className="btn-confirm-answer"
            onClick={handleConfirm}
            disabled={!canConfirm}
            data-testid="btn-confirm-answer"
          >
            <CheckIcon size={14} />
            <span>Confirm Answer</span>
          </button>
        </div>
      </div>

      {/* ── 3. Approved Answer & Basket Commitment ── */}
      {isApproved ? (
        <div className="vqa-approved-card" data-testid="vqa-approved-card">
          <div className="approved-card-header">
            <div className="approved-badge-group">
              <span className="approved-badge">
                <CheckIcon size={13} /> Approved Answer
              </span>
              <span className="approved-vid-fid tabular-nums">
                {vqaActiveResult.video_id} · Frame {vqaActiveResult.frame_id}
              </span>
            </div>

            <button
              type="button"
              className="btn-revoke-approval"
              onClick={handleClearApproval}
              title="Revoke approval"
              data-testid="btn-clear-approval"
            >
              <ClearIcon size={12} />
              <span>Revoke</span>
            </button>
          </div>

          <div className="approved-answer-text" data-testid="vqa-approved-text">
            {vqaApprovedAnswer}
          </div>

          {/* Submission Basket Action */}
          <div className="approved-basket-row">
            {!isFixture ? (
              <button
                type="button"
                className={`btn-add-vqa-basket ${isAlreadyInBasket ? 'is-in-basket' : ''}`}
                onClick={handleAddToBasket}
                disabled={isAlreadyInBasket}
                data-testid="btn-add-vqa-basket"
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '6px',
                  background: isAlreadyInBasket ? 'rgba(16, 185, 129, 0.22)' : undefined,
                  borderColor: isAlreadyInBasket ? '#10b981' : undefined,
                  color: isAlreadyInBasket ? '#10b981' : undefined,
                  transition: 'all 0.2s cubic-bezier(0.16, 1, 0.3, 1)',
                }}
              >
                {isAlreadyInBasket ? <CheckIcon size={14} color="#10b981" /> : <BasketIcon size={14} />}
                <span>
                  {isAlreadyInBasket ? 'Added to Basket' : 'Add to Submission Basket'}
                </span>
              </button>
            ) : (
              <div
                className="fixture-basket-warning"
                data-testid="fixture-basket-warning"
              >
                <WarningIcon size={13} />
                <span>
                  Fixture preview mode: Candidates and answers cannot enter real submission basket.
                </span>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="vqa-unapproved-status" data-testid="vqa-unapproved-status">
          <span className="unapproved-hint">
            Answer is currently unapproved. Click "Confirm Answer" above to mark submission-ready.
          </span>
        </div>
      )}

      {/* Error Callout */}
      {vqaError && (
        <div className="vqa-error-alert" data-testid="vqa-error-alert">
          <WarningIcon size={14} />
          <span>{vqaError}</span>
        </div>
      )}
    </div>
  )
}
