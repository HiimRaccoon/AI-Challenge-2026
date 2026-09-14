import React from 'react'
import { TrakeEventSlot, SearchCandidate, TrakeVerification } from '../types/contracts'
import { CandidatePreviewThumb } from './CandidatePreviewThumb'
import {
  FilmstripIcon,
  CheckIcon,
  WarningIcon,
  InspectionTabIcon,
  BasketIcon,
  SpinnerIcon,
} from './Icons'

interface TrakeTimelineProps {
  slots: TrakeEventSlot[]
  activeSlotIndex: number | null
  videoId: string | null
  aggregateScore: number | null
  verifierScore?: number | null
  storyConsistency?: number | null
  visualVerification?: TrakeVerification | null
  validationStatus: 'valid' | 'incomplete' | 'mixed_video' | 'empty'
  isSearching: boolean
  mode: string
  isInBasket?: boolean
  onSelectSlot: (index: number) => void
  onInspectSlot: (index: number) => void
  onLockSlot: (index: number) => void
  onUnlockSlot: (index: number) => void
  onSelectAlternative: (eventIndex: number, alternativeIndex: number) => void
  onRerunSlot: (eventIndex: number) => void
  rerunningEventIndex: number | null
  onAddToBasket: () => void
}

export const TrakeTimeline: React.FC<TrakeTimelineProps> = ({
  slots,
  activeSlotIndex,
  videoId,
  aggregateScore,
  verifierScore,
  storyConsistency,
  visualVerification,
  validationStatus,
  isSearching,
  mode,
  isInBasket,
  onSelectSlot,
  onInspectSlot,
  onLockSlot,
  onUnlockSlot,
  onSelectAlternative,
  onRerunSlot,
  rerunningEventIndex,
  onAddToBasket,
}) => {
  const isFixture = mode === 'fixture'
  const isAllLocked = slots.length > 0 && slots.every((s) => s.locked)
  const isStructurallyValid = validationStatus === 'valid'
  const isReady = isStructurallyValid && isAllLocked
  const canAddToBasket = isReady && !isFixture

  const getValidationBadge = () => {
    switch (validationStatus) {
      case 'valid':
        if (isReady) {
          return (
            <span className="trake-status-badge badge-valid" data-testid="trake-validation-badge">
              <CheckIcon size={12} />
              <span>READY</span>
            </span>
          )
        }
        return (
          <span className="trake-status-badge badge-aligned" data-testid="trake-validation-badge">
            <CheckIcon size={12} />
            <span>ALIGNED ({slots.filter((s) => s.locked).length}/{slots.length} LOCKED)</span>
          </span>
        )
      case 'incomplete':
        return (
          <span className="trake-status-badge badge-incomplete" data-testid="trake-validation-badge">
            <WarningIcon size={12} />
            <span>INCOMPLETE</span>
          </span>
        )
      case 'mixed_video':
        return (
          <span className="trake-status-badge badge-mixed" data-testid="trake-validation-badge">
            <WarningIcon size={12} />
            <span>MIXED VIDEO</span>
          </span>
        )
      case 'empty':
      default:
        return (
          <span className="trake-status-badge badge-empty" data-testid="trake-validation-badge">
            <WarningIcon size={12} />
            <span>NO ALIGNMENT</span>
          </span>
        )
    }
  }

  const getBasketButtonTitle = () => {
    if (isFixture) return 'Fixture preview mode cannot be added to real submission'
    if (validationStatus === 'incomplete') return 'All event slots must have valid frames before adding to basket'
    if (validationStatus === 'mixed_video') return 'All frames in a TRAKE sequence must belong to the same video'
    if (validationStatus === 'empty') return 'No valid alignment available to add'
    if (!isAllLocked) return 'All event slots must be locked/accepted by operator before adding to basket'
    return 'Add verified TRAKE event sequence to submission basket'
  }

  return (
    <div className="trake-timeline-container" data-testid="trake-timeline-container">
      {/* Top Hypothesis & Control Banner */}
      <div className="trake-hypothesis-banner" data-testid="trake-hypothesis-banner">
        <div className="banner-left-meta">
          <div className="hypothesis-title-group">
            <FilmstripIcon size={16} className="text-cyan" />
            <span className="hypothesis-label">TRAKE Sequence Alignment</span>
            {isSearching && <SpinnerIcon size={13} className="icon-spin text-cyan" />}
          </div>

          <div className="hypothesis-details-row">
            {visualVerification && (
              <div
                className="meta-pill"
                data-testid="trake-visual-verifier-status"
                title={(visualVerification.visual_verifier_reasons || []).filter(Boolean).join(', ') || undefined}
              >
                <span className="meta-pill-label">Visual:</span>
                <span className="meta-pill-value tabular-nums">
                  {visualVerification.visual_verifier_available
                    ? `${(visualVerification.micro_clip_support * 100).toFixed(0)}% · ${(visualVerification.visual_frame_ids || []).flat().length} frames`
                    : 'fallback'}
                  {visualVerification.clip_manual_required ? ' · manual' : ''}
                </span>
              </div>
            )}

            <div className="meta-pill">
              <span className="meta-pill-label">Hypothesis:</span>
              <span className="meta-pill-value text-cyan tabular-nums" data-testid="trake-hypothesis-vid">
                {videoId || 'Unassigned'}
              </span>
            </div>

            {aggregateScore != null && (
              <div className="meta-pill">
                <span className="meta-pill-label">Alignment Score:</span>
                <span className="meta-pill-value tabular-nums" data-testid="trake-hypothesis-score">
                  {aggregateScore.toFixed(4)}
                </span>
              </div>
            )}

            {verifierScore != null && (
              <div className="meta-pill">
                <span className="meta-pill-label">Verifier:</span>
                <span className="meta-pill-value tabular-nums" data-testid="trake-verifier-score">
                  {(verifierScore * 100).toFixed(0)}%
                </span>
              </div>
            )}

            {storyConsistency != null && (
              <div className="meta-pill">
                <span className="meta-pill-label">Story:</span>
                <span className="meta-pill-value tabular-nums" data-testid="trake-story-score">
                  {(storyConsistency * 100).toFixed(0)}%
                </span>
              </div>
            )}

            <div className="meta-pill">
              <span className="meta-pill-label">Events:</span>
              <span className="meta-pill-value tabular-nums">{slots.length} positions</span>
            </div>
          </div>
        </div>

        <div className="banner-right-actions">
          {getValidationBadge()}

          <button
            type="button"
            className={`trake-btn-add-basket ${isInBasket ? 'is-in-basket-active' : ''}`}
            onClick={onAddToBasket}
            disabled={!canAddToBasket && !isInBasket}
            title={isInBasket ? 'Sequence already added to Submission Basket' : getBasketButtonTitle()}
            data-testid="trake-add-basket-btn"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              background: isInBasket ? 'rgba(16, 185, 129, 0.22)' : undefined,
              borderColor: isInBasket ? '#10b981' : undefined,
              color: isInBasket ? '#10b981' : undefined,
              transition: 'all 0.2s cubic-bezier(0.16, 1, 0.3, 1)',
            }}
          >
            {isInBasket ? <CheckIcon size={14} color="#10b981" /> : <BasketIcon size={14} />}
            <span>{isInBasket ? 'Sequence in Basket' : 'Add Sequence to Basket'}</span>
          </button>
        </div>
      </div>

      {/* Semantic Event Strip (0..N-1) */}
      <div className="trake-slots-strip" data-testid="trake-timeline-strip">
        {slots.map((slot) => {
          const isActive = activeSlotIndex === slot.event_index
          const hasFrame = slot.frame_id !== null && slot.video_id !== null
          const isSlotInvalid = slot.validation_status === 'incompatible_video'
          const slotCand: SearchCandidate | null = hasFrame
            ? {
                query_id: 'trake',
                video_id: slot.video_id!,
                frame_id: slot.frame_id!,
                timestamp_ms: slot.timestamp_ms || 0,
                source: 'trake',
                rank: slot.event_index + 1,
                score: slot.score,
                certified_anchor_frame_id:
                  slot.certified_anchor_frame_id ?? slot.frame_id!,
                certified_anchor_timestamp_ms:
                  slot.certified_anchor_timestamp_ms ?? slot.timestamp_ms ?? 0,
                anchor_offset: slot.anchor_offset ?? 0,
                cumulative_offset: 0,
              }
            : null

          return (
            <div
              key={`slot-${slot.event_index}`}
              className={`trake-slot-card ${isActive ? 'slot-active' : ''} ${
                slot.locked ? 'slot-locked' : ''
              } ${isSlotInvalid ? 'slot-invalid' : ''}`}
              onClick={() => onSelectSlot(slot.event_index)}
              onDoubleClick={() => onInspectSlot(slot.event_index)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter') onInspectSlot(slot.event_index)
                else if (e.key === ' ') onSelectSlot(slot.event_index)
              }}
              data-testid={`trake-slot-card-${slot.event_index}`}
            >
              {/* Slot Header */}
              <div className="slot-card-header">
                <div className="slot-index-pill tabular-nums">#{slot.event_index + 1}</div>
                <div className="slot-label-text" title={slot.event_label}>
                  {slot.event_label}
                </div>
              </div>

              {/* Slot Thumbnail Stage */}
              <div className="slot-thumb-stage">
                {slotCand ? (
                  <CandidatePreviewThumb
                    candidate={slotCand}
                    alt={`${slot.video_id} frame ${slot.frame_id}`}
                    className="slot-thumb-img"
                    onErrorFallback={
                      <div className="slot-fallback-matte">
                        <span className="fallback-vid">{slot.video_id}</span>
                        <span className="fallback-fid tabular-nums">Frame {slot.frame_id}</span>
                      </div>
                    }
                  />
                ) : (
                  <div className="slot-empty-matte">
                    <span className="empty-slot-title">Pending Alignment</span>
                    <span className="empty-slot-hint">Run Search & Align</span>
                  </div>
                )}
              </div>

              {/* Slot Caption & Metadata */}
              <div className="slot-caption-pane">
                <div className="slot-meta-row">
                  <span className="slot-vid-name">{slot.video_id || 'No video'}</span>
                  <span className="slot-fid-name tabular-nums">
                    {slot.frame_id !== null ? `Frame ${slot.frame_id}` : 'Unassigned'}
                  </span>
                </div>

                <div className="slot-score-lock-row">
                  <span className="slot-score-name tabular-nums">
                    {slot.score != null ? `Score ${slot.score.toFixed(3)}` : 'Score —'}
                  </span>
                  <button
                    type="button"
                    className={`slot-lock-toggle-btn ${slot.locked ? 'is-locked' : 'is-unlocked'}`}
                    onClick={(e) => {
                      e.stopPropagation()
                      if (slot.locked) {
                        onUnlockSlot(slot.event_index)
                      } else {
                        onLockSlot(slot.event_index)
                      }
                    }}
                    title={
                      slot.locked
                        ? 'Locked: Frame protected from re-alignment. Click to unlock.'
                        : 'Unlocked: Click to lock and protect frame.'
                    }
                    aria-label={
                      slot.locked
                        ? `Unlock event ${slot.event_index + 1}`
                        : `Lock event ${slot.event_index + 1}`
                    }
                    data-testid={`slot-lock-btn-${slot.event_index}`}
                  >
                    <span>{slot.locked ? 'Locked' : 'Lock'}</span>
                  </button>
                </div>
              </div>

              {slot.alternatives && slot.alternatives.length > 0 && (
                <div
                  className="slot-alternatives-row"
                  data-testid={`slot-alternatives-${slot.event_index}`}
                  style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', padding: '0 8px 8px' }}
                >
                  <span style={{ width: '100%', fontSize: '10px', color: '#94a3b8' }}>
                    Alternatives
                  </span>
                  {slot.alternatives.map((alternative, alternativeIndex) => (
                    <button
                      key={`${alternative.frame_id}-${alternative.timestamp_ms}`}
                      type="button"
                      className="slot-lock-toggle-btn is-unlocked"
                      disabled={slot.locked || alternative.frame_id === slot.frame_id}
                      onClick={(event) => {
                        event.stopPropagation()
                        onSelectAlternative(slot.event_index, alternativeIndex)
                      }}
                      title={`Use frame ${alternative.frame_id} at ${(alternative.timestamp_ms / 1000).toFixed(1)}s`}
                      data-testid={`slot-alternative-${slot.event_index}-${alternativeIndex}`}
                    >
                      F{alternative.frame_id}
                    </button>
                  ))}
                </div>
              )}

              {/* Slot Actions Footer */}
              <div className="slot-card-footer">
                <button
                  type="button"
                  className="slot-inspect-btn"
                  disabled={slot.locked || rerunningEventIndex !== null}
                  onClick={(event) => {
                    event.stopPropagation()
                    onRerunSlot(slot.event_index)
                  }}
                  title={
                    slot.locked
                      ? 'Unlock this event before rerunning it.'
                      : 'Retrieve alternatives for this event only; locked events remain unchanged.'
                  }
                  data-testid={`slot-rerun-btn-${slot.event_index}`}
                >
                  {rerunningEventIndex === slot.event_index && (
                    <SpinnerIcon size={12} className="icon-spin" />
                  )}
                  <span>{rerunningEventIndex === slot.event_index ? 'Rerunning' : 'Rerun Event'}</span>
                </button>
                <button
                  type="button"
                  className="slot-inspect-btn"
                  onClick={(e) => {
                    e.stopPropagation()
                    onInspectSlot(slot.event_index)
                  }}
                  title="Inspect frame in detail & exact-step"
                  data-testid={`slot-inspect-btn-${slot.event_index}`}
                >
                  <InspectionTabIcon size={12} />
                  <span>Inspect Event</span>
                </button>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
