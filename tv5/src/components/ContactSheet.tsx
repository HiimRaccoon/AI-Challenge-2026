import React from 'react'
import { useAppDispatch, useAppState } from '../state/AppContext'
import { SearchCandidate } from '../types/contracts'
import { CandidatePreviewThumb } from './CandidatePreviewThumb'
import { SpinnerIcon } from './Icons'

interface FrameTileProps {
  candidate: SearchCandidate
  isSelected: boolean
  onClick: () => void
}

const FrameTile: React.FC<FrameTileProps> = ({ candidate, isSelected, onClick }) => {
  return (
    <div
      className={`contact-frame-tile ${isSelected ? 'tile-active-selection' : ''}`}
      onClick={onClick}
      data-testid={`candidate-card-${candidate.rank}`}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          onClick()
        }
      }}
    >
      {/* 16:9 Visual Frame (88–95% dominance) */}
      <div className="frame-image-stage">
        <CandidatePreviewThumb
          candidate={candidate}
          alt={`${candidate.video_id} frame ${candidate.frame_id}`}
          className="frame-image-element"
          loading="lazy"
          onErrorFallback={
            <div className="frame-fallback-matte">
              <span className="fallback-status">Preview unavailable</span>
              <span className="fallback-sub monospace">{candidate.video_id}</span>
            </div>
          }
        />

        {/* Overlaid Rank Chip */}
        <span className="frame-rank-chip monospace">#{candidate.rank}</span>

        {/* Overlaid Bottom Identity */}
        <div className="frame-identity-overlay">
          <span className="identity-vid monospace">{candidate.video_id}</span>
          <span className="identity-fid monospace">F:{candidate.frame_id}</span>
        </div>
      </div>

      {/* Subtle Metadata bar for test assertions & technical disclosure */}
      <div className="frame-meta-bar">
        <span className="meta-score monospace">
          SCORE: {candidate.score != null ? candidate.score.toFixed(4) : 'N/A'}
        </span>
        <span className="meta-time monospace">
          {(candidate.timestamp_ms / 1000).toFixed(1)}s
        </span>
      </div>
    </div>
  )
}

export const ContactSheet: React.FC = () => {
  const { candidates, activeCandidate, isSearching } = useAppState()
  const dispatch = useAppDispatch()

  const handleSelectCandidate = (candidate: SearchCandidate) => {
    dispatch({ type: 'SELECT_CANDIDATE', payload: candidate })
  }

  return (
    <div className="retrieval-contact-sheet" data-testid="candidate-grid">
      {/* Canvas Header */}
      <div className="sheet-header">
        <div className="sheet-title-group">
          <h2 className="sheet-title">Results</h2>
          {candidates.length > 0 && (
            <span className="sheet-count-pill monospace">
              {candidates.length} candidates
            </span>
          )}
        </div>
        <span className="sheet-instruction">Click candidate to inspect in original video</span>
      </div>

      {/* Main Results Contact Matrix or Quiet Empty View */}
      {isSearching ? (
        <div className="sheet-status-state">
          <SpinnerIcon size={24} className="icon-spin text-crimson" />
          <span className="state-text">Retrieving multimodal candidate frames...</span>
        </div>
      ) : candidates.length === 0 ? (
        <div className="sheet-status-state sheet-idle-view">
          <p className="idle-prompt">
            Enter a query in the search rail to retrieve candidate frames.
          </p>
        </div>
      ) : (
        <div className="contact-sheet-matrix">
          {candidates.map((cand) => (
            <FrameTile
              key={`${cand.video_id}-${cand.frame_id}-${cand.rank}`}
              candidate={cand}
              isSelected={
                activeCandidate?.video_id === cand.video_id &&
                activeCandidate?.frame_id === cand.frame_id
              }
              onClick={() => handleSelectCandidate(cand)}
            />
          ))}
        </div>
      )}
    </div>
  )
}
