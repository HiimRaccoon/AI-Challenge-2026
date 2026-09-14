import React from 'react'
import { useAppDispatch, useAppState } from '../state/AppContext'
import { FilmstripIcon } from './Icons'
import { ExactNeighborThumb } from './ExactNeighborThumb'

export const TemporalStrip: React.FC = () => {
  const { anchorCandidate, cumulativeOffset, exactNeighbors, mode } = useAppState()
  const dispatch = useAppDispatch()
  const isFixture = mode === 'fixture'

  if (!anchorCandidate) {
    return null
  }

  const handleSelectOffset = (targetOffset: number) => {
    if (targetOffset === cumulativeOffset) return
    dispatch({ type: 'EXACT_STEP_START', payload: { offset: targetOffset } })
  }

  const visibleOffsets = [-2, -1, 0, 1, 2]

  return (
    <div className="workspace-temporal-strip" data-testid="context-strip">
      <div className="temporal-strip-header">
        <div className="strip-title-pane">
          <FilmstripIcon size={13} className="text-crimson" />
          <span className="strip-title-text">Temporal Sequence</span>
        </div>
        <div className="strip-anchor-meta monospace">
          {anchorCandidate.video_id} · F:{anchorCandidate.frame_id} (Anchor)
        </div>
      </div>

      <div className="temporal-track-container">
        {visibleOffsets.map((relOff) => {
          const effOff = cumulativeOffset + relOff
          const stepData = exactNeighbors?.steps?.find((s) => s.offset === effOff)
          const fid = stepData?.frame?.frame_id
          const isCurrent = effOff === cumulativeOffset
          const isAnchor = effOff === 0
          const isDegraded = !stepData || !stepData.frame || stepData.degraded_reason !== null

          return (
            <button
              key={`temporal-cell-${effOff}`}
              type="button"
              className={`temporal-cell ${isCurrent ? 'cell-current-offset' : ''} ${
                isAnchor ? 'cell-anchor-frame' : ''
              }`}
              onClick={() => !isDegraded && handleSelectOffset(effOff)}
              disabled={isDegraded}
              title={
                stepData?.frame
                  ? `Offset ${effOff >= 0 ? `+${effOff}` : effOff} (F:${fid})`
                  : 'Unavailable'
              }
            >
              <div className="temporal-thumb-frame">
                <ExactNeighborThumb
                  videoId={anchorCandidate.video_id}
                  anchorFrameId={anchorCandidate.frame_id}
                  anchorTimestampMs={anchorCandidate.timestamp_ms}
                  certifiedAnchorFrameId={anchorCandidate.certified_anchor_frame_id ?? anchorCandidate.frame_id}
                  certifiedAnchorTimestampMs={anchorCandidate.certified_anchor_timestamp_ms ?? anchorCandidate.timestamp_ms}
                  anchorOffset={anchorCandidate.anchor_offset ?? 0}
                  cumulativeOffset={cumulativeOffset}
                  relOffset={relOff}
                  stepData={stepData}
                  isFixture={isFixture}
                  className="temporal-thumb-img"
                />
              </div>

              <div className="temporal-cell-label">
                <span
                  className={`offset-text monospace ${
                    isCurrent ? 'text-crimson' : isAnchor ? 'text-gold' : ''
                  }`}
                >
                  {effOff === 0 ? '0' : effOff > 0 ? `+${effOff}` : effOff}
                </span>
                <span className="frame-text monospace">
                  {stepData?.frame ? fid : !stepData ? '...' : 'N/A'}
                </span>
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
