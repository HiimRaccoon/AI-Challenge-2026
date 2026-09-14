import React, { useEffect, useRef, useState, useCallback } from 'react'
import { useAppDispatch, useAppState } from '../state/AppContext'
import { getVideoStreamUrl } from '../api/tv4Client'
import { getFixturePreviewDataUri } from '../fixtures/fixtureData'
import {
  VideoIcon,
  FrameIcon,
  StepPrevIcon,
  StepNextIcon,
  AnchorIcon,
  CheckIcon,
  WarningIcon,
  SpinnerIcon,
} from './Icons'

export const MediaInspector: React.FC = () => {
  const {
    activeCandidate,
    anchorCandidate,
    cumulativeOffset,
    isStepping,
    stepError,
    currentStep,
    exactImageBlobUrl,
    isImageLoading,
    imageError,
    mode,
  } = useAppState()

  const dispatch = useAppDispatch()
  const videoRef = useRef<HTMLVideoElement>(null)
  const [playbackTime, setPlaybackTime] = useState<number>(0)
  const [frameImgFailed, setFrameImgFailed] = useState(false)
  const [showProofDetails, setShowProofDetails] = useState(false)
  const isFixture = mode === 'fixture'

  // Reset image error state when active candidate or offset changes
  useEffect(() => {
    setFrameImgFailed(false)
  }, [activeCandidate, cumulativeOffset])

  // Seek video when active candidate changes
  useEffect(() => {
    if (activeCandidate && videoRef.current) {
      const targetSec = activeCandidate.timestamp_ms / 1000
      videoRef.current.currentTime = targetSec
      setPlaybackTime(targetSec)
    }
  }, [activeCandidate])

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setPlaybackTime(videoRef.current.currentTime)
    }
  }

  // Exact stepping logic: dispatches step intent to coordinator
  const handleStep = useCallback(
    (targetOffset: number) => {
      dispatch({ type: 'EXACT_STEP_START', payload: { offset: targetOffset } })
    },
    [dispatch]
  )

  // Keyboard navigation for stepping
  useEffect(() => {
    const handleKeyDown = (e: globalThis.KeyboardEvent) => {
      if (
        document.activeElement?.tagName === 'INPUT' ||
        document.activeElement?.tagName === 'TEXTAREA' ||
        document.activeElement?.tagName === 'SELECT'
      ) {
        return
      }

      if (e.key === 'ArrowLeft') {
        e.preventDefault()
        handleStep(cumulativeOffset - 1)
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        handleStep(cumulativeOffset + 1)
      } else if (e.key === 'Home' || e.key === '0') {
        e.preventDefault()
        handleStep(0)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [cumulativeOffset, handleStep])

  if (!activeCandidate || !anchorCandidate) {
    return (
      <div className="media-inspector-pane closed" data-testid="workspace-empty" aria-hidden="true" />
    )
  }

  const inspectedFrameId =
    cumulativeOffset === 0 && !currentStep?.frame
      ? anchorCandidate.frame_id
      : currentStep?.frame?.frame_id

  const inspectedTimestampMs =
    cumulativeOffset === 0 && !currentStep?.frame
      ? anchorCandidate.timestamp_ms
      : currentStep?.frame?.timestamp_ms

  const isSelectable = isFixture
    ? false
    : cumulativeOffset === 0
    ? true
    : currentStep?.frame?.submission_selectable === true

  const isBoundary =
    currentStep && (!currentStep.frame || currentStep.degraded_reason !== null)

  const videoStreamUrl = getVideoStreamUrl(anchorCandidate.video_id)

  return (
    <aside className="media-inspector-pane open" data-testid="inspection-workspace">
      {/* Inspector Header */}
      <div className="inspector-top-bar">
        <div className="inspector-identity-group">
          <span className="inspector-heading-text">Media Inspection</span>
          <span className="inspector-anchor-id monospace">
            {anchorCandidate.video_id} · F:{anchorCandidate.frame_id}
          </span>
        </div>

        <div className="inspector-offset-tag monospace" data-testid="cumulative-offset">
          {cumulativeOffset === 0
            ? 'Anchor (0)'
            : cumulativeOffset > 0
            ? `+${cumulativeOffset}`
            : cumulativeOffset}
        </div>
      </div>

      {/* Dual Media Viewport Area */}
      <div className="inspector-stages-flow">
        {/* 1. Context Video Playback */}
        <div className="stage-card video-card">
          <div className="card-top-bar">
            <div className="card-title-left">
              <VideoIcon size={13} className="text-crimson" />
              <span className="card-label">Original Video</span>
            </div>
            <span className="card-timecode monospace">{playbackTime.toFixed(3)}s</span>
          </div>

          <div className="video-player-container">
            <video
              ref={videoRef}
              src={videoStreamUrl}
              controls
              onTimeUpdate={handleTimeUpdate}
              className="inspector-video-element"
              data-testid="original-video-player"
            />
          </div>
        </div>

        {/* 2. Canonical Exact Frame Stage */}
        <div className="stage-card frame-card">
          <div className="card-top-bar">
            <div className="card-title-left">
              <FrameIcon size={13} className="text-crimson" />
              <span className="card-label">Canonical Exact Frame</span>
            </div>

            <span
              className={`selectability-tag ${
                isBoundary
                  ? 'tag-boundary'
                  : isSelectable
                  ? 'tag-selectable'
                  : isFixture
                  ? 'tag-fixture-preview'
                  : 'tag-non-selectable'
              }`}
              data-testid="submission-selectable-badge"
            >
              {isBoundary ? (
                <>
                  <WarningIcon size={10} /> Boundary
                </>
              ) : isSelectable ? (
                <>
                  <CheckIcon size={10} /> Selectable
                </>
              ) : isFixture ? (
                <>
                  <WarningIcon size={10} /> Preview
                </>
              ) : (
                <>
                  <WarningIcon size={10} /> Non-Selectable
                </>
              )}
            </span>
          </div>

          <div className="exact-frame-container">
            {isImageLoading && (
              <div className="frame-loading-overlay">
                <SpinnerIcon size={20} className="icon-spin text-crimson" />
                <span>Decoding frame...</span>
              </div>
            )}

            {!frameImgFailed && exactImageBlobUrl ? (
              <img
                src={exactImageBlobUrl}
                alt={`Exact frame ${inspectedFrameId}`}
                className="inspector-frame-element"
                data-testid="exact-frame-image"
                onError={() => setFrameImgFailed(true)}
              />
            ) : !frameImgFailed && isFixture ? (
              <img
                src={getFixturePreviewDataUri(
                  anchorCandidate.video_id,
                  inspectedFrameId ?? anchorCandidate.frame_id
                )}
                alt={`Exact frame ${inspectedFrameId}`}
                className="inspector-frame-element"
                data-testid="exact-frame-image"
                onError={() => setFrameImgFailed(true)}
              />
            ) : (
              <div className="inspector-frame-fallback">
                <span>Exact frame preview unavailable</span>
                <span className="monospace">F:{inspectedFrameId ?? 'UNKNOWN'}</span>
              </div>
            )}
          </div>

          {/* Stepping Transport Controls */}
          <div className="transport-stepping-bar">
            <button
              type="button"
              className="transport-btn"
              onClick={() => handleStep(cumulativeOffset - 1)}
              disabled={isStepping}
              data-testid="btn-step-prev"
              title="Step -1 frame (Left Arrow)"
            >
              <StepPrevIcon size={13} />
              <span>-1</span>
            </button>

            <button
              type="button"
              className="transport-btn anchor-reset-btn"
              onClick={() => handleStep(0)}
              disabled={isStepping || cumulativeOffset === 0}
              data-testid="btn-step-anchor"
              title="Reset to anchor (Home / 0)"
            >
              <AnchorIcon size={13} />
              <span>Anchor</span>
            </button>

            <button
              type="button"
              className="transport-btn"
              onClick={() => handleStep(cumulativeOffset + 1)}
              disabled={isStepping}
              data-testid="btn-step-next"
              title="Step +1 frame (Right Arrow)"
            >
              <span>+1</span>
              <StepNextIcon size={13} />
            </button>
          </div>
        </div>
      </div>

      {/* Canonical Technical Identity Section */}
      <div className="technical-identity-section" data-testid="canonical-identity-card">
        <div className="identity-data-grid">
          <div className="identity-field">
            <span className="field-name">Video ID</span>
            <span className="field-value monospace" data-testid="inspected-video-id">
              {anchorCandidate.video_id}
            </span>
          </div>

          <div className="identity-field highlight-field">
            <span className="field-name">Frame ID</span>
            <span className="field-value frame-val monospace" data-testid="inspected-frame-id">
              {inspectedFrameId ?? 'UNKNOWN'}
            </span>
          </div>

          <div className="identity-field">
            <span className="field-name">Timestamp</span>
            <span className="field-value monospace" data-testid="inspected-timestamp-ms">
              {inspectedTimestampMs !== undefined ? `${inspectedTimestampMs} ms` : 'N/A'}
            </span>
          </div>

          <div className="identity-field">
            <span className="field-name">PTS</span>
            <span className="field-value monospace">
              {currentStep?.frame?.pts ?? 'ANCHOR'}
            </span>
          </div>
        </div>

        {/* Expandable Technical Proof Details */}
        <div className="proof-disclosure-area">
          <button
            type="button"
            className="disclosure-btn"
            onClick={() => setShowProofDetails(!showProofDetails)}
          >
            <span>{showProofDetails ? '▾ Hide Proof Details' : '▸ Technical Proof Details'}</span>
          </button>

          {showProofDetails && (
            <div className="disclosure-content monospace">
              <div>Time Base: {currentStep?.frame?.time_base || '1/12800'}</div>
              <div>Run ID: {currentStep?.frame?.preprocess_run_id || 'run_v1_batch1'}</div>
              {currentStep?.frame?.certification_id && (
                <div>Cert ID: {currentStep.frame.certification_id}</div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Alert Banner for boundary/degraded offsets */}
      {(stepError || imageError || isBoundary) && (
        <div className="inspector-alert-callout" data-testid="step-alert-banner">
          <WarningIcon size={14} />
          <span>
            {stepError || imageError || currentStep?.degraded_reason || 'Boundary reached'}
          </span>
        </div>
      )}
    </aside>
  )
}
