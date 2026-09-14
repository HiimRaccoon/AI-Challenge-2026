import React, { useState, useEffect } from 'react'
import { ExactStep } from '../types/contracts'
import { getFixturePreviewDataUri } from '../fixtures/fixtureData'
import { loadExactImage } from '../api/exactImageCache'
import { SpinnerIcon } from './Icons'

export interface ExactNeighborThumbProps {
  videoId: string
  anchorFrameId: number
  anchorTimestampMs: number
  certifiedAnchorFrameId?: number | null
  certifiedAnchorTimestampMs?: number | null
  anchorOffset?: number | null
  cumulativeOffset: number
  relOffset: number
  stepData?: ExactStep | null
  isFixture: boolean
  alt?: string
  className?: string
}

export const ExactNeighborThumb: React.FC<ExactNeighborThumbProps> = ({
  videoId,
  anchorFrameId,
  anchorTimestampMs,
  certifiedAnchorFrameId,
  certifiedAnchorTimestampMs,
  anchorOffset,
  cumulativeOffset,
  relOffset,
  stepData,
  isFixture,
  alt,
  className = 'neighbor-thumb-img',
}) => {
  const [blobUrl, setBlobUrl] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [hasError, setHasError] = useState<boolean>(false)

  const isStepMatchingVideo = stepData?.frame?.video_id === videoId
  const hasValidLiveFrame =
    !isFixture &&
    isStepMatchingVideo &&
    stepData &&
    stepData.frame &&
    stepData.degraded_reason === null
  const expectedFrameId = isStepMatchingVideo ? stepData?.frame?.frame_id : undefined
  const expectedVideoId = isStepMatchingVideo ? stepData?.frame?.video_id : undefined

  const rootAnchorFrameId = certifiedAnchorFrameId ?? anchorFrameId
  const rootAnchorTimestampMs = certifiedAnchorTimestampMs ?? anchorTimestampMs
  const totalCumulativeOffset = (anchorOffset ?? 0) + cumulativeOffset

  // Unconditional hook execution at top level
  useEffect(() => {
    if (!hasValidLiveFrame || expectedFrameId === undefined || expectedVideoId === undefined) {
      return
    }

    let isMounted = true
    setIsLoading(true)
    setHasError(false)

    loadExactImage(
      {
        video_id: videoId,
        frame_id: rootAnchorFrameId,
        timestamp_ms: rootAnchorTimestampMs,
        certified_anchor_frame_id: rootAnchorFrameId,
        certified_anchor_timestamp_ms: rootAnchorTimestampMs,
        cumulative_offset: totalCumulativeOffset,
        offsets: [relOffset],
      },
      { frame_id: expectedFrameId, video_id: expectedVideoId }
    )
      .then((res) => {
        if (isMounted) {
          setBlobUrl(res.blobUrl)
          setIsLoading(false)
        }
      })
      .catch(() => {
        if (isMounted) {
          setHasError(true)
          setIsLoading(false)
        }
      })

    return () => {
      isMounted = false
    }
  }, [
    hasValidLiveFrame,
    videoId,
    rootAnchorFrameId,
    rootAnchorTimestampMs,
    totalCumulativeOffset,
    relOffset,
    expectedFrameId,
    expectedVideoId,
  ])

  // 1. Fixture Mode: Deterministic synthetic preview (isolated from Live)
  if (isFixture) {
    const syntheticFid = Math.max(0, rootAnchorFrameId + cumulativeOffset + relOffset)
    const previewUri = getFixturePreviewDataUri(videoId, syntheticFid)
    return (
      <img
        src={previewUri}
        alt={alt || `Frame ${syntheticFid}`}
        className={className}
        loading="lazy"
        decoding="async"
      />
    )
  }

  // 2. Step is missing / loading
  if (stepData === undefined || stepData === null) {
    return (
      <div className="neighbor-thumb-skeleton">
        <SpinnerIcon size={14} className="icon-spin text-cyan" />
      </div>
    )
  }

  // 3. Step degraded or boundary
  if (stepData.degraded_reason !== null || !stepData.frame) {
    return (
      <div className="neighbor-fallback-matte">
        <span className="tabular-nums">{stepData.degraded_reason || 'Boundary'}</span>
      </div>
    )
  }

  if (isLoading && !blobUrl) {
    return (
      <div className="neighbor-thumb-skeleton">
        <SpinnerIcon size={14} className="icon-spin text-cyan" />
      </div>
    )
  }

  if (hasError || !blobUrl) {
    return (
      <div className="neighbor-fallback-matte">
        <span className="tabular-nums">Unavailable</span>
      </div>
    )
  }

  return (
    <img
      src={blobUrl}
      alt={alt || `Frame ${expectedFrameId}`}
      className={className}
      loading="lazy"
      decoding="async"
      onError={() => setHasError(true)}
    />
  )
}
