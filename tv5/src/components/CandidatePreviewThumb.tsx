import React, { useEffect, useState } from 'react'
import { SearchCandidate } from '../types/contracts'
import { getBestKeyframeUrl } from '../api/tv4Client'
import { isFixtureModeActive, getFixturePreviewDataUri } from '../fixtures/fixtureData'
import { loadExactImage } from '../api/exactImageCache'

interface CandidatePreviewThumbProps {
  candidate: SearchCandidate
  alt?: string
  className?: string
  loading?: 'lazy' | 'eager'
  onErrorFallback?: React.ReactNode
}

export const CandidatePreviewThumb: React.FC<CandidatePreviewThumbProps> = ({
  candidate,
  alt,
  className = '',
  loading = 'lazy',
  onErrorFallback,
}) => {
  const isExactCorrected =
    (candidate.anchor_offset != null && candidate.anchor_offset !== 0) ||
    (candidate.certified_anchor_frame_id != null &&
      candidate.certified_anchor_frame_id !== candidate.frame_id)

  const [hasError, setHasError] = useState(false)
  const [exactBlobUrl, setExactBlobUrl] = useState<string | null>(null)
  const [coarseUrl, setCoarseUrl] = useState<string | null>(null)
  const [isLoadingExact, setIsLoadingExact] = useState(false)

  const rootAnchorFid = candidate.certified_anchor_frame_id ?? candidate.frame_id
  const rootAnchorTs = candidate.certified_anchor_timestamp_ms ?? candidate.timestamp_ms ?? 0
  const offset = candidate.anchor_offset ?? 0

  useEffect(() => {
    setHasError(false)

    if (!isExactCorrected) {
      setExactBlobUrl(null)
      setIsLoadingExact(false)
      setCoarseUrl(null)
      let isSubscribed = true
      getBestKeyframeUrl(candidate.video_id, candidate.frame_id, candidate.timestamp_ms)
        .then((url) => {
          if (isSubscribed) setCoarseUrl(url)
        })
        .catch(() => {
          if (isSubscribed) setHasError(true)
        })
      return () => {
        isSubscribed = false
      }
    }

    setCoarseUrl(null)

    if (isFixtureModeActive()) {
      setExactBlobUrl(getFixturePreviewDataUri(candidate.video_id, candidate.frame_id))
      setIsLoadingExact(false)
      return
    }

    let isSubscribed = true
    setIsLoadingExact(true)

    loadExactImage(
      {
        video_id: candidate.video_id,
        frame_id: rootAnchorFid,
        timestamp_ms: rootAnchorTs,
        certified_anchor_frame_id: rootAnchorFid,
        certified_anchor_timestamp_ms: rootAnchorTs,
        cumulative_offset: offset,
        offsets: [0],
      },
      {
        frame_id: candidate.frame_id,
        video_id: candidate.video_id,
      }
    )
      .then((res) => {
        if (!isSubscribed) return
        if (res.blobUrl) {
          setExactBlobUrl(res.blobUrl)
          setHasError(false)
        } else {
          setHasError(true)
        }
      })
      .catch(() => {
        if (!isSubscribed) return
        setHasError(true)
      })
      .finally(() => {
        if (isSubscribed) {
          setIsLoadingExact(false)
        }
      })

    return () => {
      isSubscribed = false
    }
  }, [candidate.video_id, candidate.frame_id, rootAnchorFid, rootAnchorTs, offset, isExactCorrected])

  if (hasError) {
    return onErrorFallback ? (
      <>{onErrorFallback}</>
    ) : (
      <div className="candidate-fallback-matte">
        <span className="fallback-title">Preview unavailable</span>
      </div>
    )
  }

  if (isExactCorrected) {
    if (isLoadingExact && !exactBlobUrl) {
      return (
        <div className="candidate-loading-matte">
          <span className="fallback-title">Loading...</span>
        </div>
      )
    }

    if (exactBlobUrl) {
      return (
        <img
          src={exactBlobUrl}
          alt={alt || `${candidate.video_id} frame ${candidate.frame_id}`}
          className={className}
          loading={loading}
          decoding="async"
          onError={() => setHasError(true)}
        />
      )
    }
  }

  // Coarse untouched retrieval candidate: uses pre-extracted keyframe thumbnail
  if (!coarseUrl) {
    return (
      <div className="candidate-loading-matte">
        <span className="fallback-title">Loading...</span>
      </div>
    )
  }
  return (
    <img
      src={coarseUrl}
      alt={alt || `${candidate.video_id} frame ${candidate.frame_id}`}
      className={className}
      loading={loading}
      decoding="async"
      onError={() => setHasError(true)}
    />
  )
}
