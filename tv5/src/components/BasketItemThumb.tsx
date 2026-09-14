import React, { useState, useEffect } from 'react'
import { getBestKeyframeUrl, getThumbnailUrl, getSelectedFrameUrl } from '../api/tv4Client'

interface BasketItemThumbProps {
  videoId: string
  frameId: number
  timestampMs?: number
}

export const BasketItemThumb: React.FC<BasketItemThumbProps> = ({
  videoId,
  frameId,
  timestampMs,
}) => {
  const [imgSrc, setImgSrc] = useState<string | null>(null)
  const [hasError, setHasError] = useState(false)

  useEffect(() => {
    let isSubscribed = true
    setHasError(false)
    getBestKeyframeUrl(videoId, frameId, timestampMs)
      .then((url) => {
        if (isSubscribed) {
          setImgSrc(url)
        }
      })
      .catch(() => {
        if (isSubscribed) {
          setImgSrc(getThumbnailUrl(videoId, frameId))
        }
      })
    return () => {
      isSubscribed = false
    }
  }, [videoId, frameId, timestampMs])

  const mins = timestampMs != null ? Math.floor(timestampMs / 60000) : 0
  const secs = timestampMs != null ? String(Math.floor((timestampMs % 60000) / 1000)).padStart(2, '0') : '00'

  if (hasError || !imgSrc) {
    return (
      <div
        style={{
          width: '100%',
          height: '100%',
          background: 'linear-gradient(135deg, #0f172a, #1e293b)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#94a3b8',
          fontSize: '10px',
          fontWeight: 600,
        }}
      >
        <span style={{ color: '#00e5ff' }}>F.{frameId}</span>
        <span style={{ fontSize: '9px', opacity: 0.8 }}>{mins}:{secs}</span>
      </div>
    )
  }

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative', background: '#000' }}>
      <img
        src={imgSrc}
        alt={`${videoId} frame ${frameId}`}
        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
        loading="lazy"
        onError={(e) => {
          const target = e.currentTarget
          if (!target.src.includes('/frames/')) {
            target.src = getSelectedFrameUrl(videoId, frameId)
          } else {
            setHasError(true)
          }
        }}
      />
      {timestampMs !== undefined && (
        <span
          style={{
            position: 'absolute',
            bottom: '2px',
            right: '2px',
            background: 'rgba(0, 0, 0, 0.8)',
            color: '#fff',
            fontSize: '9px',
            padding: '1px 3px',
            borderRadius: '2px',
            fontFamily: 'monospace',
          }}
        >
          {mins}:{secs}
        </span>
      )}
    </div>
  )
}
