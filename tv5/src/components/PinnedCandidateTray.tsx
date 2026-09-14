import React, { useState } from 'react'
import { useAppDispatch, useAppState } from '../state/AppContext'
import { PinIcon, BasketPlusIcon, CheckIcon, ClearIcon } from './Icons'
import { getThumbnailUrl, getSelectedFrameUrl } from '../api/tv4Client'
import { SearchCandidate } from '../types/contracts'
import { telemetry } from '../utils/telemetry'

export const PinnedCandidateTray: React.FC = () => {
  const dispatch = useAppDispatch()
  const { pinnedCandidates, submissionBasket, activeCandidate } = useAppState()
  const [isCollapsed, setIsCollapsed] = useState(false)

  if (!pinnedCandidates || pinnedCandidates.length === 0) {
    return null
  }

  const handleSelectCandidate = (cand: SearchCandidate) => {
    dispatch({ type: 'SELECT_CANDIDATE', payload: cand })
    dispatch({ type: 'SET_ACTIVE_TAB', payload: 'inspection' })
    telemetry.record({
      action: 'SELECT_PINNED_CANDIDATE',
      taskMode: 'KIS',
      videoId: cand.video_id,
      frameId: cand.frame_id,
    })
  }

  const handleUnpin = (cand: SearchCandidate, e: React.MouseEvent) => {
    e.stopPropagation()
    dispatch({
      type: 'UNPIN_CANDIDATE',
      payload: { video_id: cand.video_id, frame_id: cand.frame_id },
    })
    telemetry.record({
      action: 'UNPIN_CANDIDATE',
      taskMode: 'KIS',
      videoId: cand.video_id,
      frameId: cand.frame_id,
    })
  }

  const handleClearAll = (e: React.MouseEvent) => {
    e.stopPropagation()
    dispatch({ type: 'CLEAR_PINNED_CANDIDATES' })
  }

  const handleToggleBasket = (cand: SearchCandidate, e: React.MouseEvent) => {
    e.stopPropagation()
    const isInBasket = submissionBasket.some(
      (b) => b.video_id === cand.video_id && b.frame_id === cand.frame_id
    )
    if (isInBasket) {
      dispatch({
        type: 'REMOVE_FROM_BASKET',
        payload: { video_id: cand.video_id, frame_id: cand.frame_id },
      })
    } else {
      dispatch({
        type: 'ADD_KIS_TO_BASKET',
        payload: { candidate: cand },
      })
    }
  }

  return (
    <div
      className="pinned-candidates-tray"
      style={{
        position: 'fixed',
        bottom: '16px',
        right: '16px',
        zIndex: 9999,
        background: 'rgba(15, 23, 42, 0.94)',
        backdropFilter: 'blur(12px)',
        border: '1px solid rgba(0, 229, 255, 0.4)',
        borderRadius: '8px',
        boxShadow: '0 10px 30px rgba(0, 0, 0, 0.7), 0 0 16px rgba(0, 229, 255, 0.2)',
        maxWidth: 'calc(100vw - 32px)',
        width: isCollapsed ? 'auto' : 'auto',
        transition: 'all 0.2s ease',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '6px 10px',
          background: 'rgba(0, 229, 255, 0.1)',
          borderBottom: isCollapsed ? 'none' : '1px solid rgba(255, 255, 255, 0.08)',
          gap: '12px',
          cursor: 'pointer',
        }}
        onClick={() => setIsCollapsed(!isCollapsed)}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <PinIcon size={14} color="#00e5ff" />
          <span style={{ fontSize: '11.5px', fontWeight: 700, color: '#00e5ff', letterSpacing: '0.3px' }}>
            Khay ghim
          </span>
          <span
            style={{
              background: '#00e5ff',
              color: '#000',
              fontWeight: 800,
              fontSize: '10px',
              padding: '1px 6px',
              borderRadius: '10px',
            }}
          >
            {pinnedCandidates.length}
          </span>
          <span style={{ fontSize: '10px', color: '#64748b', marginLeft: '4px' }}>
            (Phím P để ghim)
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          {!isCollapsed && (
            <button
              type="button"
              onClick={handleClearAll}
              style={{
                background: 'rgba(239, 68, 68, 0.15)',
                color: '#f87171',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                borderRadius: '4px',
                padding: '2px 6px',
                fontSize: '10px',
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '3px',
              }}
              title="Gỡ tất cả video trong khay ghim"
            >
              <ClearIcon size={11} color="#f87171" />
              <span>Xóa hết</span>
            </button>
          )}

          <button
            type="button"
            style={{
              background: 'transparent',
              border: 'none',
              color: '#94a3b8',
              fontSize: '11px',
              cursor: 'pointer',
              padding: '0 4px',
              fontWeight: 700,
            }}
            title={isCollapsed ? 'Mở rộng khay' : 'Thu nhỏ khay'}
          >
            {isCollapsed ? '▲' : '▼'}
          </button>
        </div>
      </div>

      {/* Body: Horizontal Filmstrip */}
      {!isCollapsed && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '8px 10px',
            overflowX: 'auto',
            maxWidth: '650px',
            scrollbarWidth: 'thin',
          }}
        >
          {pinnedCandidates.map((cand) => {
            const isActive =
              activeCandidate?.video_id === cand.video_id &&
              activeCandidate?.frame_id === cand.frame_id
            const isInBasket = submissionBasket.some(
              (b) => b.video_id === cand.video_id && b.frame_id === cand.frame_id
            )
            const mins = Math.floor((cand.timestamp_ms || 0) / 60000)
            const secs = String(Math.floor(((cand.timestamp_ms || 0) % 60000) / 1000)).padStart(2, '0')

            return (
              <div
                key={`pinned-${cand.video_id}-${cand.frame_id}`}
                onClick={() => handleSelectCandidate(cand)}
                style={{
                  flex: '0 0 120px',
                  borderRadius: '6px',
                  overflow: 'hidden',
                  background: isActive ? 'rgba(0, 229, 255, 0.2)' : 'rgba(30, 41, 59, 0.7)',
                  border: isActive ? '2px solid #00e5ff' : '1px solid rgba(255, 255, 255, 0.12)',
                  boxShadow: isActive ? '0 0 12px rgba(0, 229, 255, 0.4)' : 'none',
                  cursor: 'pointer',
                  position: 'relative',
                  transition: 'all 0.15s ease',
                }}
                title={`Click để chuyển sang soi ${cand.video_id} (Frame ${cand.frame_id})`}
              >
                {/* Image Thumb */}
                <div style={{ width: '100%', height: '68px', position: 'relative', background: '#000' }}>
                  <img
                    src={getThumbnailUrl(cand.video_id, cand.frame_id)}
                    alt={`${cand.video_id} F.${cand.frame_id}`}
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                    loading="lazy"
                    onError={(e) => {
                      const target = e.currentTarget
                      if (!target.src.includes('/frames/')) {
                        target.src = getSelectedFrameUrl(cand.video_id, cand.frame_id)
                      }
                    }}
                  />

                  {/* Top-Right Unpin 'X' Button */}
                  <button
                    type="button"
                    onClick={(e) => handleUnpin(cand, e)}
                    style={{
                      position: 'absolute',
                      top: '2px',
                      right: '2px',
                      background: 'rgba(0, 0, 0, 0.75)',
                      color: '#f87171',
                      border: '1px solid rgba(239, 68, 68, 0.4)',
                      borderRadius: '50%',
                      width: '18px',
                      height: '18px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      cursor: 'pointer',
                      fontSize: '10px',
                      fontWeight: 800,
                      padding: 0,
                    }}
                    title="Bỏ ghim video này"
                  >
                    ✕
                  </button>

                  {/* Bottom-Left Timecode */}
                  <span
                    style={{
                      position: 'absolute',
                      bottom: '2px',
                      left: '3px',
                      background: 'rgba(0,0,0,0.75)',
                      color: '#fff',
                      fontSize: '8.5px',
                      padding: '1px 3px',
                      borderRadius: '2px',
                      fontFamily: 'monospace',
                    }}
                  >
                    {mins}:{secs}
                  </span>

                  {/* Bottom-Right Basket Button */}
                  <button
                    type="button"
                    onClick={(e) => handleToggleBasket(cand, e)}
                    style={{
                      position: 'absolute',
                      bottom: '2px',
                      right: '2px',
                      background: isInBasket ? 'rgba(16, 185, 129, 0.9)' : 'rgba(15, 23, 42, 0.75)',
                      color: isInBasket ? '#fff' : '#00e5ff',
                      border: isInBasket ? '1px solid #10b981' : '1px solid rgba(0, 229, 255, 0.4)',
                      borderRadius: '3px',
                      padding: '1px 4px',
                      fontSize: '8.5px',
                      fontWeight: 700,
                      cursor: 'pointer',
                    }}
                    title={isInBasket ? 'Đã lưu trong giỏ (Click để xóa)' : 'Lưu vào giỏ'}
                  >
                    {isInBasket ? <CheckIcon size={10} color="#fff" /> : <BasketPlusIcon size={10} color="#00e5ff" />}
                  </button>
                </div>

                {/* Footer Label */}
                <div
                  style={{
                    padding: '3px 5px',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    fontSize: '9.5px',
                    color: isActive ? '#00e5ff' : '#cbd5e1',
                    fontWeight: isActive ? 700 : 500,
                    background: 'rgba(15, 23, 42, 0.85)',
                  }}
                >
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {cand.video_id}
                  </span>
                  <span style={{ color: '#94a3b8', fontSize: '8.5px' }}>
                    F.{cand.frame_id}
                  </span>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
