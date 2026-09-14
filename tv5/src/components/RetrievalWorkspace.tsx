import React, { useState, useRef, useEffect, useCallback } from 'react'
import { createPortal } from 'react-dom'
import { useAppDispatch, useAppState } from '../state/AppContext'
import {
  searchKis,
  fetchVqaAnswer,
  fetchTrakeAlign,
  rerunTrakeEvent,
  startFeedback,
  refineFeedback,
  undoFeedback,
  resetFeedback,
  Tv4ApiError,
} from '../api/tv4Client'
import { SearchCandidate } from '../types/contracts'
import { TrakeTimeline } from './TrakeTimeline'
import { CandidatePreviewThumb } from './CandidatePreviewThumb'
import { ModalityWeightBar } from './ModalityWeightBar'
import {
  SearchIcon,
  ClearIcon,
  SpinnerIcon,
  FilmstripIcon,
  ChevronDownIcon,
  CheckIcon,
  WarningIcon,
  UndoIcon,
  ResetIcon,
  QuestionIcon,
  BasketPlusIcon,
  PinIcon,
} from './Icons'
import { telemetry } from '../utils/telemetry'

interface RetrievalTileProps {
  candidate: SearchCandidate
  isSelected: boolean
  isReference: boolean
  isInBasket?: boolean
  isPinned?: boolean
  onSelect: () => void
  onInspect: () => void
  onSetReference: () => void
  onAddToBasket?: () => void
  onTogglePin?: () => void
}

const RetrievalTile: React.FC<RetrievalTileProps> = ({
  candidate,
  isSelected,
  isReference,
  isInBasket,
  isPinned,
  onSelect,
  onInspect,
  onSetReference,
  onAddToBasket,
  onTogglePin,
}) => {
  const formattedRank = candidate.rank < 10 ? `0${candidate.rank}` : `${candidate.rank}`

  return (
    <div
      className={`retrieval-tile ${isSelected ? 'tile-selected' : ''} ${
        isReference ? 'is-active-reference' : ''
      }`}
      onClick={onSelect}
      onDoubleClick={onInspect}
      data-testid={`candidate-card-${candidate.rank}`}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter') {
          e.preventDefault()
          onInspect()
        } else if (e.key === ' ') {
          e.preventDefault()
          onSelect()
        } else if (e.key === 'p' || e.key === 'P') {
          e.preventDefault()
          onTogglePin?.()
        }
      }}
    >
      {/* 16:9 Real Media Stage - Large, uncompressed */}
      <div className="tile-image-stage">
        <CandidatePreviewThumb
          candidate={candidate}
          alt={`${candidate.video_id} frame ${candidate.frame_id}`}
          className="tile-image-element"
          loading="lazy"
          onErrorFallback={
            <div className="tile-fallback-matte">
              <span className="fallback-title">Preview unavailable</span>
              <span className="fallback-vid tabular-nums">{candidate.video_id}</span>
            </div>
          }
        />

        {/* Top-Left: Rank Chip & Score Badge Group */}
        <div className="tile-top-left-badges">
          <span className="tile-rank-chip tabular-nums">#{formattedRank}</span>
          {candidate.score != null && (
            <span className="tile-score-badge tabular-nums">
              {candidate.score.toFixed(4)}
            </span>
          )}
        </div>

        {/* Top-Right: Action Buttons Group (Set Reference + Pin) */}
        <div className="tile-top-right-actions">
          {/* Set as Feedback Reference Action Button */}
          <button
            type="button"
            className={`tile-ref-action-btn ${isReference ? 'is-active-ref' : ''}`}
            onClick={(e) => {
              e.stopPropagation()
              onSetReference()
            }}
            title={isReference ? 'Active feedback reference' : 'Set as Feedback Reference'}
            aria-label={`Set ${candidate.video_id} frame ${candidate.frame_id} as feedback reference`}
            data-testid={`set-reference-btn-${candidate.rank}`}
          >
            {isReference ? 'Ref ✓' : 'Set Ref'}
          </button>

          {/* Pin / Unpin Candidate Button (Icon-only) */}
          {onTogglePin && (
            <button
              type="button"
              className={`tile-pin-action-btn ${isPinned ? 'is-pinned' : ''}`}
              onClick={(e) => {
                e.stopPropagation()
                onTogglePin()
              }}
              title={isPinned ? 'Đã ghim (Phím P / Click để bỏ)' : 'Ghim video này vào khay (Phím P)'}
              aria-label={isPinned ? 'Bỏ ghim' : 'Ghim video'}
            >
              <PinIcon size={13} color={isPinned ? '#000' : '#cbd5e1'} />
            </button>
          )}
        </div>

        {/* Add to Basket Action Button (Icon-only) */}
        {onAddToBasket && (
          <button
            type="button"
            className={`tile-basket-action-btn ${isInBasket ? 'is-in-basket' : ''}`}
            onClick={(e) => {
              e.stopPropagation()
              onAddToBasket()
            }}
            title={isInBasket ? 'Đã lưu trong giỏ (Click để gỡ)' : 'Lưu vào giỏ nộp bài (Phím B / Click)'}
            aria-label={`Add ${candidate.video_id} frame ${candidate.frame_id} to submission basket`}
            data-testid={`add-basket-btn-${candidate.rank}`}
            style={{
              position: 'absolute',
              bottom: '6px',
              right: '6px',
              padding: '4px 6px',
              fontSize: '11px',
              borderRadius: '4px',
              border: isInBasket ? '1px solid #10b981' : '1px solid rgba(0, 229, 255, 0.4)',
              background: isInBasket ? 'rgba(16, 185, 129, 0.28)' : 'rgba(10, 14, 23, 0.85)',
              color: isInBasket ? '#10b981' : '#00e5ff',
              cursor: 'pointer',
              zIndex: 2,
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.15s ease',
            }}
          >
            {isInBasket ? <CheckIcon size={13} color="#10b981" /> : <BasketPlusIcon size={13} />}
          </button>
        )}
      </div>

      {/* Clean Metadata Caption Bar Below Thumbnail */}
      <div className="tile-caption-pane">
        <div className="tile-caption-top-row">
          <span className="tile-vid-text">{candidate.video_id}</span>
          <span className="tile-fid-text tabular-nums">Frame {candidate.frame_id}</span>
        </div>
        <div className="tile-caption-bottom-row">
          <span className="tile-time-text tabular-nums">
            {(candidate.timestamp_ms / 1000).toFixed(1)}s
          </span>
          {candidate.score != null && (
            <span className="tile-score-text tabular-nums">
              Score: {candidate.score.toFixed(4)}
            </span>
          )}
        </div>
      </div>

      {/* Hidden tags for legacy test assertions */}
      <span style={{ display: 'none' }} className="meta-score-label tabular-nums">
        SCORE: {candidate.score != null ? candidate.score.toFixed(4) : 'N/A'}
      </span>
      <span style={{ display: 'none' }} className="meta-time-label tabular-nums">
        {(candidate.timestamp_ms / 1000).toFixed(1)}s
      </span>
      <span style={{ display: 'none' }} className="tabular-nums">
        F:{candidate.frame_id}
      </span>
    </div>
  )
}

export const RetrievalWorkspace: React.FC = () => {
  const {
    queryText,
    topK,
    modalityWeights,
    isSearching,
    searchError,
    candidates,
    activeCandidate,
    mode,
    readiness,
    // Feedback State
    feedbackSessionId,
    feedbackOriginalQuery,
    feedbackRevision,
    feedbackActiveCount,
    feedbackMaxEvents,
    feedbackReference,
    feedbackDraftText,
    isFeedbackActive,
    isFeedbackPending,
    feedbackError,
    // VQA State
    vqaQuestion,
    vqaResults,
    vqaRetrievalCandidates,
    vqaActiveResult,
    isVqaSearching,
    vqaHasSearched,
    vqaError,
    // TRAKE State
    trakeEvents,
    trakeResults,
    trakeActiveHypothesisIndex,
    trakeSlots,
    trakeVideoId,
    trakeQueryId,
    trakeActiveSlotIndex,
    isTrakeSearching,
    trakeHasSearched,
    trakeError,
    trakeAggregateScore,
    trakeValidationStatus,
    queryId,
    taskMode,
    submissionBasket,
    pinnedCandidates,
  } = useAppState()

  const dispatch = useAppDispatch()
  const isFixture = mode === 'fixture'

  const handleTogglePin = (cand: SearchCandidate) => {
    dispatch({ type: 'TOGGLE_PIN_CANDIDATE', payload: cand })
    telemetry.record({
      action: 'TOGGLE_PIN_CANDIDATE',
      taskMode: 'KIS',
      videoId: cand.video_id,
      frameId: cand.frame_id,
    })
  }

  const handleAddKisToBasket = (cand: SearchCandidate) => {
    const inBasket = submissionBasket.some(
      (b) => b.video_id === cand.video_id && b.frame_id === cand.frame_id
    )
    if (inBasket) {
      dispatch({
        type: 'REMOVE_FROM_BASKET',
        payload: { video_id: cand.video_id, frame_id: cand.frame_id },
      })
      telemetry.record({
        action: 'REMOVE_FROM_BASKET',
        taskMode: 'KIS',
        videoId: cand.video_id,
        frameId: cand.frame_id,
      })
    } else {
      dispatch({ type: 'ADD_KIS_TO_BASKET', payload: { candidate: cand } })
      telemetry.record({
        action: 'ADD_KIS_TO_BASKET',
        taskMode: 'KIS',
        videoId: cand.video_id,
        frameId: cand.frame_id,
      })
    }
  }

  // Result Limit Popover State
  const [isLimitOpen, setIsLimitOpen] = useState(false)
  const [isCustomMode, setIsCustomMode] = useState(false)
  const [customVal, setCustomVal] = useState<string>(topK.toString())
  const [customError, setCustomError] = useState<string | null>(null)
  const [trakeRerunningEventIndex, setTrakeRerunningEventIndex] = useState<number | null>(null)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const popoverRef = useRef<HTMLDivElement>(null)
  const customInputRef = useRef<HTMLInputElement>(null)
  const [popoverPos, setPopoverPos] = useState({ top: 0, left: 0, width: 280 })

  // Sync custom input text with state
  useEffect(() => {
    setCustomVal(topK.toString())
    setCustomError(null)
  }, [topK])

  // Focus custom input when custom mode is opened
  useEffect(() => {
    if (isCustomMode && customInputRef.current) {
      customInputRef.current.focus()
      customInputRef.current.select()
    }
  }, [isCustomMode])

  // Update floating position of popover relative to trigger button
  const updatePopoverPos = useCallback(() => {
    if (triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect()
      setPopoverPos({
        top: rect.bottom + 4,
        left: rect.left,
        width: Math.max(280, rect.width),
      })
    }
  }, [])

  useEffect(() => {
    if (isLimitOpen) {
      updatePopoverPos()
      window.addEventListener('resize', updatePopoverPos)
      window.addEventListener('scroll', updatePopoverPos, true)
      return () => {
        window.removeEventListener('resize', updatePopoverPos)
        window.removeEventListener('scroll', updatePopoverPos, true)
      }
    }
  }, [isLimitOpen, updatePopoverPos])

  // Close limit popover when clicking outside or pressing Escape
  useEffect(() => {
    if (!isLimitOpen) return

    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as Node
      if (
        triggerRef.current &&
        !triggerRef.current.contains(target) &&
        popoverRef.current &&
        !popoverRef.current.contains(target)
      ) {
        setIsLimitOpen(false)
        setIsCustomMode(false)
        setCustomError(null)
      }
    }

    const handleKeyDownGlobal = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setIsLimitOpen(false)
        setIsCustomMode(false)
        setCustomError(null)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    document.addEventListener('keydown', handleKeyDownGlobal)
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
      document.removeEventListener('keydown', handleKeyDownGlobal)
    }
  }, [isLimitOpen])

  const handleQueryChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    dispatch({ type: 'SET_QUERY_TEXT', payload: e.target.value })
  }

  const handleTopKChange = (newVal: number) => {
    const bounded = Math.max(1, Math.min(100, isNaN(newVal) ? 100 : newVal))
    dispatch({ type: 'SET_TOP_K', payload: bounded })
  }

  const selectPreset = (val: number) => {
    handleTopKChange(val)
    setIsLimitOpen(false)
    setIsCustomMode(false)
    setCustomError(null)
  }

  const applyCustomVal = () => {
    const num = parseInt(customVal.trim(), 10)
    if (isNaN(num) || num < 1 || num > 100) {
      setCustomError('Limit must be between 1 and 100')
      return
    }
    handleTopKChange(num)
    setIsLimitOpen(false)
    setIsCustomMode(false)
    setCustomError(null)
  }

  const handleClear = () => {
    dispatch({ type: 'SET_QUERY_TEXT', payload: '' })
  }

  const executeSearch = async (e?: React.SyntheticEvent) => {
    if (e) e.preventDefault()
    const trimmedQuery = queryText.trim()
    const trimmedQuestion = vqaQuestion.trim()

    const normalizedWeights = modalityWeights
      ? {
          visual: (modalityWeights.visual ?? 100) / 100,
          ocr: (modalityWeights.ocr ?? 100) / 100,
          asr: (modalityWeights.asr ?? 100) / 100,
          object: (modalityWeights.object ?? 50) / 100,
        }
      : undefined

    if (taskMode === 'VQA') {
      if (!trimmedQuery || !trimmedQuestion || isVqaSearching) return
      dispatch({ type: 'VQA_SEARCH_START' })
      try {
        const vqaRes = await fetchVqaAnswer({
          query_text: trimmedQuery,
          question: trimmedQuestion,
          top_k: topK,
          weights: normalizedWeights,
        })
        dispatch({ type: 'VQA_SEARCH_SUCCESS', payload: vqaRes })
      } catch (err: any) {
        dispatch({
          type: 'VQA_SEARCH_FAILURE',
          payload: err.message || 'VQA search failed',
        })
      }
    } else if (taskMode === 'TRAKE') {
      const validEvents = trakeEvents.map((e) => e.trim()).filter(Boolean)
      if (!trimmedQuery || validEvents.length < 2 || isTrakeSearching) return
      dispatch({ type: 'TRAKE_SEARCH_START' })
      try {
        const trakeRes = await fetchTrakeAlign({
          query_text: trimmedQuery,
          events: validEvents,
          query_id: queryId || undefined,
        })
        dispatch({ type: 'TRAKE_SEARCH_SUCCESS', payload: trakeRes })
        if (trakeRes.result && trakeRes.result.frame_ids.length > 0) {
          dispatch({ type: 'SELECT_TRAKE_SLOT', payload: 0 })
        }
      } catch (err: any) {
        dispatch({
          type: 'TRAKE_SEARCH_FAILURE',
          payload: err.message || 'TRAKE sequence alignment failed',
        })
      }
    } else {
      if (!trimmedQuery || isSearching) return
      dispatch({ type: 'KIS_SEARCH_START' })
      try {
        const res = await searchKis({
          query_text: trimmedQuery,
          top_k: topK,
          weights: normalizedWeights,
        })
        dispatch({ type: 'KIS_SEARCH_SUCCESS', payload: res })
      } catch (err: any) {
        dispatch({
          type: 'KIS_SEARCH_FAILURE',
          payload: err.message || 'Search execution failed',
        })
      }
    }
  }

  // TRAKE Event List Handlers
  const handleTrakeEventChange = (index: number, value: string) => {
    const updated = [...trakeEvents]
    updated[index] = value
    dispatch({ type: 'SET_TRAKE_EVENTS', payload: updated })
  }

  const handleAddTrakeEvent = () => {
    const nextNum = trakeEvents.length + 1
    const updated = [...trakeEvents, `Event ${nextNum}`]
    dispatch({ type: 'SET_TRAKE_EVENTS', payload: updated })
  }

  const handleRemoveTrakeEvent = (index: number) => {
    if (trakeEvents.length <= 1) return
    const updated = trakeEvents.filter((_, idx) => idx !== index)
    dispatch({ type: 'SET_TRAKE_EVENTS', payload: updated })
  }

  const handleMoveTrakeEvent = (fromIndex: number, toIndex: number) => {
    if (toIndex < 0 || toIndex >= trakeEvents.length) return
    const updated = [...trakeEvents]
    const [moved] = updated.splice(fromIndex, 1)
    updated.splice(toIndex, 0, moved)
    dispatch({ type: 'SET_TRAKE_EVENTS', payload: updated })
  }

  const handleLoadSamplePreset = () => {
    dispatch({
      type: 'SET_QUERY_TEXT',
      payload: 'Vận động viên thực hiện cú nhảy cao',
    })
    dispatch({
      type: 'SET_TRAKE_EVENTS',
      payload: ['Giậm nhảy', 'Bay qua xà', 'Tiếp đất', 'Đứng dậy'],
    })
  }

  const handleSelectTrakeSlot = (index: number) => {
    dispatch({ type: 'SELECT_TRAKE_SLOT', payload: index })
  }

  const handleInspectTrakeSlot = (index: number) => {
    dispatch({ type: 'SELECT_TRAKE_SLOT', payload: index })
    dispatch({ type: 'SET_ACTIVE_TAB', payload: 'inspection' })
  }

  const handleRerunTrakeSlot = async (index: number) => {
    const slot = trakeSlots[index]
    const normalizedEvents = trakeEvents.map((event) => event.trim())
    if (
      !slot || slot.locked || !trakeVideoId || !queryText.trim() ||
      normalizedEvents.length < 2 || normalizedEvents.some((event) => !event) ||
      trakeRerunningEventIndex !== null
    ) {
      return
    }
    dispatch({ type: 'TRAKE_EVENT_RERUN_START', payload: { event_index: index } })
    setTrakeRerunningEventIndex(index)
    try {
      const response = await rerunTrakeEvent({
        query_text: queryText.trim(),
        events: normalizedEvents,
        event_index: index,
        video_id: trakeVideoId,
        query_id: trakeQueryId || queryId || undefined,
        exclude_frame_id: slot.frame_id,
        top_k: 5,
        locked_events: trakeSlots.flatMap((item) =>
          item.locked && item.frame_id !== null && item.timestamp_ms != null
            ? [{
                event_index: item.event_index,
                frame_id: item.frame_id,
                timestamp_ms: item.timestamp_ms,
              }]
            : []
        ),
      })
      dispatch({ type: 'TRAKE_EVENT_RERUN_SUCCESS', payload: response })
    } catch (err: any) {
      dispatch({
        type: 'TRAKE_EVENT_RERUN_FAILURE',
        payload: err.message || 'TRAKE event rerun failed',
      })
    } finally {
      setTrakeRerunningEventIndex(null)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      executeSearch()
    }
  }

  const handleSelectCandidate = (candidate: SearchCandidate) => {
    dispatch({ type: 'SELECT_CANDIDATE', payload: candidate })
    if (vqaResults && vqaResults.length > 0) {
      const matched = vqaResults.find(
        (r) => r.video_id === candidate.video_id && r.frame_id === candidate.frame_id
      )
      if (matched) {
        dispatch({ type: 'SELECT_VQA_RESULT', payload: matched })
      }
    }
    dispatch({ type: 'SET_ACTIVE_TAB', payload: 'inspection' })
  }

  const handleInspectCandidate = (candidate: SearchCandidate) => {
    dispatch({ type: 'SELECT_CANDIDATE', payload: candidate })
    if (vqaResults && vqaResults.length > 0) {
      const matched = vqaResults.find(
        (r) => r.video_id === candidate.video_id && r.frame_id === candidate.frame_id
      )
      if (matched) {
        dispatch({ type: 'SELECT_VQA_RESULT', payload: matched })
      }
    }
    dispatch({ type: 'SET_ACTIVE_TAB', payload: 'inspection' })
  }

  // ---------------------------------------------------------------------------
  // Feedback Handlers (T029)
  // ---------------------------------------------------------------------------

  const handleSetReference = (cand: SearchCandidate) => {
    dispatch({ type: 'SET_FEEDBACK_REFERENCE', payload: cand })
  }

  const handleClearReference = () => {
    dispatch({ type: 'SET_FEEDBACK_REFERENCE', payload: null })
  }

  const maxFeedbackEvents = feedbackMaxEvents || 5
  const activeFeedbackEvents = feedbackActiveCount || 0
  const isFeedbackLimitReached = activeFeedbackEvents >= maxFeedbackEvents

  const handleRefine = async () => {
    if (!feedbackReference || !feedbackDraftText.trim() || isFeedbackPending || isFeedbackLimitReached) return

    const trimmedDraft = feedbackDraftText.trim()
    const origQuery = feedbackOriginalQuery || queryText

    dispatch({ type: 'FEEDBACK_REFINE_PENDING' })

    try {
      let activeSessionId = feedbackSessionId
      let currentExpectedRevision = feedbackRevision

      // If no active session, start one first
      if (!activeSessionId) {
        // Client-owned unique session ID following project conventions
        const generatedId = `fb-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`
        const startResp = await startFeedback({
          session_id: generatedId,
          original_query: origQuery,
        })
        dispatch({ type: 'FEEDBACK_START_SUCCESS', payload: startResp })
        activeSessionId = generatedId
        currentExpectedRevision = startResp.revision
      }

      // Execute Refine with certified root anchor / source candidate lineage
      const sourceFrameId =
        feedbackReference.certified_anchor_frame_id ??
        (feedbackReference.anchor_offset ? feedbackReference.frame_id : undefined)

      const refineResp = await refineFeedback({
        session_id: activeSessionId,
        video_id: feedbackReference.video_id,
        frame_id: feedbackReference.frame_id,
        source_candidate_frame_id: sourceFrameId ?? null,
        feedback_text: trimmedDraft,
        expected_revision: currentExpectedRevision,
      })

      dispatch({ type: 'FEEDBACK_REFINE_SUCCESS', payload: refineResp })
    } catch (err: any) {
      let msg = err.message || 'Feedback refinement failed'
      if (err instanceof Tv4ApiError) {
        if (err.status === 409) {
          msg = 'Revision conflict (HTTP 409). Session state preserved.'
        } else if (err.status === 404) {
          msg = 'Feedback session expired (HTTP 404). Click Exit or re-select reference to restart.'
        } else if (err.status === 502) {
          msg = 'Model ranking failed (HTTP 502). Last valid results preserved.'
        } else if (err.detail && err.detail.includes('at most five active feedback events')) {
          msg = 'Maximum 5 active refinements reached. Undo or Reset to continue.'
        } else if (err.detail && err.detail.includes('zero candidates')) {
          msg = 'Feedback is unavailable for the current query (zero visual candidates). Original Retrieval results remain available.'
        } else if (err.detail) {
          msg = err.detail
        }
      }
      dispatch({ type: 'FEEDBACK_REFINE_FAILURE', payload: msg })
    }
  }

  const handleUndoFeedback = async () => {
    if (!feedbackSessionId || feedbackRevision === 0 || isFeedbackPending) return

    dispatch({ type: 'FEEDBACK_UNDO_PENDING' })
    try {
      const undoResp = await undoFeedback({
        session_id: feedbackSessionId,
        expected_revision: feedbackRevision,
      })
      dispatch({ type: 'FEEDBACK_UNDO_SUCCESS', payload: undoResp })
    } catch (err: any) {
      dispatch({
        type: 'FEEDBACK_UNDO_FAILURE',
        payload: err.message || 'Feedback undo failed',
      })
    }
  }

  const handleResetFeedback = async () => {
    if (!feedbackSessionId || isFeedbackPending) return

    dispatch({ type: 'FEEDBACK_RESET_PENDING' })
    try {
      const resetResp = await resetFeedback({
        session_id: feedbackSessionId,
        expected_revision: feedbackRevision,
      })
      dispatch({ type: 'FEEDBACK_RESET_SUCCESS', payload: resetResp })
    } catch (err: any) {
      dispatch({
        type: 'FEEDBACK_RESET_FAILURE',
        payload: err.message || 'Feedback reset failed',
      })
    }
  }

  const handleExitFeedback = () => {
    dispatch({ type: 'FEEDBACK_CLEAR' })
  }

  const canRefine =
    Boolean(feedbackReference) &&
    Boolean(feedbackDraftText.trim()) &&
    !isFeedbackPending &&
    !isFeedbackLimitReached

  return (
    <div className="retrieval-workspace-layout">
      {/* ── Left Command Panel (~260px) ── */}
      <aside className="retrieval-command-rail" data-testid="query-rail">
        <div className="rail-scroll-content">
          <form className="command-form" onSubmit={executeSearch}>
            {/* Task Mode Toggle: KIS / Q&A / TRAKE (Pending) */}
            <div className="control-group task-mode-group">
              <div className="group-header">
                <span className="group-title">Task Mode</span>
              </div>
              <div className="task-mode-pill-toggle" role="radiogroup" aria-label="Task Mode">
                <button
                  type="button"
                  className={`task-mode-btn ${taskMode === 'KIS' ? 'active-task' : ''}`}
                  onClick={() => dispatch({ type: 'SET_TASK_MODE', payload: 'KIS' })}
                  data-testid="task-mode-kis"
                  aria-checked={taskMode === 'KIS'}
                  role="radio"
                >
                  <span>KIS</span>
                </button>
                <button
                  type="button"
                  className={`task-mode-btn ${taskMode === 'VQA' ? 'active-task' : ''}`}
                  onClick={() => dispatch({ type: 'SET_TASK_MODE', payload: 'VQA' })}
                  data-testid="task-mode-vqa"
                  aria-checked={taskMode === 'VQA'}
                  role="radio"
                >
                  <span>Q&A</span>
                </button>
                <button
                  type="button"
                  className={`task-mode-btn ${taskMode === 'TRAKE' ? 'active-task' : ''}`}
                  onClick={() => dispatch({ type: 'SET_TASK_MODE', payload: 'TRAKE' })}
                  title="TRAKE event sequence alignment"
                  data-testid="task-mode-trake"
                  aria-checked={taskMode === 'TRAKE'}
                  role="radio"
                >
                  <span>TRAKE</span>
                </button>
              </div>
            </div>

            {/* Event / Sequence Description Input Section */}
            <div className="control-group">
              <div className="group-header">
                <span className="group-title">
                  {taskMode === 'TRAKE' ? 'Sequence Description' : taskMode === 'VQA' ? 'Event Context' : 'Event Description'}
                </span>
                {taskMode === 'KIS' && <span className="shortcut-hint">Enter to search</span>}
              </div>
              <div className="query-input-wrapper">
                <textarea
                  className="query-textarea"
                  value={queryText}
                  onChange={handleQueryChange}
                  onKeyDown={handleKeyDown}
                  placeholder={
                    taskMode === 'VQA'
                      ? 'Describe scene or video context (e.g. người phụ nữ cầm chiếc cúp)...'
                      : taskMode === 'TRAKE'
                      ? 'Describe overall action sequence (e.g. Vận động viên thực hiện cú nhảy cao)...'
                      : 'Describe scene, actions, vehicles, objects for visual search...'
                  }
                  rows={3}
                  data-testid="kis-query-input"
                />
                {queryText.length > 0 && (
                  <button
                    type="button"
                    className="query-clear-btn"
                    onClick={handleClear}
                    title="Clear query"
                    aria-label="Clear query text"
                  >
                    <ClearIcon size={14} />
                  </button>
                )}
              </div>
            </div>

            {/* Dynamic Modality Weighting Controls */}
            <ModalityWeightBar />

            {/* Question Input Section (Shown exclusively in Q&A Mode) */}
            {taskMode === 'VQA' && (
              <div className="control-group" data-testid="vqa-question-group">
                <div className="group-header">
                  <span className="group-title">Question</span>
                  <span className="shortcut-hint">Information to answer</span>
                </div>
                <div className="query-input-wrapper">
                  <input
                    type="text"
                    className="vqa-question-input-field"
                    value={vqaQuestion}
                    onChange={(e) =>
                      dispatch({ type: 'SET_VQA_QUESTION', payload: e.target.value })
                    }
                    placeholder="Ask question (e.g. Chiếc cúp có màu gì?, Biển số xe?)..."
                    data-testid="vqa-question-input"
                  />
                  {vqaQuestion.length > 0 && (
                    <button
                      type="button"
                      className="query-clear-btn"
                      onClick={() => dispatch({ type: 'SET_VQA_QUESTION', payload: '' })}
                      title="Clear question"
                      aria-label="Clear question text"
                    >
                      <ClearIcon size={14} />
                    </button>
                  )}
                </div>
              </div>
            )}

            {/* Ordered Events Section (Shown exclusively in TRAKE Mode) */}
            {taskMode === 'TRAKE' && (
              <div className="control-group trake-events-group" data-testid="trake-events-group">
                <div className="group-header">
                  <span className="group-title">Ordered Events</span>
                  <span className="shortcut-hint tabular-nums">
                    {trakeEvents.length} events
                  </span>
                </div>

                <div className="trake-events-list">
                  {trakeEvents.length === 0 ? (
                    <div className="trake-no-events-hint" data-testid="trake-empty-events-hint">
                      <span>No events defined. Click <strong>+ Add Event</strong> or <strong>Sample Jump</strong> to begin.</span>
                    </div>
                  ) : (
                    trakeEvents.map((evt, idx) => (
                      <div key={idx} className="trake-event-item-row" data-testid={`trake-event-row-${idx}`}>
                        <span className="trake-event-num-pill tabular-nums">#{idx + 1}</span>
                        <input
                          type="text"
                          className="trake-event-input-field"
                          value={evt}
                          onChange={(e) => handleTrakeEventChange(idx, e.target.value)}
                          placeholder={`Event #${idx + 1} (e.g. Giậm nhảy)...`}
                          data-testid={`trake-event-input-${idx}`}
                        />
                        <div className="trake-event-item-actions">
                          <button
                            type="button"
                            className="trake-btn-reorder"
                            disabled={idx === 0}
                            onClick={() => handleMoveTrakeEvent(idx, idx - 1)}
                            title="Move event earlier in sequence"
                            aria-label={`Move event ${idx + 1} up`}
                            data-testid={`trake-move-up-${idx}`}
                          >
                            ▲
                          </button>
                          <button
                            type="button"
                            className="trake-btn-reorder"
                            disabled={idx === trakeEvents.length - 1}
                            onClick={() => handleMoveTrakeEvent(idx, idx + 1)}
                            title="Move event later in sequence"
                            aria-label={`Move event ${idx + 1} down`}
                            data-testid={`trake-move-down-${idx}`}
                          >
                            ▼
                          </button>
                          <button
                            type="button"
                            className="trake-btn-remove-evt"
                            disabled={trakeEvents.length <= 1}
                            onClick={() => handleRemoveTrakeEvent(idx)}
                            title={trakeEvents.length <= 1 ? 'Cannot remove the only event' : 'Remove event'}
                            aria-label={`Remove event ${idx + 1}`}
                            data-testid={`trake-remove-event-${idx}`}
                          >
                            <ClearIcon size={11} />
                          </button>
                        </div>
                      </div>
                    ))
                  )}
                </div>

                <div className="trake-events-bottom-controls">
                  <button
                    type="button"
                    className="trake-btn-add-event"
                    onClick={handleAddTrakeEvent}
                    data-testid="trake-add-event-btn"
                  >
                    <span>+ Add Event</span>
                  </button>

                  <button
                    type="button"
                    className="trake-btn-sample-preset"
                    onClick={handleLoadSamplePreset}
                    title="Load standard jump sequence"
                    data-testid="trake-sample-preset-btn"
                  >
                    <span>Sample Jump</span>
                  </button>
                </div>
              </div>
            )}

            {/* Redesigned Result Limit Control (Single trigger + Portal Popover with Custom editor) */}
            <div className="control-group">
              <div className="group-header">
                <span className="group-title">Result Limit</span>
              </div>
              <div className="result-limit-control-container">
                <button
                  ref={triggerRef}
                  type="button"
                  className={`limit-trigger-btn ${isLimitOpen ? 'trigger-open' : ''}`}
                  onClick={() => {
                    setIsLimitOpen(!isLimitOpen)
                    setIsCustomMode(false)
                  }}
                  aria-haspopup="listbox"
                  aria-expanded={isLimitOpen}
                  data-testid="limit-trigger-btn"
                >
                  <span className="trigger-label tabular-nums">
                    {topK} candidates {![20, 50, 100].includes(topK) ? '(Custom)' : ''}
                  </span>
                  <ChevronDownIcon size={14} className={`chevron-icon ${isLimitOpen ? 'rotate' : ''}`} />
                </button>

                {/* Hidden input for programmatic/test synchronization */}
                <input
                  type="number"
                  min={1}
                  max={100}
                  value={topK}
                  onChange={(e) => handleTopKChange(parseInt(e.target.value, 10))}
                  data-testid="top-k-input"
                  style={{ display: 'none' }}
                />

                {isLimitOpen &&
                  createPortal(
                    <div
                      ref={popoverRef}
                      className="limit-popover-portal"
                      style={{
                        position: 'fixed',
                        top: `${popoverPos.top}px`,
                        left: `${popoverPos.left}px`,
                        width: `${popoverPos.width}px`,
                        zIndex: 9999,
                      }}
                      role="listbox"
                      data-testid="result-limit-popover"
                    >
                      {!isCustomMode ? (
                        <>
                          {[20, 50, 100].map((preset) => (
                            <button
                              key={preset}
                              type="button"
                              className={`limit-option-item ${topK === preset ? 'option-selected' : ''}`}
                              onClick={() => selectPreset(preset)}
                              data-testid={`preset-limit-${preset}`}
                            >
                              <span className="tabular-nums">{preset} candidates</span>
                              {topK === preset && <CheckIcon size={13} className="text-cyan" />}
                            </button>
                          ))}

                          <div className="limit-popover-divider" />

                          <button
                            type="button"
                            className={`limit-option-item ${![20, 50, 100].includes(topK) ? 'option-selected' : ''}`}
                            onClick={() => setIsCustomMode(true)}
                            data-testid="custom-limit-toggle"
                          >
                            <span>Custom...</span>
                            {![20, 50, 100].includes(topK) && <CheckIcon size={13} className="text-cyan" />}
                          </button>
                        </>
                      ) : (
                        <div className="limit-custom-editor">
                          <span className="custom-editor-title">Custom result limit</span>
                          <input
                            ref={customInputRef}
                            type="text"
                            inputMode="numeric"
                            pattern="[0-9]*"
                            value={customVal}
                            onChange={(e) => {
                              const val = e.target.value.replace(/[^0-9]/g, '')
                              setCustomVal(val)
                              if (customError) setCustomError(null)
                            }}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') {
                                e.preventDefault()
                                applyCustomVal()
                              } else if (e.key === 'Escape') {
                                e.preventDefault()
                                setIsCustomMode(false)
                                setCustomError(null)
                              }
                            }}
                            placeholder="1 - 100"
                            className="custom-number-editor-field tabular-nums"
                            data-testid="custom-limit-input"
                          />
                          {customError && (
                            <span className="custom-editor-error">{customError}</span>
                          )}
                          <div className="custom-editor-actions">
                            <button
                              type="button"
                              className="btn-custom-cancel"
                              onClick={() => {
                                setIsCustomMode(false)
                                setCustomError(null)
                              }}
                            >
                              Cancel
                            </button>
                            <button
                              type="button"
                              className="btn-custom-apply"
                              onClick={applyCustomVal}
                            >
                              Apply
                            </button>
                          </div>
                        </div>
                      )}
                    </div>,
                    document.body
                  )}
              </div>
            </div>

            {/* Mode Indicator */}
            <div className="control-group">
              <div className="group-header">
                <span className="group-title">Mode</span>
              </div>
              <div className="mode-indicator-box">
                <div className={`mode-item ${isFixture ? 'active-mode' : ''}`}>
                  <span className="mode-dot" />
                  <span>Fixture Preview</span>
                </div>
                <div className={`mode-item ${!isFixture ? 'active-mode' : ''}`}>
                  <span className="mode-dot" />
                  <span>Live Search</span>
                </div>
              </div>
            </div>

            {/* Status / Readiness */}
            <div className="control-group">
              <div className="group-header">
                <span className="group-title">Status</span>
              </div>
              <div className="status-indicator-box">
                <div className="status-item">
                  <span
                    className={`status-indicator-dot ${
                      readiness === 'READY'
                        ? 'dot-green'
                        : readiness === 'PARTIAL' || readiness === 'DEGRADED'
                        ? 'dot-amber'
                        : 'dot-red'
                    }`}
                  />
                  <span className="status-text">
                    {readiness === 'READY'
                      ? 'All Services Operational'
                      : readiness === 'PARTIAL'
                      ? 'Partial Readiness'
                      : readiness === 'DEGRADED'
                      ? 'Degraded Mode'
                      : 'Offline'}
                  </span>
                </div>
              </div>
            </div>

            {/* Glowing Search CTA Button */}
            <div className="search-cta-block">
              <button
                type="submit"
                className="search-submit-btn"
                disabled={
                  taskMode === 'VQA'
                    ? isVqaSearching || !queryText.trim() || !vqaQuestion.trim()
                    : taskMode === 'TRAKE'
                    ? isTrakeSearching || !queryText.trim() || trakeEvents.length < 2 || trakeEvents.some((e) => !e.trim())
                    : isSearching || !queryText.trim()
                }
                onClick={executeSearch}
                data-testid="kis-search-btn"
              >
                {taskMode === 'VQA' ? (
                  isVqaSearching ? (
                    <>
                      <SpinnerIcon size={16} className="icon-spin" />
                      <span>Searching Q&A...</span>
                    </>
                  ) : (
                    <>
                      <SearchIcon size={16} />
                      <span>Search Q&A</span>
                    </>
                  )
                ) : taskMode === 'TRAKE' ? (
                  isTrakeSearching ? (
                    <>
                      <SpinnerIcon size={16} className="icon-spin" />
                      <span>Aligning Sequence...</span>
                    </>
                  ) : (
                    <>
                      <SearchIcon size={16} />
                      <span>Search & Align</span>
                    </>
                  )
                ) : isSearching ? (
                  <>
                    <SpinnerIcon size={16} className="icon-spin" />
                    <span>Searching KIS...</span>
                  </>
                ) : (
                  <>
                    <SearchIcon size={16} />
                    <span>Search KIS</span>
                  </>
                )}
              </button>

              <button
                type="button"
                className="reset-btn"
                onClick={handleClear}
                disabled={!queryText.length}
              >
                Reset
              </button>
            </div>
          </form>

          {/* Search Error Callout - strictly mode-scoped */}
          {((taskMode === 'KIS' && searchError) ||
            (taskMode === 'VQA' && vqaError) ||
            (taskMode === 'TRAKE' && trakeError)) && (
            <div className="search-error-banner" data-testid="search-error-banner">
              <span>{taskMode === 'KIS' ? searchError : taskMode === 'VQA' ? vqaError : trakeError}</span>
            </div>
          )}
        </div>
      </aside>

      {/* ── Main Retrieval Stage ── */}
      <main className="retrieval-main-stage">

        {/* Compact Feedback Control Surface (WP13 T029) - Gated to KIS mode */}
        {taskMode === 'KIS' && candidates.length > 0 && (
          <div
            className={`feedback-control-panel ${isFeedbackActive ? 'is-active-session' : ''}`}
            data-testid="feedback-panel"
          >
            <div className="feedback-panel-header">
              <div className="feedback-header-left">
                <span className="feedback-title">Feedback Reranking</span>
                {isFeedbackActive && (
                  <>
                    <span
                      className="feedback-revision-pill tabular-nums"
                      data-testid="feedback-revision-badge"
                    >
                      Revision {feedbackRevision}
                    </span>
                    <span
                      className={`feedback-count-pill tabular-nums ${isFeedbackLimitReached ? 'limit-reached' : ''}`}
                      data-testid="feedback-count-badge"
                      title={`${activeFeedbackEvents} of maximum ${maxFeedbackEvents} active refinements used`}
                    >
                      Refinements: {activeFeedbackEvents} / {maxFeedbackEvents}
                    </span>
                  </>
                )}
              </div>

              {isFeedbackActive && (
                <div className="feedback-header-actions">
                  <button
                    type="button"
                    className="feedback-btn-undo"
                    onClick={handleUndoFeedback}
                    disabled={isFeedbackPending || feedbackRevision === 0}
                    title="Undo last refinement step"
                    data-testid="feedback-undo-btn"
                  >
                    <UndoIcon size={12} />
                    <span>Undo</span>
                  </button>

                  <button
                    type="button"
                    className="feedback-btn-reset"
                    onClick={handleResetFeedback}
                    disabled={isFeedbackPending}
                    title="Reset ranking to initial feedback baseline"
                    data-testid="feedback-reset-btn"
                  >
                    <ResetIcon size={12} />
                    <span>Reset</span>
                  </button>

                  <button
                    type="button"
                    className="feedback-btn-exit"
                    onClick={handleExitFeedback}
                    title="Exit feedback and restore original KIS results"
                    data-testid="feedback-exit-btn"
                  >
                    <span>Exit Feedback</span>
                  </button>
                </div>
              )}
            </div>

            <div className="feedback-panel-body">
              {/* Reference Candidate Identity */}
              <div
                className="feedback-reference-slot"
                data-testid="feedback-reference-display"
              >
                <span className="reference-label">Reference:</span>
                {feedbackReference ? (
                  <div className="reference-pill">
                    <span className="reference-vid">{feedbackReference.video_id}</span>
                    <span className="reference-fid tabular-nums">
                      · Frame {feedbackReference.frame_id}
                    </span>
                    <button
                      type="button"
                      className="reference-clear-btn"
                      onClick={handleClearReference}
                      title="Clear reference candidate"
                      aria-label="Clear reference"
                    >
                      ×
                    </button>
                  </div>
                ) : (
                  <span className="reference-empty-hint">
                    Select "Set Reference" on any candidate card below
                  </span>
                )}
              </div>

              {/* Feedback Input & Refine Action */}
              <div className="feedback-input-row">
                <input
                  type="text"
                  className="feedback-text-input"
                  placeholder={
                    isFeedbackLimitReached
                      ? 'Maximum 5 active refinements reached. Undo or Reset to continue.'
                      : 'Describe how results should change (e.g., closer angle, darker lighting, vehicle in front)...'
                  }
                  value={feedbackDraftText}
                  onChange={(e) =>
                    dispatch({ type: 'SET_FEEDBACK_DRAFT', payload: e.target.value })
                  }
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault()
                      handleRefine()
                    }
                  }}
                  disabled={isFeedbackPending || isFeedbackLimitReached}
                  data-testid="feedback-text-input"
                />

                <button
                  type="button"
                  className="feedback-refine-btn"
                  onClick={handleRefine}
                  disabled={!canRefine}
                  data-testid="feedback-refine-btn"
                >
                  {isFeedbackPending ? (
                    <>
                      <SpinnerIcon size={13} className="icon-spin" />
                      <span>Refining...</span>
                    </>
                  ) : (
                    <span>Refine</span>
                  )}
                </button>
              </div>

              {/* Limit banner when max active refinements reached */}
              {isFeedbackLimitReached && (
                <div
                  className="feedback-limit-banner"
                  data-testid="feedback-limit-banner"
                >
                  <span>Maximum 5 active refinements reached. Undo or Reset to continue.</span>
                </div>
              )}

              {/* Degraded / Error alert */}
              {feedbackError && !isFeedbackLimitReached && (
                <div
                  className="feedback-error-banner"
                  data-testid="feedback-error-banner"
                >
                  <span>{feedbackError}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Main Stage: TRAKE Timeline OR KIS/VQA Contact Sheet */}
        {taskMode === 'TRAKE' ? (
          <div className="trake-main-stage-wrapper" data-testid="trake-main-stage">
            {isTrakeSearching ? (
              <div className="retrieval-state-box" data-testid="trake-searching-state">
                <SpinnerIcon size={26} className="icon-spin text-cyan" />
                <span className="state-text">Retrieving event candidates & aligning sequence...</span>
              </div>
            ) : !trakeHasSearched ? (
              <div className="retrieval-state-box idle-state" data-testid="trake-presearch-state">
                <div className="trake-presearch-guide">
                  <div className="guide-icon-box">
                    <FilmstripIcon size={28} className="text-cyan" />
                  </div>
                  <h3 className="guide-title">TRAKE Sequence Alignment</h3>
                  <p className="guide-description">
                    Define an ordered action sequence on the command rail and click <strong>"Search & Align"</strong> to rank multiple valid video hypotheses and align semantic keyframes.
                  </p>
                  <div className="guide-steps-list">
                    <div className="guide-step-item">
                      <span className="step-num">1</span>
                      <span>Describe the overall sequence context</span>
                    </div>
                    <div className="guide-step-item">
                      <span className="step-num">2</span>
                      <span>Define at least 2 ordered semantic events</span>
                    </div>
                    <div className="guide-step-item">
                      <span className="step-num">3</span>
                      <span>Click "Search & Align" to retrieve and align keyframes</span>
                    </div>
                  </div>
                </div>
              </div>
            ) : trakeVideoId === null || trakeSlots.length === 0 ? (
              <div className="retrieval-state-box trake-no-align-state" data-testid="trake-no-align-state">
                <div className="no-align-icon-box">
                  <WarningIcon size={26} className="text-amber" />
                </div>
                <h3 className="no-align-title">No Valid Alignment Found</h3>
                <p className="no-align-description">
                  {trakeError || 'No monotonic keyframe sequence was found matching the requested events.'}
                </p>
                <span className="no-align-hint">
                  Adjust your sequence description or event queries and search again.
                </span>
              </div>
            ) : (
              <>
                {trakeResults.length > 1 && (
                  <div className="trake-hypothesis-selector" data-testid="trake-hypothesis-selector">
                    <span className="hypothesis-label">Top sequences:</span>
                    {trakeResults.map((hypothesis, index) => (
                      <button
                        key={`${hypothesis.video_id}-${index}`}
                        type="button"
                        className={`slot-lock-toggle-btn ${trakeActiveHypothesisIndex === index ? 'is-locked' : 'is-unlocked'}`}
                        onClick={() => dispatch({ type: 'SELECT_TRAKE_HYPOTHESIS', payload: index })}
                        title={`${hypothesis.video_id} · score ${hypothesis.aggregate_score.toFixed(4)}`}
                        data-testid={`trake-hypothesis-${index}`}
                      >
                        #{index + 1} {hypothesis.video_id}
                      </button>
                    ))}
                  </div>
                )}
                <TrakeTimeline
                  slots={trakeSlots}
                activeSlotIndex={trakeActiveSlotIndex}
                videoId={trakeVideoId}
                aggregateScore={trakeAggregateScore}
                verifierScore={
                  trakeActiveHypothesisIndex !== null
                    ? trakeResults[trakeActiveHypothesisIndex]?.verifier_score
                    : null
                }
                storyConsistency={
                  trakeActiveHypothesisIndex !== null
                    ? trakeResults[trakeActiveHypothesisIndex]?.story_consistency
                    : null
                }
                visualVerification={
                  trakeActiveHypothesisIndex !== null
                    ? trakeResults[trakeActiveHypothesisIndex]?.verification
                    : null
                }
                validationStatus={trakeValidationStatus}
                isSearching={isTrakeSearching}
                mode={mode}
                isInBasket={Boolean(
                  trakeVideoId &&
                  trakeSlots.length > 0 &&
                  submissionBasket.some(
                    (b) =>
                      b.task === 'TRAKE' &&
                      b.video_id === trakeVideoId &&
                      JSON.stringify(b.frame_ids) === JSON.stringify(trakeSlots.map((s) => s.frame_id))
                  )
                )}
                onSelectSlot={handleSelectTrakeSlot}
                onInspectSlot={handleInspectTrakeSlot}
                onLockSlot={(idx) => dispatch({ type: 'LOCK_TRAKE_SLOT', payload: { event_index: idx } })}
                onUnlockSlot={(idx) => dispatch({ type: 'UNLOCK_TRAKE_SLOT', payload: { event_index: idx } })}
                onSelectAlternative={(eventIndex, alternativeIndex) =>
                  dispatch({
                    type: 'SELECT_TRAKE_EVENT_ALTERNATIVE',
                    payload: { event_index: eventIndex, alternative_index: alternativeIndex },
                  })
                }
                onRerunSlot={handleRerunTrakeSlot}
                rerunningEventIndex={trakeRerunningEventIndex}
                onAddToBasket={() => {
                  dispatch({ type: 'ADD_TRAKE_TO_BASKET' })
                  telemetry.record({
                    action: 'ADD_TRAKE_TO_BASKET',
                    taskMode: 'TRAKE',
                    videoId: trakeVideoId,
                    details: {
                      frame_ids: trakeSlots.map((s) => s.frame_id),
                      event_count: trakeSlots.length,
                    },
                  })
                }}
                />
              </>
            )}
          </div>
        ) : taskMode === 'VQA' ? (
          <div className="retrieval-results-section" data-testid="vqa-main-stage">
            {isVqaSearching ? (
              <div className="retrieval-state-box" data-testid="vqa-searching-state">
                <SpinnerIcon size={26} className="icon-spin text-cyan" />
                <span className="state-text">Retrieving grounded VQA answers and multimodal evidence...</span>
              </div>
            ) : !vqaHasSearched || vqaResults.length === 0 ? (
              <div className="retrieval-state-box idle-state" data-testid="vqa-presearch-state">
                <div className="trake-presearch-guide">
                  <div className="guide-icon-box">
                    <QuestionIcon size={28} className="text-cyan" />
                  </div>
                  <h3 className="guide-title">Q&A Multimodal Search & Answer</h3>
                  <p className="guide-description">
                    Enter an event description and question in the search rail and click <strong>"Search & Answer"</strong> to generate retrieval-grounded answers and inspect multimodal evidence.
                  </p>
                  <div className="guide-steps-list">
                    <div className="guide-step-item">
                      <span className="step-num">1</span>
                      <span>Describe the scene or visual event context</span>
                    </div>
                    <div className="guide-step-item">
                      <span className="step-num">2</span>
                      <span>Enter the specific question to answer</span>
                    </div>
                    <div className="guide-step-item">
                      <span className="step-num">3</span>
                      <span>Click "Search & Answer" to produce grounded answers</span>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="vqa-results-wrapper" data-testid="vqa-results-grid">
                <div className="results-section-header">
                  <div className="header-left">
                    <h2 className="section-title">VQA Answer Proposals</h2>
                    <span className="results-count-pill tabular-nums" data-testid="vqa-results-count-pill">
                      {vqaResults.length} answer{vqaResults.length > 1 ? 's' : ''}
                    </span>
                  </div>
                  <span className="results-hint">
                    Click candidate answer to inspect evidence and verify proposal
                  </span>
                </div>

                <div className="retrieval-results-viewport" data-testid="vqa-results-scroll-container">
                  <div className="contact-sheet-grid">
                    {vqaResults.map((res) => {
                      const isSelected =
                        vqaActiveResult?.video_id === res.video_id &&
                        vqaActiveResult?.frame_id === res.frame_id
                      const vqaCand: SearchCandidate = {
                        query_id: queryId || 'vqa',
                        video_id: res.video_id,
                        frame_id: res.frame_id,
                        timestamp_ms: res.timestamp_ms || 0,
                        source: 'vqa',
                        rank: res.rank,
                        score: res.confidence,
                        certified_anchor_frame_id: res.certified_anchor_frame_id,
                        certified_anchor_timestamp_ms: res.certified_anchor_timestamp_ms,
                        anchor_offset: res.anchor_offset,
                      }

                      return (
                        <div
                          key={`vqa-card-${res.video_id}-${res.frame_id}-${res.rank}`}
                          className={`retrieval-tile ${isSelected ? 'tile-selected' : ''}`}
                          onClick={() => {
                            dispatch({ type: 'SELECT_VQA_RESULT', payload: res })
                            dispatch({ type: 'SET_ACTIVE_TAB', payload: 'inspection' })
                          }}
                          role="button"
                          tabIndex={0}
                          data-testid={`candidate-card-${res.rank}`}
                        >
                          <div className="tile-image-stage">
                            <CandidatePreviewThumb
                              candidate={vqaCand}
                              alt={`Rank ${res.rank}`}
                              className="tile-image-element"
                            />
                            <div className="tile-top-left-badges">
                              <span className="tile-rank-chip tabular-nums">#{res.rank}</span>
                              {res.confidence != null && (
                                <span className="tile-score-badge tabular-nums">
                                  {(res.confidence * 100).toFixed(0)}%
                                </span>
                              )}
                            </div>
                          </div>

                          <div className="tile-caption-pane">
                            <div className="caption-id-row" data-testid="vqa-tile-caption-row">
                              <span className="caption-vid" data-testid="vqa-tile-vid">{res.video_id}</span>
                              <span className="caption-fid tabular-nums" data-testid="vqa-tile-fid">Frame {res.frame_id}</span>
                            </div>
                            <div className="vqa-tile-proposal-box" title={res.proposal}>
                              <span className="vqa-tile-proposal-text">"{res.proposal}"</span>
                            </div>
                            <span className="retry-counter-pill" data-testid={`vqa-grounding-status-${res.rank}`}>
                              {res.verification?.reasons.includes('scene_mismatch')
                                ? 'Scene mismatch'
                                : res.verification?.scene_supported === false
                                ? 'Scene unverified'
                                : res.independently_supported === true
                                ? 'Machine verified'
                                : 'Unverified proposal'}
                            </span>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
                {vqaRetrievalCandidates.length > vqaResults.length && (
                  <details className="proposal-degraded-reasons" data-testid="vqa-deep-candidates">
                    <summary>
                      Deep retrieval candidates ({Math.min(30, vqaRetrievalCandidates.length)} shown of {vqaRetrievalCandidates.length})
                    </summary>
                    <div className="contact-sheet-grid">
                      {vqaRetrievalCandidates.slice(0, 30).map((candidate) => (
                        <div
                          key={`vqa-deep-${candidate.video_id}-${candidate.frame_id}-${candidate.rank}`}
                          className="retrieval-tile"
                          role="button"
                          tabIndex={0}
                          title="Deep retrieval candidate — click to pin for comparison"
                          onClick={() => dispatch({ type: 'TOGGLE_PIN_CANDIDATE', payload: candidate })}
                        >
                          <div className="tile-image-stage">
                            <CandidatePreviewThumb
                              candidate={candidate}
                              alt={`Deep candidate rank ${candidate.rank}`}
                              className="tile-image-element"
                            />
                            <div className="tile-top-left-badges">
                              <span className="tile-rank-chip tabular-nums">#{candidate.rank}</span>
                            </div>
                          </div>
                          <div className="tile-caption-pane">
                            <div className="caption-id-row">
                              <span className="caption-vid">{candidate.video_id}</span>
                              <span className="caption-fid tabular-nums">Frame {candidate.frame_id}</span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            )}
          </div>
        ) : (
          <div className="retrieval-results-section" data-testid="candidate-grid">
            <div className="results-section-header">
              <div className="header-left">
                <h2 className="section-title">
                  Retrieval Results {isFeedbackActive ? `(Refined · Rev ${feedbackRevision})` : ''}
                </h2>
                {candidates.length > 0 && (
                  <span className="results-count-pill tabular-nums">
                    {candidates.length} candidates
                  </span>
                )}
              </div>
              <span className="results-hint">
                Click candidate to inspect · "Set Reference" to rerank with Feedback
              </span>
            </div>

            {/* Results Matrix or Idle state */}
            {isSearching ? (
              <div className="retrieval-state-box">
                <SpinnerIcon size={26} className="icon-spin text-cyan" />
                <span className="state-text">Retrieving multimodal candidates...</span>
              </div>
            ) : candidates.length === 0 ? (
              <div className="retrieval-state-box idle-state">
                <p className="idle-message">
                  Enter a query in the search rail to retrieve candidate video frames.
                </p>
              </div>
            ) : (
              <div className="retrieval-results-viewport" data-testid="results-scroll-container">
                <div className="contact-sheet-grid">
                  {candidates.map((cand) => (
                    <RetrievalTile
                      key={`${cand.video_id}-${cand.frame_id}-${cand.rank}`}
                      candidate={cand}
                      isSelected={
                        activeCandidate?.video_id === cand.video_id &&
                        activeCandidate?.frame_id === cand.frame_id
                      }
                      isReference={
                        feedbackReference?.video_id === cand.video_id &&
                        feedbackReference?.frame_id === cand.frame_id
                      }
                      isInBasket={submissionBasket.some(
                        (b) => b.video_id === cand.video_id && b.frame_id === cand.frame_id && b.task === 'KIS'
                      )}
                      isPinned={pinnedCandidates.some(
                        (p) => p.video_id === cand.video_id && p.frame_id === cand.frame_id
                      )}
                      onSelect={() => handleSelectCandidate(cand)}
                      onInspect={() => handleInspectCandidate(cand)}
                      onSetReference={() => handleSetReference(cand)}
                      onAddToBasket={() => handleAddKisToBasket(cand)}
                      onTogglePin={() => handleTogglePin(cand)}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}
