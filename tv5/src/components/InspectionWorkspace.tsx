import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import { useAppDispatch, useAppState } from '../state/AppContext'
import { getVideoStreamUrl, getThumbnailUrl, getSelectedFrameUrl, fetchVideoMeta, VideoMeta, fetchVideoKeyframes, VideoKeyframe } from '../api/tv4Client'
import { getFixturePreviewDataUri } from '../fixtures/fixtureData'
import { ExactNeighborThumb } from './ExactNeighborThumb'
import { CandidatePreviewThumb } from './CandidatePreviewThumb'
import { SearchCandidate } from '../types/contracts'
import {
  VideoIcon,
  FrameIcon,
  StepPrevIcon,
  StepNextIcon,
  AnchorIcon,
  CheckIcon,
  WarningIcon,
  SpinnerIcon,
  ArrowLeftIcon,
  QuestionIcon,
  FilmstripIcon,
  BasketPlusIcon,
  SettingsIcon,
  PinIcon,
  FullscreenIcon,
  MinimizeIcon,
} from './Icons'
import { EvidenceInspector } from './EvidenceInspector'
import { VqaAnswerPanel } from './VqaAnswerPanel'
import { telemetry } from '../utils/telemetry'

export const InspectionWorkspace: React.FC = () => {
  const {
    activeCandidate,
    anchorCandidate,
    candidates,
    kisActiveCandidate,
    cumulativeOffset,
    isStepping,
    stepError,
    currentStep,
    exactNeighbors,
    exactImageBlobUrl,
    isImageLoading,
    imageError,
    mode,
    vqaResults,
    vqaActiveResult,
    queryText,
    vqaQuestion,
    taskMode,
    queryId,
    trakeSlots,
    trakeActiveSlotIndex,
    trakeVideoId,
    submissionBasket,
    pinnedCandidates,
  } = useAppState()

  const dispatch = useAppDispatch()

  const activeTrakeSlot =
    trakeActiveSlotIndex !== null && trakeActiveSlotIndex < trakeSlots.length
      ? trakeSlots[trakeActiveSlotIndex]
      : null

  const targetCandidate: SearchCandidate | null =
    taskMode === 'TRAKE'
      ? activeTrakeSlot && activeTrakeSlot.video_id && activeTrakeSlot.frame_id !== null
        ? {
            query_id: queryId || 'trake',
            video_id: activeTrakeSlot.video_id,
            frame_id: activeTrakeSlot.frame_id,
            timestamp_ms: activeTrakeSlot.timestamp_ms || 0,
            source: 'trake',
            rank: activeTrakeSlot.event_index + 1,
            score: activeTrakeSlot.score,
            certified_anchor_frame_id: activeTrakeSlot.certified_anchor_frame_id ?? activeTrakeSlot.frame_id,
            certified_anchor_timestamp_ms: activeTrakeSlot.certified_anchor_timestamp_ms ?? activeTrakeSlot.timestamp_ms ?? 0,
            anchor_offset: activeTrakeSlot.anchor_offset ?? 0,
            cumulative_offset: 0,
          }
        : null
      : taskMode === 'VQA'
      ? vqaActiveResult
        ? {
            query_id: queryId || 'vqa',
            video_id: vqaActiveResult.video_id,
            frame_id: vqaActiveResult.frame_id,
            timestamp_ms: vqaActiveResult.timestamp_ms || 0,
            source: 'vqa',
            rank: vqaActiveResult.rank,
            score: vqaActiveResult.confidence,
            certified_anchor_frame_id: vqaActiveResult.certified_anchor_frame_id,
            certified_anchor_timestamp_ms: vqaActiveResult.certified_anchor_timestamp_ms,
            anchor_offset: vqaActiveResult.anchor_offset,
          }
        : null
      : kisActiveCandidate || (candidates.length > 0 ? candidates[0] : null)

  const videoRef = useRef<HTMLVideoElement>(null)
  const [playbackTime, setPlaybackTime] = useState<number>(0)
  const [frameImgFailed, setFrameImgFailed] = useState(false)
  const [showProofDetails, setShowProofDetails] = useState(false)
  const [videoMeta, setVideoMeta] = useState<VideoMeta | null>(null)
  const [playbackRate, setPlaybackRate] = useState<number>(() => {
    const saved = localStorage.getItem('tv5_playback_rate')
    return saved ? Math.max(0.1, parseFloat(saved)) : 1.0
  })
  const [seekStepSeconds, setSeekStepSeconds] = useState<number>(() => {
    const saved = localStorage.getItem('tv5_seek_step')
    return saved ? Math.max(1, parseInt(saved, 10)) : 5
  })
  const [autoPlayOnSelect, setAutoPlayOnSelect] = useState<boolean>(() => {
    const saved = localStorage.getItem('tv5_autoplay')
    return saved !== null ? saved === 'true' : true
  })
  const [isPlaying, setIsPlaying] = useState<boolean>(false)
  const [showSettingsPopover, setShowSettingsPopover] = useState<boolean>(false)
  const [isCustomSpeedOpen, setIsCustomSpeedOpen] = useState<boolean>(false)
  const [customSpeedInput, setCustomSpeedInput] = useState<string>('')
  const [isCustomSeekOpen, setIsCustomSeekOpen] = useState<boolean>(false)
  const [customSeekInput, setCustomSeekInput] = useState<string>('')
  const [videoKeyframes, setVideoKeyframes] = useState<VideoKeyframe[]>([])
  const [isLoadingKeyframes, setIsLoadingKeyframes] = useState<boolean>(false)
  const [showHoverScrubber, setShowHoverScrubber] = useState<boolean>(() => {
    const saved = localStorage.getItem('tv5_show_hover_scrubber')
    return saved !== null ? saved === 'true' : true
  })
  const scrubberTrackRef = useRef<HTMLDivElement>(null)
  const [isDraggingScrubber, setIsDraggingScrubber] = useState<boolean>(false)
  const isDraggingRef = useRef<boolean>(false)
  const [hoverPos, setHoverPos] = useState<{
    x: number
    percent: number
    timeSec: number
    frameId: number
    nearestKeyframe?: VideoKeyframe
  } | null>(null)
  const [activeBottomTab, setActiveBottomTab] = useState<'scenes' | 'neighbors'>('scenes')
  const [isBottomCollapsed, setIsBottomCollapsed] = useState<boolean>(false)
  const [selectedTimelineKeyframe, setSelectedTimelineKeyframe] = useState<VideoKeyframe | null>(null)
  const scenesStripRef = useRef<HTMLDivElement>(null)
  const togglePlaybackBasketRef = useRef<() => void>()
  const settingsPopoverRef = useRef<HTMLDivElement>(null)
  const settingsBtnRef = useRef<HTMLButtonElement>(null)
  const videoViewportRef = useRef<HTMLDivElement>(null)
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false)
  const isFixture = mode === 'fixture'

  // Sync fullscreen state
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(document.fullscreenElement === videoViewportRef.current)
    }
    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange)
    }
  }, [])

  const toggleFullscreen = useCallback(() => {
    if (!videoViewportRef.current) return
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => {})
    } else {
      videoViewportRef.current.requestFullscreen().catch(() => {})
    }
  }, [])

  // Auto-close settings popover when clicking outside
  useEffect(() => {
    if (!showSettingsPopover) return

    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as Node
      if (
        settingsPopoverRef.current &&
        !settingsPopoverRef.current.contains(target) &&
        settingsBtnRef.current &&
        !settingsBtnRef.current.contains(target)
      ) {
        setShowSettingsPopover(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [showSettingsPopover])

  const getMetaDuration = useCallback(() => {
    const meta = videoMeta as any
    return meta?.duration || meta?.duration_s || (meta?.duration_ms ? meta.duration_ms / 1000 : 0)
  }, [videoMeta])

  const videoDuration = videoRef.current?.duration || getMetaDuration()

  const updateScrubberPosition = useCallback(
    (clientX: number, shouldSeekVideo: boolean) => {
      if (!scrubberTrackRef.current) return
      const rect = scrubberTrackRef.current.getBoundingClientRect()
      const offsetX = Math.max(0, Math.min(clientX - rect.left, rect.width))
      const percent = rect.width > 0 ? offsetX / rect.width : 0
      const duration = videoRef.current?.duration || getMetaDuration()
      const timeSec = percent * duration
      const fps = videoMeta?.fps || 25
      const frameId = Math.round(timeSec * fps)

      let nearestKeyframe: VideoKeyframe | undefined = undefined
      if (videoKeyframes && videoKeyframes.length > 0) {
        let minDiff = Infinity
        for (const kf of videoKeyframes) {
          const kfTimeSec = kf.timestamp_ms / 1000
          const diff = Math.abs(kfTimeSec - timeSec)
          if (diff < minDiff) {
            minDiff = diff
            nearestKeyframe = kf
          }
        }
      }

      setHoverPos({
        x: offsetX,
        percent,
        timeSec,
        frameId,
        nearestKeyframe,
      })

      if (shouldSeekVideo && videoRef.current) {
        videoRef.current.currentTime = timeSec
        setPlaybackTime(timeSec)
      }
    },
    [videoMeta, videoKeyframes, getMetaDuration]
  )

  const handleScrubberMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    e.preventDefault()
    isDraggingRef.current = true
    setIsDraggingScrubber(true)
    updateScrubberPosition(e.clientX, true)
  }

  const handleScrubberMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (isDraggingRef.current) return
    updateScrubberPosition(e.clientX, false)
  }

  const handleScrubberMouseLeave = () => {
    if (!isDraggingRef.current) {
      setHoverPos(null)
    }
  }

  const handleScrubberClick = (e: React.MouseEvent<HTMLDivElement>) => {
    updateScrubberPosition(e.clientX, true)
  }

  // Global mousemove and mouseup listeners for seamless dragging across screen
  useEffect(() => {
    const handleGlobalMouseMove = (e: MouseEvent) => {
      if (!isDraggingRef.current) return
      e.preventDefault()
      updateScrubberPosition(e.clientX, true)
    }

    const handleGlobalMouseUp = (e: MouseEvent) => {
      if (isDraggingRef.current) {
        isDraggingRef.current = false
        setIsDraggingScrubber(false)
        updateScrubberPosition(e.clientX, true)
      }
    }

    window.addEventListener('mousemove', handleGlobalMouseMove)
    window.addEventListener('mouseup', handleGlobalMouseUp)
    return () => {
      window.removeEventListener('mousemove', handleGlobalMouseMove)
      window.removeEventListener('mouseup', handleGlobalMouseUp)
    }
  }, [updateScrubberPosition])

  // Reset image error state when active candidate or offset changes
  useEffect(() => {
    setFrameImgFailed(false)
  }, [activeCandidate, cumulativeOffset])

  // Exact stepping logic: dispatches step intent to single authoritative coordinator
  const handleStep = useCallback(
    (targetOffset: number) => {
      dispatch({ type: 'EXACT_STEP_START', payload: { offset: targetOffset } })
    },
    [dispatch]
  )

  const togglePlay = useCallback(() => {
    if (!videoRef.current) return
    if (videoRef.current.paused) {
      const playResult = videoRef.current.play()
      if (playResult && typeof playResult.catch === 'function') {
        playResult.catch(() => {})
      }
    } else {
      videoRef.current.pause()
    }
  }, [])

  const seekRelative = useCallback((deltaSec: number) => {
    if (!videoRef.current) return
    const newTime = Math.max(0, Math.min(videoRef.current.duration || 9999, videoRef.current.currentTime + deltaSec))
    videoRef.current.currentTime = newTime
    setPlaybackTime(newTime)
  }, [])

  const updateSpeed = useCallback((rate: number) => {
    if (isNaN(rate) || rate <= 0) return
    const rounded = Math.round(rate * 100) / 100
    setPlaybackRate(rounded)
    if (videoRef.current) {
      videoRef.current.playbackRate = rounded
    }
    localStorage.setItem('tv5_playback_rate', String(rounded))
  }, [])

  const updateSeekStep = useCallback((sec: number) => {
    if (isNaN(sec) || sec <= 0) return
    const intSec = Math.round(sec)
    setSeekStepSeconds(intSec)
    localStorage.setItem('tv5_seek_step', String(intSec))
  }, [])

  const updateAutoplay = useCallback((val: boolean) => {
    setAutoPlayOnSelect(val)
    localStorage.setItem('tv5_autoplay', String(val))
  }, [])

  const jumpToAnchor = useCallback(() => {
    if (!videoRef.current) return
    const cand = anchorCandidate || targetCandidate || activeCandidate || (candidates.length > 0 ? candidates[0] : null)
    if (!cand) return
    const targetSec = cand.timestamp_ms / 1000
    videoRef.current.currentTime = targetSec
    setPlaybackTime(targetSec)
    setSelectedTimelineKeyframe(null)
    handleStep(0)
  }, [anchorCandidate, targetCandidate, activeCandidate, candidates, handleStep])

  // Seek and optionally autoplay video when active candidate changes
  useEffect(() => {
    if (activeCandidate && videoRef.current) {
      const targetSec = activeCandidate.timestamp_ms / 1000
      videoRef.current.currentTime = targetSec
      setPlaybackTime(targetSec)
      videoRef.current.playbackRate = playbackRate
      if (autoPlayOnSelect) {
        const playResult = videoRef.current.play()
        if (playResult && typeof playResult.then === 'function') {
          playResult.then(() => setIsPlaying(true)).catch(() => setIsPlaying(false))
        }
      }
    }
  }, [activeCandidate, autoPlayOnSelect, playbackRate])

  // Sync video currentTime when stepping exact frames
  useEffect(() => {
    if (currentStep?.frame && videoRef.current) {
      const targetSec = currentStep.frame.timestamp_ms / 1000
      if (Math.abs(videoRef.current.currentTime - targetSec) > 0.005) {
        videoRef.current.currentTime = targetSec
        setPlaybackTime(targetSec)
      }
    }
  }, [currentStep])

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setPlaybackTime(videoRef.current.currentTime)
      setIsPlaying(!videoRef.current.paused)
    }
  }

  // Keyboard navigation for stepping, playback and fast-forward
  useEffect(() => {
    const handleKeyDown = (e: globalThis.KeyboardEvent) => {
      if (
        document.activeElement?.tagName === 'INPUT' ||
        document.activeElement?.tagName === 'TEXTAREA' ||
        document.activeElement?.tagName === 'SELECT'
      ) {
        return
      }

      if (e.key === ' ' || e.code === 'Space') {
        e.preventDefault()
        togglePlay()
      } else if (e.key === 'j' || e.key === 'J') {
        e.preventDefault()
        seekRelative(-seekStepSeconds)
      } else if (e.key === 'k' || e.key === 'K') {
        e.preventDefault()
        togglePlay()
      } else if (e.key === 'l' || e.key === 'L') {
        e.preventDefault()
        seekRelative(seekStepSeconds)
      } else if (e.key === '1') {
        e.preventDefault()
        updateSpeed(1.0)
      } else if (e.key === '2') {
        e.preventDefault()
        updateSpeed(1.5)
      } else if (e.key === '3') {
        e.preventDefault()
        updateSpeed(2.0)
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault()
        if (e.ctrlKey || e.altKey) {
          handleStep(cumulativeOffset - 1)
        } else if (e.shiftKey) {
          seekRelative(-seekStepSeconds * 2)
        } else {
          seekRelative(-seekStepSeconds)
        }
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        if (e.ctrlKey || e.altKey) {
          handleStep(cumulativeOffset + 1)
        } else if (e.shiftKey) {
          seekRelative(seekStepSeconds * 2)
        } else {
          seekRelative(seekStepSeconds)
        }
      } else if (e.key === ',' || e.key === '<') {
        e.preventDefault()
        handleStep(cumulativeOffset - 1)
      } else if (e.key === '.' || e.key === '>') {
        e.preventDefault()
        handleStep(cumulativeOffset + 1)
      } else if (e.key === 'Home' || e.key === '0') {
        e.preventDefault()
        jumpToAnchor()
      } else if (e.key === 'b' || e.key === 'B') {
        e.preventDefault()
        togglePlaybackBasketRef.current?.()
      } else if (e.key === 'p' || e.key === 'P') {
        e.preventDefault()
        if (targetCandidate) {
          dispatch({ type: 'TOGGLE_PIN_CANDIDATE', payload: targetCandidate })
        }
      } else if (e.key === 'f' || e.key === 'F') {
        e.preventDefault()
        toggleFullscreen()
      }
    }

    window.addEventListener('keydown', handleKeyDown, { capture: true })
    return () => window.removeEventListener('keydown', handleKeyDown, { capture: true })
  }, [cumulativeOffset, handleStep, togglePlay, seekRelative, updateSpeed, jumpToAnchor, seekStepSeconds, toggleFullscreen, targetCandidate, dispatch])

  const handlePrevTrakeEvent = () => {
    if (trakeActiveSlotIndex === null || trakeActiveSlotIndex <= 0) return
    const newIdx = trakeActiveSlotIndex - 1
    dispatch({ type: 'SELECT_TRAKE_SLOT', payload: newIdx })
  }

  const handleNextTrakeEvent = () => {
    if (trakeActiveSlotIndex === null || trakeActiveSlotIndex >= trakeSlots.length - 1) return
    const newIdx = trakeActiveSlotIndex + 1
    dispatch({ type: 'SELECT_TRAKE_SLOT', payload: newIdx })
  }

  if (!targetCandidate) {
    return (
      <div className="inspection-workspace-empty" data-testid="workspace-empty">
        <div className="empty-message-box">
          <p className="empty-title">
            {taskMode === 'TRAKE'
              ? 'No TRAKE Event Frame Available for Inspection'
              : taskMode === 'VQA'
              ? 'No VQA Answer Selected for Inspection'
              : 'No Candidate Selected for Inspection'}
          </p>
          <p className="empty-subtitle">
            {taskMode === 'TRAKE'
              ? 'Run Search & Align in the Retrieval workspace to align event slots.'
              : taskMode === 'VQA'
              ? 'Run Search & Answer in the Retrieval workspace to generate grounded answers.'
              : 'Perform a search in the Retrieval tab and select a candidate to inspect.'}
          </p>
          <button
            type="button"
            className="return-retrieval-btn"
            onClick={() => dispatch({ type: 'SET_ACTIVE_TAB', payload: 'retrieval' })}
          >
            <ArrowLeftIcon size={14} />
            <span>Go to Retrieval</span>
          </button>
        </div>
      </div>
    )
  }

  const isCurrentStepMatching = currentStep?.frame?.video_id === targetCandidate.video_id

  const inspectedFrameId =
    cumulativeOffset === 0 && (!currentStep?.frame || !isCurrentStepMatching)
      ? targetCandidate.frame_id
      : isCurrentStepMatching
      ? currentStep?.frame?.frame_id
      : targetCandidate.frame_id

  const inspectedTimestampMs =
    cumulativeOffset === 0 && (!currentStep?.frame || !isCurrentStepMatching)
      ? targetCandidate.timestamp_ms
      : isCurrentStepMatching
      ? currentStep?.frame?.timestamp_ms
      : targetCandidate.timestamp_ms

  const isSelectable = isFixture
    ? false
    : cumulativeOffset === 0
    ? true
    : isCurrentStepMatching && currentStep?.frame?.submission_selectable === true

  const isBoundary =
    currentStep && (!currentStep.frame || currentStep.degraded_reason !== null)

  const videoStreamUrl = getVideoStreamUrl(targetCandidate.video_id)
  const currentPlaybackFrame = Math.round(playbackTime * (videoMeta?.fps || 25.0))

  useEffect(() => {
    if (targetCandidate?.video_id) {
      fetchVideoMeta(targetCandidate.video_id)
        .then((meta) => setVideoMeta(meta))
        .catch(() => setVideoMeta({ video_id: targetCandidate.video_id, fps: 25.0 }))

      setIsLoadingKeyframes(true)
      fetchVideoKeyframes(targetCandidate.video_id)
        .then((kfs) => {
          setVideoKeyframes(kfs)
          setIsLoadingKeyframes(false)
        })
        .catch(() => {
          setVideoKeyframes([])
          setIsLoadingKeyframes(false)
        })
    }
  }, [targetCandidate?.video_id])

  // Reset selected timeline preview keyframe on candidate switch
  useEffect(() => {
    setSelectedTimelineKeyframe(null)
  }, [targetCandidate?.video_id, targetCandidate?.frame_id])

  const handleSelectTimelineKeyframe = useCallback(
    (kf: VideoKeyframe) => {
      setSelectedTimelineKeyframe(kf)
      if (videoRef.current) {
        const sec = kf.timestamp_ms / 1000
        videoRef.current.currentTime = sec
        setPlaybackTime(sec)
      }
    },
    []
  )

  // Enable mouse wheel horizontal scrubbing over the scenes filmstrip
  useEffect(() => {
    const el = scenesStripRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      if (e.deltaY !== 0 || e.deltaX !== 0) {
        e.preventDefault()
        el.scrollLeft += e.deltaY !== 0 ? e.deltaY * 1.5 : e.deltaX * 1.5
      }
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [videoKeyframes, activeBottomTab, isBottomCollapsed])

  // Anchor Frame ID vs Previewed Frame ID
  const anchorFid = anchorCandidate?.frame_id ?? targetCandidate.frame_id
  const targetFid = selectedTimelineKeyframe?.frame_id ?? (inspectedFrameId ?? targetCandidate.frame_id)

  const bestMatchingKeyframe = useMemo(() => {
    if (!videoKeyframes || videoKeyframes.length === 0) return null
    const exact = videoKeyframes.find((k) => k.frame_id === targetFid)
    if (exact) return exact
    return videoKeyframes.reduce((closest, kf) => {
      return Math.abs(kf.frame_id - targetFid) < Math.abs(closest.frame_id - targetFid) ? kf : closest
    }, videoKeyframes[0])
  }, [videoKeyframes, targetFid])

  // Auto-scroll the target / anchor keyframe into center view whenever candidate or keyframes load
  useEffect(() => {
    if (activeBottomTab === 'scenes' && !isBottomCollapsed && videoKeyframes.length > 0) {
      const timer = setTimeout(() => {
        const activeEl = scenesStripRef.current?.querySelector('.is-target-keyframe') as HTMLElement
        if (activeEl && scenesStripRef.current) {
          activeEl.scrollIntoView({
            behavior: 'smooth',
            inline: 'center',
            block: 'nearest',
          })
        }
      }, 70)
      return () => clearTimeout(timer)
    }
  }, [targetCandidate?.video_id, targetCandidate?.frame_id, videoKeyframes, activeBottomTab, isBottomCollapsed, bestMatchingKeyframe])

  const isCurrentPlaybackInBasket = Boolean(
    targetCandidate &&
    submissionBasket.some(
      (b) => b.video_id === targetCandidate.video_id && b.frame_id === currentPlaybackFrame
    )
  )

  const isSelectedFrameInBasket = Boolean(
    targetCandidate &&
    submissionBasket.some(
      (b) => b.video_id === targetCandidate.video_id && b.frame_id === (inspectedFrameId ?? targetCandidate.frame_id)
    )
  )

  const handleToggleCurrentPlaybackFrameBasket = useCallback(() => {
    if (!targetCandidate) return
    const videoId = targetCandidate.video_id
    const frameId = currentPlaybackFrame
    const inBasket = submissionBasket.some(
      (b) => b.video_id === videoId && b.frame_id === frameId
    )
    if (inBasket) {
      dispatch({
        type: 'REMOVE_FROM_BASKET',
        payload: { video_id: videoId, frame_id: frameId },
      })
      telemetry.record({
        action: 'REMOVE_FROM_BASKET',
        taskMode,
        videoId,
        frameId,
      })
    } else {
      dispatch({
        type: 'ADD_TO_BASKET',
        payload: {
          query_id: queryId || targetCandidate.query_id || 'manual_query',
          video_id: videoId,
          frame_id: frameId,
          timestamp_ms: Math.round(playbackTime * 1000),
          added_at_utc: new Date().toISOString(),
          task: taskMode || 'KIS',
        },
      })
      telemetry.record({
        action: 'ADD_TO_BASKET',
        taskMode,
        videoId,
        frameId,
      })
    }
  }, [currentPlaybackFrame, dispatch, playbackTime, queryId, submissionBasket, targetCandidate, taskMode])

  const handleToggleSelectedFrameBasket = useCallback(() => {
    if (!targetCandidate) return
    const videoId = targetCandidate.video_id
    const frameId = inspectedFrameId ?? targetCandidate.frame_id
    const inBasket = submissionBasket.some(
      (b) => b.video_id === videoId && b.frame_id === frameId
    )
    if (inBasket) {
      dispatch({
        type: 'REMOVE_FROM_BASKET',
        payload: { video_id: videoId, frame_id: frameId },
      })
      telemetry.record({
        action: 'REMOVE_FROM_BASKET',
        taskMode,
        videoId,
        frameId,
      })
    } else {
      dispatch({
        type: 'ADD_TO_BASKET',
        payload: {
          query_id: queryId || targetCandidate.query_id || 'manual_query',
          video_id: videoId,
          frame_id: frameId,
          timestamp_ms: inspectedTimestampMs ?? targetCandidate.timestamp_ms ?? 0,
          added_at_utc: new Date().toISOString(),
          task: taskMode || 'KIS',
        },
      })
      telemetry.record({
        action: 'ADD_TO_BASKET',
        taskMode,
        videoId,
        frameId,
      })
    }
  }, [dispatch, inspectedFrameId, inspectedTimestampMs, queryId, submissionBasket, targetCandidate, taskMode])

  const handleToggleKeyframeBasket = useCallback(
    (kf: VideoKeyframe) => {
      if (!targetCandidate) return
      const videoId = targetCandidate.video_id
      const frameId = kf.frame_id
      const inBasket = submissionBasket.some(
        (b) => b.video_id === videoId && b.frame_id === frameId
      )
      if (inBasket) {
        dispatch({
          type: 'REMOVE_FROM_BASKET',
          payload: { video_id: videoId, frame_id: frameId },
        })
        telemetry.record({
          action: 'REMOVE_FROM_BASKET',
          taskMode,
          videoId,
          frameId,
        })
      } else {
        dispatch({
          type: 'ADD_TO_BASKET',
          payload: {
            query_id: queryId || targetCandidate.query_id || 'manual_query',
            video_id: videoId,
            frame_id: frameId,
            timestamp_ms: kf.timestamp_ms,
            added_at_utc: new Date().toISOString(),
            task: taskMode || 'KIS',
          },
        })
        telemetry.record({
          action: 'ADD_TO_BASKET',
          taskMode,
          videoId,
          frameId,
        })
      }
    },
    [dispatch, queryId, submissionBasket, targetCandidate, taskMode]
  )

  useEffect(() => {
    togglePlaybackBasketRef.current = handleToggleCurrentPlaybackFrameBasket
  }, [handleToggleCurrentPlaybackFrameBasket])

  const handleSelectCandidate = (cand: SearchCandidate) => {
    dispatch({ type: 'SELECT_CANDIDATE', payload: cand })
  }

  return (
    <div className="inspection-workspace-layout" data-testid="inspection-workspace">
      {/* ── 1. Left Candidate Shortlist Column (~235px) ── */}
      <aside className="inspection-shortlist-rail" data-testid="inspection-shortlist-rail">
        <div className="shortlist-header">
          <span className="shortlist-title tabular-nums">
            {taskMode === 'TRAKE'
              ? `Event Slots (${trakeSlots.length})`
              : taskMode === 'VQA'
              ? `VQA Answers (${vqaResults.length})`
              : `Results (${candidates.length})`}
          </span>
          <button
            type="button"
            className="back-to-retrieval-link"
            onClick={() => dispatch({ type: 'SET_ACTIVE_TAB', payload: 'retrieval' })}
            title="Return to retrieval results"
          >
            <ArrowLeftIcon size={12} />
            <span>Back to Results</span>
          </button>
        </div>

        <div className="shortlist-cards-scroll">
          {taskMode === 'TRAKE' ? (
            trakeSlots.map((slot) => {
              const isSelected = trakeActiveSlotIndex === slot.event_index
              const hasFrame = slot.video_id && slot.frame_id !== null
              const slotCand: SearchCandidate | null = hasFrame
                ? {
                    query_id: queryId || 'trake',
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
                  key={`trake-slot-rail-${slot.event_index}`}
                  className={`shortlist-card ${isSelected ? 'card-selected' : ''}`}
                  onClick={() => {
                    dispatch({ type: 'SELECT_TRAKE_SLOT', payload: slot.event_index })
                  }}
                  role="button"
                  tabIndex={0}
                  data-testid={`trake-shortlist-card-${slot.event_index}`}
                >
                  <div className="shortlist-thumb-box">
                    {slotCand ? (
                      <CandidatePreviewThumb
                        candidate={slotCand}
                        alt={`Event ${slot.event_index + 1}`}
                        className="shortlist-thumb-img"
                        onErrorFallback={
                          <div className="shortlist-thumb-placeholder">
                            <span>#{slot.event_index + 1}</span>
                          </div>
                        }
                      />
                    ) : (
                      <div className="shortlist-thumb-placeholder">
                        <span>#{slot.event_index + 1}</span>
                      </div>
                    )}
                    <span className="shortlist-rank-tag tabular-nums">#{slot.event_index + 1}</span>
                  </div>

                  <div className="shortlist-info">
                    <span className="shortlist-vid" title={slot.event_label}>{slot.event_label}</span>
                    <span className="shortlist-fid tabular-nums">
                      {slot.frame_id !== null ? `Frame ${slot.frame_id}` : 'Unassigned'}
                    </span>
                    <div className="shortlist-sub-meta">
                      {slot.score != null && (
                        <span className="shortlist-score tabular-nums text-cyan">
                          {slot.score.toFixed(3)}
                        </span>
                      )}
                      <span className={`slot-rail-lock-badge ${slot.locked ? 'is-locked' : 'is-unlocked'}`}>
                        {slot.locked ? 'LOCKED' : 'UNLOCKED'}
                      </span>
                    </div>
                    {isSelected && <span className="shortlist-anchor-pill">Active</span>}
                  </div>
                </div>
              )
            })
          ) : taskMode === 'VQA' ? (
            vqaResults.length === 0 ? (
              <div className="shortlist-empty-note">
                <span>No VQA answers yet</span>
              </div>
            ) : (
              vqaResults.map((res) => {
                const isSelected =
                  vqaActiveResult?.video_id === res.video_id &&
                  vqaActiveResult?.frame_id === res.frame_id
                const formattedRank = res.rank < 10 ? `0${res.rank}` : `${res.rank}`
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
                    key={`vqa-rail-${res.video_id}-${res.frame_id}-${res.rank}`}
                    className={`shortlist-card ${isSelected ? 'card-selected' : ''}`}
                    onClick={() => dispatch({ type: 'SELECT_VQA_RESULT', payload: res })}
                    role="button"
                    tabIndex={0}
                    data-testid={`vqa-shortlist-card-${res.rank}`}
                  >
                    <div className="shortlist-thumb-box">
                      <CandidatePreviewThumb
                        candidate={vqaCand}
                        alt={`Rank ${res.rank}`}
                        className="shortlist-thumb-img"
                      />
                      <span className="shortlist-rank-tag tabular-nums">{formattedRank}</span>
                    </div>

                    <div className="shortlist-info">
                      <span className="shortlist-vid">{res.video_id}</span>
                      <span className="shortlist-fid tabular-nums">Frame {res.frame_id}</span>
                      <div className="shortlist-sub-meta">
                        <span className="shortlist-score tabular-nums text-cyan">
                          {res.confidence != null ? `${(res.confidence * 100).toFixed(0)}%` : ''}
                        </span>
                      </div>
                      {isSelected && <span className="shortlist-anchor-pill">Active</span>}
                    </div>
                  </div>
                )
              })
            )
          ) : (
            candidates.map((cand) => {
              const isSelected =
                targetCandidate.video_id === cand.video_id &&
                targetCandidate.frame_id === cand.frame_id
              const formattedRank = cand.rank < 10 ? `0${cand.rank}` : `${cand.rank}`

              return (
                <div
                  key={`${cand.video_id}-${cand.frame_id}-${cand.rank}`}
                  className={`shortlist-card ${isSelected ? 'card-selected' : ''}`}
                  onClick={() => handleSelectCandidate(cand)}
                  role="button"
                  tabIndex={0}
                  data-testid={`shortlist-card-${cand.rank}`}
                >
                  <div className="shortlist-thumb-box">
                    <CandidatePreviewThumb
                      candidate={cand}
                      alt={`Rank ${cand.rank}`}
                      className="shortlist-thumb-img"
                      loading="lazy"
                      onErrorFallback={<span className="shortlist-thumb-fallback">N/A</span>}
                    />
                    <span className="shortlist-rank-tag tabular-nums">{formattedRank}</span>

                    {/* Quick Pin Toggle on Shortlist Card */}
                    {(() => {
                      const isCandPinned = pinnedCandidates.some(
                        (p) => p.video_id === cand.video_id && p.frame_id === cand.frame_id
                      )
                      return (
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation()
                            dispatch({ type: 'TOGGLE_PIN_CANDIDATE', payload: cand })
                          }}
                          style={{
                            position: 'absolute',
                            top: '3px',
                            right: '3px',
                            background: isCandPinned ? 'rgba(234, 179, 8, 0.95)' : 'rgba(0, 0, 0, 0.65)',
                            border: isCandPinned ? '1px solid #eab308' : '1px solid rgba(255, 255, 255, 0.2)',
                            borderRadius: '3px',
                            padding: '2px 4px',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '2px',
                            zIndex: 2,
                          }}
                          title={isCandPinned ? 'Đã ghim (Click để bỏ)' : 'Ghim vào khay tiềm năng (Phím P)'}
                        >
                          <PinIcon size={10} color={isCandPinned ? '#000' : '#fff'} />
                        </button>
                      )
                    })()}
                  </div>

                  <div className="shortlist-info">
                    <span className="shortlist-vid">{cand.video_id}</span>
                    <span className="shortlist-fid tabular-nums">Frame {cand.frame_id}</span>
                    {cand.score != null && (
                      <span className="shortlist-score tabular-nums text-cyan">
                        {cand.score.toFixed(4)}
                      </span>
                    )}
                    {isSelected && <span className="shortlist-anchor-pill">Anchor</span>}
                  </div>
                </div>
              )
            })
          )}
        </div>
      </aside>

      {/* ── 2. Center Media Stage ── */}
      <main className="inspection-media-stage">
        {/* Top Context Strip for TRAKE Mode */}
        {taskMode === 'TRAKE' && (
          <div className="inspection-trake-context-strip" data-testid="inspection-context-strip">
            <div className="context-strip-task-tag trake-tag">
              <FilmstripIcon size={14} className="text-cyan" />
              <span>TRAKE</span>
            </div>

            <div className="context-strip-item">
              <span className="context-strip-label">Sequence:</span>
              <span className="context-strip-value" data-testid="context-strip-event">
                {queryText || 'N/A'}
              </span>
            </div>

            <div className="context-strip-item highlight-trake-event">
              <span className="context-strip-label">Active Event:</span>
              <span className="context-strip-value" data-testid="trake-context-event">
                {activeTrakeSlot
                  ? `Event #${activeTrakeSlot.event_index + 1} of ${trakeSlots.length}: "${activeTrakeSlot.event_label}"`
                  : 'None'}
              </span>
            </div>

            <div className="context-strip-item">
              <span className="context-strip-label">Hypothesis:</span>
              <span className="context-strip-value text-cyan tabular-nums" data-testid="trake-context-video">
                {trakeVideoId || targetCandidate.video_id}
              </span>
            </div>

            <div className="context-strip-item">
              <span className="context-strip-label">Status:</span>
              <span
                className={`context-strip-badge ${activeTrakeSlot?.locked ? 'badge-locked' : 'badge-unlocked'}`}
                data-testid="trake-context-lock"
              >
                {activeTrakeSlot?.locked ? 'LOCKED' : 'UNLOCKED'}
              </span>
            </div>

            {/* Event Navigation (Previous / Next Event in Sequence) */}
            <div className="context-strip-event-nav" data-testid="trake-event-nav">
              <button
                type="button"
                className="btn-event-nav"
                disabled={trakeActiveSlotIndex === null || trakeActiveSlotIndex <= 0}
                onClick={handlePrevTrakeEvent}
                title="Navigate to previous semantic event in sequence"
                data-testid="btn-trake-prev-event"
              >
                ← Prev Event
              </button>

              <button
                type="button"
                className="btn-event-nav"
                disabled={
                  trakeActiveSlotIndex === null ||
                  trakeActiveSlotIndex >= trakeSlots.length - 1
                }
                onClick={handleNextTrakeEvent}
                title="Navigate to next semantic event in sequence"
                data-testid="btn-trake-next-event"
              >
                Next Event →
              </button>
            </div>
          </div>
        )}

        {/* Top Context Strip for Q&A Mode */}
        {taskMode === 'VQA' && (
          <div className="inspection-vqa-context-strip" data-testid="inspection-context-strip">
            <div className="context-strip-task-tag">
              <QuestionIcon size={14} className="text-cyan" />
              <span>Q&A</span>
            </div>
            <div className="context-strip-item">
              <span className="context-strip-label">Event:</span>
              <span className="context-strip-value" data-testid="context-strip-event">
                {queryText || vqaActiveResult?.evidence.query_text || 'N/A'}
              </span>
            </div>
            <div className="context-strip-item highlight-q">
              <span className="context-strip-label">Question:</span>
              <span className="context-strip-value" data-testid="context-strip-question">
                {vqaQuestion || vqaActiveResult?.evidence.question || 'N/A'}
              </span>
            </div>
          </div>
        )}

        {/* Dual Video & Frame Viewports */}
        <div className="viewports-split-row">
          {/* A. Original Video Context */}
          <div ref={videoViewportRef} className="viewport-panel video-viewport">
            <div className="viewport-panel-header">
              <div className="panel-title-group">
                <VideoIcon size={16} className="text-cyan" />
                <span className="panel-title-text">Original Video</span>
                {videoMeta?.fps != null && (
                  <span
                    className="fps-badge text-cyan tabular-nums"
                    style={{
                      marginLeft: '8px',
                      fontSize: '11px',
                      fontWeight: 600,
                      background: 'rgba(0, 229, 255, 0.12)',
                      color: '#00e5ff',
                      padding: '2px 7px',
                      borderRadius: '4px',
                      border: '1px solid rgba(0, 229, 255, 0.25)',
                    }}
                    title={`Nominal: ${videoMeta.fps_nominal ?? 'N/A'}, Average: ${videoMeta.fps_average ?? 'N/A'}`}
                  >
                    {videoMeta.fps} FPS
                  </span>
                )}
              </div>
              <div className="panel-timecode-group" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span
                  className="current-playback-frame-badge tabular-nums"
                  style={{
                    fontSize: '11px',
                    fontWeight: 700,
                    background: 'rgba(255, 171, 0, 0.15)',
                    color: '#ffab00',
                    padding: '2px 7px',
                    borderRadius: '4px',
                    border: '1px solid rgba(255, 171, 0, 0.35)',
                  }}
                  title="Khung hình thực tế tại mốc thời gian đang phát"
                >
                  Frame {currentPlaybackFrame}
                </span>
                <span className="panel-timecode tabular-nums">{playbackTime.toFixed(3)}s</span>
              </div>
            </div>

            <div
              className="video-player-frame"
              onDoubleClick={toggleFullscreen}
              title="Nhấp đúp chuột để phóng to / thu nhỏ toàn màn hình (Phím F)"
            >
              <video
                ref={videoRef}
                src={videoStreamUrl}
                controls
                preload="metadata"
                onTimeUpdate={handleTimeUpdate}
                onSeeked={handleTimeUpdate}
                onLoadedMetadata={handleTimeUpdate}
                className="media-video-player"
                data-testid="original-video-player"
              />
            </div>

            {/* Interactive Hover Preview Scrubber Bar (Toggled via Settings) */}
            {showHoverScrubber && (
              <div
                className="video-hover-scrubber-bar"
                style={{
                  position: 'relative',
                  width: '100%',
                  height: '18px',
                  background: 'rgba(15, 23, 42, 0.95)',
                  padding: '4px 12px',
                  boxSizing: 'border-box',
                  display: 'flex',
                  alignItems: 'center',
                  cursor: 'pointer',
                  borderTop: '1px solid rgba(255, 255, 255, 0.08)',
                  userSelect: 'none',
                  zIndex: 45,
                }}
              >
                {/* Track container */}
                <div
                  ref={scrubberTrackRef}
                  onMouseDown={handleScrubberMouseDown}
                  onMouseMove={handleScrubberMouseMove}
                  onMouseLeave={(e) => {
                    handleScrubberMouseLeave()
                    if (!isDraggingScrubber) {
                      e.currentTarget.style.height = '6px'
                    }
                  }}
                  onClick={handleScrubberClick}
                  style={{
                    position: 'relative',
                    width: '100%',
                    height: isDraggingScrubber ? '8px' : '6px',
                    background: 'rgba(255, 255, 255, 0.18)',
                    borderRadius: '3px',
                    transition: isDraggingScrubber ? 'none' : 'height 0.15s ease',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.height = '8px'
                  }}
                >
                  {/* Played Progress Bar */}
                  <div
                    style={{
                      position: 'absolute',
                      left: 0,
                      top: 0,
                      bottom: 0,
                      width: `${videoDuration > 0 ? (playbackTime / videoDuration) * 100 : 0}%`,
                      background: 'linear-gradient(90deg, #0284c7, #00e5ff)',
                      borderRadius: '3px',
                      pointerEvents: 'none',
                    }}
                  />

                  {/* Current Playhead Handle */}
                  <div
                    style={{
                      position: 'absolute',
                      left: `calc(${videoDuration > 0 ? (playbackTime / videoDuration) * 100 : 0}% - 5px)`,
                      top: '-2px',
                      width: '10px',
                      height: '10px',
                      borderRadius: '50%',
                      background: '#00e5ff',
                      boxShadow: '0 0 6px #00e5ff',
                      pointerEvents: 'none',
                    }}
                  />

                  {/* Anchor Frame Marker */}
                  {targetCandidate?.timestamp_ms != null && videoDuration > 0 && (
                    <div
                      style={{
                        position: 'absolute',
                        left: `calc(${(targetCandidate.timestamp_ms / 1000 / videoDuration) * 100}% - 3px)`,
                        top: '-4px',
                        width: '6px',
                        height: '14px',
                        background: '#eab308',
                        borderRadius: '2px',
                        boxShadow: '0 0 8px rgba(234, 179, 8, 0.8)',
                        zIndex: 3,
                        pointerEvents: 'none',
                      }}
                      title={`Mốc Anchor Frame ${targetCandidate.frame_id}`}
                    />
                  )}

                  {/* Keyframe Tick Marks */}
                  {videoDuration > 0 &&
                    videoKeyframes.map((kf) => {
                      const kfPercent = (kf.timestamp_ms / 1000 / videoDuration) * 100
                      return (
                        <div
                          key={`scrubber-tick-${kf.frame_id}`}
                          style={{
                            position: 'absolute',
                            left: `${kfPercent}%`,
                            top: '1px',
                            width: '2px',
                            height: '4px',
                            background: 'rgba(255, 255, 255, 0.35)',
                            borderRadius: '1px',
                            pointerEvents: 'none',
                          }}
                        />
                      )
                    })}

                  {/* Hover Indicator Line */}
                  {hoverPos && (
                    <div
                      style={{
                        position: 'absolute',
                        left: `${hoverPos.percent * 100}%`,
                        top: '-2px',
                        width: '2px',
                        height: '10px',
                        background: '#fff',
                        boxShadow: '0 0 4px #fff',
                        pointerEvents: 'none',
                        zIndex: 4,
                      }}
                    />
                  )}

                  {/* Hover Thumbnail Preview Tooltip Box */}
                  {hoverPos && (() => {
                    const currentVidId = targetCandidate?.video_id || kisActiveCandidate?.video_id || candidates[0]?.video_id || ''
                    if (!currentVidId) return null
                    return (
                      <div
                        style={{
                          position: 'absolute',
                          left: (() => {
                            const tooltipW = 150
                            const trackW = scrubberTrackRef.current?.clientWidth || 400
                            let pos = hoverPos.x - tooltipW / 2
                            if (pos < 0) pos = 0
                            if (pos + tooltipW > trackW) pos = trackW - tooltipW
                            return `${pos}px`
                          })(),
                          bottom: '18px',
                          width: '150px',
                          background: 'rgba(15, 23, 42, 0.96)',
                          backdropFilter: 'blur(10px)',
                          border: '1.5px solid #00e5ff',
                          borderRadius: '6px',
                          padding: '4px',
                          boxShadow: '0 10px 30px rgba(0, 0, 0, 0.9), 0 0 12px rgba(0, 229, 255, 0.3)',
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'center',
                          gap: '3px',
                          pointerEvents: 'none',
                          zIndex: 100,
                        }}
                      >
                        {/* Thumbnail Image */}
                        <div style={{ width: '142px', height: '80px', borderRadius: '4px', overflow: 'hidden', background: '#000' }}>
                          <img
                            src={
                              hoverPos.nearestKeyframe
                                ? getThumbnailUrl(currentVidId, hoverPos.nearestKeyframe.frame_id)
                                : getSelectedFrameUrl(currentVidId, hoverPos.frameId)
                            }
                            alt={`Preview ${hoverPos.frameId}`}
                            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                            onError={(e) => {
                              const target = e.currentTarget
                              if (!target.src.includes('/frames/') && currentVidId) {
                                target.src = getSelectedFrameUrl(currentVidId, hoverPos.frameId)
                              }
                            }}
                          />
                        </div>

                        {/* Caption Info Row */}
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%', padding: '0 2px', fontSize: '9.5px' }}>
                          <span style={{ color: '#fff', fontWeight: 700, fontFamily: 'monospace' }}>
                            {Math.floor(hoverPos.timeSec / 60)}:{String(Math.floor(hoverPos.timeSec % 60)).padStart(2, '0')}.{String(Math.floor((hoverPos.timeSec % 1) * 10)).padStart(1, '0')}s
                          </span>
                          <span style={{ color: '#00e5ff', fontWeight: 600 }}>
                            {hoverPos.nearestKeyframe ? `Keyframe ${hoverPos.nearestKeyframe.frame_id}` : `Frame ${hoverPos.frameId}`}
                          </span>
                        </div>
                      </div>
                    )
                  })()}
                </div>
              </div>
            )}

            {/* Fast Video Scrubbing & Playback Toolbar with YouTube-style Settings */}
            <div
              className="video-quick-controls"
              style={{
                position: 'relative',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '6px 10px',
                background: 'rgba(15, 23, 42, 0.85)',
                borderTop: '1px solid rgba(255, 255, 255, 0.08)',
                fontSize: '11px',
                gap: '8px',
              }}
            >
              {/* Left group: Play/Pause and Fast Jumps (-Ns, +Ns liền kề) */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <button
                  type="button"
                  onClick={togglePlay}
                  style={{
                    background: isPlaying ? 'rgba(0, 229, 255, 0.2)' : 'rgba(255, 255, 255, 0.1)',
                    color: isPlaying ? '#00e5ff' : '#fff',
                    border: '1px solid rgba(255, 255, 255, 0.15)',
                    borderRadius: '4px',
                    padding: '3px 10px',
                    fontSize: '12px',
                    fontWeight: 700,
                    cursor: 'pointer',
                  }}
                  title="Space / K: Phát hoặc tạm dừng"
                >
                  {isPlaying ? '⏸' : '▶'}
                </button>

                <button
                  type="button"
                  onClick={() => seekRelative(-seekStepSeconds)}
                  style={{
                    background: 'rgba(255, 255, 255, 0.08)',
                    color: '#f8fafc',
                    border: '1px solid rgba(255, 255, 255, 0.15)',
                    borderRadius: '4px',
                    padding: '3px 9px',
                    fontWeight: 600,
                    cursor: 'pointer',
                  }}
                  title={`Phím ← (Mũi tên trái) / J: Tua lùi ${seekStepSeconds}s`}
                >
                  ⏪ -{seekStepSeconds}s
                </button>

                <button
                  type="button"
                  onClick={() => seekRelative(seekStepSeconds)}
                  style={{
                    background: 'rgba(255, 255, 255, 0.08)',
                    color: '#f8fafc',
                    border: '1px solid rgba(255, 255, 255, 0.15)',
                    borderRadius: '4px',
                    padding: '3px 9px',
                    fontWeight: 600,
                    cursor: 'pointer',
                  }}
                  title={`Phím → (Mũi tên phải) / L: Tua tới ${seekStepSeconds}s`}
                >
                  +{seekStepSeconds}s ⏩
                </button>

                {/* Vertical Divider */}
                <div style={{ width: '1px', height: '16px', background: 'rgba(255, 255, 255, 0.15)', margin: '0 4px' }} />

                {/* Dedicated Anchor Button (Icon-only) */}
                <button
                  type="button"
                  onClick={jumpToAnchor}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: 'rgba(0, 229, 255, 0.08)',
                    color: '#00e5ff',
                    border: '1px solid rgba(0, 229, 255, 0.3)',
                    borderRadius: '4px',
                    padding: '4px 8px',
                    cursor: 'pointer',
                  }}
                  title="Phím 0 / Home: Trở về mốc Anchor frame ban đầu"
                >
                  <AnchorIcon size={14} color="#00e5ff" />
                </button>

                {/* Vertical Divider */}
                <div style={{ width: '1px', height: '16px', background: 'rgba(255, 255, 255, 0.15)', margin: '0 4px' }} />

                {/* Save Current Video Playback Frame into Basket button (Icon-only) */}
                <button
                  type="button"
                  onClick={handleToggleCurrentPlaybackFrameBasket}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: isCurrentPlaybackInBasket ? 'rgba(16, 185, 129, 0.28)' : 'rgba(0, 229, 255, 0.15)',
                    color: isCurrentPlaybackInBasket ? '#10b981' : '#00e5ff',
                    border: isCurrentPlaybackInBasket ? '1px solid #10b981' : '1px solid rgba(0, 229, 255, 0.4)',
                    borderRadius: '4px',
                    padding: '4px 8px',
                    cursor: 'pointer',
                  }}
                  title={
                    isCurrentPlaybackInBasket
                      ? `Đã lưu Frame ${currentPlaybackFrame} trong Giỏ (Phím B / Click để gỡ)`
                      : `Lưu Frame ${currentPlaybackFrame} đang phát vào Giỏ (Phím B / Click để lưu)`
                  }
                >
                  {isCurrentPlaybackInBasket ? <CheckIcon size={14} color="#10b981" /> : <BasketPlusIcon size={14} color="#00e5ff" />}
                </button>

                {/* Pin Video / Candidate into Staging Tray button (Icon-only) */}
                {targetCandidate && (() => {
                  const isTargetPinned = pinnedCandidates.some(
                    (p) => p.video_id === targetCandidate.video_id && p.frame_id === targetCandidate.frame_id
                  )
                  return (
                    <button
                      type="button"
                      onClick={() => dispatch({ type: 'TOGGLE_PIN_CANDIDATE', payload: targetCandidate })}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        background: isTargetPinned ? 'rgba(234, 179, 8, 0.28)' : 'rgba(255, 255, 255, 0.08)',
                        color: isTargetPinned ? '#eab308' : '#cbd5e1',
                        border: isTargetPinned ? '1px solid #eab308' : '1px solid rgba(255, 255, 255, 0.15)',
                        borderRadius: '4px',
                        padding: '4px 8px',
                        cursor: 'pointer',
                      }}
                      title={
                        isTargetPinned
                          ? `Đã ghim ${targetCandidate.video_id} (Phím P / Click để gỡ)`
                          : `Ghim ${targetCandidate.video_id} vào khay tiềm năng (Phím P / Click để ghim)`
                      }
                    >
                      <PinIcon size={14} color={isTargetPinned ? '#eab308' : '#cbd5e1'} />
                    </button>
                  )
                })()}
              </div>

              {/* Right group: Speed Badge & YouTube Settings Button */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span
                  style={{
                    color: '#38bdf8',
                    fontWeight: 700,
                    background: 'rgba(56, 189, 248, 0.12)',
                    padding: '2px 6px',
                    borderRadius: '4px',
                    border: '1px solid rgba(56, 189, 248, 0.25)',
                    fontSize: '11px',
                  }}
                  title="Tốc độ phát hiện tại"
                >
                  {playbackRate}x
                </span>

                <button
                  ref={settingsBtnRef}
                  type="button"
                  onClick={() => setShowSettingsPopover(!showSettingsPopover)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: showSettingsPopover ? 'rgba(0, 229, 255, 0.25)' : 'rgba(255, 255, 255, 0.08)',
                    color: showSettingsPopover ? '#00e5ff' : '#cbd5e1',
                    border: showSettingsPopover ? '1px solid #00e5ff' : '1px solid rgba(255, 255, 255, 0.15)',
                    borderRadius: '4px',
                    padding: '4px 8px',
                    cursor: 'pointer',
                  }}
                  title="Cài đặt tốc độ và số giây tua video (kiểu YouTube)"
                >
                  <SettingsIcon size={15} color={showSettingsPopover ? '#00e5ff' : '#cbd5e1'} />
                </button>

                <button
                  type="button"
                  onClick={toggleFullscreen}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    background: isFullscreen ? 'rgba(0, 229, 255, 0.25)' : 'rgba(255, 255, 255, 0.08)',
                    color: isFullscreen ? '#00e5ff' : '#cbd5e1',
                    border: isFullscreen ? '1px solid #00e5ff' : '1px solid rgba(255, 255, 255, 0.15)',
                    borderRadius: '4px',
                    padding: '4px 8px',
                    cursor: 'pointer',
                  }}
                  title={isFullscreen ? 'Thu nhỏ lại (Phím F / Esc)' : 'Phóng to toàn màn hình (Phím F / Double click)'}
                >
                  {isFullscreen ? <MinimizeIcon size={14} color="#00e5ff" /> : <FullscreenIcon size={14} color="#cbd5e1" />}
                </button>
              </div>

              {/* YouTube-style Settings Popover Card (Compact & Sleek) */}
              {showSettingsPopover && (
                <div
                  ref={settingsPopoverRef}
                  className="youtube-settings-popover"
                  style={{
                    position: 'absolute',
                    bottom: 'calc(100% + 6px)',
                    right: '6px',
                    width: '235px',
                    background: 'rgba(15, 23, 42, 0.96)',
                    backdropFilter: 'blur(12px)',
                    border: '1px solid rgba(0, 229, 255, 0.35)',
                    borderRadius: '6px',
                    boxShadow: '0 8px 28px rgba(0, 0, 0, 0.85)',
                    padding: '8px 10px',
                    zIndex: 100,
                    color: '#f1f5f9',
                    fontSize: '10.5px',
                  }}
                >
                  {/* Popover Header */}
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                      paddingBottom: '5px',
                      marginBottom: '6px',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '5px', fontWeight: 700, fontSize: '11px', color: '#00e5ff' }}>
                      <SettingsIcon size={12} color="#00e5ff" />
                      <span>Cài đặt video</span>
                    </div>
                    <button
                      type="button"
                      onClick={() => setShowSettingsPopover(false)}
                      style={{
                        background: 'transparent',
                        border: 'none',
                        color: '#94a3b8',
                        cursor: 'pointer',
                        fontSize: '12px',
                        fontWeight: 700,
                        lineHeight: 1,
                        padding: '0 2px',
                      }}
                      title="Đóng"
                    >
                      ✕
                    </button>
                  </div>

                  {/* Section 1: Playback Speed */}
                  <div style={{ marginBottom: '6px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px', fontSize: '10px', fontWeight: 600, color: '#94a3b8' }}>
                      <span>Tốc độ:</span>
                      <span style={{ color: '#00e5ff', fontWeight: 700 }}>{playbackRate}x</span>
                    </div>
                    <div style={{ display: 'flex', gap: '3px' }}>
                      {[0.5, 1.0, 1.5, 2.0, 3.0].map((rate) => (
                        <button
                          key={`popover-speed-${rate}`}
                          type="button"
                          onClick={() => updateSpeed(rate)}
                          style={{
                            flex: 1,
                            background: playbackRate === rate ? 'rgba(0, 229, 255, 0.3)' : 'rgba(255, 255, 255, 0.07)',
                            color: playbackRate === rate ? '#00e5ff' : '#cbd5e1',
                            border: playbackRate === rate ? '1px solid #00e5ff' : '1px solid rgba(255, 255, 255, 0.1)',
                            borderRadius: '3px',
                            padding: '2px 0',
                            fontSize: '9.5px',
                            fontWeight: 600,
                            cursor: 'pointer',
                          }}
                        >
                          {rate}x
                        </button>
                      ))}
                      <button
                        type="button"
                        onClick={() => setIsCustomSpeedOpen(!isCustomSpeedOpen)}
                        style={{
                          background: isCustomSpeedOpen ? 'rgba(0, 229, 255, 0.35)' : 'rgba(255, 255, 255, 0.07)',
                          color: isCustomSpeedOpen ? '#00e5ff' : '#94a3b8',
                          border: isCustomSpeedOpen ? '1px solid #00e5ff' : '1px solid rgba(255, 255, 255, 0.1)',
                          borderRadius: '3px',
                          padding: '2px 5px',
                          fontSize: '9.5px',
                          fontWeight: 600,
                          cursor: 'pointer',
                        }}
                        title="Tùy chọn tốc độ khác"
                      >
                        ✏️
                      </button>
                    </div>

                    {/* Inline Custom Speed Input */}
                    {isCustomSpeedOpen && (
                      <div style={{ display: 'flex', gap: '4px', marginTop: '4px' }}>
                        <input
                          type="number"
                          step="0.1"
                          min="0.1"
                          max="16"
                          placeholder="Tốc độ (VD: 1.75)"
                          value={customSpeedInput}
                          onChange={(e) => setCustomSpeedInput(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' && customSpeedInput) {
                              const val = parseFloat(customSpeedInput)
                              if (!isNaN(val) && val > 0) {
                                updateSpeed(val)
                                setCustomSpeedInput('')
                                setIsCustomSpeedOpen(false)
                              }
                            }
                          }}
                          autoFocus
                          style={{
                            flex: 1,
                            background: 'rgba(0, 0, 0, 0.4)',
                            border: '1px solid rgba(0, 229, 255, 0.3)',
                            borderRadius: '3px',
                            color: '#fff',
                            padding: '2px 5px',
                            fontSize: '10px',
                          }}
                        />
                        <button
                          type="button"
                          onClick={() => {
                            if (customSpeedInput) {
                              const val = parseFloat(customSpeedInput)
                              if (!isNaN(val) && val > 0) {
                                updateSpeed(val)
                                setCustomSpeedInput('')
                                setIsCustomSpeedOpen(false)
                              }
                            }
                          }}
                          style={{
                            background: 'rgba(0, 229, 255, 0.2)',
                            color: '#00e5ff',
                            border: '1px solid #00e5ff',
                            borderRadius: '3px',
                            padding: '2px 6px',
                            fontSize: '9.5px',
                            fontWeight: 700,
                            cursor: 'pointer',
                          }}
                        >
                          Lưu
                        </button>
                      </div>
                    )}
                  </div>

                  {/* Section 2: Seek Jump Step */}
                  <div style={{ marginBottom: '6px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px', fontSize: '10px', fontWeight: 600, color: '#94a3b8' }}>
                      <span>Tua nhanh (phím ←/→):</span>
                      <span style={{ color: '#00e5ff', fontWeight: 700 }}>±{seekStepSeconds}s</span>
                    </div>
                    <div style={{ display: 'flex', gap: '3px' }}>
                      {[1, 2, 5, 10, 15].map((sec) => (
                        <button
                          key={`popover-seek-${sec}`}
                          type="button"
                          onClick={() => updateSeekStep(sec)}
                          style={{
                            flex: 1,
                            background: seekStepSeconds === sec ? 'rgba(0, 229, 255, 0.3)' : 'rgba(255, 255, 255, 0.07)',
                            color: seekStepSeconds === sec ? '#00e5ff' : '#cbd5e1',
                            border: seekStepSeconds === sec ? '1px solid #00e5ff' : '1px solid rgba(255, 255, 255, 0.1)',
                            borderRadius: '3px',
                            padding: '2px 0',
                            fontSize: '9.5px',
                            fontWeight: 600,
                            cursor: 'pointer',
                          }}
                        >
                          {sec}s
                        </button>
                      ))}
                      <button
                        type="button"
                        onClick={() => setIsCustomSeekOpen(!isCustomSeekOpen)}
                        style={{
                          background: isCustomSeekOpen ? 'rgba(0, 229, 255, 0.35)' : 'rgba(255, 255, 255, 0.07)',
                          color: isCustomSeekOpen ? '#00e5ff' : '#94a3b8',
                          border: isCustomSeekOpen ? '1px solid #00e5ff' : '1px solid rgba(255, 255, 255, 0.1)',
                          borderRadius: '3px',
                          padding: '2px 5px',
                          fontSize: '9.5px',
                          fontWeight: 600,
                          cursor: 'pointer',
                        }}
                        title="Tùy chọn số giây tua khác"
                      >
                        ✏️
                      </button>
                    </div>

                    {/* Inline Custom Seek Input */}
                    {isCustomSeekOpen && (
                      <div style={{ display: 'flex', gap: '4px', marginTop: '4px' }}>
                        <input
                          type="number"
                          step="1"
                          min="1"
                          max="300"
                          placeholder="Số giây (VD: 7)"
                          value={customSeekInput}
                          onChange={(e) => setCustomSeekInput(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' && customSeekInput) {
                              const val = parseInt(customSeekInput, 10)
                              if (!isNaN(val) && val > 0) {
                                updateSeekStep(val)
                                setCustomSeekInput('')
                                setIsCustomSeekOpen(false)
                              }
                            }
                          }}
                          autoFocus
                          style={{
                            flex: 1,
                            background: 'rgba(0, 0, 0, 0.4)',
                            border: '1px solid rgba(0, 229, 255, 0.3)',
                            borderRadius: '3px',
                            color: '#fff',
                            padding: '2px 5px',
                            fontSize: '10px',
                          }}
                        />
                        <button
                          type="button"
                          onClick={() => {
                            if (customSeekInput) {
                              const val = parseInt(customSeekInput, 10)
                              if (!isNaN(val) && val > 0) {
                                updateSeekStep(val)
                                setCustomSeekInput('')
                                setIsCustomSeekOpen(false)
                              }
                            }
                          }}
                          style={{
                            background: 'rgba(0, 229, 255, 0.2)',
                            color: '#00e5ff',
                            border: '1px solid #00e5ff',
                            borderRadius: '3px',
                            padding: '2px 6px',
                            fontSize: '9.5px',
                            fontWeight: 700,
                            cursor: 'pointer',
                          }}
                        >
                          Lưu
                        </button>
                      </div>
                    )}
                  </div>

                  {/* Section 3: Hover Preview Scrubber Toggle */}
                  <div
                    style={{
                      borderTop: '1px solid rgba(255, 255, 255, 0.08)',
                      paddingTop: '5px',
                      marginBottom: '5px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                    }}
                  >
                    <span style={{ fontSize: '10px', color: '#cbd5e1', fontWeight: 500 }}>Lia chuột xem cảnh:</span>
                    <button
                      type="button"
                      onClick={() => {
                        const next = !showHoverScrubber
                        setShowHoverScrubber(next)
                        localStorage.setItem('tv5_show_hover_scrubber', String(next))
                      }}
                      style={{
                        background: showHoverScrubber ? 'rgba(0, 229, 255, 0.25)' : 'rgba(255, 255, 255, 0.1)',
                        color: showHoverScrubber ? '#00e5ff' : '#94a3b8',
                        border: showHoverScrubber ? '1px solid #00e5ff' : '1px solid rgba(255, 255, 255, 0.15)',
                        borderRadius: '10px',
                        padding: '1px 6px',
                        fontSize: '9px',
                        fontWeight: 700,
                        cursor: 'pointer',
                      }}
                    >
                      {showHoverScrubber ? 'BẬT' : 'TẮT'}
                    </button>
                  </div>

                  {/* Section 4: Autoplay Toggle */}
                  <div
                    style={{
                      borderTop: '1px solid rgba(255, 255, 255, 0.08)',
                      paddingTop: '5px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                    }}
                  >
                    <span style={{ fontSize: '10px', color: '#cbd5e1', fontWeight: 500 }}>Tự động phát (AutoPlay):</span>
                    <button
                      type="button"
                      onClick={() => updateAutoplay(!autoPlayOnSelect)}
                      style={{
                        background: autoPlayOnSelect ? 'rgba(0, 229, 255, 0.25)' : 'rgba(255, 255, 255, 0.1)',
                        color: autoPlayOnSelect ? '#00e5ff' : '#94a3b8',
                        border: autoPlayOnSelect ? '1px solid #00e5ff' : '1px solid rgba(255, 255, 255, 0.15)',
                        borderRadius: '10px',
                        padding: '1px 6px',
                        fontSize: '9px',
                        fontWeight: 700,
                        cursor: 'pointer',
                      }}
                    >
                      {autoPlayOnSelect ? 'BẬT' : 'TẮT'}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* B. Exact Canonical Frame */}
          <div className="viewport-panel frame-viewport">
            <div className="viewport-panel-header">
              <div className="panel-title-group">
                <FrameIcon size={16} className="text-cyan" />
                <span className="panel-title-text">Exact Canonical Frame</span>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span
                  className={`selectability-badge ${
                    isBoundary
                      ? 'badge-boundary'
                      : isSelectable
                      ? 'badge-selectable'
                      : isFixture
                      ? 'badge-fixture-preview'
                      : 'badge-non-selectable'
                  }`}
                  data-testid="submission-selectable-badge"
                >
                  {isBoundary ? (
                    <>
                      <WarningIcon size={12} /> Boundary
                    </>
                  ) : isSelectable ? (
                    <>
                      <CheckIcon size={12} /> Selectable
                    </>
                  ) : isFixture ? (
                    <>
                      <WarningIcon size={12} /> Preview
                    </>
                  ) : (
                    <>
                      <WarningIcon size={12} /> Non-Selectable
                    </>
                  )}
                </span>
              </div>
            </div>

            <div className="exact-frame-image-stage">
              {isImageLoading && (
                <div className="frame-loading-curtain">
                  <SpinnerIcon size={24} className="icon-spin text-cyan" />
                  <span>Decoding frame PTS...</span>
                </div>
              )}

              {!frameImgFailed && exactImageBlobUrl ? (
                <img
                  src={exactImageBlobUrl}
                  alt={`Exact frame ${inspectedFrameId}`}
                  className="exact-frame-img"
                  data-testid="exact-frame-image"
                  decoding="async"
                  onError={() => setFrameImgFailed(true)}
                />
              ) : !frameImgFailed && isFixture ? (
                <img
                  src={getFixturePreviewDataUri(
                    targetCandidate.video_id,
                    inspectedFrameId ?? targetCandidate.frame_id
                  )}
                  alt={`Exact frame ${inspectedFrameId}`}
                  className="exact-frame-img"
                  data-testid="exact-frame-image"
                  decoding="async"
                  onError={() => setFrameImgFailed(true)}
                />
              ) : (
                <div className="frame-error-matte">
                  <span>Exact frame preview unavailable</span>
                  <span className="tabular-nums">Frame {inspectedFrameId ?? 'UNKNOWN'}</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* C. Video Scenes Timeline & Exact Neighbors Panel (Bottom) */}
        <div className="exact-neighbors-panel" data-testid="context-strip" style={{ transition: 'all 0.2s ease' }}>
          <div className="neighbors-panel-header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '6px 12px' }}>
            {/* Left: Tab Selectors */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <button
                type="button"
                onClick={() => { setActiveBottomTab('scenes'); setIsBottomCollapsed(false); }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '5px',
                  background: activeBottomTab === 'scenes' && !isBottomCollapsed ? 'rgba(0, 229, 255, 0.18)' : 'rgba(255, 255, 255, 0.05)',
                  color: activeBottomTab === 'scenes' && !isBottomCollapsed ? '#00e5ff' : '#94a3b8',
                  border: activeBottomTab === 'scenes' && !isBottomCollapsed ? '1px solid #00e5ff' : '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: '4px',
                  padding: '4px 10px',
                  fontSize: '11px',
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
                title="Xem toàn bộ các cảnh / keyframe trong video này và click nhảy cảnh tức thì"
              >
                <FilmstripIcon size={13} color={activeBottomTab === 'scenes' && !isBottomCollapsed ? '#00e5ff' : '#94a3b8'} />
                <span>Toàn bộ cảnh video ({videoKeyframes.length})</span>
              </button>

              <button
                type="button"
                onClick={() => { setActiveBottomTab('neighbors'); setIsBottomCollapsed(false); }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '5px',
                  background: activeBottomTab === 'neighbors' && !isBottomCollapsed ? 'rgba(0, 229, 255, 0.18)' : 'rgba(255, 255, 255, 0.05)',
                  color: activeBottomTab === 'neighbors' && !isBottomCollapsed ? '#00e5ff' : '#94a3b8',
                  border: activeBottomTab === 'neighbors' && !isBottomCollapsed ? '1px solid #00e5ff' : '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: '4px',
                  padding: '4px 10px',
                  fontSize: '11px',
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
                title="Soi từng frame liền kề (±1, ±2 frame)"
              >
                <span>Soi frame lân cận</span>
              </button>
            </div>

            {/* Right: Actions & Collapse Button */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              {/* Stepping controls for neighbors tab */}
              {activeBottomTab === 'neighbors' && !isBottomCollapsed && (
                <div className="neighbors-transport-controls" style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                  <button
                    type="button"
                    onClick={() => {
                      if (targetCandidate) {
                        const playbackFid = currentPlaybackFrame
                        const playbackTs = Math.round(playbackTime * 1000)
                        dispatch({
                          type: 'SELECT_CANDIDATE',
                          payload: {
                            ...targetCandidate,
                            frame_id: playbackFid,
                            timestamp_ms: playbackTs,
                            certified_anchor_frame_id: playbackFid,
                            certified_anchor_timestamp_ms: playbackTs,
                            anchor_offset: 0,
                            cumulative_offset: 0,
                          },
                        })
                      }
                    }}
                    style={{
                      background: 'rgba(0, 229, 255, 0.12)',
                      color: '#00e5ff',
                      border: '1px solid rgba(0, 229, 255, 0.35)',
                      borderRadius: '4px',
                      padding: '3px 8px',
                      fontSize: '10.5px',
                      fontWeight: 600,
                      cursor: 'pointer',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '4px',
                    }}
                    title="Đặt frame đang phát trên video làm tâm điểm soi lân cận"
                  >
                    <span>🎯 Mốc F.{currentPlaybackFrame}</span>
                  </button>

                  <div style={{ width: '1px', height: '14px', background: 'rgba(255, 255, 255, 0.15)', margin: '0 2px' }} />

                  <button
                    type="button"
                    className="step-btn"
                    onClick={() => handleStep(cumulativeOffset - 1)}
                    disabled={isStepping}
                    title="Lùi 1 frame (Ctrl+← / ,)"
                  >
                    <StepPrevIcon size={13} />
                    <span>Lùi</span>
                  </button>

                  <button
                    type="button"
                    className="step-btn anchor-btn"
                    onClick={() => handleStep(0)}
                    disabled={isStepping || cumulativeOffset === 0}
                    title="Về Anchor ban đầu"
                  >
                    <AnchorIcon size={13} />
                  </button>

                  <button
                    type="button"
                    className="step-btn"
                    onClick={() => handleStep(cumulativeOffset + 1)}
                    disabled={isStepping}
                    title="Tiến 1 frame (Ctrl+→ / .)"
                  >
                    <span>Tiến</span>
                    <StepNextIcon size={13} />
                  </button>

                  <div className="offset-indicator-chip tabular-nums text-cyan" style={{ fontSize: '11px', padding: '2px 6px' }}>
                    {cumulativeOffset === 0 ? '0' : cumulativeOffset > 0 ? `+${cumulativeOffset}` : `${cumulativeOffset}`}
                  </div>
                </div>
              )}

              {/* Tab 1 (Scenes timeline) Controls: Về Anchor and Use This Frame */}
              {activeBottomTab === 'scenes' && selectedTimelineKeyframe !== null && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                  {/* Button 1: Jump back to original anchor */}
                  <button
                    type="button"
                    onClick={jumpToAnchor}
                    style={{
                      background: 'rgba(0, 229, 255, 0.12)',
                      color: '#00e5ff',
                      border: '1px solid rgba(0, 229, 255, 0.35)',
                      borderRadius: '4px',
                      padding: '3px 8px',
                      fontSize: '11px',
                      fontWeight: 600,
                      cursor: 'pointer',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '4px',
                    }}
                    title={`Quay về Frame Anchor ban đầu (${anchorFid})`}
                  >
                    <AnchorIcon size={12} color="#00e5ff" />
                    <span>Về Anchor (F.{anchorFid})</span>
                  </button>

                  {/* Button 2: Use This Frame / Commit as new canonical prediction */}
                  {taskMode === 'KIS' && selectedTimelineKeyframe.frame_id !== anchorFid && (
                    <button
                      type="button"
                      onClick={() => {
                        dispatch({
                          type: 'SELECT_CANDIDATE',
                          payload: {
                            ...targetCandidate,
                            frame_id: selectedTimelineKeyframe.frame_id,
                            timestamp_ms: selectedTimelineKeyframe.timestamp_ms,
                            anchor_offset: 0,
                            cumulative_offset: 0,
                          },
                        })
                        setSelectedTimelineKeyframe(null)
                        telemetry.record({
                          action: 'COMMIT_TIMELINE_KEYFRAME',
                          taskMode: 'KIS',
                          videoId: targetCandidate.video_id,
                          frameId: selectedTimelineKeyframe.frame_id,
                        })
                      }}
                      style={{
                        background: 'rgba(0, 229, 255, 0.22)',
                        border: '1px solid #00e5ff',
                        color: '#00e5ff',
                        padding: '3px 9px',
                        borderRadius: '4px',
                        fontSize: '11px',
                        fontWeight: 700,
                        cursor: 'pointer',
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '4px',
                      }}
                      title={`Xác nhận chọn Frame ${selectedTimelineKeyframe.frame_id} làm Anchor / Frame nộp bài mới`}
                    >
                      <CheckIcon size={12} color="#00e5ff" />
                      <span>Use This Frame (F.{selectedTimelineKeyframe.frame_id})</span>
                    </button>
                  )}
                </div>
              )}

              {/* Exact Canonical Commit Button for KIS on Tab 2 (Neighbors) */}
              {activeBottomTab === 'neighbors' && taskMode === 'KIS' && inspectedFrameId !== undefined && inspectedFrameId !== targetCandidate.frame_id && (
                <button
                  type="button"
                  className="btn-use-for-event"
                  onClick={() => {
                    if (inspectedFrameId === undefined) return
                    const proof = currentStep?.frame ?? undefined
                    dispatch({
                      type: 'COMMIT_KIS_FRAME',
                      payload: {
                        frame_id: inspectedFrameId,
                        timestamp_ms: inspectedTimestampMs,
                        proof,
                      },
                    })
                  }}
                  title={`Commit Frame ${inspectedFrameId} as canonical prediction for ${targetCandidate.video_id}`}
                >
                  <CheckIcon size={13} />
                  <span>Use This Frame</span>
                </button>
              )}

              {/* Add to Submission Basket Button for KIS */}
              {taskMode === 'KIS' && (() => {
                const finalFid = inspectedFrameId ?? targetCandidate.frame_id
                const isInBasket = submissionBasket.some(
                  (b) => b.video_id === targetCandidate.video_id && b.frame_id === finalFid && (b.task || 'KIS') === 'KIS'
                )
                return (
                  <button
                    type="button"
                    className="btn-use-for-event"
                    onClick={() => {
                      if (isInBasket) {
                        dispatch({
                          type: 'REMOVE_FROM_BASKET',
                          payload: { video_id: targetCandidate.video_id, frame_id: finalFid },
                        })
                        telemetry.record({
                          action: 'REMOVE_FROM_BASKET',
                          taskMode: 'KIS',
                          videoId: targetCandidate.video_id,
                          frameId: finalFid,
                        })
                      } else {
                        const finalTime = inspectedTimestampMs ?? targetCandidate.timestamp_ms
                        dispatch({
                          type: 'ADD_KIS_TO_BASKET',
                          payload: {
                            candidate: {
                              ...targetCandidate,
                              frame_id: finalFid,
                              timestamp_ms: finalTime,
                            },
                          },
                        })
                        telemetry.record({
                          action: 'ADD_KIS_TO_BASKET_FROM_INSPECTION',
                          taskMode: 'KIS',
                          videoId: targetCandidate.video_id,
                          frameId: finalFid,
                        })
                      }
                    }}
                    title={isInBasket ? `Frame ${finalFid} đã trong giỏ (Click để gỡ)` : `Thêm Frame ${finalFid} vào giỏ nộp bài`}
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      background: isInBasket ? 'rgba(16, 185, 129, 0.22)' : 'rgba(0, 229, 255, 0.15)',
                      borderColor: isInBasket ? '#10b981' : 'var(--color-cyan, #00e5ff)',
                      color: isInBasket ? '#10b981' : 'var(--color-cyan, #00e5ff)',
                      padding: '4px 8px',
                      borderRadius: '4px',
                      cursor: 'pointer',
                    }}
                  >
                    {isInBasket ? <CheckIcon size={14} color="#10b981" /> : <BasketPlusIcon size={14} />}
                  </button>
                )
              })()}

              {/* Collapse Toggle Button */}
              <button
                type="button"
                onClick={() => setIsBottomCollapsed(!isBottomCollapsed)}
                style={{
                  background: 'rgba(255, 255, 255, 0.06)',
                  color: '#94a3b8',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: '4px',
                  padding: '4px 8px',
                  fontSize: '10px',
                  cursor: 'pointer',
                  fontWeight: 600,
                }}
                title={isBottomCollapsed ? 'Mở rộng dải phim' : 'Thu gọn dải phim để xem video to hơn'}
              >
                {isBottomCollapsed ? '▲ Mở rộng' : '▼ Thu gọn'}
              </button>
            </div>
          </div>

          {/* Content (Rendered only when not collapsed) */}
          {!isBottomCollapsed && (
            <div style={{ padding: '8px 12px 10px' }}>
              {activeBottomTab === 'scenes' ? (
                /* Tab 1: Video Keyframes Timeline Strip */
                <div
                  ref={scenesStripRef}
                  className="video-scenes-strip"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    overflowX: 'auto',
                    paddingBottom: '6px',
                    scrollbarWidth: 'thin',
                  }}
                >
                  {isLoadingKeyframes ? (
                    <div style={{ color: '#94a3b8', fontSize: '11px', padding: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <SpinnerIcon size={16} />
                      <span>Đang tải các cảnh trong video...</span>
                    </div>
                  ) : videoKeyframes.length === 0 ? (
                    <div style={{ color: '#64748b', fontSize: '11px', padding: '16px' }}>
                      Không có keyframe nào được tìm thấy cho video này.
                    </div>
                  ) : (
                    videoKeyframes.map((kf) => {
                      const isAnchor = kf.frame_id === anchorFid
                      const isSelectedPreview = selectedTimelineKeyframe !== null && kf.frame_id === selectedTimelineKeyframe.frame_id
                      const isTarget = isSelectedPreview || (selectedTimelineKeyframe === null && (kf.frame_id === (bestMatchingKeyframe?.frame_id ?? targetFid) || isAnchor))
                      const mins = Math.floor(kf.timestamp_ms / 60000)
                      const secs = String(Math.floor((kf.timestamp_ms % 60000) / 1000)).padStart(2, '0')
                      return (
                        <div
                          key={`kf-${kf.frame_id}`}
                          className={`keyframe-timeline-card ${isTarget ? 'is-target-keyframe' : ''}`}
                          onClick={() => handleSelectTimelineKeyframe(kf)}
                          style={{
                            flex: '0 0 130px',
                            cursor: 'pointer',
                            borderRadius: '6px',
                            overflow: 'hidden',
                            background: isSelectedPreview
                              ? 'rgba(56, 189, 248, 0.25)'
                              : isAnchor
                              ? 'rgba(0, 229, 255, 0.2)'
                              : 'rgba(15, 23, 42, 0.6)',
                            border: isSelectedPreview
                              ? '2px solid #38bdf8'
                              : isAnchor
                              ? '2px solid #00e5ff'
                              : '1px solid rgba(255, 255, 255, 0.1)',
                            boxShadow: isSelectedPreview
                              ? '0 0 14px rgba(56, 189, 248, 0.65)'
                              : isAnchor
                              ? '0 0 12px rgba(0, 229, 255, 0.5)'
                              : 'none',
                            transition: 'all 0.15s ease',
                          }}
                          title={
                            isAnchor
                              ? `Frame Anchor gốc ${kf.frame_id} (${mins}:${secs})`
                              : isSelectedPreview
                              ? `Đang xem trước Frame ${kf.frame_id} (${mins}:${secs}) - Bấm "Use This Frame" để lưu`
                              : `Click để nhảy đến xem Frame ${kf.frame_id} (${mins}:${secs})`
                          }
                        >
                          <div style={{ width: '100%', height: '74px', position: 'relative', background: '#000' }}>
                            <img
                              src={
                                kf.keyframe_path
                                  ? `http://127.0.0.1:8000/keyframe-image/${targetCandidate.video_id}/${kf.keyframe_path.split('/').pop()}`
                                  : getThumbnailUrl(targetCandidate.video_id, kf.frame_id)
                              }
                              alt={`Frame ${kf.frame_id}`}
                              style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                              loading="lazy"
                              onError={(e) => {
                                const target = e.currentTarget
                                if (!target.src.includes('/frames/')) {
                                  target.src = getSelectedFrameUrl(targetCandidate.video_id, kf.frame_id)
                                }
                              }}
                            />
                            <span
                              style={{
                                position: 'absolute',
                                bottom: '3px',
                                right: '4px',
                                background: 'rgba(0,0,0,0.75)',
                                color: '#fff',
                                fontSize: '9px',
                                padding: '1px 4px',
                                borderRadius: '2px',
                                fontFamily: 'monospace',
                              }}
                            >
                              {mins}:{secs}
                            </span>
                            {/* Quick Basket button on Keyframe */}
                            {(() => {
                              const isKfInBasket = submissionBasket.some(
                                (b) => b.video_id === targetCandidate.video_id && b.frame_id === kf.frame_id
                              )
                              return (
                                <button
                                  type="button"
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    handleToggleKeyframeBasket(kf)
                                  }}
                                  style={{
                                    position: 'absolute',
                                    top: '3px',
                                    right: '3px',
                                    background: isKfInBasket ? 'rgba(16, 185, 129, 0.9)' : 'rgba(15, 23, 42, 0.75)',
                                    color: isKfInBasket ? '#fff' : '#00e5ff',
                                    border: isKfInBasket ? '1px solid #10b981' : '1px solid rgba(0, 229, 255, 0.4)',
                                    borderRadius: '3px',
                                    padding: '1px 4px',
                                    fontSize: '9px',
                                    fontWeight: 700,
                                    cursor: 'pointer',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '2px',
                                  }}
                                  title={isKfInBasket ? 'Đã lưu trong giỏ (Click để xóa)' : 'Thêm Keyframe này vào giỏ'}
                                >
                                  {isKfInBasket ? '✓ Giỏ' : '+ Giỏ'}
                                </button>
                              )
                            })()}
                          </div>
                          <div
                            style={{
                              padding: '3px 6px',
                              display: 'flex',
                              justifyContent: 'space-between',
                              alignItems: 'center',
                              fontSize: '10px',
                              color: isSelectedPreview ? '#38bdf8' : isAnchor ? '#00e5ff' : '#cbd5e1',
                              fontWeight: isAnchor || isSelectedPreview ? 700 : 500,
                            }}
                          >
                            <span>F.{kf.frame_id}</span>
                            <div style={{ display: 'flex', gap: '3px' }}>
                              {isAnchor && (
                                <span style={{ fontSize: '9px', background: '#00e5ff', color: '#000', padding: '0 4px', borderRadius: '2px', fontWeight: 700 }}>
                                  Anchor
                                </span>
                              )}
                              {isSelectedPreview && !isAnchor && (
                                <span style={{ fontSize: '9px', background: '#38bdf8', color: '#000', padding: '0 4px', borderRadius: '2px', fontWeight: 700 }}>
                                  Active
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                      )
                    })
                  )}
                </div>
              ) : (
                /* Tab 2: Original Canonical Neighbors 5-Card Filmstrip */
                <div className="neighbors-thumbnails-track">
                  {[-2, -1, 0, 1, 2].map((relOff) => {
                    const effOff = cumulativeOffset + relOff
                    const stepData = exactNeighbors?.steps?.find((s) => s.offset === effOff)
                    const fid = stepData?.frame?.frame_id
                    const isCurrent = effOff === cumulativeOffset
                    const isAnchor = effOff === 0
                    const isDegraded = !stepData || !stepData.frame || stepData.degraded_reason !== null

                    return (
                      <div
                        key={`neighbor-card-${effOff}`}
                        className={`neighbor-card ${isCurrent ? 'neighbor-current' : ''} ${
                          isAnchor ? 'neighbor-anchor' : ''
                        } ${isDegraded ? 'neighbor-degraded' : ''}`}
                        onClick={() => !isDegraded && handleStep(effOff)}
                        role="button"
                        tabIndex={0}
                        data-testid={`neighbor-card-${effOff}`}
                        style={{ position: 'relative', cursor: isDegraded ? 'not-allowed' : 'pointer' }}
                        title={stepData?.frame ? `Click để chuyển sang Frame ${fid} (Offset ${effOff >= 0 ? `+${effOff}` : effOff})` : 'Unavailable'}
                      >
                        <div className="neighbor-thumb-box" style={{ position: 'relative' }}>
                          <ExactNeighborThumb
                            videoId={targetCandidate.video_id}
                            anchorFrameId={anchorCandidate?.frame_id ?? targetCandidate.frame_id}
                            anchorTimestampMs={anchorCandidate?.timestamp_ms ?? targetCandidate.timestamp_ms}
                            certifiedAnchorFrameId={
                              anchorCandidate?.certified_anchor_frame_id ??
                              targetCandidate.certified_anchor_frame_id ??
                              anchorCandidate?.frame_id ??
                              targetCandidate.frame_id
                            }
                            certifiedAnchorTimestampMs={
                              anchorCandidate?.certified_anchor_timestamp_ms ??
                              targetCandidate.certified_anchor_timestamp_ms ??
                              anchorCandidate?.timestamp_ms ??
                              targetCandidate.timestamp_ms
                            }
                            anchorOffset={anchorCandidate?.anchor_offset ?? targetCandidate.anchor_offset ?? 0}
                            cumulativeOffset={cumulativeOffset}
                            relOffset={relOff}
                            stepData={stepData}
                            isFixture={isFixture}
                          />

                          {/* Quick Basket button on Neighbor Card */}
                          {stepData?.frame && (() => {
                            const nFid = stepData.frame.frame_id
                            const isNeighborInBasket = submissionBasket.some(
                              (b) => b.video_id === targetCandidate.video_id && b.frame_id === nFid
                            )
                            return (
                              <button
                                type="button"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  if (isNeighborInBasket) {
                                    dispatch({
                                      type: 'REMOVE_FROM_BASKET',
                                      payload: { video_id: targetCandidate.video_id, frame_id: nFid },
                                    })
                                  } else {
                                    dispatch({
                                      type: 'ADD_KIS_TO_BASKET',
                                      payload: {
                                        candidate: {
                                          ...targetCandidate,
                                          frame_id: nFid,
                                          timestamp_ms: stepData.frame?.timestamp_ms ?? targetCandidate.timestamp_ms,
                                        },
                                      },
                                    })
                                  }
                                }}
                                style={{
                                  position: 'absolute',
                                  top: '2px',
                                  right: '2px',
                                  background: isNeighborInBasket ? 'rgba(16, 185, 129, 0.9)' : 'rgba(15, 23, 42, 0.75)',
                                  color: isNeighborInBasket ? '#fff' : '#00e5ff',
                                  border: isNeighborInBasket ? '1px solid #10b981' : '1px solid rgba(0, 229, 255, 0.4)',
                                  borderRadius: '3px',
                                  padding: '1px 4px',
                                  fontSize: '9px',
                                  fontWeight: 700,
                                  cursor: 'pointer',
                                  zIndex: 2,
                                }}
                                title={isNeighborInBasket ? 'Đã lưu (Click để gỡ)' : 'Lưu Frame này vào giỏ'}
                              >
                                {isNeighborInBasket ? '✓ Giỏ' : '+ Giỏ'}
                              </button>
                            )
                          })()}
                        </div>
                        <div className="neighbor-meta tabular-nums">
                          <span className="meta-fid-title">
                            {stepData?.frame
                              ? `Frame ${fid}`
                              : !stepData
                              ? 'Loading...'
                              : 'Unavailable'}
                          </span>
                          {isAnchor && <span className="meta-role-pill pill-anchor">Anchor</span>}
                          {isCurrent && !isAnchor && (
                            <span className="meta-role-pill pill-current">Current</span>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Main Q&A Working Region: ONLY rendered in VQA mode */}
        {taskMode === 'VQA' && (
          <div className="inspection-main-qa-region" data-testid="inspection-main-qa-region">
            <div className="qa-evidence-column">
              <EvidenceInspector evidence={vqaActiveResult?.evidence || null} />
            </div>
            <div className="qa-answer-column">
              <VqaAnswerPanel />
            </div>
          </div>
        )}
      </main>

      {/* ── 3. Right Technical Panel (~260px) ── */}
      <aside className="inspection-technical-panel" data-testid="canonical-identity-card">
        <div className="panel-header">
          <span className="panel-title">Proof & Identity</span>
        </div>

        <div className="tech-fields-grid">
          <div className="tech-field">
            <span className="field-label">Video ID</span>
            <span className="field-value text-cyan" data-testid="inspected-video-id">
              {targetCandidate.video_id}
            </span>
          </div>

          <div className="tech-field">
            <span className="field-label">FPS</span>
            <span className="field-value text-cyan tabular-nums" data-testid="inspected-fps">
              {videoMeta?.fps ? `${videoMeta.fps} fps` : '25.0 fps'}
            </span>
          </div>

          <div className="tech-field highlight-field" style={{ borderLeft: '3px solid #ffab00' }}>
            <span className="field-label" style={{ color: '#ffab00' }}>Current Video Frame</span>
            <span className="field-value tabular-nums" style={{ color: '#ffab00', fontWeight: 700 }}>
              {currentPlaybackFrame}
            </span>
          </div>

          <div className="tech-field highlight-field">
            <span className="field-label">Selected Frame ID</span>
            <span className="field-value tabular-nums" data-testid="inspected-frame-id">
              {inspectedFrameId ?? 'UNKNOWN'}
            </span>
          </div>

          <div className="tech-field">
            <span className="field-label">Timestamp</span>
            <span className="field-value tabular-nums" data-testid="inspected-timestamp-ms">
              {inspectedTimestampMs !== undefined ? `${inspectedTimestampMs} ms` : 'N/A'}
            </span>
          </div>

          <div className="tech-field">
            <span className="field-label">PTS</span>
            <span className="field-value tabular-nums">
              {currentStep?.frame?.pts ?? 'ANCHOR'}
            </span>
          </div>

          {taskMode === 'TRAKE' && activeTrakeSlot && (
            <div className="tech-field highlight-field" data-testid="trake-assigned-slot-box">
              <span className="field-label">Assigned Event</span>
              <span className="field-value text-cyan" data-testid="trake-slot-assigned-info">
                #{activeTrakeSlot.event_index + 1}: {activeTrakeSlot.event_label}
              </span>
            </div>
          )}
        </div>

        {/* Primary Basket Action Buttons (Save Video Playback Frame or Selected Frame) */}
        <div style={{ marginTop: '14px', marginBottom: '14px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <button
            type="button"
            onClick={handleToggleCurrentPlaybackFrameBasket}
            style={{
              padding: '8px 12px',
              borderRadius: '6px',
              background: isCurrentPlaybackInBasket
                ? 'rgba(16, 185, 129, 0.22)'
                : 'linear-gradient(135deg, rgba(0, 229, 255, 0.22), rgba(0, 180, 216, 0.32))',
              border: isCurrentPlaybackInBasket ? '1px solid #10b981' : '1px solid #00e5ff',
              color: isCurrentPlaybackInBasket ? '#10b981' : '#00e5ff',
              fontWeight: 700,
              fontSize: '12px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '6px',
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.35)',
              transition: 'all 0.15s ease',
            }}
            title="Phím B: Lưu hoặc Xóa Frame đang phát trên video vào giỏ nộp bài"
          >
            {isCurrentPlaybackInBasket ? <CheckIcon size={14} color="#10b981" /> : <BasketPlusIcon size={14} />}
            <span>
              {isCurrentPlaybackInBasket
                ? `✓ Frame ${currentPlaybackFrame} trong Giỏ (Bấm để gỡ)`
                : `📥 Lưu Frame đang phát (F.${currentPlaybackFrame}) vào Giỏ`}
            </span>
          </button>

          <button
            type="button"
            onClick={handleToggleSelectedFrameBasket}
            style={{
              padding: '6px 10px',
              borderRadius: '6px',
              background: isSelectedFrameInBasket ? 'rgba(16, 185, 129, 0.18)' : 'rgba(255, 255, 255, 0.06)',
              border: isSelectedFrameInBasket ? '1px solid #10b981' : '1px solid rgba(255, 255, 255, 0.15)',
              color: isSelectedFrameInBasket ? '#10b981' : '#cbd5e1',
              fontWeight: 600,
              fontSize: '11px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '6px',
              transition: 'all 0.15s ease',
            }}
            title="Lưu Frame được chọn qua Soi Frame vào giỏ nộp bài"
          >
            {isSelectedFrameInBasket ? <CheckIcon size={12} color="#10b981" /> : <BasketPlusIcon size={12} />}
            <span>
              {isSelectedFrameInBasket
                ? `✓ Frame ${inspectedFrameId} trong Giỏ (Bấm để gỡ)`
                : `Lưu Frame đã chọn (F.${inspectedFrameId}) vào Giỏ`}
            </span>
          </button>
        </div>

        {/* Collapsible Technical Proof Details */}
        <div className="proof-details-disclosure">
          <button
            type="button"
            className="disclosure-toggle"
            onClick={() => setShowProofDetails(!showProofDetails)}
          >
            <span>{showProofDetails ? '▾ Hide Proof Details' : '▸ Technical Proof Details'}</span>
          </button>

          {showProofDetails && (
            <div className="disclosure-body tabular-nums">
              <div>FPS Nominal: {videoMeta?.fps_nominal ?? '25.0'}</div>
              <div>FPS Average: {videoMeta?.fps_average ?? '25.0'}</div>
              {videoMeta?.width_px && videoMeta?.height_px && (
                <div>Resolution: {videoMeta.width_px}x{videoMeta.height_px}</div>
              )}
              {videoMeta?.frame_count && (
                <div>Total Frames: {videoMeta.frame_count}</div>
              )}
              <div>Time Base: {currentStep?.frame?.time_base || videoMeta?.time_base || '1/12800'}</div>
              <div>Run ID: {currentStep?.frame?.preprocess_run_id || 'run_v1_batch1'}</div>
              {currentStep?.frame?.certification_id && (
                <div>Cert ID: {currentStep.frame.certification_id}</div>
              )}
            </div>
          )}
        </div>

        {/* Warning Banner for Boundary / Degradation */}
        {(stepError || imageError || isBoundary) && (
          <div className="tech-alert-banner" data-testid="step-alert-banner">
            <WarningIcon size={14} />
            <span>
              {stepError || imageError || currentStep?.degraded_reason || 'Boundary reached'}
            </span>
          </div>
        )}
      </aside>
    </div>
  )
}
