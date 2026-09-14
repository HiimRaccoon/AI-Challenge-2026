import { useEffect, useRef } from 'react'
import { useAppDispatch, useAppState } from './AppContext'
import { fetchExactNeighbors } from '../api/tv4Client'
import { loadExactImage, clearExactImageCacheForAnchor } from '../api/exactImageCache'
import { generateFixtureNeighbors, getFixturePreviewDataUri } from '../fixtures/fixtureData'

/**
 * Single authoritative frontend orchestration owner for:
 * candidate selection -> exact-neighbor fetch -> current exact-image fetch -> stale-request protection -> shared state update.
 */
export function useExactInspectionCoordinator(): void {
  const {
    mode,
    anchorCandidate,
    cumulativeOffset,
  } = useAppState()
  const dispatch = useAppDispatch()

  const reqSeq = useRef<number>(0)

  useEffect(() => {
    if (!anchorCandidate) {
      return
    }

    const rootAnchorFrameId =
      anchorCandidate.certified_anchor_frame_id ?? anchorCandidate.frame_id
    const rootAnchorTimestampMs =
      anchorCandidate.certified_anchor_timestamp_ms ?? anchorCandidate.timestamp_ms
    const baseOffset = anchorCandidate.anchor_offset ?? 0
    const totalCumulativeOffset = baseOffset + cumulativeOffset

    const currentAnchorKey = `${anchorCandidate.video_id}:${rootAnchorFrameId}`

    // 1. Fixture Mode: Isolated deterministic preview simulation
    if (mode === 'fixture') {
      const fixtureNeighbors = generateFixtureNeighbors(anchorCandidate, cumulativeOffset)
      dispatch({ type: 'EXACT_STEP_SUCCESS', payload: fixtureNeighbors })

      const inspectedFid = Math.max(0, anchorCandidate.frame_id + cumulativeOffset)
      const previewUrl = getFixturePreviewDataUri(anchorCandidate.video_id, inspectedFid)
      dispatch({
        type: 'EXACT_IMAGE_SUCCESS',
        payload: {
          blobUrl: previewUrl,
          headers: {
            video_id: anchorCandidate.video_id,
            frame_id: inspectedFid,
            pts: inspectedFid * 512,
            time_base: '1/12800',
            timestamp_ms: anchorCandidate.timestamp_ms + cumulativeOffset * 40,
            preprocess_run_id: 'fixture_preview_run_v1',
            certification_id: 'fixture-preview-non-canonical',
            submission_selectable: false,
          },
        },
      })
      return
    }

    // 2. Live Mode: Single Authoritative Fetch Lifecycle with Stale Protection
    reqSeq.current += 1
    const seq = reqSeq.current

    // Clear old cache when switching anchors
    clearExactImageCacheForAnchor(currentAnchorKey)

    dispatch({ type: 'EXACT_IMAGE_START' })

    const neighborPayload = {
      video_id: anchorCandidate.video_id,
      frame_id: rootAnchorFrameId,
      timestamp_ms: rootAnchorTimestampMs,
      certified_anchor_frame_id: rootAnchorFrameId,
      certified_anchor_timestamp_ms: rootAnchorTimestampMs,
      cumulative_offset: totalCumulativeOffset,
      offsets: [-2, -1, 0, 1, 2],
    }

    fetchExactNeighbors(neighborPayload)
      .then(async (neighborRes) => {
        if (seq !== reqSeq.current) return

        // Normalize step offsets so they match UI coordinates (relative to candidate + cumulativeOffset)
        const normalizedSteps = neighborRes.steps.map((s) => ({
          ...s,
          offset: s.offset - baseOffset,
        }))
        const normalizedRes = {
          ...neighborRes,
          anchor_frame_id: anchorCandidate.frame_id,
          steps: normalizedSteps,
        }

        dispatch({ type: 'EXACT_STEP_SUCCESS', payload: normalizedRes })

        // Find authoritative step for the currently inspected offset
        const currentStep = normalizedSteps.find((s) => s.offset === cumulativeOffset)
        if (!currentStep?.frame) {
          // A 200 neighbors response can truthfully contain a degraded step
          // (for example WP09's mapping_mismatch).  It has no proof record,
          // so an exact JPEG request would necessarily return 409.  Surface
          // the reason locally instead of sending a knowingly invalid call.
          dispatch({
            type: 'EXACT_IMAGE_FAILURE',
            payload: currentStep?.degraded_reason || neighborRes.degraded_reason || 'Exact frame image unavailable',
          })
          return
        }

        const expectedFrame = {
          frame_id: currentStep.frame.frame_id,
          video_id: currentStep.frame.video_id,
        }

        try {
          const imgRes = await loadExactImage(
            {
              video_id: anchorCandidate.video_id,
              frame_id: rootAnchorFrameId,
              timestamp_ms: rootAnchorTimestampMs,
              certified_anchor_frame_id: rootAnchorFrameId,
              certified_anchor_timestamp_ms: rootAnchorTimestampMs,
              cumulative_offset: totalCumulativeOffset,
              offsets: [0], // exactly one offset relative to totalCumulativeOffset
            },
            expectedFrame
          )

          if (seq !== reqSeq.current) return
          dispatch({ type: 'EXACT_IMAGE_SUCCESS', payload: imgRes })
        } catch (imgErr: any) {
          if (seq !== reqSeq.current) return
          dispatch({
            type: 'EXACT_IMAGE_FAILURE',
            payload: imgErr.message || 'Exact frame image unavailable',
          })
        }
      })
      .catch((err: any) => {
        if (seq !== reqSeq.current) return
        dispatch({
          type: 'EXACT_STEP_FAILURE',
          payload: err.message || 'Exact frame step failed',
        })
        dispatch({
          type: 'EXACT_IMAGE_FAILURE',
          payload: err.message || 'Exact frame image unavailable',
        })
      })
  }, [
    mode,
    anchorCandidate?.video_id,
    anchorCandidate?.certified_anchor_frame_id ?? anchorCandidate?.frame_id,
    anchorCandidate?.certified_anchor_timestamp_ms ?? anchorCandidate?.timestamp_ms,
    anchorCandidate?.anchor_offset,
    cumulativeOffset,
    dispatch,
  ])
}
