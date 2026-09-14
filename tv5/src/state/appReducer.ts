import {
  BasketItem,
  ExactImageResult,
  ExactNeighborResponse,
  FeedbackResponse,
  KisResponse,
  ReadinessStatus,
  SearchCandidate,
  SystemMode,
  TV4HealthResponse,
  VqaResponse,
  VqaResult,
} from '../types/contracts'
import { AppState, WorkspaceTab } from './appState'
import { generateFixtureNeighbors } from '../fixtures/fixtureData'

export type AppAction =
  | { type: 'SET_ACTIVE_TAB'; payload: WorkspaceTab }
  | { type: 'SET_TASK_MODE'; payload: import('./appState').TaskMode }
  | { type: 'SET_MODE'; payload: SystemMode }
  | { type: 'SET_HEALTH'; payload: { health: TV4HealthResponse | null; error?: string } }
  | { type: 'SET_READINESS'; payload: ReadinessStatus }
  | { type: 'SET_QUERY_TEXT'; payload: string }
  | { type: 'SET_TOP_K'; payload: number }
  | { type: 'SET_MODALITY_WEIGHTS'; payload: Partial<import('./appState').ModalityWeights> }
  | { type: 'RESET_MODALITY_WEIGHTS' }
  | { type: 'KIS_SEARCH_START' }
  | { type: 'KIS_SEARCH_SUCCESS'; payload: KisResponse }
  | { type: 'KIS_SEARCH_FAILURE'; payload: string }
  | { type: 'SELECT_CANDIDATE'; payload: SearchCandidate }
  | { type: 'EXACT_STEP_START'; payload: { offset: number } }
  | { type: 'EXACT_STEP_SUCCESS'; payload: ExactNeighborResponse }
  | { type: 'EXACT_STEP_FAILURE'; payload: string }
  | { type: 'EXACT_IMAGE_START' }
  | { type: 'EXACT_IMAGE_SUCCESS'; payload: ExactImageResult }
  | { type: 'EXACT_IMAGE_FAILURE'; payload: string }
  | { type: 'RESET_TO_ANCHOR' }
  // Feedback Actions (T020/T028 Seam)
  | { type: 'SET_FEEDBACK_REFERENCE'; payload: SearchCandidate | null }
  | { type: 'SET_FEEDBACK_DRAFT'; payload: string }
  | { type: 'FEEDBACK_START_PENDING' }
  | { type: 'FEEDBACK_START_SUCCESS'; payload: FeedbackResponse }
  | { type: 'FEEDBACK_START_FAILURE'; payload: string }
  | { type: 'FEEDBACK_REFINE_PENDING' }
  | { type: 'FEEDBACK_REFINE_SUCCESS'; payload: FeedbackResponse }
  | { type: 'FEEDBACK_REFINE_FAILURE'; payload: string }
  | { type: 'FEEDBACK_UNDO_PENDING' }
  | { type: 'FEEDBACK_UNDO_SUCCESS'; payload: FeedbackResponse }
  | { type: 'FEEDBACK_UNDO_FAILURE'; payload: string }
  | { type: 'FEEDBACK_RESET_PENDING' }
  | { type: 'FEEDBACK_RESET_SUCCESS'; payload: FeedbackResponse }
  | { type: 'FEEDBACK_RESET_FAILURE'; payload: string }
  | { type: 'FEEDBACK_CLEAR' }
  // Basket Actions
  | { type: 'ADD_TO_BASKET'; payload: BasketItem }
  | { type: 'ADD_KIS_TO_BASKET'; payload?: { candidate?: SearchCandidate } }
  | { type: 'CLEAR_BASKET' }
  | { type: 'REMOVE_FROM_BASKET'; payload: { video_id: string; frame_id: number } }
  // Pin / Staging Tray Actions
  | { type: 'TOGGLE_PIN_CANDIDATE'; payload: SearchCandidate }
  | { type: 'PIN_CANDIDATE'; payload: SearchCandidate }
  | { type: 'UNPIN_CANDIDATE'; payload: { video_id: string; frame_id?: number } }
  | { type: 'CLEAR_PINNED_CANDIDATES' }
  | { type: 'TOGGLE_KEYBOARD_HELP' }
  | { type: 'SET_KEYBOARD_HELP'; payload: boolean }
  // VQA Actions (WP11 / T016 / T030 / T031)
  | { type: 'SET_VQA_QUESTION'; payload: string }
  | { type: 'VQA_SEARCH_START' }
  | { type: 'VQA_SEARCH_SUCCESS'; payload: VqaResponse }
  | { type: 'VQA_SEARCH_FAILURE'; payload: string }
  | { type: 'SELECT_VQA_RESULT'; payload: VqaResult }
  | { type: 'SET_VQA_DRAFT_ANSWER'; payload: string }
  | { type: 'CONFIRM_VQA_ANSWER' }
  | { type: 'CLEAR_VQA_APPROVAL' }
  | { type: 'ADD_VQA_TO_BASKET' }
  // TRAKE Actions (WP12 / T032 / T033)
  | { type: 'SET_TRAKE_EVENTS'; payload: string[] }
  | { type: 'TRAKE_SEARCH_START' }
  | { type: 'TRAKE_SEARCH_SUCCESS'; payload: import('../types/contracts').TrakeResponse }
  | { type: 'TRAKE_SEARCH_FAILURE'; payload: string }
  | { type: 'TRAKE_EVENT_RERUN_START'; payload: { event_index: number } }
  | {
      type: 'TRAKE_EVENT_RERUN_SUCCESS'
      payload: import('../types/contracts').TrakeRerunResponse
    }
  | { type: 'TRAKE_EVENT_RERUN_FAILURE'; payload: string }
  | { type: 'SELECT_TRAKE_HYPOTHESIS'; payload: number }
  | {
      type: 'SELECT_TRAKE_EVENT_ALTERNATIVE'
      payload: { event_index: number; alternative_index: number }
    }
  | { type: 'LOCK_TRAKE_SLOT'; payload: { event_index: number } }
  | { type: 'UNLOCK_TRAKE_SLOT'; payload: { event_index: number } }
  | {
      type: 'CORRECT_TRAKE_SLOT'
      payload: {
        event_index: number
        frame_id: number
        timestamp_ms?: number
        proof?: import('../types/contracts').ExactFrameProof
      }
    }
  | {
      type: 'COMMIT_KIS_FRAME'
      payload: {
        frame_id: number
        timestamp_ms?: number
        proof?: import('../types/contracts').ExactFrameProof
      }
    }
  | {
      type: 'COMMIT_VQA_FRAME'
      payload: {
        frame_id: number
        timestamp_ms?: number
        proof?: import('../types/contracts').ExactFrameProof
      }
    }
  | { type: 'SELECT_TRAKE_SLOT'; payload: number | null }
  | { type: 'ADD_TRAKE_TO_BASKET' }
  // Session Recovery
  | { type: 'RESTORE_SESSION_STATE'; payload: Partial<AppState> }

export function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_ACTIVE_TAB': {
      return {
        ...state,
        activeTab: action.payload,
      }
    }

    case 'SET_TASK_MODE': {
      const nextMode = action.payload
      if (nextMode === state.taskMode) return state

      // Re-bind active inspection candidate strictly to the target mode's current item
      let nextActiveCand: SearchCandidate | null = null
      if (nextMode === 'TRAKE') {
        const slotIdx = state.trakeActiveSlotIndex ?? (state.trakeSlots.length > 0 ? 0 : null)
        const slot = slotIdx !== null ? state.trakeSlots[slotIdx] : null
        if (slot && slot.video_id && slot.frame_id !== null) {
          nextActiveCand = {
            query_id: state.queryId || 'trake',
            video_id: slot.video_id,
            frame_id: slot.frame_id,
            timestamp_ms: slot.timestamp_ms || 0,
            source: 'trake',
            rank: slot.event_index + 1,
            score: slot.score,
            certified_anchor_frame_id: slot.certified_anchor_frame_id ?? slot.frame_id,
            certified_anchor_timestamp_ms: slot.certified_anchor_timestamp_ms ?? slot.timestamp_ms ?? 0,
            anchor_offset: slot.anchor_offset ?? 0,
            cumulative_offset: 0,
          }
        }
      } else if (nextMode === 'VQA') {
        if (state.vqaActiveResult) {
          nextActiveCand = {
            query_id: state.queryId || 'vqa',
            video_id: state.vqaActiveResult.video_id,
            frame_id: state.vqaActiveResult.frame_id,
            timestamp_ms: state.vqaActiveResult.timestamp_ms || 0,
            source: 'vqa',
            rank: state.vqaActiveResult.rank,
            score: state.vqaActiveResult.confidence,
            certified_anchor_frame_id: state.vqaActiveResult.certified_anchor_frame_id ?? state.vqaActiveResult.frame_id,
            certified_anchor_timestamp_ms: state.vqaActiveResult.certified_anchor_timestamp_ms ?? state.vqaActiveResult.timestamp_ms ?? 0,
            anchor_offset: state.vqaActiveResult.anchor_offset ?? 0,
            cumulative_offset: 0,
          }
        }
      } else {
        // KIS mode: strictly bind to KIS candidate state
        if (state.candidates.length > 0) {
          const validKisCand = state.kisActiveCandidate
            ? state.candidates.find(
                (c) =>
                  c.video_id === state.kisActiveCandidate?.video_id &&
                  c.frame_id === state.kisActiveCandidate?.frame_id
              ) || state.candidates[0]
            : state.candidates[0]
          nextActiveCand = validKisCand
        }
      }

      return {
        ...state,
        taskMode: nextMode,
        activeCandidate: nextActiveCand,
        anchorCandidate: nextActiveCand,
        cumulativeOffset: 0,
        currentStep: null,
        exactNeighbors: null,
        exactImageBlobUrl: null,
        exactImageHeaders: null,
      }
    }

    case 'SET_MODE': {
      return {
        ...state,
        mode: action.payload,
      }
    }

    case 'SET_HEALTH': {
      const { health, error } = action.payload
      let readiness: ReadinessStatus = state.readiness
      if (error || !health) {
        readiness = 'OFFLINE'
      } else if (health.mode === 'fixture') {
        readiness = 'PARTIAL'
      } else if (health.status === 'ok') {
        readiness = 'PARTIAL'
      } else {
        readiness = 'DEGRADED'
      }

      return {
        ...state,
        tv4Health: health,
        healthError: error ?? null,
        mode: health?.mode ?? state.mode,
        preprocessRunId: health?.preprocess_run_id ?? null,
        readiness,
      }
    }

    case 'SET_READINESS': {
      return {
        ...state,
        readiness: action.payload,
      }
    }

    case 'SET_QUERY_TEXT': {
      let trakeValidationStatus = state.trakeValidationStatus
      let trakeHasSearched = state.trakeHasSearched
      if (state.taskMode === 'TRAKE' && action.payload !== state.queryText && state.trakeSlots.length > 0) {
        trakeValidationStatus = 'incomplete'
        trakeHasSearched = false
      }
      return {
        ...state,
        queryText: action.payload,
        trakeValidationStatus,
        trakeHasSearched,
      }
    }

    case 'SET_TOP_K': {
      const clamped = Math.max(1, Math.min(100, action.payload))
      return {
        ...state,
        topK: clamped,
      }
    }

    case 'SET_MODALITY_WEIGHTS': {
      return {
        ...state,
        modalityWeights: {
          ...state.modalityWeights,
          ...action.payload,
        },
      }
    }

    case 'RESET_MODALITY_WEIGHTS': {
      return {
        ...state,
        modalityWeights: {
          visual: 100,
          ocr: 100,
          asr: 100,
          object: 50,
        },
      }
    }

    case 'KIS_SEARCH_START': {
      return {
        ...state,
        taskMode: 'KIS',
        isSearching: true,
        searchError: null,
      }
    }

    case 'KIS_SEARCH_SUCCESS': {
      const candidatesList = action.payload.candidates || []
      const defaultActive = candidatesList.length > 0 ? candidatesList[0] : null
      return {
        ...state,
        isSearching: false,
        searchError: null,
        queryId: action.payload.query_id,
        candidates: candidatesList,
        originalCandidates: candidatesList,
        kisActiveCandidate: defaultActive,
        activeCandidate: state.taskMode === 'KIS' ? defaultActive : state.activeCandidate,
        anchorCandidate: state.taskMode === 'KIS' ? defaultActive : state.anchorCandidate,
        // Reset feedback state on new KIS query
        isFeedbackActive: false,
        feedbackSessionId: null,
        feedbackOriginalQuery: null,
        feedbackRevision: 0,
        feedbackReference: null,
        feedbackDraftText: '',
        feedbackError: null,
      }
    }

    case 'KIS_SEARCH_FAILURE': {
      return {
        ...state,
        isSearching: false,
        searchError: action.payload,
      }
    }

    case 'SELECT_CANDIDATE': {
      // Invariant: Selecting a candidate establishes anchor and resets cumulativeOffset to 0
      const initialNeighbors =
        state.mode === 'fixture'
          ? generateFixtureNeighbors(action.payload, 0)
          : null
      const initialStep =
        initialNeighbors?.steps.find((s) => s.offset === 0) || null

      const matchedVqa = state.vqaResults.find(
        (r) => r.video_id === action.payload.video_id && r.frame_id === action.payload.frame_id
      )

      return {
        ...state,
        kisActiveCandidate: state.taskMode === 'KIS' ? action.payload : state.kisActiveCandidate,
        activeCandidate: action.payload,
        anchorCandidate: action.payload,
        cumulativeOffset: 0,
        isStepping: false,
        stepError: null,
        exactNeighbors: initialNeighbors,
        currentStep: initialStep,
        exactImageBlobUrl: null,
        exactImageHeaders: null,
        isImageLoading: false,
        imageError: null,
        // VQA Invariants: sync active VQA result and invalidate approval on candidate switch
        vqaActiveResult: matchedVqa || state.vqaActiveResult,
        vqaDraftAnswer: matchedVqa ? matchedVqa.proposal : state.vqaDraftAnswer,
        vqaApprovedAnswer: null, // Approval invalidated on candidate switch
      }
    }

    case 'EXACT_STEP_START': {
      return {
        ...state,
        cumulativeOffset: action.payload.offset,
        isStepping: true,
        stepError: null,
      }
    }

    case 'EXACT_STEP_SUCCESS': {
      const step = action.payload.steps.find((s) => s.offset === state.cumulativeOffset) || null
      return {
        ...state,
        exactNeighbors: action.payload,
        currentStep: step,
        isStepping: false,
        stepError: null,
        // Invalidate VQA approval if exact frame has moved away from certified anchor
        vqaApprovedAnswer: state.cumulativeOffset === 0 ? state.vqaApprovedAnswer : null,
      }
    }

    case 'EXACT_STEP_FAILURE': {
      return {
        ...state,
        isStepping: false,
        stepError: action.payload,
      }
    }

    case 'EXACT_IMAGE_START': {
      return {
        ...state,
        isImageLoading: true,
        imageError: null,
      }
    }

    case 'EXACT_IMAGE_SUCCESS': {
      return {
        ...state,
        isImageLoading: false,
        imageError: null,
        exactImageBlobUrl: action.payload.blobUrl,
        exactImageHeaders: action.payload.headers,
      }
    }

    case 'EXACT_IMAGE_FAILURE': {
      return {
        ...state,
        isImageLoading: false,
        imageError: action.payload,
      }
    }

    case 'RESET_TO_ANCHOR': {
      return {
        ...state,
        cumulativeOffset: 0,
      }
    }

    // -------------------------------------------------------------------------
    // Feedback State Transitions (WP13 T028 / T029)
    // -------------------------------------------------------------------------

    case 'SET_FEEDBACK_REFERENCE': {
      if (!action.payload) {
        return {
          ...state,
          feedbackReference: null,
        }
      }
      // Revalidation: Reference candidate must exist in active candidate pool
      const exists = state.candidates.some(
        (c) => c.video_id === action.payload!.video_id && c.frame_id === action.payload!.frame_id
      )
      if (!exists) {
        return {
          ...state,
          feedbackReference: null,
          feedbackError: 'Selected reference candidate is not in the active candidate pool',
        }
      }
      return {
        ...state,
        feedbackReference: action.payload,
        feedbackError: null,
      }
    }

    case 'SET_FEEDBACK_DRAFT': {
      return {
        ...state,
        feedbackDraftText: action.payload,
      }
    }

    case 'FEEDBACK_START_PENDING': {
      return {
        ...state,
        isFeedbackPending: true,
        feedbackError: null,
      }
    }

    case 'FEEDBACK_START_SUCCESS': {
      const resp = action.payload
      return {
        ...state,
        isFeedbackActive: true,
        isFeedbackPending: false,
        feedbackError: null,
        feedbackSessionId: resp.session_id,
        // Invariant: original query remains immutable
        feedbackOriginalQuery: state.feedbackOriginalQuery || state.queryText,
        feedbackRevision: resp.revision,
        feedbackActiveCount: resp.active_feedback_count ?? 0,
        feedbackMaxEvents: resp.max_active_feedback_events ?? 5,
        candidates: resp.candidates || [],
        originalCandidates:
          state.originalCandidates.length > 0 ? state.originalCandidates : state.candidates,
        feedbackExpiresAtUtc: resp.expires_at_utc ?? null,
      }
    }

    case 'FEEDBACK_START_FAILURE': {
      return {
        ...state,
        isFeedbackPending: false,
        feedbackError: action.payload,
      }
    }

    case 'FEEDBACK_REFINE_PENDING': {
      return {
        ...state,
        isFeedbackPending: true,
        feedbackError: null,
      }
    }

    case 'FEEDBACK_REFINE_SUCCESS': {
      const resp = action.payload
      // Revalidate active reference candidate in new candidate pool
      const updatedRef = state.feedbackReference
        ? resp.candidates.find(
            (c) =>
              c.video_id === state.feedbackReference?.video_id &&
              c.frame_id === state.feedbackReference?.frame_id
          ) || null
        : null

      return {
        ...state,
        isFeedbackPending: false,
        feedbackError: null,
        feedbackRevision: resp.revision,
        feedbackActiveCount: resp.active_feedback_count ?? Math.min(state.feedbackMaxEvents || 5, state.feedbackActiveCount + 1),
        feedbackMaxEvents: resp.max_active_feedback_events ?? 5,
        candidates: resp.candidates || [],
        feedbackReference: updatedRef,
        feedbackDraftText: '',
      }
    }

    case 'FEEDBACK_REFINE_FAILURE': {
      const isLimitErr =
        action.payload.includes('at most five active feedback events') ||
        action.payload.includes('Maximum 5 active refinements reached')
      return {
        ...state,
        isFeedbackPending: false,
        feedbackActiveCount: isLimitErr ? (state.feedbackMaxEvents || 5) : state.feedbackActiveCount,
        feedbackError: isLimitErr
          ? 'Maximum 5 active refinements reached. Undo or Reset to continue.'
          : action.payload,
      }
    }

    case 'FEEDBACK_UNDO_PENDING': {
      return {
        ...state,
        isFeedbackPending: true,
        feedbackError: null,
      }
    }

    case 'FEEDBACK_UNDO_SUCCESS': {
      const resp = action.payload
      const updatedRef = state.feedbackReference
        ? resp.candidates.find(
            (c) =>
              c.video_id === state.feedbackReference?.video_id &&
              c.frame_id === state.feedbackReference?.frame_id
          ) || null
        : null
      return {
        ...state,
        isFeedbackPending: false,
        feedbackError: null,
        feedbackRevision: resp.revision,
        feedbackActiveCount: resp.active_feedback_count ?? Math.max(0, state.feedbackActiveCount - 1),
        feedbackMaxEvents: resp.max_active_feedback_events ?? 5,
        candidates: resp.candidates || [],
        feedbackReference: updatedRef,
      }
    }

    case 'FEEDBACK_UNDO_FAILURE': {
      return {
        ...state,
        isFeedbackPending: false,
        feedbackError: action.payload,
      }
    }

    case 'FEEDBACK_RESET_PENDING': {
      return {
        ...state,
        isFeedbackPending: true,
        feedbackError: null,
      }
    }

    case 'FEEDBACK_RESET_SUCCESS': {
      const resp = action.payload
      return {
        ...state,
        isFeedbackPending: false,
        feedbackError: null,
        feedbackRevision: resp.revision,
        feedbackActiveCount: resp.active_feedback_count ?? 0,
        feedbackMaxEvents: resp.max_active_feedback_events ?? 5,
        candidates: resp.candidates || [],
        feedbackReference: null,
        feedbackDraftText: '',
      }
    }

    case 'FEEDBACK_RESET_FAILURE': {
      return {
        ...state,
        isFeedbackPending: false,
        feedbackError: action.payload,
      }
    }

    case 'FEEDBACK_CLEAR': {
      return {
        ...state,
        isFeedbackActive: false,
        feedbackSessionId: null,
        feedbackOriginalQuery: null,
        feedbackRevision: 0,
        feedbackActiveCount: 0,
        feedbackMaxEvents: 5,
        feedbackReference: null,
        feedbackDraftText: '',
        feedbackError: null,
        feedbackExpiresAtUtc: null,
        candidates: state.originalCandidates.length > 0 ? state.originalCandidates : state.candidates,
      }
    }

    // -------------------------------------------------------------------------
    // Basket Operations (Isolated from Feedback)
    // -------------------------------------------------------------------------

    case 'ADD_TO_BASKET': {
      const exists = state.submissionBasket.some(
        (b) => b.video_id === action.payload.video_id && b.frame_id === action.payload.frame_id
      )
      if (exists) return state
      if (state.submissionBasket.length >= 100) return state
      return {
        ...state,
        submissionBasket: [...state.submissionBasket, action.payload],
      }
    }

    case 'ADD_KIS_TO_BASKET': {
      const cand = action.payload?.candidate || state.activeCandidate || state.kisActiveCandidate
      if (!cand || !cand.video_id || cand.frame_id === undefined || cand.frame_id === null || cand.frame_id < 0) {
        return state
      }
      if (state.submissionBasket.length >= 100) {
        return state
      }
      if (!cand.query_id) return state
      const item: BasketItem = {
        query_id: cand.query_id,
        video_id: cand.video_id,
        frame_id: cand.frame_id,
        timestamp_ms: cand.timestamp_ms || 0,
        added_at_utc: new Date().toISOString(),
        task: 'KIS',
      }
      const exists = state.submissionBasket.some(
        (b) => b.video_id === item.video_id && b.frame_id === item.frame_id && b.task === 'KIS' && b.query_id === item.query_id
      )
      if (exists) return state
      return {
        ...state,
        submissionBasket: [...state.submissionBasket, item],
      }
    }

    case 'CLEAR_BASKET': {
      return {
        ...state,
        submissionBasket: [],
      }
    }

    case 'REMOVE_FROM_BASKET': {
      return {
        ...state,
        submissionBasket: state.submissionBasket.filter(
          (b) => !(b.video_id === action.payload.video_id && b.frame_id === action.payload.frame_id)
        ),
      }
    }

    case 'TOGGLE_PIN_CANDIDATE': {
      const cand = action.payload
      const exists = state.pinnedCandidates.some(
        (p) => p.video_id === cand.video_id && p.frame_id === cand.frame_id
      )
      if (exists) {
        return {
          ...state,
          pinnedCandidates: state.pinnedCandidates.filter(
            (p) => !(p.video_id === cand.video_id && p.frame_id === cand.frame_id)
          ),
        }
      }
      return {
        ...state,
        pinnedCandidates: [...state.pinnedCandidates, cand],
      }
    }

    case 'PIN_CANDIDATE': {
      const cand = action.payload
      const exists = state.pinnedCandidates.some(
        (p) => p.video_id === cand.video_id && p.frame_id === cand.frame_id
      )
      if (exists) return state
      return {
        ...state,
        pinnedCandidates: [...state.pinnedCandidates, cand],
      }
    }

    case 'UNPIN_CANDIDATE': {
      const { video_id, frame_id } = action.payload
      return {
        ...state,
        pinnedCandidates: state.pinnedCandidates.filter((p) => {
          if (frame_id !== undefined) {
            return !(p.video_id === video_id && p.frame_id === frame_id)
          }
          return p.video_id !== video_id
        }),
      }
    }

    case 'CLEAR_PINNED_CANDIDATES': {
      return {
        ...state,
        pinnedCandidates: [],
      }
    }

    case 'TOGGLE_KEYBOARD_HELP': {
      return {
        ...state,
        isKeyboardHelpOpen: !state.isKeyboardHelpOpen,
      }
    }

    case 'SET_KEYBOARD_HELP': {
      return {
        ...state,
        isKeyboardHelpOpen: action.payload,
      }
    }

    // -------------------------------------------------------------------------
    // VQA Operations (WP11 / T016 / T030 / T031)
    // -------------------------------------------------------------------------

    case 'SET_VQA_QUESTION': {
      return {
        ...state,
        vqaQuestion: action.payload,
        vqaApprovedAnswer: null, // Invalidate previous approval on question change
      }
    }

    case 'VQA_SEARCH_START': {
      return {
        ...state,
        taskMode: 'VQA',
        isVqaSearching: true,
        vqaError: null,
        vqaApprovedAnswer: null, // Invalidate previous approval
      }
    }

    case 'VQA_SEARCH_SUCCESS': {
      const results = action.payload.results || []
      const active = results.length > 0 ? results[0] : null
      let nextActiveCand: SearchCandidate | null = null
      if (active) {
        nextActiveCand = {
          query_id: action.payload.query_id || state.queryId || 'vqa',
          video_id: active.video_id,
          frame_id: active.frame_id,
          timestamp_ms: active.timestamp_ms || 0,
          source: 'vqa',
          rank: active.rank,
          score: active.confidence,
          certified_anchor_frame_id: active.certified_anchor_frame_id ?? active.frame_id,
          certified_anchor_timestamp_ms: active.certified_anchor_timestamp_ms ?? active.timestamp_ms ?? 0,
          anchor_offset: active.anchor_offset ?? 0,
          cumulative_offset: 0,
        }
      }
      return {
        ...state,
        isVqaSearching: false,
        vqaHasSearched: true,
        vqaResults: results,
        vqaRetrievalCandidates: action.payload.retrieval_candidates || [],
        vqaActiveResult: active,
        vqaDraftAnswer: active?.proposal || '',
        vqaApprovedAnswer: null, // CRITICAL P0 INVARIANT: Machine proposal != operator approval
        vqaError: null,
        activeCandidate: state.taskMode === 'VQA' ? nextActiveCand : state.activeCandidate,
        anchorCandidate: state.taskMode === 'VQA' ? nextActiveCand : state.anchorCandidate,
        cumulativeOffset: state.taskMode === 'VQA' ? 0 : state.cumulativeOffset,
        isStepping: false,
        stepError: null,
        exactNeighbors: state.taskMode === 'VQA' ? null : state.exactNeighbors,
        currentStep: state.taskMode === 'VQA' ? null : state.currentStep,
        exactImageBlobUrl: state.taskMode === 'VQA' ? null : state.exactImageBlobUrl,
        exactImageHeaders: state.taskMode === 'VQA' ? null : state.exactImageHeaders,
        isImageLoading: false,
        imageError: null,
      }
    }

    case 'VQA_SEARCH_FAILURE': {
      return {
        ...state,
        isVqaSearching: false,
        vqaHasSearched: true,
        vqaError: action.payload,
      }
    }

    case 'SELECT_VQA_RESULT': {
      const active = action.payload
      const rootAnchorFrameId =
        active.certified_anchor_frame_id ?? active.frame_id
      const rootAnchorTimestampMs =
        active.certified_anchor_timestamp_ms ?? active.timestamp_ms ?? 0
      const anchorOffset = active.anchor_offset ?? 0

      const cand: SearchCandidate = {
        query_id: state.queryId || 'vqa',
        video_id: active.video_id,
        frame_id: active.frame_id,
        timestamp_ms: active.timestamp_ms || 0,
        source: 'vqa',
        rank: active.rank,
        score: active.confidence,
        certified_anchor_frame_id: rootAnchorFrameId,
        certified_anchor_timestamp_ms: rootAnchorTimestampMs,
        anchor_offset: anchorOffset,
        cumulative_offset: 0,
      }
      return {
        ...state,
        vqaActiveResult: active,
        vqaDraftAnswer: active.proposal || '',
        vqaApprovedAnswer: null, // Invalidate previous approval on candidate switch
        activeCandidate: cand,
        anchorCandidate: cand,
        cumulativeOffset: 0,
        isStepping: false,
        stepError: null,
        exactNeighbors: null,
        currentStep: null,
        exactImageBlobUrl: null,
        exactImageHeaders: null,
        isImageLoading: false,
        imageError: null,
      }
    }

    case 'SET_VQA_DRAFT_ANSWER': {
      return {
        ...state,
        vqaDraftAnswer: action.payload,
      }
    }

    case 'CONFIRM_VQA_ANSWER': {
      if (!state.vqaDraftAnswer.trim() || !state.vqaActiveResult) {
        return state
      }
      return {
        ...state,
        vqaApprovedAnswer: state.vqaDraftAnswer.trim(), // Exact operator text preserved verbatim
      }
    }

    case 'CLEAR_VQA_APPROVAL': {
      return {
        ...state,
        vqaApprovedAnswer: null,
      }
    }

    case 'ADD_VQA_TO_BASKET': {
      // Invariants: only valid approved answer can enter basket; fixture cannot enter real basket
      if (!state.vqaApprovedAnswer || !state.vqaActiveResult || state.mode === 'fixture') {
        return state
      }
      const vqaQueryId =
        state.activeCandidate?.query_id ||
        state.queryId ||
        state.vqaActiveResult.evidence.query_id
      if (!vqaQueryId) return state
      const item: BasketItem = {
        query_id: vqaQueryId,
        video_id: state.vqaActiveResult.video_id,
        frame_id: state.vqaActiveResult.frame_id,
        timestamp_ms: state.vqaActiveResult.timestamp_ms,
        added_at_utc: new Date().toISOString(),
        task: 'VQA',
        answer: state.vqaApprovedAnswer,
      }
      const exists = state.submissionBasket.some(
        (b) =>
          b.video_id === item.video_id &&
          b.frame_id === item.frame_id &&
          b.answer === item.answer &&
          b.query_id === item.query_id
      )
      if (exists) return state
      return {
        ...state,
        submissionBasket: [...state.submissionBasket, item],
      }
    }

    // ── TRAKE Reducer Branches (WP12 / T032 / T033) ──
    case 'SET_TRAKE_EVENTS': {
      const events = action.payload
      const newSlots: import('../types/contracts').TrakeEventSlot[] = events.map((label, idx) => {
        const existing = state.trakeSlots[idx]
        if (existing && existing.locked && existing.event_label === label) {
          return existing
        }
        return {
          event_index: idx,
          event_label: label,
          video_id: null,
          frame_id: null,
          timestamp_ms: null,
          score: null,
          locked: false,
          validation_status: 'missing',
        }
      })
      return {
        ...state,
        trakeEvents: events,
        trakeResults: [],
        trakeActiveHypothesisIndex: null,
        trakeSlots: newSlots,
        trakeHasSearched: false,
        trakeActiveSlotIndex: events.length > 0 ? 0 : null,
        trakeValidationStatus: newSlots.length === 0 ? 'empty' : 'incomplete',
      }
    }

    case 'TRAKE_SEARCH_START': {
      return {
        ...state,
        taskMode: 'TRAKE',
        isTrakeSearching: true,
        trakeError: null,
      }
    }

    case 'TRAKE_SEARCH_SUCCESS': {
      const resp = action.payload
      if (!resp.result) {
        const resetSlots = state.trakeSlots.map((s) => ({
          ...s,
          video_id: null,
          frame_id: null,
          timestamp_ms: null,
          score: null,
          validation_status: 'missing' as const,
          certified_anchor_frame_id: null,
          certified_anchor_timestamp_ms: null,
          anchor_offset: 0,
        }))
        return {
          ...state,
          isTrakeSearching: false,
          trakeHasSearched: true,
          trakeVideoId: null,
          trakeQueryId: resp.query_id,
          trakeResults: [],
          trakeActiveHypothesisIndex: null,
          trakeSlots: resetSlots,
          trakeAggregateScore: null,
          trakeError: resp.message || 'No valid alignment found',
          trakeValidationStatus: 'empty',
        }
      }

      const res = resp.result
      const slotSource: import('../types/contracts').TrakeEventSlot[] =
        state.trakeSlots.length > 0
          ? state.trakeSlots
          : res.frame_ids.map((fid, i) => ({
              event_index: i,
              event_label: `Event ${i + 1}`,
              video_id: res.video_id,
              frame_id: fid,
              locked: false,
              validation_status: 'missing' as const,
            }))

      const updatedSlots: import('../types/contracts').TrakeEventSlot[] = slotSource.map((slot, idx) => {
        if (slot.locked && slot.frame_id !== null) {
          // Lock protection: retain existing frame
          const isMixed = slot.video_id !== null && slot.video_id !== res.video_id
          return {
            ...slot,
            validation_status: isMixed ? 'incompatible_video' : 'valid',
          }
        }
        const fid = res.frame_ids[idx] ?? null
        const ts =
          (res.timestamps_ms && res.timestamps_ms[idx] !== undefined)
            ? res.timestamps_ms[idx]
            : (res.candidates && res.candidates[idx]?.timestamp_ms !== undefined)
            ? res.candidates[idx].timestamp_ms
            : slot.timestamp_ms ?? 0
        const sc = res.event_scores[idx] ?? null
        const cand = res.candidates && res.candidates[idx]
        const eventResult = res.events?.find((event) => event.event_index === idx) ?? res.events?.[idx]
        return {
          ...slot,
          video_id: res.video_id,
          frame_id: fid,
          timestamp_ms: ts,
          score: sc,
          validation_status: fid !== null ? 'valid' : 'missing',
          certified_anchor_frame_id: cand?.certified_anchor_frame_id ?? fid,
          certified_anchor_timestamp_ms: cand?.certified_anchor_timestamp_ms ?? ts,
          anchor_offset: cand?.anchor_offset ?? 0,
          alternatives: eventResult?.alternatives ?? [],
        }
      })

      const hasMixed = updatedSlots.some((s) => s.validation_status === 'incompatible_video')
      const allValid =
        updatedSlots.length === (state.trakeEvents.length || res.frame_ids.length) &&
        updatedSlots.every((s) => s.frame_id !== null && s.validation_status === 'valid')

      let overallStatus: 'valid' | 'incomplete' | 'mixed_video' | 'empty' = 'incomplete'
      if (hasMixed) {
        overallStatus = 'mixed_video'
      } else if (allValid) {
        overallStatus = 'valid'
      }

      const activeIdx = state.trakeActiveSlotIndex ?? 0
      const activeSlot = updatedSlots[activeIdx] || (updatedSlots.length > 0 ? updatedSlots[0] : null)
      let nextActiveCand: SearchCandidate | null = null
      if (activeSlot && activeSlot.video_id && activeSlot.frame_id !== null) {
        nextActiveCand = {
          query_id: resp.query_id || state.queryId || 'trake',
          video_id: activeSlot.video_id,
          frame_id: activeSlot.frame_id,
          timestamp_ms: activeSlot.timestamp_ms || 0,
          source: 'trake',
          rank: activeSlot.event_index + 1,
          score: activeSlot.score,
          certified_anchor_frame_id: activeSlot.certified_anchor_frame_id ?? activeSlot.frame_id,
          certified_anchor_timestamp_ms: activeSlot.certified_anchor_timestamp_ms ?? activeSlot.timestamp_ms ?? 0,
          anchor_offset: activeSlot.anchor_offset ?? 0,
          cumulative_offset: 0,
        }
      }

      return {
        ...state,
        isTrakeSearching: false,
        trakeHasSearched: true,
        trakeVideoId: res.video_id,
        trakeQueryId: resp.query_id,
        trakeResults: resp.results && resp.results.length > 0 ? resp.results : [res],
        trakeActiveHypothesisIndex: 0,
        trakeSlots: updatedSlots,
        trakeAggregateScore: res.aggregate_score,
        trakeError: null,
        trakeValidationStatus: overallStatus,
        activeCandidate: state.taskMode === 'TRAKE' ? nextActiveCand : state.activeCandidate,
        anchorCandidate: state.taskMode === 'TRAKE' ? nextActiveCand : state.anchorCandidate,
        cumulativeOffset: state.taskMode === 'TRAKE' ? 0 : state.cumulativeOffset,
        exactNeighbors: state.taskMode === 'TRAKE' ? null : state.exactNeighbors,
        currentStep: state.taskMode === 'TRAKE' ? null : state.currentStep,
        exactImageBlobUrl: state.taskMode === 'TRAKE' ? null : state.exactImageBlobUrl,
        exactImageHeaders: state.taskMode === 'TRAKE' ? null : state.exactImageHeaders,
      }
    }

    case 'SELECT_TRAKE_HYPOTHESIS': {
      const hypothesis = state.trakeResults[action.payload]
      if (!hypothesis) return state
      const next = appReducer(state, {
        type: 'TRAKE_SEARCH_SUCCESS',
        payload: {
          query_id: state.trakeQueryId || state.queryId || 'trake',
          result: hypothesis,
          results: state.trakeResults,
        },
      })
      return {
        ...next,
        trakeActiveHypothesisIndex: action.payload,
      }
    }

    case 'SELECT_TRAKE_EVENT_ALTERNATIVE': {
      const { event_index, alternative_index } = action.payload
      const slot = state.trakeSlots[event_index]
      const alternative = slot?.alternatives?.[alternative_index]
      if (!slot || slot.locked || !alternative) return state

      const videoId = alternative.candidate?.video_id ?? slot.video_id ?? state.trakeVideoId
      if (!videoId || (state.trakeVideoId !== null && videoId !== state.trakeVideoId)) return state

      const previous = state.trakeSlots[event_index - 1]
      const next = state.trakeSlots[event_index + 1]
      if (
        (previous?.timestamp_ms != null && alternative.timestamp_ms <= previous.timestamp_ms) ||
        (next?.timestamp_ms != null && alternative.timestamp_ms >= next.timestamp_ms)
      ) {
        return state
      }

      const previousAlternative =
        slot.frame_id !== null && slot.timestamp_ms != null
          ? {
              frame_id: slot.frame_id,
              timestamp_ms: slot.timestamp_ms,
              score: slot.score,
            }
          : null
      const remainingAlternatives = (slot.alternatives ?? []).filter(
        (item) => item.frame_id !== alternative.frame_id
      )
      const alternatives = [
        ...(previousAlternative ? [previousAlternative] : []),
        ...remainingAlternatives,
      ].filter(
        (item, index, items) =>
          items.findIndex((candidate) => candidate.frame_id === item.frame_id) === index
      ).slice(0, 5)

      const updatedSlots = state.trakeSlots.map((item, index) =>
        index === event_index
          ? {
              ...item,
              video_id: videoId,
              frame_id: alternative.frame_id,
              timestamp_ms: alternative.timestamp_ms,
              score: alternative.score ?? alternative.candidate?.score ?? item.score,
              certified_anchor_frame_id:
                alternative.candidate?.certified_anchor_frame_id ?? alternative.frame_id,
              certified_anchor_timestamp_ms:
                alternative.candidate?.certified_anchor_timestamp_ms ?? alternative.timestamp_ms,
              anchor_offset: alternative.candidate?.anchor_offset ?? 0,
              alternatives,
              validation_status: 'valid' as const,
            }
          : item
      )
      const selected = updatedSlots[event_index]
      const activeCandidate =
        state.taskMode === 'TRAKE' && state.trakeActiveSlotIndex === event_index
          ? {
              query_id: state.trakeQueryId || state.queryId || 'trake',
              video_id: videoId,
              frame_id: alternative.frame_id,
              timestamp_ms: alternative.timestamp_ms,
              source: 'trake',
              rank: event_index + 1,
              score: selected.score,
              certified_anchor_frame_id: selected.certified_anchor_frame_id ?? alternative.frame_id,
              certified_anchor_timestamp_ms:
                selected.certified_anchor_timestamp_ms ?? alternative.timestamp_ms,
              anchor_offset: selected.anchor_offset ?? 0,
              cumulative_offset: 0,
            } satisfies SearchCandidate
          : state.activeCandidate

      return {
        ...state,
        trakeSlots: updatedSlots,
        trakeActiveHypothesisIndex: null,
        trakeAggregateScore: updatedSlots.reduce(
          (sum, item) => sum + (item.score ?? 0),
          0
        ),
        activeCandidate,
        anchorCandidate:
          state.taskMode === 'TRAKE' && state.trakeActiveSlotIndex === event_index
            ? activeCandidate
            : state.anchorCandidate,
        cumulativeOffset:
          state.taskMode === 'TRAKE' && state.trakeActiveSlotIndex === event_index
            ? 0
            : state.cumulativeOffset,
      }
    }

    case 'TRAKE_SEARCH_FAILURE': {
      return {
        ...state,
        isTrakeSearching: false,
        trakeHasSearched: true,
        trakeError: action.payload,
      }
    }

    case 'TRAKE_EVENT_RERUN_START': {
      const slot = state.trakeSlots[action.payload.event_index]
      if (!slot || slot.locked) return state
      return { ...state, trakeError: null }
    }

    case 'TRAKE_EVENT_RERUN_SUCCESS': {
      const { event_index, video_id, alternatives } = action.payload
      const slot = state.trakeSlots[event_index]
      if (!slot || slot.locked || video_id !== state.trakeVideoId) return state
      return {
        ...state,
        trakeSlots: state.trakeSlots.map((item, index) =>
          index === event_index ? { ...item, alternatives } : item
        ),
        trakeError: null,
      }
    }

    case 'TRAKE_EVENT_RERUN_FAILURE': {
      return { ...state, trakeError: action.payload }
    }

    case 'LOCK_TRAKE_SLOT': {
      const { event_index } = action.payload
      const updated = state.trakeSlots.map((s, idx) =>
        idx === event_index ? { ...s, locked: true } : s
      )
      return {
        ...state,
        trakeSlots: updated,
      }
    }

    case 'UNLOCK_TRAKE_SLOT': {
      const { event_index } = action.payload
      const updated = state.trakeSlots.map((s, idx) =>
        idx === event_index ? { ...s, locked: false } : s
      )
      return {
        ...state,
        trakeSlots: updated,
      }
    }

    case 'CORRECT_TRAKE_SLOT': {
      const { event_index, frame_id, timestamp_ms, proof } = action.payload
      const targetSlot = state.trakeSlots[event_index]
      if (targetSlot && targetSlot.locked) {
        // Locked slot cannot be modified without explicit unlock first
        return state
      }
      const targetVid = proof?.video_id || targetSlot?.video_id || state.trakeVideoId
      const isMixed =
        state.trakeVideoId !== null && targetVid !== null && targetVid !== state.trakeVideoId

      const certifiedRootAnchor =
        state.anchorCandidate?.certified_anchor_frame_id ??
        targetSlot?.certified_anchor_frame_id ??
        targetSlot?.frame_id ??
        frame_id
      const certifiedRootTimestamp =
        state.anchorCandidate?.certified_anchor_timestamp_ms ??
        targetSlot?.certified_anchor_timestamp_ms ??
        targetSlot?.timestamp_ms ??
        timestamp_ms ??
        0
      const currentCumulative = state.cumulativeOffset ?? 0
      const existingAnchorOffset =
        state.anchorCandidate?.anchor_offset ?? targetSlot?.anchor_offset ?? 0
      const newAnchorOffset = existingAnchorOffset + currentCumulative

      const updated = state.trakeSlots.map((s, idx) => {
        if (idx !== event_index) return s
        return {
          ...s,
          frame_id,
          timestamp_ms: timestamp_ms ?? proof?.timestamp_ms ?? s.timestamp_ms,
          video_id: targetVid,
          exact_proof: proof ?? s.exact_proof,
          validation_status: isMixed ? ('incompatible_video' as const) : ('valid' as const),
          certified_anchor_frame_id: certifiedRootAnchor,
          certified_anchor_timestamp_ms: certifiedRootTimestamp,
          anchor_offset: newAnchorOffset,
        }
      })

      const hasMixed = updated.some((s) => s.validation_status === 'incompatible_video')
      const allValid =
        updated.length === state.trakeEvents.length &&
        updated.every((s) => s.frame_id !== null && s.validation_status === 'valid')

      let overallStatus: 'valid' | 'incomplete' | 'mixed_video' | 'empty' = 'incomplete'
      if (hasMixed) {
        overallStatus = 'mixed_video'
      } else if (allValid) {
        overallStatus = 'valid'
      }

      // Authoritative sync: update active candidate if this corrected slot is currently active
      let updatedActiveCandidate = state.activeCandidate
      let updatedAnchorCandidate = state.anchorCandidate
      if (targetVid && state.trakeActiveSlotIndex === event_index) {
        const syncedCandidate: SearchCandidate = {
          query_id: state.queryId || 'trake',
          video_id: targetVid,
          frame_id: frame_id,
          timestamp_ms: timestamp_ms ?? proof?.timestamp_ms ?? 0,
          source: 'trake',
          rank: event_index + 1,
          score: targetSlot?.score ?? null,
          certified_anchor_frame_id: certifiedRootAnchor,
          certified_anchor_timestamp_ms: certifiedRootTimestamp,
          anchor_offset: newAnchorOffset,
          cumulative_offset: 0,
        }
        updatedActiveCandidate = syncedCandidate
        updatedAnchorCandidate = syncedCandidate
      }

      const initialNeighbors =
        state.mode === 'fixture' && updatedActiveCandidate
          ? generateFixtureNeighbors(updatedActiveCandidate, 0)
          : null
      const initialStep =
        initialNeighbors?.steps.find((s) => s.offset === 0) || null

      return {
        ...state,
        trakeSlots: updated,
        trakeValidationStatus: overallStatus,
        activeCandidate: updatedActiveCandidate,
        anchorCandidate: updatedAnchorCandidate,
        cumulativeOffset: 0,
        currentStep: initialStep,
        exactNeighbors: initialNeighbors,
        exactImageBlobUrl: null,
        exactImageHeaders: null,
      }
    }

    case 'SELECT_TRAKE_SLOT': {
      const slotIdx = action.payload
      const targetSlot =
        slotIdx !== null && slotIdx < state.trakeSlots.length
          ? state.trakeSlots[slotIdx]
          : null
      let nextActiveCand = state.activeCandidate
      let nextAnchorCand = state.anchorCandidate
      if (targetSlot && targetSlot.video_id && targetSlot.frame_id !== null) {
        const cand: SearchCandidate = {
          query_id: state.queryId || 'trake',
          video_id: targetSlot.video_id,
          frame_id: targetSlot.frame_id,
          timestamp_ms: targetSlot.timestamp_ms || 0,
          source: 'trake',
          rank: targetSlot.event_index + 1,
          score: targetSlot.score,
          certified_anchor_frame_id:
            targetSlot.certified_anchor_frame_id ?? targetSlot.frame_id,
          certified_anchor_timestamp_ms:
            targetSlot.certified_anchor_timestamp_ms ?? targetSlot.timestamp_ms ?? 0,
          anchor_offset: targetSlot.anchor_offset ?? 0,
          cumulative_offset: 0,
        }
        nextActiveCand = cand
        nextAnchorCand = cand
      }
      return {
        ...state,
        trakeActiveSlotIndex: slotIdx,
        activeCandidate: nextActiveCand,
        anchorCandidate: nextAnchorCand,
        cumulativeOffset: 0,
        currentStep: null,
        exactNeighbors: null,
        exactImageBlobUrl: null,
        exactImageHeaders: null,
      }
    }

    case 'ADD_TRAKE_TO_BASKET': {
      // P0 Invariants: fixture cannot enter basket; all N slots must be valid, from single video, and locked/accepted
      const isAllLocked = state.trakeSlots.length > 0 && state.trakeSlots.every((s) => s.locked)
      if (
        state.mode === 'fixture' ||
        !state.trakeVideoId ||
        state.trakeSlots.length === 0 ||
        state.trakeSlots.length !== state.trakeEvents.length ||
        state.trakeValidationStatus !== 'valid' ||
        !isAllLocked ||
        state.trakeSlots.some(
          (s) =>
            s.frame_id === null ||
            s.validation_status !== 'valid' ||
            s.video_id !== state.trakeVideoId
        )
      ) {
        return state
      }
      if (!state.trakeQueryId) return state

      const item: BasketItem = {
        query_id: state.trakeQueryId,
        video_id: state.trakeVideoId,
        frame_id: state.trakeSlots[0].frame_id!,
        timestamp_ms: state.trakeSlots[0].timestamp_ms ?? undefined,
        added_at_utc: new Date().toISOString(),
        task: 'TRAKE',
        frame_ids: state.trakeSlots.map((s) => s.frame_id!),
        event_labels: state.trakeSlots.map((s) => s.event_label),
      }

      const exists = state.submissionBasket.some(
        (b) =>
          b.video_id === item.video_id &&
          b.task === 'TRAKE' &&
          b.query_id === item.query_id &&
          JSON.stringify(b.frame_ids) === JSON.stringify(item.frame_ids)
      )
      if (exists) return state

      return {
        ...state,
        submissionBasket: [...state.submissionBasket, item],
      }
    }

    case 'COMMIT_KIS_FRAME': {
      if (!state.activeCandidate) return state
      const { frame_id, timestamp_ms } = action.payload
      const targetRank = state.activeCandidate.rank
      const targetVid = state.activeCandidate.video_id

      // Preserve certified root anchor lineage
      const prevBaseOffset = state.anchorCandidate?.anchor_offset ?? 0
      const newAnchorOffset = prevBaseOffset + state.cumulativeOffset

      const rootAnchorFrameId =
        state.anchorCandidate?.certified_anchor_frame_id ??
        state.anchorCandidate?.frame_id ??
        state.activeCandidate.certified_anchor_frame_id ??
        state.activeCandidate.frame_id

      const rootAnchorTimestampMs =
        state.anchorCandidate?.certified_anchor_timestamp_ms ??
        state.anchorCandidate?.timestamp_ms ??
        state.activeCandidate.certified_anchor_timestamp_ms ??
        state.activeCandidate.timestamp_ms

      const newCand: SearchCandidate = {
        ...state.activeCandidate,
        frame_id,
        timestamp_ms: timestamp_ms ?? state.activeCandidate.timestamp_ms,
        certified_anchor_frame_id: rootAnchorFrameId,
        certified_anchor_timestamp_ms: rootAnchorTimestampMs,
        anchor_offset: newAnchorOffset,
        cumulative_offset: 0,
      }

      const updatedCandidates = state.candidates.map((c) => {
        if (c.rank === targetRank && c.video_id === targetVid) {
          return newCand
        }
        return c
      })

      const updatedOriginal = state.originalCandidates.map((c) => {
        if (c.rank === targetRank && c.video_id === targetVid) {
          return newCand
        }
        return c
      })

      const updatedRef =
        state.feedbackReference &&
        state.feedbackReference.rank === targetRank &&
        state.feedbackReference.video_id === targetVid
          ? newCand
          : state.feedbackReference

      return {
        ...state,
        candidates: updatedCandidates,
        originalCandidates: updatedOriginal,
        kisActiveCandidate: newCand,
        activeCandidate: newCand,
        anchorCandidate: newCand,
        feedbackReference: updatedRef,
        cumulativeOffset: 0,
      }
    }

    case 'COMMIT_VQA_FRAME': {
      if (!state.vqaActiveResult) return state
      const { frame_id, timestamp_ms } = action.payload
      const targetRank = state.vqaActiveResult.rank
      const targetVid = state.vqaActiveResult.video_id

      // Preserve certified root anchor lineage for VQA
      const prevBaseOffset =
        state.vqaActiveResult.anchor_offset ??
        state.anchorCandidate?.anchor_offset ??
        0
      const newAnchorOffset = prevBaseOffset + state.cumulativeOffset

      const rootAnchorFrameId =
        state.vqaActiveResult.certified_anchor_frame_id ??
        state.anchorCandidate?.certified_anchor_frame_id ??
        state.anchorCandidate?.frame_id ??
        state.vqaActiveResult.frame_id

      const rootAnchorTimestampMs =
        state.vqaActiveResult.certified_anchor_timestamp_ms ??
        state.anchorCandidate?.certified_anchor_timestamp_ms ??
        state.anchorCandidate?.timestamp_ms ??
        state.vqaActiveResult.timestamp_ms

      const updatedActiveResult: import('../types/contracts').VqaResult = {
        ...state.vqaActiveResult,
        frame_id,
        timestamp_ms: timestamp_ms ?? state.vqaActiveResult.timestamp_ms,
        certified_anchor_frame_id: rootAnchorFrameId,
        certified_anchor_timestamp_ms: rootAnchorTimestampMs,
        anchor_offset: newAnchorOffset,
      }

      const updatedVqaResults = state.vqaResults.map((r) => {
        if (r.rank === targetRank && r.video_id === targetVid) {
          return updatedActiveResult
        }
        return r
      })

      const newCand: SearchCandidate = {
        query_id: state.queryId || 'vqa',
        video_id: targetVid,
        frame_id,
        timestamp_ms: timestamp_ms ?? 0,
        source: 'vqa',
        rank: targetRank,
        score: updatedActiveResult.confidence,
        certified_anchor_frame_id: rootAnchorFrameId,
        certified_anchor_timestamp_ms: rootAnchorTimestampMs,
        anchor_offset: newAnchorOffset,
        cumulative_offset: 0,
      }

      return {
        ...state,
        vqaResults: updatedVqaResults,
        vqaActiveResult: updatedActiveResult,
        activeCandidate: newCand,
        anchorCandidate: newCand,
        vqaApprovedAnswer: null, // Critical: Invalidate previous approval when frame identity is changed
        cumulativeOffset: 0,
        currentStep: null,
        exactNeighbors: null,
        exactImageBlobUrl: null,
        exactImageHeaders: null,
      }
    }

    // --- Session State Recovery ---
    case 'RESTORE_SESSION_STATE': {
      return {
        ...state,
        ...action.payload,
      }
    }

    default:
      return state
  }
}
