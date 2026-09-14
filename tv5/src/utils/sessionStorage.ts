import { AppState } from '../state/appState'

const SESSION_STORAGE_KEY = 'tv5_session_backup_v1'

export interface PersistedSessionData {
  timestamp: string
  taskMode: AppState['taskMode']
  activeTab: AppState['activeTab']
  queryText: string
  queryId: string | null
  candidates: AppState['candidates']
  originalCandidates: AppState['originalCandidates']
  kisActiveCandidate: AppState['kisActiveCandidate']
  submissionBasket: AppState['submissionBasket']
  pinnedCandidates: AppState['pinnedCandidates']
  vqaQuestion: string
  vqaResults: AppState['vqaResults']
  vqaRetrievalCandidates: AppState['vqaRetrievalCandidates']
  vqaDraftAnswer: string
  trakeEvents: string[]
  trakeResults?: AppState['trakeResults']
  trakeActiveHypothesisIndex?: number | null
  trakeSlots: AppState['trakeSlots']
  trakeVideoId: string | null
  modalityWeights?: AppState['modalityWeights']
}

export function saveSessionState(state: AppState): void {
  try {
    const data: PersistedSessionData = {
      timestamp: new Date().toISOString(),
      taskMode: state.taskMode,
      activeTab: state.activeTab,
      queryText: state.queryText,
      queryId: state.queryId,
      candidates: (state.candidates || []).slice(0, 100),
      originalCandidates: (state.originalCandidates || []).slice(0, 100),
      kisActiveCandidate: state.kisActiveCandidate,
      submissionBasket: state.submissionBasket || [],
      pinnedCandidates: state.pinnedCandidates || [],
      vqaQuestion: state.vqaQuestion || '',
      vqaResults: state.vqaResults || [],
      vqaRetrievalCandidates: state.vqaRetrievalCandidates || [],
      vqaDraftAnswer: state.vqaDraftAnswer || '',
      trakeEvents: state.trakeEvents || [],
      trakeResults: state.trakeResults || [],
      trakeActiveHypothesisIndex: state.trakeActiveHypothesisIndex,
      trakeSlots: state.trakeSlots || [],
      trakeVideoId: state.trakeVideoId,
      modalityWeights: state.modalityWeights,
    }
    localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(data))
  } catch (err) {
    console.warn('[SessionStorage] Failed to save session:', err)
  }
}

export function loadSavedSessionState(): Partial<AppState> | null {
  try {
    const raw = localStorage.getItem(SESSION_STORAGE_KEY)
    if (!raw) return null
    const parsed: PersistedSessionData = JSON.parse(raw)
    return {
      taskMode: parsed.taskMode || 'KIS',
      activeTab: parsed.activeTab || 'retrieval',
      queryText: parsed.queryText || '',
      queryId: parsed.queryId || null,
      candidates: parsed.candidates || [],
      originalCandidates: parsed.originalCandidates || [],
      kisActiveCandidate: parsed.kisActiveCandidate || null,
      submissionBasket: parsed.submissionBasket || [],
      pinnedCandidates: parsed.pinnedCandidates || [],
      vqaQuestion: parsed.vqaQuestion || '',
      vqaResults: parsed.vqaResults || [],
      vqaRetrievalCandidates: parsed.vqaRetrievalCandidates || [],
      vqaDraftAnswer: parsed.vqaDraftAnswer || '',
      trakeEvents: parsed.trakeEvents || [],
      trakeResults: parsed.trakeResults || [],
      trakeActiveHypothesisIndex: parsed.trakeActiveHypothesisIndex ?? null,
      trakeSlots: parsed.trakeSlots || [],
      trakeVideoId: parsed.trakeVideoId || null,
      modalityWeights: parsed.modalityWeights || {
        visual: 100,
        ocr: 100,
        asr: 100,
        object: 50,
      },
    }
  } catch (err) {
    console.warn('[SessionStorage] Failed to load session:', err)
    return null
  }
}

export function exportSessionAsJson(state: AppState): string {
  const data: PersistedSessionData = {
    timestamp: new Date().toISOString(),
    taskMode: state.taskMode,
    activeTab: state.activeTab,
    queryText: state.queryText,
    queryId: state.queryId,
    candidates: state.candidates,
    originalCandidates: state.originalCandidates,
    kisActiveCandidate: state.kisActiveCandidate,
    submissionBasket: state.submissionBasket,
    pinnedCandidates: state.pinnedCandidates,
    vqaQuestion: state.vqaQuestion,
    vqaResults: state.vqaResults,
    vqaRetrievalCandidates: state.vqaRetrievalCandidates,
    vqaDraftAnswer: state.vqaDraftAnswer,
    trakeEvents: state.trakeEvents,
    trakeResults: state.trakeResults,
    trakeActiveHypothesisIndex: state.trakeActiveHypothesisIndex,
    trakeSlots: state.trakeSlots,
    trakeVideoId: state.trakeVideoId,
    modalityWeights: state.modalityWeights,
  }
  return JSON.stringify(data, null, 2)
}

export function downloadSessionBackup(state: AppState, filename = 'aic2026_session_backup.json'): void {
  const jsonStr = exportSessionAsJson(state)
  const blob = new Blob([jsonStr], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

export function importSessionFromJson(jsonStr: string): Partial<AppState> {
  const parsed: PersistedSessionData = JSON.parse(jsonStr)
  return {
    taskMode: parsed.taskMode || 'KIS',
    activeTab: parsed.activeTab || 'retrieval',
    queryText: parsed.queryText || '',
    queryId: parsed.queryId || null,
    candidates: parsed.candidates || [],
    originalCandidates: parsed.originalCandidates || [],
    kisActiveCandidate: parsed.kisActiveCandidate || null,
    submissionBasket: parsed.submissionBasket || [],
    pinnedCandidates: parsed.pinnedCandidates || [],
    vqaQuestion: parsed.vqaQuestion || '',
    vqaResults: parsed.vqaResults || [],
    vqaRetrievalCandidates: parsed.vqaRetrievalCandidates || [],
    vqaDraftAnswer: parsed.vqaDraftAnswer || '',
    trakeEvents: parsed.trakeEvents || [],
    trakeResults: parsed.trakeResults || [],
    trakeActiveHypothesisIndex: parsed.trakeActiveHypothesisIndex ?? null,
    trakeSlots: parsed.trakeSlots || [],
    trakeVideoId: parsed.trakeVideoId || null,
    modalityWeights: parsed.modalityWeights || {
      visual: 100,
      ocr: 100,
      asr: 100,
      object: 50,
    },
  }
}
