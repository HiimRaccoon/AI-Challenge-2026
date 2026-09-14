import React, { useEffect, useRef } from 'react'
import { ProductHeader } from './components/ProductHeader'
import { RetrievalWorkspace } from './components/RetrievalWorkspace'
import { InspectionWorkspace } from './components/InspectionWorkspace'
import { EvidenceSubmissionWorkspace } from './components/EvidenceSubmissionWorkspace'
import { EvaluationWorkspace } from './components/EvaluationWorkspace'
import { KeyboardHelpModal } from './components/KeyboardHelpModal'
import { PinnedCandidateTray } from './components/PinnedCandidateTray'
import { useAppDispatch, useAppState } from './state/AppContext'
import { fetchHealth } from './api/tv4Client'
import { useExactInspectionCoordinator } from './state/useExactInspectionCoordinator'
import { saveSessionState, loadSavedSessionState } from './utils/sessionStorage'

export const App: React.FC = () => {
  const dispatch = useAppDispatch()
  const state = useAppState()
  const { activeTab, isKeyboardHelpOpen } = state

  // Single authoritative owner for exact-neighbor & exact-frame lifecycle
  useExactInspectionCoordinator()

  // Load persisted session on initial mount
  const hasRestoredSessionRef = useRef(false)
  useEffect(() => {
    if (!hasRestoredSessionRef.current) {
      hasRestoredSessionRef.current = true
      const saved = loadSavedSessionState()
      if (saved) {
        dispatch({ type: 'RESTORE_SESSION_STATE', payload: saved })
      }
    }
  }, [dispatch])

  // Debounced auto-save session state on change
  useEffect(() => {
    if (!hasRestoredSessionRef.current) return
    const timer = setTimeout(() => {
      saveSessionState(state)
    }, 500)
    return () => clearTimeout(timer)
  }, [state])

  useEffect(() => {
    let isMounted = true
    const checkHealth = () => {
      fetchHealth()
        .then((health) => {
          if (isMounted) {
            dispatch({ type: 'SET_HEALTH', payload: { health } })
          }
        })
        .catch((err) => {
          if (isMounted) {
            dispatch({
              type: 'SET_HEALTH',
              payload: { health: null, error: err.message || 'Service unavailable' },
            })
          }
        })
    }

    checkHealth()
    const interval = setInterval(checkHealth, 5000)

    return () => {
      isMounted = false
      clearInterval(interval)
    }
  }, [dispatch])

  // Global focus-safe keyboard listener for '?' help modal toggle
  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      const isInput =
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable)

      if (e.key === '?' || (e.shiftKey && e.key === '/')) {
        if (!isInput) {
          e.preventDefault()
          dispatch({ type: 'TOGGLE_KEYBOARD_HELP' })
        }
      } else if (e.key === 'Escape' && isKeyboardHelpOpen) {
        e.preventDefault()
        dispatch({ type: 'SET_KEYBOARD_HELP', payload: false })
      }
    }

    window.addEventListener('keydown', handleGlobalKeyDown)
    return () => {
      window.removeEventListener('keydown', handleGlobalKeyDown)
    }
  }, [dispatch, isKeyboardHelpOpen])

  return (
    <div className="multimodal-workstation">
      <ProductHeader />
      <div className="workspace-tab-viewport">
        {activeTab === 'retrieval' && <RetrievalWorkspace />}
        {activeTab === 'inspection' && <InspectionWorkspace />}
        {activeTab === 'evidence' && <EvidenceSubmissionWorkspace />}
        {activeTab === 'evaluation' && <EvaluationWorkspace />}
      </div>
      <KeyboardHelpModal
        isOpen={isKeyboardHelpOpen}
        onClose={() => dispatch({ type: 'SET_KEYBOARD_HELP', payload: false })}
      />
      <PinnedCandidateTray />
    </div>
  )
}
