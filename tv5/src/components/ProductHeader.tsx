import React from 'react'
import { useAppDispatch, useAppState } from '../state/AppContext'
import { WorkspaceTab } from '../state/appState'
import {
  PureDatabaseLogoIcon,
  RetrievalTabIcon,
  InspectionTabIcon,
  EvidenceTabIcon,
  CheckIcon,
} from './Icons'

export const ProductHeader: React.FC = () => {
  const {
    mode,
    readiness,
    candidates,
    isSearching,
    tv4Health,
    activeTab,
    taskMode,
    trakeSlots,
    isTrakeSearching,
    vqaResults,
    isVqaSearching,
  } = useAppState()
  const dispatch = useAppDispatch()
  const isFixture = mode === 'fixture'
  const isDegraded = readiness === 'DEGRADED' || readiness === 'OFFLINE' || readiness === 'PARTIAL'

  const handleTabChange = (tab: WorkspaceTab) => {
    dispatch({ type: 'SET_ACTIVE_TAB', payload: tab })
  }

  const taskBadgeLabel = taskMode === 'VQA' ? 'Q&A' : taskMode === 'TRAKE' ? 'TRAKE' : 'KIS'

  return (
    <header className="product-header">
      {/* Left: Pure Database Cylinder Logo & Title */}
      <div className="header-brand-pane">
        <div className="brand-logo-badge" title="Team SS009.Q24 Database Node">
          <PureDatabaseLogoIcon size={22} className="text-cyan" />
        </div>
        <div className="brand-text-block">
          <div className="brand-title-row">
            <h1 className="system-title">Team SS009.Q24 Multimodal Retrieval System</h1>
            <span className="mode-tag" data-testid="header-task-badge">{taskBadgeLabel}</span>
          </div>
          <span className="system-subtitle">Precision Search. Smarter Decisions.</span>
        </div>
      </div>

      {/* Center: Three Primary Navigation Tabs */}
      <div className="header-nav-center">
        <nav className="header-nav-tabs" aria-label="Workstation Workspaces">
          <button
            type="button"
            className={`nav-tab-btn ${activeTab === 'retrieval' ? 'tab-active' : ''}`}
            onClick={() => handleTabChange('retrieval')}
            data-testid="tab-retrieval"
          >
            <RetrievalTabIcon size={16} />
            <span>Retrieval</span>
          </button>

          <button
            type="button"
            className={`nav-tab-btn ${activeTab === 'inspection' ? 'tab-active' : ''}`}
            onClick={() => handleTabChange('inspection')}
            data-testid="tab-inspection"
          >
            <InspectionTabIcon size={16} />
            <span>Inspection</span>
          </button>

          <button
            type="button"
            className={`nav-tab-btn ${activeTab === 'evidence' ? 'tab-active' : ''}`}
            onClick={() => handleTabChange('evidence')}
            data-testid="tab-evidence"
          >
            <EvidenceTabIcon size={16} />
            <span>Evidence / Submission</span>
          </button>

          <button
            type="button"
            className={`nav-tab-btn ${activeTab === 'evaluation' ? 'tab-active' : ''}`}
            onClick={() => handleTabChange('evaluation')}
            data-testid="tab-evaluation"
          >
            <CheckIcon size={16} />
            <span>Evaluation / Stats</span>
          </button>
        </nav>
      </div>

      {/* Right: Operational Status & Candidate Count */}
      <div className="header-status-pane">
        <button
          type="button"
          className="btn-keyboard-help-trigger"
          onClick={() => dispatch({ type: 'TOGGLE_KEYBOARD_HELP' })}
          title="Keyboard Shortcuts (?)"
          aria-label="Open keyboard shortcuts help modal"
          data-testid="btn-keyboard-help"
          style={{
            padding: '4px 8px',
            background: 'rgba(0, 229, 255, 0.1)',
            border: '1px solid rgba(0, 229, 255, 0.3)',
            borderRadius: '4px',
            color: 'var(--color-cyan, #00e5ff)',
            fontWeight: 'bold',
            fontSize: '13px',
            cursor: 'pointer',
          }}
        >
          ?
        </button>
        {/* Subtle Operational Status */}
        {isFixture ? (
          <span className="status-pill pill-fixture">
            <span className="status-dot dot-fixture" />
            <span>Fixture Preview</span>
          </span>
        ) : (
          <span className="status-pill pill-live">
            <span className="status-dot dot-live" />
            <span>Live Infrastructure</span>
          </span>
        )}

        {isDegraded && (
          <span
            className={`status-pill pill-${readiness.toLowerCase().replace('_', '-')}`}
            data-testid="readiness-status"
          >
            {readiness}
          </span>
        )}

        {/* Diagnostic hooks for automated regression verification */}
        <span
          className="diagnostic-hook"
          data-testid="tv4-health-status"
          style={{ display: 'none' }}
        >
          {tv4Health ? 'ONLINE' : 'OFFLINE'}
        </span>
        {tv4Health?.preprocess_run_id && (
          <span
            className="diagnostic-hook"
            data-testid="preprocess-run-id"
            style={{ display: 'none' }}
          >
            {tv4Health.preprocess_run_id}
          </span>
        )}

        {/* Task-Scoped Result / Event Count */}
        <div className="candidate-count-chip" data-testid="header-count-chip">
          <span className="count-label">
            {taskMode === 'TRAKE' ? 'Events' : taskMode === 'VQA' ? 'Answers' : 'Results'}
          </span>
          <span className="count-value tabular-nums">
            {taskMode === 'TRAKE'
              ? isTrakeSearching
                ? '...'
                : trakeSlots.length
              : taskMode === 'VQA'
              ? isVqaSearching
                ? '...'
                : vqaResults.length
              : isSearching
              ? '...'
              : candidates.length}
          </span>
        </div>
      </div>
    </header>
  )
}
