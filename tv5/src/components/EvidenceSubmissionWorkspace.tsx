import React, { useRef, useState } from 'react'
import { useAppDispatch, useAppState } from '../state/AppContext'
import { EvidenceInspector } from './EvidenceInspector'
import { VqaAnswerPanel } from './VqaAnswerPanel'
import {
  EvidenceTabIcon,
  BasketIcon,
  ClearIcon,
  WarningIcon,
  CheckIcon,
  QuestionIcon,
  FilmstripIcon,
  ExportIcon,
} from './Icons'
import { exportBasketToPerQueryCsv, queryCsvFilename, triggerBrowserDownload } from '../utils/submissionExporter'
import { downloadSessionBackup, importSessionFromJson } from '../utils/sessionStorage'
import { BasketItemThumb } from './BasketItemThumb'
import { telemetry } from '../utils/telemetry'

export const EvidenceSubmissionWorkspace: React.FC = () => {
  const state = useAppState()
  const {
    taskMode,
    vqaActiveResult,
    submissionBasket,
    mode,
    trakeSlots,
    trakeVideoId,
    trakeEvents,
    trakeValidationStatus,
    queryText,
    activeCandidate,
  } = state

  const dispatch = useAppDispatch()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [sessionNotice, setSessionNotice] = useState<string | null>(null)
  const isFixture = mode === 'fixture'

  const kisItems = submissionBasket.filter((b) => (b.task || 'KIS') === 'KIS')
  const vqaItems = submissionBasket.filter((b) => b.task === 'VQA')
  const trakeItems = submissionBasket.filter((b) => b.task === 'TRAKE')

  const currentTaskItems =
    taskMode === 'KIS' ? kisItems : taskMode === 'VQA' ? vqaItems : trakeItems
  const currentTaskLabel = taskMode === 'VQA' ? 'Q&A' : taskMode

  const handleRemoveFromBasket = (videoId: string, frameId: number) => {
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
  }

  const handleExportTaskCsv = (task: 'KIS' | 'VQA' | 'TRAKE') => {
    const items = submissionBasket.filter((b) => (b.task || 'KIS') === task)
    if (items.length === 0) return
    // One CSV per query_id -- items from different queries must never be
    // combined into one file (README_RETRIEVAL_HANDOFF / RUN_BOOK section 20).
    const csvByQuery = exportBasketToPerQueryCsv(submissionBasket, task)
    const filenames: string[] = []
    for (const [queryId, csvContent] of Object.entries(csvByQuery)) {
      const filename = queryCsvFilename(queryId)
      triggerBrowserDownload(csvContent, filename)
      filenames.push(filename)
    }
    telemetry.record({
      action: 'EXPORT_SUBMISSION_CSV',
      taskMode: task,
      details: { rowCount: items.length, queryCount: filenames.length, filenames },
    })
  }

  const handleClearBasket = () => {
    dispatch({ type: 'CLEAR_BASKET' })
    telemetry.record({
      action: 'CLEAR_BASKET',
      taskMode,
    })
  }

  const handleExportSessionBackup = () => {
    const filename = `aic2026_session_${taskMode}_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.json`
    downloadSessionBackup(state, filename)
    setSessionNotice(`✓ Đã xuất file sao lưu phiên làm việc: ${filename}`)
    setTimeout(() => setSessionNotice(null), 4000)
  }

  const handleImportSessionFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (event) => {
      try {
        const content = event.target?.result as string
        const restored = importSessionFromJson(content)
        dispatch({ type: 'RESTORE_SESSION_STATE', payload: restored })
        setSessionNotice(`✓ Đã phục hồi phiên thi đấu thành công từ file: ${file.name}`)
        setTimeout(() => setSessionNotice(null), 4000)
      } catch (err) {
        setSessionNotice(`✕ Lỗi khi đọc file sao lưu: File không đúng định dạng JSON`)
        setTimeout(() => setSessionNotice(null), 4000)
      }
    }
    reader.readAsText(file)
    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const activeEvidence = taskMode === 'VQA' ? vqaActiveResult?.evidence ?? null : null

  // TRAKE Readiness calculation
  const isTrakeAllLocked = trakeSlots.length > 0 && trakeSlots.every((s) => s.locked)
  const isTrakeStructurallyValid = trakeValidationStatus === 'valid'
  const isTrakeReady = isTrakeStructurallyValid && isTrakeAllLocked

  return (
    <div className="evidence-workspace-layout" data-testid="evidence-workspace">
      {/* Workspace Banner with Session Backup Actions */}
      <div className="evidence-foundation-banner" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div className="banner-icon-box">
            <EvidenceTabIcon size={28} className="text-cyan" />
          </div>
          <div className="banner-text-block">
            <h2 className="banner-title">Evidence & Submission Management</h2>
            <p className="banner-description">
              Review multimodal proof, verify canonical candidates, adopt advisory proposals, and assemble competition submission packages.
            </p>
          </div>
        </div>

        {/* Session Backup / Restore Buttons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexShrink: 0 }}>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            onChange={handleImportSessionFile}
            style={{ display: 'none' }}
          />

          <button
            type="button"
            onClick={handleExportSessionBackup}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 12px',
              background: 'rgba(0, 229, 255, 0.12)',
              color: '#00e5ff',
              border: '1px solid rgba(0, 229, 255, 0.35)',
              borderRadius: '6px',
              fontSize: '11.5px',
              fontWeight: 600,
              cursor: 'pointer',
            }}
            title="Tải về file sao lưu toàn bộ giỏ bài, khay ghim và kết quả tìm kiếm (.json)"
          >
            <span>💾 Sao lưu Session</span>
          </button>

          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 12px',
              background: 'rgba(255, 255, 255, 0.08)',
              color: '#cbd5e1',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '6px',
              fontSize: '11.5px',
              fontWeight: 600,
              cursor: 'pointer',
            }}
            title="Tải lại phiên làm việc từ file sao lưu (.json)"
          >
            <span>📥 Phục hồi Session</span>
          </button>
        </div>
      </div>

      {/* Session Notice Banner */}
      {sessionNotice && (
        <div
          style={{
            margin: '0 0 12px 0',
            padding: '8px 14px',
            background: sessionNotice.startsWith('✓') ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
            border: sessionNotice.startsWith('✓') ? '1px solid #10b981' : '1px solid #ef4444',
            borderRadius: '6px',
            color: sessionNotice.startsWith('✓') ? '#10b981' : '#ef4444',
            fontSize: '12px',
            fontWeight: 600,
          }}
        >
          {sessionNotice}
        </div>
      )}

      {/* 2. Top Review Row (2-Column Decision Area: Task-Scoped Verification on Left ~59%, Submission Basket on Right ~41%) */}
      <div className="evidence-top-review-row">
        {/* Left Column: Task-Specific Review Panel */}
        <div className="evidence-review-left-col">
          {taskMode === 'VQA' ? (
            vqaActiveResult ? (
              <VqaAnswerPanel />
            ) : (
              <div className="evidence-card-notice" data-testid="no-vqa-notice">
                <div className="notice-icon-box">
                  <QuestionIcon size={20} className="text-cyan" />
                </div>
                <div className="notice-text-content">
                  <span className="notice-title">No Active VQA Query</span>
                  <p className="notice-text">
                    Enter a question in the Retrieval workspace to generate retrieval-grounded VQA answers and multimodal evidence.
                  </p>
                </div>
              </div>
            )
          ) : taskMode === 'TRAKE' ? (
            <div className="evidence-task-review-card" data-testid="evidence-trake-review">
              <div className="review-card-header">
                <div className="review-title-group">
                  <FilmstripIcon size={16} className="text-cyan" />
                  <span className="review-title">TRAKE Sequence Status</span>
                </div>
                <span
                  className={`trake-status-badge ${
                    isTrakeReady
                      ? 'badge-valid'
                      : isTrakeStructurallyValid
                      ? 'badge-aligned'
                      : 'badge-incomplete'
                  }`}
                >
                  {isTrakeReady
                    ? 'READY'
                    : isTrakeStructurallyValid
                    ? `ALIGNED (${trakeSlots.filter((s) => s.locked).length}/${trakeSlots.length} LOCKED)`
                    : 'INCOMPLETE'}
                </span>
              </div>
              <div className="review-card-body">
                <div className="review-field-row">
                  <span className="review-field-label">Sequence:</span>
                  <span className="review-field-val text-primary">{queryText || 'No sequence text'}</span>
                </div>
                <div className="review-field-row">
                  <span className="review-field-label">Video Hypothesis:</span>
                  <span className="review-field-val text-cyan tabular-nums">{trakeVideoId || 'Unassigned'}</span>
                </div>
                <div className="review-field-row">
                  <span className="review-field-label">Event Positions:</span>
                  <span className="review-field-val tabular-nums">
                    {trakeSlots.length} of {trakeEvents.length} events ({trakeSlots.filter((s) => s.locked).length} locked)
                  </span>
                </div>
                {isFixture && (
                  <div className="review-notice-banner">
                    <WarningIcon size={13} />
                    <span>Fixture Mode: Aligned sequences can be reviewed in workstation preview.</span>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="evidence-task-review-card" data-testid="evidence-kis-review">
              <div className="review-card-header">
                <div className="review-title-group">
                  <CheckIcon size={16} className="text-cyan" />
                  <span className="review-title">KIS Visual Search Status</span>
                </div>
              </div>
              <div className="review-card-body">
                <div className="review-field-row">
                  <span className="review-field-label">Active Query:</span>
                  <span className="review-field-val text-primary">{queryText || 'No query text'}</span>
                </div>
                {activeCandidate ? (
                  <div className="review-field-row">
                    <span className="review-field-label">Selected Candidate:</span>
                    <span className="review-field-val text-cyan tabular-nums">
                      {activeCandidate.video_id} · Frame {activeCandidate.frame_id}
                    </span>
                  </div>
                ) : (
                  <div className="review-field-row">
                    <span className="review-field-label">Selection:</span>
                    <span className="review-field-val text-muted">No candidate currently selected</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Submission Basket & Packaging Validator */}
        <div className="evidence-review-right-col">
          <div className="submission-basket-card" data-testid="submission-basket-card">
            <div className="basket-card-header">
              <div className="basket-title-group">
                <BasketIcon size={16} className="text-cyan" />
                <span className="basket-title">Submission Basket</span>
              </div>
              <span className="basket-count-badge tabular-nums" data-testid="basket-count-badge">
                {submissionBasket.length} / 100 predictions
              </span>
            </div>

            <div className="basket-card-body">
              {isFixture && (
                <div className="basket-fixture-alert" data-testid="basket-fixture-alert">
                  <WarningIcon size={13} />
                  <span>
                    Fixture Preview Mode Active — Items are non-submitting demo fixtures.
                  </span>
                </div>
              )}

              {submissionBasket.length === 0 ? (
                <div className="basket-empty-state" data-testid="basket-empty-state">
                  <div className="basket-empty-icon-box">
                    <BasketIcon size={24} className="empty-basket-icon" />
                  </div>
                  <p className="empty-state-title">Submission basket is empty</p>
                  <p className="empty-state-text">
                    Approved VQA answers, aligned TRAKE sequences, or verified KIS selections will appear here.
                  </p>
                </div>
              ) : (
                <div className="basket-items-list" data-testid="basket-items-list">
                  {submissionBasket.map((item, idx) => (
                    <div
                      key={`${item.video_id}-${item.frame_id}-${idx}`}
                      className="basket-item-row"
                      data-testid={`basket-item-${idx}`}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '10px',
                        padding: '8px 10px',
                        borderRadius: '6px',
                        background: 'rgba(255, 255, 255, 0.03)',
                        border: '1px solid rgba(255, 255, 255, 0.08)',
                        marginBottom: '8px',
                        transition: 'all 0.15s ease',
                      }}
                    >
                      {/* Visual Keyframe Thumbnail Preview */}
                      <div
                        style={{
                          width: '92px',
                          height: '52px',
                          borderRadius: '4px',
                          overflow: 'hidden',
                          flexShrink: 0,
                          background: '#000',
                          position: 'relative',
                          cursor: 'pointer',
                          border: '1px solid rgba(0, 229, 255, 0.35)',
                        }}
                        onClick={() => {
                          dispatch({
                            type: 'SELECT_CANDIDATE',
                            payload: {
                              query_id: item.query_id,
                              video_id: item.video_id,
                              frame_id: item.frame_id,
                              timestamp_ms: item.timestamp_ms || 0,
                              source: item.task || 'KIS',
                              rank: idx + 1,
                            },
                          })
                          dispatch({ type: 'SET_ACTIVE_TAB', payload: 'inspection' })
                        }}
                        title="Click để xem lại video tại phân cảnh này"
                      >
                        <BasketItemThumb
                          videoId={item.video_id}
                          frameId={item.frame_id}
                          timestampMs={item.timestamp_ms}
                        />
                      </div>

                      <div className="basket-item-info" style={{ flex: 1, minWidth: 0 }}>
                        <div className="basket-item-id-row" style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                          <span className="item-task-badge" style={{ fontSize: '10px', padding: '1px 5px', borderRadius: '3px', background: 'rgba(0, 229, 255, 0.15)', color: '#00e5ff', fontWeight: 700 }}>
                            {item.task || 'KIS'}
                          </span>
                          <span className="item-vid text-cyan" style={{ fontWeight: 700, fontSize: '12px' }}>{item.video_id}</span>
                          <span className="item-fid tabular-nums" style={{ color: '#ffab00', fontWeight: 700, fontSize: '11px' }}>
                            {item.task === 'TRAKE' && item.frame_ids
                              ? `${item.frame_ids.length} Events`
                              : `F.${item.frame_id}`}
                          </span>
                        </div>
                        {item.task === 'VQA' && item.answer && (
                          <div className="basket-item-answer" data-testid={`basket-answer-${idx}`} style={{ fontSize: '11px', color: '#10b981', marginBottom: '2px' }}>
                            <span className="answer-prefix" style={{ fontWeight: 600 }}>Ans: </span>
                            <span className="answer-text">"{item.answer}"</span>
                          </div>
                        )}
                        {item.task === 'TRAKE' && item.frame_ids && (
                          <div className="basket-item-trake-chain" data-testid={`basket-trake-chain-${idx}`} style={{ fontSize: '11px', color: '#cbd5e1', marginBottom: '2px' }}>
                            <span className="chain-prefix" style={{ fontWeight: 600 }}>Frames: </span>
                            <span className="chain-sequence tabular-nums">
                              {item.frame_ids.join(' → ')}
                            </span>
                          </div>
                        )}
                        <div className="basket-item-meta tabular-nums" style={{ fontSize: '10px', color: '#64748b' }}>
                          <span>Thêm lúc {new Date(item.added_at_utc).toLocaleTimeString()}</span>
                        </div>
                      </div>

                      <button
                        type="button"
                        className="btn-remove-basket-item"
                        onClick={() => handleRemoveFromBasket(item.video_id, item.frame_id)}
                        title="Xóa frame này khỏi giỏ"
                        aria-label={`Remove ${item.video_id} frame ${item.frame_id}`}
                        data-testid={`btn-remove-basket-${idx}`}
                        style={{
                          background: 'rgba(239, 68, 68, 0.12)',
                          border: '1px solid rgba(239, 68, 68, 0.3)',
                          borderRadius: '4px',
                          color: '#ef4444',
                          padding: '6px',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          transition: 'all 0.15s ease',
                        }}
                      >
                        <ClearIcon size={13} />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {submissionBasket.length > 0 && (
              <div className="basket-card-footer">
                <div className="basket-summary-status">
                  <CheckIcon size={14} className="text-cyan" />
                  <span className="summary-text tabular-nums">
                    {submissionBasket.length} total prediction{submissionBasket.length > 1 ? 's' : ''} in basket ({kisItems.length} KIS, {vqaItems.length} Q&A, {trakeItems.length} TRAKE)
                  </span>
                </div>
                <div className="basket-export-actions" style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '8px', marginTop: '12px' }}>
                  {/* Primary Export Button for Active Task Mode */}
                  <button
                    type="button"
                    className="btn-export-csv"
                    onClick={() => handleExportTaskCsv(taskMode)}
                    disabled={currentTaskItems.length === 0}
                    data-testid="btn-export-csv"
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '6px',
                      padding: '7px 14px',
                      background: currentTaskItems.length > 0 ? 'var(--color-cyan, #00e5ff)' : 'rgba(255, 255, 255, 0.08)',
                      color: currentTaskItems.length > 0 ? '#0a0e17' : '#64748b',
                      border: 'none',
                      borderRadius: '6px',
                      fontWeight: 600,
                      cursor: currentTaskItems.length > 0 ? 'pointer' : 'not-allowed',
                      fontSize: '12px',
                      transition: 'all 0.15s ease',
                    }}
                  >
                    <ExportIcon size={14} color={currentTaskItems.length > 0 ? '#0a0e17' : '#64748b'} />
                    <span>Export {currentTaskLabel} CSV ({currentTaskItems.length})</span>
                  </button>

                  {/* Secondary Export Buttons for other tasks present in basket */}
                  {taskMode !== 'KIS' && kisItems.length > 0 && (
                    <button
                      type="button"
                      onClick={() => handleExportTaskCsv('KIS')}
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '5px',
                        padding: '6px 10px',
                        background: 'rgba(0, 229, 255, 0.1)',
                        color: '#00e5ff',
                        border: '1px solid rgba(0, 229, 255, 0.3)',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        fontSize: '11.5px',
                        fontWeight: 500,
                      }}
                    >
                      <ExportIcon size={13} color="#00e5ff" />
                      <span>Export KIS ({kisItems.length})</span>
                    </button>
                  )}

                  {taskMode !== 'VQA' && vqaItems.length > 0 && (
                    <button
                      type="button"
                      onClick={() => handleExportTaskCsv('VQA')}
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '5px',
                        padding: '6px 10px',
                        background: 'rgba(0, 229, 255, 0.1)',
                        color: '#00e5ff',
                        border: '1px solid rgba(0, 229, 255, 0.3)',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        fontSize: '11.5px',
                        fontWeight: 500,
                      }}
                    >
                      <ExportIcon size={13} color="#00e5ff" />
                      <span>Export Q&A ({vqaItems.length})</span>
                    </button>
                  )}

                  {taskMode !== 'TRAKE' && trakeItems.length > 0 && (
                    <button
                      type="button"
                      onClick={() => handleExportTaskCsv('TRAKE')}
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '5px',
                        padding: '6px 10px',
                        background: 'rgba(0, 229, 255, 0.1)',
                        color: '#00e5ff',
                        border: '1px solid rgba(0, 229, 255, 0.3)',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        fontSize: '11.5px',
                        fontWeight: 500,
                      }}
                    >
                      <ExportIcon size={13} color="#00e5ff" />
                      <span>Export TRAKE ({trakeItems.length})</span>
                    </button>
                  )}

                  <button
                    type="button"
                    className="btn-clear-basket"
                    onClick={handleClearBasket}
                    data-testid="btn-clear-basket"
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '4px',
                      padding: '6px 10px',
                      background: 'rgba(255, 75, 75, 0.12)',
                      color: '#ff4b4b',
                      border: '1px solid rgba(255, 75, 75, 0.3)',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      fontSize: '11.5px',
                      marginLeft: 'auto',
                    }}
                  >
                    <ClearIcon size={12} color="#ff4b4b" />
                    <span>Clear All</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 3. Bottom Supporting Row: Multimodal Evidence Pack Inspector (Only rendered in VQA mode when evidence is available) */}
      {taskMode === 'VQA' && activeEvidence && (
        <div className="evidence-bottom-supporting-row">
          <EvidenceInspector evidence={activeEvidence} />
        </div>
      )}
    </div>
  )
}
