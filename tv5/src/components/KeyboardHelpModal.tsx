import React from 'react'
import { CheckIcon, ClearIcon, QuestionIcon } from './Icons'

export interface ShortcutDefinition {
  category: string
  keyCombo: string
  description: string
  focusSafe: boolean
}

export const SHORTCUT_REGISTRY: ShortcutDefinition[] = [
  { category: 'Search & Query', keyCombo: 'Ctrl+Enter / Cmd+Enter', description: 'Submit search query', focusSafe: false },
  { category: 'Staging / Pin', keyCombo: 'P', description: 'Ghim / Bỏ ghim video tiềm năng vào Khay ghim', focusSafe: true },
  { category: 'Submission', keyCombo: 'B', description: 'Lưu / Gỡ nhanh frame đang xem vào Giỏ nộp bài', focusSafe: true },
  { category: 'Frame Inspection', keyCombo: 'Ctrl+Left / Ctrl+Right', description: 'Soi bước từng frame liền kề (-1 / +1 frame)', focusSafe: true },
  { category: 'Frame Inspection', keyCombo: ', / .', description: 'Bước lùi / tiến 1 frame lân cận', focusSafe: true },
  { category: 'Frame Inspection', keyCombo: 'Home / 0', description: 'Reset về mốc Anchor frame ban đầu', focusSafe: true },
  { category: 'Media Player', keyCombo: 'Space / K', description: 'Play / Pause video', focusSafe: true },
  { category: 'Media Player', keyCombo: 'Left / Right (J / L)', description: 'Tua lùi / tiến video theo số giây đã cài', focusSafe: true },
  { category: 'Media Player', keyCombo: 'F / Double click', description: 'Phóng to / Thu nhỏ toàn màn hình (kèm thanh lia chuột)', focusSafe: true },
  { category: 'Media Player', keyCombo: '1 / 2 / 3', description: 'Đổi tốc độ phát 1.0x / 1.5x / 2.0x', focusSafe: true },
  { category: 'General', keyCombo: 'Escape', description: 'Đóng modal / popup / Thoát toàn màn hình', focusSafe: true },
  { category: 'Help Guide', keyCombo: '?', description: 'Mở bảng hướng dẫn phím tắt', focusSafe: true },
]

interface KeyboardHelpModalProps {
  isOpen: boolean
  onClose: () => void
}

export const KeyboardHelpModal: React.FC<KeyboardHelpModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null

  return (
    <div
      className="modal-overlay"
      data-testid="keyboard-help-modal"
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(4, 8, 16, 0.82)',
        backdropFilter: 'blur(10px)',
        WebkitBackdropFilter: 'blur(10px)',
        zIndex: 99999,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '20px',
        animation: 'modalFadeIn 0.2s cubic-bezier(0.16, 1, 0.3, 1)',
      }}
    >
      <div
        className="modal-content"
        onClick={(e) => e.stopPropagation()}
        style={{
          width: '100%',
          maxWidth: '720px',
          background: 'linear-gradient(180deg, #0f172a 0%, #0a0f1d 100%)',
          border: '1px solid rgba(0, 229, 255, 0.35)',
          borderRadius: '12px',
          boxShadow: '0 25px 65px rgba(0, 0, 0, 0.85), 0 0 35px rgba(0, 229, 255, 0.15)',
          overflow: 'hidden',
          animation: 'modalPopIn 0.25s cubic-bezier(0.16, 1, 0.3, 1)',
        }}
      >
        {/* Header */}
        <div
          className="modal-header"
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '16px 22px',
            background: 'rgba(0, 229, 255, 0.04)',
            borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div
              style={{
                width: '32px',
                height: '32px',
                borderRadius: '8px',
                background: 'rgba(0, 229, 255, 0.12)',
                border: '1px solid rgba(0, 229, 255, 0.3)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <QuestionIcon size={18} color="#00e5ff" />
            </div>
            <h2 className="modal-title" style={{ margin: 0, fontSize: '16px', fontWeight: 700, color: '#f8fafc', letterSpacing: '0.3px' }}>
              Keyboard Shortcuts Guide
            </h2>
          </div>
          <button
            type="button"
            className="close-btn"
            onClick={onClose}
            data-testid="close-keyboard-help"
            title="Close (Escape)"
            style={{
              background: 'transparent',
              border: 'none',
              color: '#94a3b8',
              cursor: 'pointer',
              padding: '6px',
              borderRadius: '6px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.15s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = '#ff4b4b'
              e.currentTarget.style.background = 'rgba(255, 75, 75, 0.15)'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = '#94a3b8'
              e.currentTarget.style.background = 'transparent'
            }}
          >
            <ClearIcon size={16} />
          </button>
        </div>

        {/* Body */}
        <div className="modal-body" style={{ padding: '20px 22px' }}>
          <div
            className="modal-intro"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '10px 14px',
              background: 'rgba(16, 185, 129, 0.08)',
              borderLeft: '3px solid #10b981',
              borderRadius: '4px',
              fontSize: '12.5px',
              color: '#cbd5e1',
              marginBottom: '16px',
              lineHeight: 1.45,
            }}
          >
            <CheckIcon size={14} color="#10b981" />
            <span>
              All operator shortcuts include <strong>focus guards</strong>: they do not trigger destructively while typing in search, answers, or text fields.
            </span>
          </div>

          <div className="shortcuts-table-wrap" style={{ borderRadius: '8px', overflow: 'hidden', border: '1px solid rgba(255, 255, 255, 0.07)' }}>
            <table className="shortcuts-table" style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
              <thead>
                <tr style={{ background: '#131c2e', borderBottom: '1px solid rgba(255, 255, 255, 0.1)', color: '#94a3b8', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.6px' }}>
                  <th style={{ padding: '10px 14px', fontWeight: 600 }}>Category</th>
                  <th style={{ padding: '10px 14px', fontWeight: 600 }}>Key Combo</th>
                  <th style={{ padding: '10px 14px', fontWeight: 600 }}>Description</th>
                  <th style={{ padding: '10px 14px', fontWeight: 600 }}>Focus Guarded</th>
                </tr>
              </thead>
              <tbody>
                {SHORTCUT_REGISTRY.map((s, i) => (
                  <tr
                    key={i}
                    style={{
                      borderBottom: i < SHORTCUT_REGISTRY.length - 1 ? '1px solid rgba(255, 255, 255, 0.04)' : 'none',
                      background: i % 2 === 0 ? 'rgba(255, 255, 255, 0.015)' : 'transparent',
                      transition: 'background 0.12s ease',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = 'rgba(0, 229, 255, 0.06)'
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = i % 2 === 0 ? 'rgba(255, 255, 255, 0.015)' : 'transparent'
                    }}
                  >
                    <td style={{ padding: '10px 14px' }}>
                      <span
                        className="badge-category"
                        style={{
                          background: 'rgba(0, 229, 255, 0.1)',
                          color: '#00e5ff',
                          border: '1px solid rgba(0, 229, 255, 0.25)',
                          borderRadius: '4px',
                          padding: '3px 8px',
                          fontSize: '11px',
                          fontWeight: 600,
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {s.category}
                      </span>
                    </td>
                    <td style={{ padding: '10px 14px' }}>
                      <kbd
                        className="key-combo"
                        style={{
                          background: '#192438',
                          border: '1px solid rgba(255, 255, 255, 0.18)',
                          borderBottom: '2px solid rgba(0, 229, 255, 0.5)',
                          borderRadius: '4px',
                          padding: '3px 7px',
                          color: '#ffffff',
                          fontFamily: 'Consolas, Monaco, monospace',
                          fontSize: '12px',
                          fontWeight: 600,
                          boxShadow: '0 2px 4px rgba(0, 0, 0, 0.3)',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {s.keyCombo}
                      </kbd>
                    </td>
                    <td style={{ padding: '10px 14px', color: '#e2e8f0' }}>{s.description}</td>
                    <td style={{ padding: '10px 14px' }}>
                      {s.focusSafe ? (
                        <span className="guard-safe" style={{ color: '#10b981', display: 'inline-flex', alignItems: 'center', gap: '5px', fontSize: '12px', fontWeight: 500 }}>
                          <CheckIcon size={12} color="#10b981" />
                          <span>Yes (Focus Safe)</span>
                        </span>
                      ) : (
                        <span style={{ color: '#94a3b8', fontSize: '12px' }}>Contextual</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}
