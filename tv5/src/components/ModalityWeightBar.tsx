import React, { useState } from 'react'
import { useAppState, useAppDispatch } from '../state/AppContext'
import { ModalityWeights } from '../state/appState'

export const ModalityWeightBar: React.FC = () => {
  const { modalityWeights } = useAppState()
  const dispatch = useAppDispatch()
  const [isExpanded, setIsExpanded] = useState(false)

  const weights = modalityWeights || {
    visual: 100,
    ocr: 100,
    asr: 100,
    object: 50,
  }

  const handleSliderChange = (modality: keyof ModalityWeights, val: number) => {
    dispatch({
      type: 'SET_MODALITY_WEIGHTS',
      payload: { [modality]: val },
    })
  }

  const applyPreset = (preset: ModalityWeights) => {
    dispatch({
      type: 'SET_MODALITY_WEIGHTS',
      payload: preset,
    })
  }

  const isBalanced =
    weights.visual === 100 && weights.ocr === 100 && weights.asr === 100 && weights.object === 50
  const isVisualMax =
    weights.visual === 100 && weights.ocr === 20 && weights.asr === 10 && weights.object === 50
  const isOcrMax =
    weights.visual === 30 && weights.ocr === 100 && weights.asr === 10 && weights.object === 20
  const isAsrMax =
    weights.visual === 20 && weights.ocr === 20 && weights.asr === 100 && weights.object === 10

  return (
    <div
      className="modality-weight-bar"
      style={{
        background: 'rgba(15, 23, 42, 0.65)',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        borderRadius: '8px',
        padding: '6px 10px',
        margin: '6px 0 10px 0',
        transition: 'all 0.2s ease',
      }}
    >
      {/* Header / Compact Bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '8px',
          fontSize: '11px',
          fontWeight: 600,
        }}
      >
        <div
          onClick={() => setIsExpanded(!isExpanded)}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            cursor: 'pointer',
            userSelect: 'none',
            flexWrap: 'wrap',
          }}
          title="Bấm để mở rộng / thu gọn thanh chỉnh trọng số tìm kiếm"
        >
          <span style={{ color: '#00e5ff', display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span style={{ fontSize: '12px' }}>⚖️</span> Trọng số:
          </span>

          <span
            style={{
              color: '#38bdf8',
              background: 'rgba(56, 189, 248, 0.12)',
              padding: '1px 5px',
              borderRadius: '4px',
              border: '1px solid rgba(56, 189, 248, 0.25)',
              fontSize: '10px',
            }}
          >
            🖼️ Ảnh: <strong>{weights.visual}%</strong>
          </span>

          <span
            style={{
              color: '#34d399',
              background: 'rgba(52, 211, 153, 0.12)',
              padding: '1px 5px',
              borderRadius: '4px',
              border: '1px solid rgba(52, 211, 153, 0.25)',
              fontSize: '10px',
            }}
          >
            🔤 Chữ OCR: <strong>{weights.ocr}%</strong>
          </span>

          <span
            style={{
              color: '#fbbf24',
              background: 'rgba(251, 191, 36, 0.12)',
              padding: '1px 5px',
              borderRadius: '4px',
              border: '1px solid rgba(251, 191, 36, 0.25)',
              fontSize: '10px',
            }}
          >
            🎙️ Thoại ASR: <strong>{weights.asr}%</strong>
          </span>

          <span
            style={{
              color: '#a78bfa',
              background: 'rgba(167, 139, 250, 0.12)',
              padding: '1px 5px',
              borderRadius: '4px',
              border: '1px solid rgba(167, 139, 250, 0.25)',
              fontSize: '10px',
            }}
          >
            📦 Vật thể: <strong>{weights.object}%</strong>
          </span>
        </div>

        <button
          type="button"
          onClick={() => setIsExpanded(!isExpanded)}
          style={{
            background: 'rgba(255, 255, 255, 0.06)',
            color: '#94a3b8',
            border: '1px solid rgba(255, 255, 255, 0.12)',
            borderRadius: '4px',
            padding: '2px 7px',
            fontSize: '10px',
            fontWeight: 700,
            cursor: 'pointer',
          }}
        >
          {isExpanded ? '▲ Thu gọn' : '▼ Tùy chỉnh'}
        </button>
      </div>

      {/* Expanded Sliders & Presets Panel */}
      {isExpanded && (
        <div style={{ marginTop: '10px', paddingTop: '8px', borderTop: '1px solid rgba(255, 255, 255, 0.08)' }}>
          {/* Quick Presets */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '10px', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '10.5px', color: '#64748b', fontWeight: 600 }}>Cấu hình nhanh:</span>

            <button
              type="button"
              onClick={() => applyPreset({ visual: 100, ocr: 100, asr: 100, object: 50 })}
              style={{
                background: isBalanced ? 'rgba(0, 229, 255, 0.25)' : 'rgba(255, 255, 255, 0.05)',
                color: isBalanced ? '#00e5ff' : '#cbd5e1',
                border: isBalanced ? '1px solid #00e5ff' : '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '4px',
                padding: '2px 8px',
                fontSize: '10.5px',
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              ⚖️ Cân bằng (Mặc định)
            </button>

            <button
              type="button"
              onClick={() => applyPreset({ visual: 100, ocr: 20, asr: 10, object: 50 })}
              style={{
                background: isVisualMax ? 'rgba(56, 189, 248, 0.25)' : 'rgba(255, 255, 255, 0.05)',
                color: isVisualMax ? '#38bdf8' : '#cbd5e1',
                border: isVisualMax ? '1px solid #38bdf8' : '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '4px',
                padding: '2px 8px',
                fontSize: '10.5px',
                fontWeight: 600,
                cursor: 'pointer',
              }}
              title="Ưu tiên tối đa hình ảnh và màu sắc cảnh"
            >
              🖼️ Ưu tiên Ảnh (Visual Max)
            </button>

            <button
              type="button"
              onClick={() => applyPreset({ visual: 30, ocr: 100, asr: 10, object: 20 })}
              style={{
                background: isOcrMax ? 'rgba(52, 211, 153, 0.25)' : 'rgba(255, 255, 255, 0.05)',
                color: isOcrMax ? '#34d399' : '#cbd5e1',
                border: isOcrMax ? '1px solid #34d399' : '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '4px',
                padding: '2px 8px',
                fontSize: '10.5px',
                fontWeight: 600,
                cursor: 'pointer',
              }}
              title="Ưu tiên tối đa chữ viết / bảng hiệu / biển số xe"
            >
              🔤 Ưu tiên Chữ (OCR Max)
            </button>

            <button
              type="button"
              onClick={() => applyPreset({ visual: 20, ocr: 20, asr: 100, object: 10 })}
              style={{
                background: isAsrMax ? 'rgba(251, 191, 36, 0.25)' : 'rgba(255, 255, 255, 0.05)',
                color: isAsrMax ? '#fbbf24' : '#cbd5e1',
                border: isAsrMax ? '1px solid #fbbf24' : '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '4px',
                padding: '2px 8px',
                fontSize: '10.5px',
                fontWeight: 600,
                cursor: 'pointer',
              }}
              title="Ưu tiên tối đa giọng nói / hội thoại / phỏng vấn"
            >
              🎙️ Ưu tiên Thoại (ASR Max)
            </button>
          </div>

          {/* Interactive Sliders Grid */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '10px 16px',
            }}
          >
            {/* Visual Slider */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#38bdf8', fontWeight: 600 }}>
                <span>🖼️ Visual (Hình ảnh)</span>
                <span className="tabular-nums">{weights.visual}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                step="5"
                value={weights.visual}
                onChange={(e) => handleSliderChange('visual', parseInt(e.target.value, 10))}
                style={{ accentColor: '#38bdf8', cursor: 'pointer', height: '5px' }}
              />
            </div>

            {/* OCR Slider */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#34d399', fontWeight: 600 }}>
                <span>🔤 OCR (Chữ trong video)</span>
                <span className="tabular-nums">{weights.ocr}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                step="5"
                value={weights.ocr}
                onChange={(e) => handleSliderChange('ocr', parseInt(e.target.value, 10))}
                style={{ accentColor: '#34d399', cursor: 'pointer', height: '5px' }}
              />
            </div>

            {/* ASR Slider */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#fbbf24', fontWeight: 600 }}>
                <span>🎙️ ASR (Thuyết minh / Thoại)</span>
                <span className="tabular-nums">{weights.asr}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                step="5"
                value={weights.asr}
                onChange={(e) => handleSliderChange('asr', parseInt(e.target.value, 10))}
                style={{ accentColor: '#fbbf24', cursor: 'pointer', height: '5px' }}
              />
            </div>

            {/* Object Slider */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '3px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#a78bfa', fontWeight: 600 }}>
                <span>📦 Object (Vật thể phát hiện)</span>
                <span className="tabular-nums">{weights.object}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                step="5"
                value={weights.object}
                onChange={(e) => handleSliderChange('object', parseInt(e.target.value, 10))}
                style={{ accentColor: '#a78bfa', cursor: 'pointer', height: '5px' }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
