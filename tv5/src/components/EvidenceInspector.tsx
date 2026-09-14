import React, { useState } from 'react'
import { EvidencePack } from '../types/contracts'
import {
  OcrIcon,
  AsrIcon,
  ObjectIcon,
  FilmstripIcon,
  CheckIcon,
  WarningIcon,
} from './Icons'

interface EvidenceInspectorProps {
  evidence: EvidencePack | null | undefined
}

type EvidenceTab = 'ocr' | 'asr' | 'object' | 'metadata' | 'frames' | 'provenance'

export const EvidenceInspector: React.FC<EvidenceInspectorProps> = ({ evidence }) => {
  const [activeTab, setActiveTab] = useState<EvidenceTab>('ocr')

  if (!evidence) {
    return (
      <div className="evidence-inspector-empty" data-testid="evidence-inspector-empty">
        <span className="empty-text">No evidence pack available for this candidate.</span>
      </div>
    )
  }

  const ocrCount = evidence.ocr_evidence?.length ?? 0
  const asrCount = evidence.asr_evidence?.length ?? 0
  const objectCount = evidence.object_evidence?.length ?? 0
  const metaCount = evidence.metadata_evidence?.length ?? 0
  const framesCount = evidence.selected_frames?.length ?? 0

  const ocrAvail = evidence.availability?.ocr || (ocrCount > 0 ? 'available' : 'empty')
  const asrAvail = evidence.availability?.asr || (asrCount > 0 ? 'available' : 'empty')
  const objAvail = evidence.availability?.object || (objectCount > 0 ? 'available' : 'empty')

  return (
    <div className="evidence-inspector-card" data-testid="evidence-inspector">
      <div className="evidence-inspector-header">
        <div className="inspector-title-row">
          <span className="inspector-title">Multimodal Evidence Pack</span>
          <span className="inspector-source-tag tabular-nums" data-testid="evidence-source-tag">
            Evidence Source: {evidence.video_id} · Frame {evidence.frame_id}
          </span>
        </div>

        {/* Evidence Modality Tabs */}
        <div className="evidence-modality-tabs" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'ocr'}
            className={`modality-tab-btn ${activeTab === 'ocr' ? 'is-active' : ''}`}
            onClick={() => setActiveTab('ocr')}
            data-testid="tab-evidence-ocr"
          >
            <OcrIcon size={13} />
            <span>OCR ({ocrCount})</span>
            {ocrAvail === 'available' ? (
              <span className="avail-dot dot-green" title="OCR Available" />
            ) : (
              <span className="avail-dot dot-gray" title="OCR Empty" />
            )}
          </button>

          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'asr'}
            className={`modality-tab-btn ${activeTab === 'asr' ? 'is-active' : ''}`}
            onClick={() => setActiveTab('asr')}
            data-testid="tab-evidence-asr"
          >
            <AsrIcon size={13} />
            <span>ASR ({asrCount})</span>
            {asrAvail === 'available' ? (
              <span className="avail-dot dot-green" title="ASR Available" />
            ) : (
              <span className="avail-dot dot-gray" title="ASR Empty" />
            )}
          </button>

          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'object'}
            className={`modality-tab-btn ${activeTab === 'object' ? 'is-active' : ''}`}
            onClick={() => setActiveTab('object')}
            data-testid="tab-evidence-object"
          >
            <ObjectIcon size={13} />
            <span>Objects ({objectCount})</span>
            {objAvail === 'available' ? (
              <span className="avail-dot dot-green" title="Objects Available" />
            ) : (
              <span className="avail-dot dot-gray" title="Objects Empty" />
            )}
          </button>

          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'frames'}
            className={`modality-tab-btn ${activeTab === 'frames' ? 'is-active' : ''}`}
            onClick={() => setActiveTab('frames')}
            data-testid="tab-evidence-frames"
          >
            <FilmstripIcon size={13} />
            <span>Frames ({framesCount})</span>
          </button>

          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'metadata'}
            className={`modality-tab-btn ${activeTab === 'metadata' ? 'is-active' : ''}`}
            onClick={() => setActiveTab('metadata')}
            data-testid="tab-evidence-metadata"
          >
            <span>Meta ({metaCount})</span>
          </button>

          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'provenance'}
            className={`modality-tab-btn ${activeTab === 'provenance' ? 'is-active' : ''}`}
            onClick={() => setActiveTab('provenance')}
            data-testid="tab-evidence-provenance"
          >
            <span>Provenance</span>
          </button>
        </div>
      </div>

      {/* Tab Body Viewport */}
      <div className="evidence-inspector-body">
        {/* ── 1. OCR Evidence Tab ── */}
        {activeTab === 'ocr' && (
          <div className="evidence-tab-pane" data-testid="pane-evidence-ocr">
            {ocrCount === 0 ? (
              <div className="evidence-empty-branch">
                <span className="empty-branch-text">
                  No OCR text detections found for this frame window.
                </span>
                <span className="branch-status-pill pill-empty">Status: Empty</span>
              </div>
            ) : (
              <div className="evidence-records-list">
                {evidence.ocr_evidence.map((det) => (
                  <div key={det.detection_id} className="evidence-item-card ocr-card">
                    <div className="item-card-top">
                      <span className="ocr-text-raw">"{det.raw_text}"</span>
                      {det.confidence != null && (
                        <span className="item-confidence-tag tabular-nums">
                          Conf: {(det.confidence * 100).toFixed(1)}%
                        </span>
                      )}
                    </div>
                    {det.normalized_text && det.normalized_text !== det.raw_text && (
                      <div className="item-norm-row">
                        <span className="norm-label">Normalized:</span>
                        <span className="norm-text">{det.normalized_text}</span>
                      </div>
                    )}
                    <div className="item-geo-row tabular-nums">
                      <span className="geo-label">BBox (norm):</span>
                      <span className="geo-coords">
                        [{det.bbox_xyxy_norm.map((n) => n.toFixed(3)).join(', ')}]
                      </span>
                      {det.model_name && (
                        <span className="item-model-tag">Model: {det.model_name}</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── 2. ASR Evidence Tab ── */}
        {activeTab === 'asr' && (
          <div className="evidence-tab-pane" data-testid="pane-evidence-asr">
            {asrCount === 0 ? (
              <div className="evidence-empty-branch">
                <span className="empty-branch-text">
                  No ASR audio transcripts available for this temporal window.
                </span>
                <span className="branch-status-pill pill-empty">Status: Empty</span>
              </div>
            ) : (
              <div className="evidence-records-list">
                {evidence.asr_evidence.map((seg) => (
                  <div key={seg.segment_id} className="evidence-item-card asr-card">
                    {/* Top Meta Row */}
                    <div className="asr-card-top-row">
                      <div className="asr-timestamp-badge tabular-nums">
                        {(seg.start_ms / 1000).toFixed(1)}s → {(seg.end_ms / 1000).toFixed(1)}s
                      </div>
                      <div className="asr-meta-right-group">
                        {seg.confidence != null && (
                          <span className="item-confidence-tag tabular-nums">
                            Conf: {(seg.confidence * 100).toFixed(1)}%
                          </span>
                        )}
                        {seg.language && (
                          <span className="item-lang-tag">
                            {seg.language.toUpperCase()}
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Primary Transcript Content */}
                    <div className="asr-transcript-box">
                      <span className="asr-text-content">"{seg.text}"</span>
                    </div>

                    {/* Word-Level Tokens Detail (Secondary) */}
                    {seg.words && seg.words.length > 0 && (
                      <div className="asr-words-section">
                        <span className="asr-words-heading">Word Tokens:</span>
                        <div className="asr-words-chips">
                          {seg.words.map((w, idx) => (
                            <span
                              key={idx}
                              className="asr-word-pill tabular-nums"
                              title={
                                w.start_ms != null
                                  ? `${(w.start_ms / 1000).toFixed(2)}s`
                                  : undefined
                              }
                            >
                              <span className="word-text">{w.word}</span>
                              {w.probability != null && (
                                <span className="word-prob">
                                  {(w.probability * 100).toFixed(0)}%
                                </span>
                              )}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── 3. Object Evidence Tab ── */}
        {activeTab === 'object' && (
          <div className="evidence-tab-pane" data-testid="pane-evidence-object">
            {objectCount === 0 ? (
              <div className="evidence-empty-branch">
                <span className="empty-branch-text">
                  No object detection bounding boxes for this frame.
                </span>
                <span className="branch-status-pill pill-empty">Status: Empty</span>
              </div>
            ) : (
              <div className="evidence-records-list">
                {evidence.object_evidence.map((obj) => (
                  <div key={obj.detection_id} className="evidence-item-card object-card">
                    <div className="item-card-top">
                      <span className="obj-label-text">{obj.label}</span>
                      {obj.canonical_label && obj.canonical_label !== obj.label && (
                        <span className="obj-canon-tag">({obj.canonical_label})</span>
                      )}
                      {obj.confidence != null && (
                        <span className="item-confidence-tag tabular-nums">
                          {(obj.confidence * 100).toFixed(1)}%
                        </span>
                      )}
                    </div>
                    <div className="item-geo-row tabular-nums">
                      <span className="geo-label">BBox (norm):</span>
                      <span className="geo-coords">
                        [{obj.bbox_xyxy_norm.map((n) => n.toFixed(3)).join(', ')}]
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── 4. Canonical Frames Evidence Tab ── */}
        {activeTab === 'frames' && (
          <div className="evidence-tab-pane" data-testid="pane-evidence-frames">
            {framesCount === 0 ? (
              <div className="evidence-empty-branch">
                <span className="empty-branch-text">No multi-frame sequence records.</span>
              </div>
            ) : (
              <div className="evidence-frames-track">
                {evidence.selected_frames.map((fr, idx) => (
                  <div key={idx} className="evidence-frame-pill tabular-nums">
                    <span className="frame-pill-id">Frame {fr.frame_id}</span>
                    <span className="frame-pill-time">
                      {(fr.timestamp_ms / 1000).toFixed(1)}s
                    </span>
                    {fr.submission_selectable ? (
                      <span className="frame-selectable-tag">
                        <CheckIcon size={10} /> Selectable
                      </span>
                    ) : (
                      <span className="frame-preview-tag">
                        <WarningIcon size={10} /> Preview
                      </span>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── 5. Metadata Tab ── */}
        {activeTab === 'metadata' && (
          <div className="evidence-tab-pane" data-testid="pane-evidence-metadata">
            {metaCount === 0 ? (
              <div className="evidence-empty-branch">
                <span className="empty-branch-text">No structured catalogue metadata.</span>
              </div>
            ) : (
              <div className="evidence-records-list">
                {evidence.metadata_evidence.map((meta) => (
                  <div key={meta.metadata_id} className="evidence-item-card meta-card">
                    <div className="item-card-top">
                      <span className="meta-source-name">{meta.source}</span>
                      <span className="meta-id-tag">{meta.metadata_id}</span>
                    </div>
                    <pre className="meta-json-view">
                      {JSON.stringify(meta.values, null, 2)}
                    </pre>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── 6. Provenance Tab ── */}
        {activeTab === 'provenance' && (
          <div className="evidence-tab-pane" data-testid="pane-evidence-provenance">
            <div className="provenance-details-table tabular-nums">
              <div className="prov-row">
                <span className="prov-key">Query ID:</span>
                <span className="prov-val">{evidence.query_id}</span>
              </div>
              <div className="prov-row">
                <span className="prov-key">Video ID:</span>
                <span className="prov-val text-cyan">{evidence.video_id}</span>
              </div>
              <div className="prov-row">
                <span className="prov-key">Frame ID:</span>
                <span className="prov-val">{evidence.frame_id}</span>
              </div>
              <div className="prov-row">
                <span className="prov-key">Keyframe Path:</span>
                <span className="prov-val">{evidence.keyframe_path || 'N/A'}</span>
              </div>
              <div className="prov-row">
                <span className="prov-key">Submission Selectable:</span>
                <span className="prov-val">
                  {evidence.provenance?.submission_selectable === false ? 'No (Preview)' : 'Yes'}
                </span>
              </div>
              <div className="prov-row">
                <span className="prov-key">Raw Provenance:</span>
                <span className="prov-val">
                  {JSON.stringify(evidence.provenance || {})}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
