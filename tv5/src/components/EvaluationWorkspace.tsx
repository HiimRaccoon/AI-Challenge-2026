import React, { useState, useEffect } from 'react'
import { useAppState } from '../state/AppContext'
import { telemetry, ClientTelemetryRecord } from '../utils/telemetry'
import { triggerBrowserDownload } from '../utils/submissionExporter'
import {
  CheckIcon,
  FilmstripIcon,
  QuestionIcon,
  DownloadIcon,
} from './Icons'

const formatLocalTime = (isoString: string): string => {
  try {
    const d = new Date(isoString)
    if (isNaN(d.getTime())) return isoString.slice(11, 19)
    const hours = String(d.getHours()).padStart(2, '0')
    const minutes = String(d.getMinutes()).padStart(2, '0')
    const seconds = String(d.getSeconds()).padStart(2, '0')
    return `${hours}:${minutes}:${seconds}`
  } catch {
    return isoString.slice(11, 19)
  }
}

export const EvaluationWorkspace: React.FC = () => {
  const { mode, preprocessRunId } = useAppState()
  const [telemetryLogs, setTelemetryLogs] = useState<ClientTelemetryRecord[]>([])

  useEffect(() => {
    setTelemetryLogs(telemetry.getRecords())
    const interval = setInterval(() => {
      setTelemetryLogs(telemetry.getRecords())
    }, 1000)
    return () => clearInterval(interval)
  }, [])

  const handleDownloadTelemetry = () => {
    const jsonl = telemetry.exportJsonl()
    triggerBrowserDownload(
      jsonl || '{"message":"No telemetry recorded yet"}',
      `telemetry_${new Date().toISOString().slice(0, 10)}.jsonl`,
      'application/x-ndjson;charset=utf-8;'
    )
  }

  return (
    <div className="evaluation-workspace-layout" data-testid="evaluation-workspace" style={{ padding: '24px', overflowY: 'auto', height: '100%', color: 'var(--color-text-primary, #e6edf3)' }}>
      {/* 1. Header Banner */}
      <div style={{ marginBottom: '24px', borderBottom: '1px solid rgba(255, 255, 255, 0.1)', paddingBottom: '16px' }}>
        <h2 style={{ fontSize: '20px', fontWeight: 'bold', color: 'var(--color-cyan, #00e5ff)', margin: '0 0 6px 0' }}>
          Evaluation, Telemetry & Preprocessing Statistics
        </h2>
        <p style={{ margin: 0, color: 'var(--color-text-muted, #8b949e)', fontSize: '13px' }}>
          Official AIC 2026 metric calculation engine, preprocessed corpus statistics, and operational telemetry inspector.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '20px' }}>
        {/* Card 1: Official Metric Engine (T046) */}
        <div style={{ background: '#111722', border: '1px solid rgba(0, 229, 255, 0.2)', borderRadius: '8px', padding: '16px' }} data-testid="metric-engine-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
            <CheckIcon size={18} className="text-cyan" />
            <h3 style={{ margin: 0, fontSize: '15px', color: 'var(--color-cyan, #00e5ff)' }}>Official AIC 2026 Metric Formulas</h3>
          </div>
          <div style={{ fontSize: '13px', lineHeight: '1.6', color: '#c9d1d9' }}>
            <div style={{ marginBottom: '10px', padding: '8px', background: '#0a0e17', borderRadius: '4px' }}>
              <strong>KIS Score:</strong> Binary hit evaluation (1.0 if predicted frame falls in ground-truth target interval, else 0.0).
            </div>
            <div style={{ marginBottom: '10px', padding: '8px', background: '#0a0e17', borderRadius: '4px' }}>
              <strong>VQA Score:</strong> Frame in target interval <em>AND</em> semantic agreement with verified answer. (Returns <code>INCOMPLETE</code> without adjudicator).
            </div>
            <div style={{ marginBottom: '10px', padding: '8px', background: '#0a0e17', borderRadius: '4px' }}>
              <strong>TRAKE Score:</strong> Monotonic temporal order match across all N event positions within hypothesis video.
            </div>
            <div style={{ padding: '8px', background: 'rgba(0, 229, 255, 0.05)', borderLeft: '3px solid #00e5ff', borderRadius: '2px' }}>
              <strong>Organizer Golden Benchmark:</strong><br />
              <span className="tabular-nums">Final Score = 0.7400 (mean of R@1, R@5, R@20, R@50, R@100)</span>
            </div>
          </div>
        </div>

        {/* Card 2: Preprocessing Ingestion Statistics (T048) */}
        <div style={{ background: '#111722', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '8px', padding: '16px' }} data-testid="preprocessing-stats-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
            <FilmstripIcon size={18} className="text-cyan" />
            <h3 style={{ margin: 0, fontSize: '15px', color: '#fff' }}>Authoritative Corpus Statistics</h3>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '12px' }}>
            <div style={{ padding: '12px', background: '#0a0e17', borderRadius: '6px', textAlign: 'center' }}>
              <span style={{ fontSize: '22px', fontWeight: 'bold', color: 'var(--color-cyan, #00e5ff)' }} className="tabular-nums">873</span>
              <div style={{ fontSize: '12px', color: '#8b949e', marginTop: '4px' }}>Authoritative Videos</div>
            </div>
            <div style={{ padding: '12px', background: '#0a0e17', borderRadius: '6px', textAlign: 'center' }}>
              <span style={{ fontSize: '22px', fontWeight: 'bold', color: 'var(--color-cyan, #00e5ff)' }} className="tabular-nums">106,380</span>
              <div style={{ fontSize: '12px', color: '#8b949e', marginTop: '4px' }}>Preprocessed Vectors</div>
            </div>
          </div>
          <div style={{ fontSize: '12px', color: '#8b949e', lineHeight: '1.5' }}>
            <div>• <strong>Run ID:</strong> <span className="tabular-nums">{preprocessRunId || 'run_v1_batch1'}</span></div>
            <div>• <strong>Visual Models:</strong> BEiT-3, BGE-VL, MetaCLIP-2, Perception</div>
            <div>• <strong>Modality Data:</strong> OCR Bboxes, ASR Transcripts, Object Labels</div>
            <div>• <strong>System Mode:</strong> <span style={{ textTransform: 'uppercase', color: mode === 'live' ? '#3fb950' : '#d29922' }}>{mode}</span></div>
          </div>
        </div>

        {/* Card 3: Operational Telemetry Logger (T025) */}
        <div style={{ background: '#111722', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '8px', padding: '16px', gridColumn: '1 / -1' }} data-testid="telemetry-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <QuestionIcon size={18} className="text-cyan" />
              <h3 style={{ margin: 0, fontSize: '15px', color: '#fff' }}>Operational Telemetry Log (T025)</h3>
              <span style={{ fontSize: '12px', background: 'rgba(255, 255, 255, 0.1)', padding: '2px 8px', borderRadius: '10px' }} className="tabular-nums">
                {telemetryLogs.length} events
              </span>
            </div>
            <button
              type="button"
              onClick={handleDownloadTelemetry}
              data-testid="btn-download-telemetry"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
                padding: '6px 12px',
                background: 'rgba(0, 229, 255, 0.12)',
                color: 'var(--color-cyan, #00e5ff)',
                border: '1px solid rgba(0, 229, 255, 0.35)',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '12px',
                fontWeight: '600',
                transition: 'all 0.15s ease',
              }}
            >
              <DownloadIcon size={14} color="var(--color-cyan, #00e5ff)" />
              <span>Download Telemetry (JSONL)</span>
            </button>
          </div>

          <div style={{ background: '#0a0e17', borderRadius: '6px', padding: '10px', maxHeight: '180px', overflowY: 'auto', fontFamily: 'monospace', fontSize: '11px', color: '#8b949e' }}>
            {telemetryLogs.length === 0 ? (
              <div>No operator telemetry events recorded yet in this session. Perform searches, stepping, or basket actions to populate.</div>
            ) : (
              telemetryLogs.map((log, i) => (
                <div key={i} style={{ marginBottom: '4px', borderBottom: '1px solid rgba(255, 255, 255, 0.05)', paddingBottom: '3px' }}>
                  <span style={{ color: '#58a6ff' }}>[{formatLocalTime(log.timestamp)}]</span>{' '}
                  <span style={{ color: '#00e5ff', fontWeight: 'bold' }}>{log.action}</span>{' '}
                  <span style={{ color: '#e6edf3' }}>mode={log.taskMode}</span>{' '}
                  {log.details && <span>{JSON.stringify(log.details)}</span>}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
