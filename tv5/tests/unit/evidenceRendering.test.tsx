import React from 'react'
import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { EvidenceInspector } from '../../src/components/EvidenceInspector'
import { EvidencePack } from '../../src/types/contracts'

describe('T034 / T035 Evidence Rendering & Provenance UI', () => {
  const sampleEvidence: EvidencePack = {
    video_id: 'L01_V001',
    frame_id: 1050,
    timestamp_ms: 42000,
    ocr_evidence: [
      {
        detection_id: 'ocr-1',
        raw_text: 'CHUC MUNG NAM MOI',
        normalized_text: 'chuc mung nam moi',
        confidence: 0.95,
        bbox_xyxy_norm: [0.1, 0.2, 0.5, 0.4],
        model_name: 'paddleocr-v4',
      },
    ],
    asr_evidence: [
      {
        segment_id: 'asr-1',
        start_ms: 40000,
        end_ms: 45000,
        text: 'Xin kính chào quý vị đại biểu',
        confidence: 0.92,
        language: 'vi',
        words: [
          { word: 'Xin', start_ms: 40000, end_ms: 40500, probability: 0.98 },
          { word: 'kính', start_ms: 40500, end_ms: 41000, probability: 0.95 },
        ],
      },
    ],
    object_evidence: [
      {
        detection_id: 'obj-1',
        label: 'person',
        confidence: 0.88,
        bbox_xyxy_norm: [0.3, 0.1, 0.7, 0.9],
        model_name: 'yolov8x-world',
        crop_ref: 'crops/L01_V001_1050_obj1.jpg',
      },
    ],
    metadata_evidence: [
      {
        metadata_id: 'meta-1',
        source: 'video_metadata_registry',
        values: { event_type: 'festival_opening' },
      },
    ],
    selected_frames: [
      {
        video_id: 'L01_V001',
        frame_id: 1050,
        timestamp_ms: 42000,
        pts: 42000,
        time_base: '1/1000',
        preprocess_run_id: 'run_v1_batch1',
        mapping_guaranteed: true,
        submission_selectable: true,
        identity_source: 'certified_anchor',
      },
    ],
    availability: {
      ocr: 'available',
      asr: 'available',
      object: 'available',
      metadata: 'available',
    },
    provenance_summary: {
      preprocess_run_id: 'run_v1_batch1',
      wp04_run_id: 'wp04_run_v1',
      models: {
        ocr: 'paddleocr-v4',
        asr: 'whisper-large-v3',
        object: 'yolov8x-world',
      },
    },
  }

  it('renders empty state when evidence is null or undefined without fabrication', () => {
    render(<EvidenceInspector evidence={null} />)
    expect(screen.getByTestId('evidence-inspector-empty')).toBeInTheDocument()
    expect(
      screen.getByText('No evidence pack available for this candidate.')
    ).toBeInTheDocument()
  })

  it('renders OCR evidence with bounding box coordinates, text, and confidence', () => {
    render(<EvidenceInspector evidence={sampleEvidence} />)
    expect(screen.getByTestId('evidence-source-tag')).toHaveTextContent(
      'L01_V001 · Frame 1050'
    )

    // OCR Tab is default
    expect(screen.getByTestId('pane-evidence-ocr')).toBeInTheDocument()
    expect(screen.getByText('"CHUC MUNG NAM MOI"')).toBeInTheDocument()
    expect(screen.getByText('chuc mung nam moi')).toBeInTheDocument()
    expect(screen.getByText('Conf: 95.0%')).toBeInTheDocument()
    expect(screen.getByText('Model: paddleocr-v4')).toBeInTheDocument()
    expect(screen.getByText('[0.100, 0.200, 0.500, 0.400]')).toBeInTheDocument()
  })

  it('renders ASR transcripts with timestamps and word tokens', () => {
    render(<EvidenceInspector evidence={sampleEvidence} />)
    fireEvent.click(screen.getByTestId('tab-evidence-asr'))

    expect(screen.getByTestId('pane-evidence-asr')).toBeInTheDocument()
    expect(screen.getByText('40.0s → 45.0s')).toBeInTheDocument()
    expect(
      screen.getByText('"Xin kính chào quý vị đại biểu"')
    ).toBeInTheDocument()
    expect(screen.getByText('VI')).toBeInTheDocument()
    expect(screen.getByText('Xin')).toBeInTheDocument()
    expect(screen.getByText('98%')).toBeInTheDocument()
  })

  it('renders Object detections with label and bbox', () => {
    render(<EvidenceInspector evidence={sampleEvidence} />)
    fireEvent.click(screen.getByTestId('tab-evidence-object'))

    expect(screen.getByTestId('pane-evidence-object')).toBeInTheDocument()
    expect(screen.getByText('person')).toBeInTheDocument()
    expect(screen.getByText('88.0%')).toBeInTheDocument()
    expect(screen.getByText('[0.300, 0.100, 0.700, 0.900]')).toBeInTheDocument()
  })

  it('renders Frames tab with submission selectability and proof provenance', () => {
    render(<EvidenceInspector evidence={sampleEvidence} />)
    fireEvent.click(screen.getByTestId('tab-evidence-frames'))

    expect(screen.getByTestId('pane-evidence-frames')).toBeInTheDocument()
    expect(screen.getByText('Frame 1050')).toBeInTheDocument()
    expect(screen.getByText(/Selectable/i)).toBeInTheDocument()
    expect(screen.getByText('42.0s')).toBeInTheDocument()
  })

  it('handles empty branches cleanly with explicit empty status without errors', () => {
    const emptyBranchEvidence: EvidencePack = {
      video_id: 'L01_V002',
      frame_id: 200,
      timestamp_ms: 8000,
      ocr_evidence: [],
      asr_evidence: [],
      object_evidence: [],
      metadata_evidence: [],
      selected_frames: [],
      availability: {
        ocr: 'empty',
        asr: 'empty',
        object: 'empty',
        metadata: 'empty',
      },
    }

    render(<EvidenceInspector evidence={emptyBranchEvidence} />)

    // OCR empty
    expect(screen.getByText(/No OCR text detections found/i)).toBeInTheDocument()

    // ASR empty
    fireEvent.click(screen.getByTestId('tab-evidence-asr'))
    expect(
      screen.getByText(/No ASR audio transcripts available/i)
    ).toBeInTheDocument()

    // Objects empty
    fireEvent.click(screen.getByTestId('tab-evidence-object'))
    expect(
      screen.getByText(/No object detection bounding boxes for this frame/i)
    ).toBeInTheDocument()
  })
})
