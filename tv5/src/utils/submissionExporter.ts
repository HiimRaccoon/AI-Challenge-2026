/**
 * Client-Side Submission CSV and Package Exporter
 * Implements strict competition formatting:
 * - KIS: <video_id>,<frame_id>
 * - VQA: <video_id>,<frame_id>,<answer>
 * - TRAKE: <video_id>,<frame_1>,<frame_2>,...,<frame_N>
 * - Capacity: Max 100 rows per query
 * - Video ID: Non-empty, no '.mp4'
 * - Frame ID: Non-negative integers
 */
import { BasketItem } from '../types/contracts'

/**
 * A safe filename for one query's submission CSV.
 */
export function queryCsvFilename(queryId: string): string {
  const safe = queryId.replace(/[^A-Za-z0-9_-]/g, '_')
  return `${safe}.csv`
}

export interface FormattedSubmissionRow {
  task: 'KIS' | 'VQA' | 'TRAKE'
  videoId: string
  csvLine: string
}

/**
 * Format a single BasketItem into a standard RFC 4180 CSV line.
 */
export function formatBasketItemToCsvLine(item: BasketItem): string {
  const cleanVid = (item.video_id || '').trim().replace(/\.mp4$/i, '')
  const cleanFid = Math.floor(Math.max(0, item.frame_id || 0))

  if (item.task === 'VQA') {
    const rawAnswer = (item.answer || '').trim()
    // Escape quotes if answer contains commas, quotes, or newlines
    let formattedAnswer = rawAnswer
    if (rawAnswer.includes(',') || rawAnswer.includes('"') || rawAnswer.includes('\n')) {
      formattedAnswer = `"${rawAnswer.replace(/"/g, '""')}"`
    }
    return `${cleanVid},${cleanFid},${formattedAnswer}`
  }

  if (item.task === 'TRAKE' && item.frame_ids && item.frame_ids.length > 0) {
    const cleanFrames = item.frame_ids.map((f) => Math.floor(Math.max(0, f))).join(',')
    return `${cleanVid},${cleanFrames}`
  }

  // KIS format
  return `${cleanVid},${cleanFid}`
}

/**
 * Export basket items to a single multi-row CSV string (UTF-8).
 * Optionally filters strictly by task type ('KIS' | 'VQA' | 'TRAKE') to prevent mixed output.
 */
export function exportBasketToCsvString(basket: BasketItem[], taskFilter?: 'KIS' | 'VQA' | 'TRAKE'): string {
  if (!basket || basket.length === 0) return ''
  const filtered = taskFilter ? basket.filter((item) => (item.task || 'KIS') === taskFilter) : basket
  const itemsToExport = filtered.slice(0, 100)
  return itemsToExport.map(formatBasketItemToCsvLine).join('\n')
}

/**
 * Group basket items by their real competition query_id, preserving basket
 * order as rank (best-first). Each group becomes exactly one submission CSV
 * per README_RETRIEVAL_HANDOFF / RUN_BOOK section 20 -- items from different
 * queries must never be combined into one file.
 */
export function groupBasketByQuery(basket: BasketItem[]): Record<string, BasketItem[]> {
  const groups: Record<string, BasketItem[]> = {}
  basket.forEach((item) => {
    if (!item.query_id) return
    if (!groups[item.query_id]) groups[item.query_id] = []
    groups[item.query_id].push(item)
  })
  return groups
}

/**
 * Build one headerless CSV per query_id from the basket, optionally
 * restricted to one task type. Returns {query_id: csvText}.
 */
export function exportBasketToPerQueryCsv(
  basket: BasketItem[],
  taskFilter?: 'KIS' | 'VQA' | 'TRAKE'
): Record<string, string> {
  const filtered = taskFilter ? basket.filter((item) => (item.task || 'KIS') === taskFilter) : basket
  const groups = groupBasketByQuery(filtered)
  const out: Record<string, string> = {}
  for (const [queryId, items] of Object.entries(groups)) {
    out[queryId] = items.slice(0, 100).map(formatBasketItemToCsvLine).join('\n')
  }
  return out
}

/**
 * Trigger a browser download of a text/CSV file.
 */
export function triggerBrowserDownload(content: string, filename: string, mimeType = 'text/csv;charset=utf-8;'): void {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  document.body.removeChild(anchor)
  URL.revokeObjectURL(url)
}
