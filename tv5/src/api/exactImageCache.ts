import { ExactImageHeaders, ExactImageResult, ExactNeighborRequest } from '../types/contracts'
import { fetchExactImageBlob } from './tv4Client'

interface CacheEntry {
  blobUrl: string
  headers: ExactImageHeaders
}

// In-flight request deduplication map to prevent redundant concurrent fetches
const pendingRequests = new Map<string, Promise<ExactImageResult>>()

// Completed cache per anchor key
let currentAnchorKey = ''
const completedCache = new Map<string, CacheEntry>()

export function getExactImageCacheKey(
  videoId: string,
  anchorFrameId: number,
  cumulativeOffset: number,
  relOffset: number
): string {
  return `${videoId}:${anchorFrameId}:${cumulativeOffset + relOffset}`
}

export function clearExactImageCacheForAnchor(newAnchorKey: string): void {
  if (currentAnchorKey && currentAnchorKey !== newAnchorKey) {
    // Revoke previous blob URLs to avoid memory leaks
    for (const [_, entry] of completedCache) {
      if (entry.blobUrl.startsWith('blob:')) {
        try {
          URL.revokeObjectURL(entry.blobUrl)
        } catch {
          // Ignore revocation errors in test environments
        }
      }
    }
    completedCache.clear()
    pendingRequests.clear()
  }
  currentAnchorKey = newAnchorKey
}

/**
 * Shared exact-image loader with concurrent request deduplication,
 * bounded per-anchor cache, and strict proof header validation.
 */
export async function loadExactImage(
  req: ExactNeighborRequest,
  expectedFrame?: { frame_id: number; video_id: string }
): Promise<ExactImageResult> {
  const relOffset = req.offsets[0] ?? 0
  const anchorKey = `${req.video_id}:${req.certified_anchor_frame_id ?? req.frame_id}`
  clearExactImageCacheForAnchor(anchorKey)

  const key = getExactImageCacheKey(
    req.video_id,
    req.certified_anchor_frame_id ?? req.frame_id,
    req.cumulative_offset,
    relOffset
  )

  // 1. Check completed cache
  const cached = completedCache.get(key)
  if (cached) {
    return { blobUrl: cached.blobUrl, headers: cached.headers }
  }

  // 2. Check pending in-flight request for deduplication
  let promise = pendingRequests.get(key)
  if (!promise) {
    promise = (async () => {
      try {
        const result = await fetchExactImageBlob(req)

        // Fail-closed invariant: verify returned exact-image proof headers match step.frame for the target video
        if (
          expectedFrame &&
          expectedFrame.video_id === req.video_id &&
          result.headers.frame_id !== undefined
        ) {
          if (result.headers.frame_id !== expectedFrame.frame_id) {
            throw new Error(
              `Exact image proof mismatch: expected frame ${expectedFrame.frame_id}, got ${result.headers.frame_id}`
            )
          }
        }

        completedCache.set(key, { blobUrl: result.blobUrl, headers: result.headers })
        return result
      } finally {
        pendingRequests.delete(key)
      }
    })()
    pendingRequests.set(key, promise)
  }

  return promise
}

/** Helper for tests to reset cache state */
export function resetExactImageCache(): void {
  for (const [_, entry] of completedCache) {
    if (entry.blobUrl.startsWith('blob:')) {
      try {
        URL.revokeObjectURL(entry.blobUrl)
      } catch {
        // Ignore
      }
    }
  }
  completedCache.clear()
  pendingRequests.clear()
  currentAnchorKey = ''
}
