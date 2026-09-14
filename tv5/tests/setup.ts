import '@testing-library/jest-dom'
import { vi } from 'vitest'

// Mock URL.createObjectURL and URL.revokeObjectURL in jsdom
if (typeof window !== 'undefined') {
  window.URL.createObjectURL = vi.fn(() => 'blob:mock-blob-url')
  window.URL.revokeObjectURL = vi.fn()
}
