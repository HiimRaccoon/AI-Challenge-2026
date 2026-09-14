import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { App } from '../../src/App'
import { AppProvider } from '../../src/state/AppContext'

describe('T022 / T023 App Shell & State Integration', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('renders app shell with live health probe and distinct partial readiness', async () => {
    global.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.endsWith('/health')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            status: 'ok',
            mode: 'live',
            preprocess_run_id: 'run_v1_batch1',
          }),
        } as Response)
      }
      return Promise.reject(new Error(`Unhandled URL: ${url}`))
    })

    render(
      <AppProvider>
        <App />
      </AppProvider>
    )

    // Shell header brand name
    expect(
      screen.getByText('Team SS009.Q24 Multimodal Retrieval System')
    ).toBeInTheDocument()
    expect(screen.getByTestId('header-task-badge')).toHaveTextContent('KIS')

    // Wait for health check to complete
    await waitFor(() => {
      expect(screen.getByTestId('tv4-health-status')).toHaveTextContent('ONLINE')
      expect(screen.getByTestId('readiness-status')).toHaveTextContent('PARTIAL')
      expect(screen.getByTestId('preprocess-run-id')).toHaveTextContent('run_v1_batch1')
    })
  })

  it('handles TV4 offline / failure gracefully without fabricating healthy state', async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error('Network error: TV4 connection refused'))

    render(
      <AppProvider>
        <App />
      </AppProvider>
    )

    await waitFor(() => {
      expect(screen.getByTestId('tv4-health-status')).toHaveTextContent('OFFLINE')
      expect(screen.getByTestId('readiness-status')).toHaveTextContent('OFFLINE')
    })
  })
})
