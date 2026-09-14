import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { App } from '../../src/App'
import { AppProvider } from '../../src/state/AppContext'

describe('Result Limit (Top-K) & Product Identity Regressions', () => {
  it('renders correct product brand and removes old WP13 RETRIEVAL COCKPIT / TV4: ONLINE', () => {
    render(
      <AppProvider>
        <App />
      </AppProvider>
    )

    // Primary product identity
    const brandElement = screen.getByText('Team SS009.Q24 Multimodal Retrieval System')
    expect(brandElement).toBeInTheDocument()
    expect(brandElement.tagName.toLowerCase()).toBe('h1')

    // Confirms rejected branding is NOT present
    expect(screen.queryByText('WP13 RETRIEVAL COCKPIT')).not.toBeInTheDocument()
    expect(screen.queryByText('WP13 Retrieval Cockpit')).not.toBeInTheDocument()
    expect(screen.queryByText('TV4: ONLINE')).not.toBeInTheDocument()
  })

  it('updates result limit atomically without intermediate value rolling or tweening', () => {
    render(
      <AppProvider>
        <App />
      </AppProvider>
    )

    const topKInput = screen.getByTestId('top-k-input') as HTMLInputElement
    expect(topKInput.value).toBe('100')

    // Change to 20 atomically
    fireEvent.change(topKInput, { target: { value: '20' } })
    expect(topKInput.value).toBe('20')

    // Change to 50 atomically
    fireEvent.change(topKInput, { target: { value: '50' } })
    expect(topKInput.value).toBe('50')

    // Custom limit 35
    fireEvent.change(topKInput, { target: { value: '35' } })
    expect(topKInput.value).toBe('35')
  })

  it('custom result limit editor uses text numeric input (no number spinner) and validates 1..100', async () => {
    render(
      <AppProvider>
        <App />
      </AppProvider>
    )

    // Click result limit trigger to open popover
    const triggerBtn = screen.getByRole('button', { name: /candidates/i })
    fireEvent.click(triggerBtn)

    // Click 'Custom...'
    const customOption = screen.getByText('Custom...')
    fireEvent.click(customOption)

    // Verify custom editor opened with input
    const customInput = screen.getByTestId('custom-limit-input') as HTMLInputElement
    expect(customInput).toBeInTheDocument()
    expect(customInput.type).toBe('text')
    expect(customInput.getAttribute('inputmode')).toBe('numeric')
    expect(customInput.getAttribute('pattern')).toBe('[0-9]*')

    // Invalid input: empty or 150
    fireEvent.change(customInput, { target: { value: '150' } })
    const applyBtn = screen.getByRole('button', { name: /apply/i })
    fireEvent.click(applyBtn)

    // Should show inline error and keep popover open
    expect(screen.getByText('Limit must be between 1 and 100')).toBeInTheDocument()

    // Valid input: 73
    fireEvent.change(customInput, { target: { value: '73' } })
    fireEvent.click(applyBtn)

    // Popover closes, topK becomes 73
    await waitFor(() => {
      const topKInput = screen.getByTestId('top-k-input') as HTMLInputElement
      expect(topKInput.value).toBe('73')
    })
  })
})
