import React from 'react'
import { useAppDispatch, useAppState } from '../state/AppContext'
import { searchKis } from '../api/tv4Client'
import { SearchIcon, ClearIcon, SpinnerIcon } from './Icons'

export const SearchPanel: React.FC = () => {
  const { queryText, topK, isSearching, searchError } = useAppState()
  const dispatch = useAppDispatch()

  const handleQueryChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    dispatch({ type: 'SET_QUERY_TEXT', payload: e.target.value })
  }

  const handleTopKChange = (newVal: number) => {
    const bounded = Math.max(1, Math.min(100, isNaN(newVal) ? 100 : newVal))
    dispatch({ type: 'SET_TOP_K', payload: bounded })
  }

  const handleClear = () => {
    dispatch({ type: 'SET_QUERY_TEXT', payload: '' })
  }

  const executeSearch = async (e?: React.SyntheticEvent) => {
    if (e) {
      e.preventDefault()
    }
    const trimmed = queryText.trim()
    if (!trimmed || isSearching) return

    dispatch({ type: 'KIS_SEARCH_START' })
    try {
      const res = await searchKis({
        query_text: trimmed,
        top_k: topK,
      })
      dispatch({ type: 'KIS_SEARCH_SUCCESS', payload: res })
    } catch (err: any) {
      dispatch({
        type: 'KIS_SEARCH_FAILURE',
        payload: err.message || 'Search execution failed',
      })
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      executeSearch()
    }
  }

  return (
    <aside className="search-panel" data-testid="query-rail">
      <div className="search-panel-inner">
        <div className="panel-header">
          <span className="panel-title">Search & Retrieval</span>
        </div>

        <form className="search-command-form" onSubmit={executeSearch}>
          {/* Query Text Area */}
          <div className="command-input-container">
            <textarea
              className="command-textarea"
              value={queryText}
              onChange={handleQueryChange}
              onKeyDown={handleKeyDown}
              placeholder="Describe scene or action (e.g. xe máy qua ngã tư, xe buýt màu đỏ)..."
              rows={3}
              data-testid="kis-query-input"
            />
            {queryText.length > 0 && (
              <button
                type="button"
                className="command-clear-btn"
                onClick={handleClear}
                title="Clear query text"
                aria-label="Clear query text"
              >
                <ClearIcon size={12} />
              </button>
            )}
          </div>

          {/* Atomic Result Limit Selector (No rolling animation) */}
          <div className="result-limit-control">
            <label htmlFor="top-k-input" className="limit-caption">
              Results Limit
            </label>
            <div className="limit-inputs-cluster">
              <select
                className="limit-dropdown"
                value={[20, 50, 100].includes(topK) ? topK : 'custom'}
                onChange={(e) => {
                  const val = e.target.value
                  if (val !== 'custom') {
                    handleTopKChange(parseInt(val, 10))
                  }
                }}
                aria-label="Select result limit preset"
              >
                <option value={20}>20</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
                {![20, 50, 100].includes(topK) && <option value="custom">Custom</option>}
              </select>

              <input
                id="top-k-input"
                type="number"
                min={1}
                max={100}
                value={topK}
                onChange={(e) => handleTopKChange(parseInt(e.target.value, 10))}
                className="limit-number-input monospace"
                title="Custom top-K limit (1..100)"
                data-testid="top-k-input"
              />
            </div>
          </div>

          {/* Primary Search CTA Button */}
          <button
            type="submit"
            className="command-search-btn"
            disabled={isSearching || !queryText.trim()}
            onClick={executeSearch}
            data-testid="kis-search-btn"
          >
            {isSearching ? (
              <>
                <SpinnerIcon size={14} className="icon-spin" />
                <span>Searching...</span>
              </>
            ) : (
              <>
                <SearchIcon size={14} />
                <span>Search</span>
                <kbd className="command-shortcut">↵</kbd>
              </>
            )}
          </button>
        </form>

        {/* Query Summary / Error Box */}
        {searchError && (
          <div className="search-error-callout">
            <span>{searchError}</span>
          </div>
        )}

        {/* Architectural extension seam for future multimodal filters */}
        <div className="future-extension-seam" aria-hidden="true" />
      </div>
    </aside>
  )
}
