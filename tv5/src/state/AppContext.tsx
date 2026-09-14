import React, { createContext, useContext, useReducer, ReactNode, Dispatch } from 'react'
import { AppState, initialAppState } from './appState'
import { AppAction, appReducer } from './appReducer'

interface AppContextType {
  state: AppState
  dispatch: Dispatch<AppAction>
}

const AppContext = createContext<AppContextType | undefined>(undefined)

export const AppProvider: React.FC<{ children: ReactNode; initialState?: AppState }> = ({
  children,
  initialState = initialAppState,
}) => {
  const [state, dispatch] = useReducer(appReducer, initialState)

  return <AppContext.Provider value={{ state, dispatch }}>{children}</AppContext.Provider>
}

export function useAppState(): AppState {
  const context = useContext(AppContext)
  if (!context) {
    throw new Error('useAppState must be used within an AppProvider')
  }
  return context.state
}

export function useAppDispatch(): Dispatch<AppAction> {
  const context = useContext(AppContext)
  if (!context) {
    throw new Error('useAppDispatch must be used within an AppProvider')
  }
  return context.dispatch
}
