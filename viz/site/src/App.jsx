import { useState, useEffect } from 'react'
import { io } from 'socket.io-client'
import CameraFeed from './components/CameraFeed'
import StateViewer from './components/StateViewer'
import './App.css'

function App() {
  const [state, setState] = useState(null)
  const [connected, setConnected] = useState(false)
  const [overlayUpdate, setOverlayUpdate] = useState(0)

  useEffect(() => {
    // Connect to WebSocket
    const socket = io('http://localhost:8000', {
      transports: ['websocket'],
      path: '/ws/socket.io/',
    })

    socket.on('connect', () => {
      console.log('WebSocket connected')
      setConnected(true)
    })

    socket.on('disconnect', () => {
      console.log('WebSocket disconnected')
      setConnected(false)
    })

    // Listen for state updates
    socket.on('message', (data) => {
      console.log('Received message:', data)

      if (data.type === 'initial_state' || data.type === 'cache_update') {
        setState(data.data)
      } else if (data.type === 'overlay_update' || data.type === 'flag_update') {
        // Trigger overlay refresh
        setOverlayUpdate(prev => prev + 1)
      }
    })

    // Fetch initial state via REST API as fallback
    fetch('/api/state')
      .then(res => res.json())
      .then(data => {
        if (!data.error) {
          setState(data)
        }
      })
      .catch(err => console.error('Error fetching initial state:', err))

    return () => {
      socket.disconnect()
    }
  }, [])

  return (
    <div className="app">
      <header className="header">
        <h1>Chess Robot Visualization</h1>
        <div className="connection-status">
          <div className={`status-indicator ${connected ? 'connected' : ''}`}></div>
          <span>{connected ? 'Connected' : 'Disconnected'}</span>
        </div>
      </header>

      <main className="main-content">
        <div className="camera-panel">
          <div className="panel-header">Global Camera</div>
          <div className="panel-content">
            <CameraFeed endpoint="/api/stream/global" refreshInterval={100} />
          </div>
        </div>

        <div className="camera-panel">
          <div className="panel-header">Guidance Overlay</div>
          <div className="panel-content">
            <CameraFeed
              endpoint="/api/stream/overlay"
              refreshInterval={500}
              trigger={overlayUpdate}
            />
          </div>
        </div>

        <div className="camera-panel">
          <div className="panel-header">Gripper Camera</div>
          <div className="panel-content">
            <CameraFeed endpoint="/api/stream/gripper" refreshInterval={100} />
          </div>
        </div>

        <div className="state-panel">
          <div className="panel-header">Robot State</div>
          <div className="state-content">
            <StateViewer state={state} />
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
