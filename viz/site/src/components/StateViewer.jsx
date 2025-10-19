/**
 * StateViewer Component
 *
 * Displays the robot state cache with organized sections.
 * Updates reactively via WebSocket.
 */
function StateViewer({ state }) {
  if (!state) {
    return <div className="no-signal">Waiting for state data...</div>
  }

  const { game_state, robot_state, guidance_state, metadata } = state

  return (
    <div>
      {/* Game State Section */}
      <div className="state-section">
        <h3>Game State</h3>
        <div className="state-grid">
          <div className="state-item">
            <div className="state-item-label">FEN</div>
            <div className="state-item-value">
              <code>{game_state?.fen || 'N/A'}</code>
            </div>
          </div>
          <div className="state-item">
            <div className="state-item-label">Turn</div>
            <div className="state-item-value">{game_state?.turn || 'N/A'}</div>
          </div>
          <div className="state-item">
            <div className="state-item-label">Board Image</div>
            <div className="state-item-value">
              {game_state?.board_image_path || 'None'}
            </div>
          </div>
        </div>
      </div>

      {/* Robot State Section */}
      <div className="state-section">
        <h3>Robot State</h3>
        <div className="state-grid">
          <div className="state-item">
            <div className="state-item-label">Robot Turn</div>
            <div className="state-item-value">
              {robot_state?.is_robot_turn ? 'Yes' : 'No'}
            </div>
          </div>
          <div className="state-item">
            <div className="state-item-label">Holding Piece</div>
            <div className="state-item-value">
              {robot_state?.holding_piece ? 'Yes' : 'No'}
            </div>
          </div>
          <div className="state-item">
            <div className="state-item-label">Current Move</div>
            <div className="state-item-value">
              {robot_state?.current_move || 'None'}
            </div>
          </div>
          <div className="state-item">
            <div className="state-item-label">Move Type</div>
            <div className="state-item-value">
              {robot_state?.move_type || 'N/A'}
            </div>
          </div>
          <div className="state-item">
            <div className="state-item-label">Action Index</div>
            <div className="state-item-value">
              {robot_state?.action_index ?? 'N/A'}
            </div>
          </div>
          <div className="state-item">
            <div className="state-item-label">Actions Remaining</div>
            <div className="state-item-value">
              {robot_state?.actions_remaining ?? 'N/A'}
            </div>
          </div>
        </div>

        {/* Action Sequence */}
        {robot_state?.action_sequence && robot_state.action_sequence.length > 0 && (
          <div style={{ marginTop: '15px' }}>
            <div className="state-item-label" style={{ marginBottom: '8px' }}>
              Action Sequence
            </div>
            <ul className="action-list">
              {robot_state.action_sequence.map((action, index) => {
                const isCurrent = index === robot_state.action_index
                const isCompleted = index < robot_state.action_index
                const isPending = index > robot_state.action_index

                return (
                  <li
                    key={index}
                    className={`action-item ${
                      isCurrent ? 'current' : isCompleted ? 'completed' : 'pending'
                    }`}
                  >
                    <div>
                      <div className="action-type">{action.type}</div>
                      <div className="action-details">
                        {action.square || 'graveyard'} - {action.piece} ({action.status})
                      </div>
                    </div>
                    <div style={{ fontSize: '12px', color: '#999' }}>
                      #{index + 1}
                    </div>
                  </li>
                )
              })}
            </ul>
          </div>
        )}
      </div>

      {/* Guidance State Section */}
      <div className="state-section">
        <h3>Guidance State</h3>
        <div className="state-grid">
          <div className="state-item">
            <div className="state-item-label">Best Move (UCI)</div>
            <div className="state-item-value">
              {guidance_state?.best_move || 'None'}
            </div>
          </div>
          <div className="state-item">
            <div className="state-item-label">Best Move (SAN)</div>
            <div className="state-item-value">
              {guidance_state?.best_move_san || 'N/A'}
            </div>
          </div>
          <div className="state-item">
            <div className="state-item-label">Evaluation</div>
            <div className="state-item-value">
              {guidance_state?.evaluation ?? 'N/A'}
            </div>
          </div>
        </div>
      </div>

      {/* Metadata Section */}
      <div className="state-section">
        <h3>Metadata</h3>
        <div className="state-grid">
          <div className="state-item">
            <div className="state-item-label">Last Updated By</div>
            <div className="state-item-value">
              {metadata?.last_updated_by || 'Unknown'}
            </div>
          </div>
          <div className="state-item">
            <div className="state-item-label">Timestamp</div>
            <div className="state-item-value">
              {metadata?.timestamp
                ? new Date(metadata.timestamp * 1000).toLocaleString()
                : 'N/A'}
            </div>
          </div>
          <div className="state-item">
            <div className="state-item-label">Cache Version</div>
            <div className="state-item-value">
              {metadata?.cache_version || 'N/A'}
            </div>
          </div>
          <div className="state-item">
            <div className="state-item-label">Overlay Path</div>
            <div className="state-item-value">
              {metadata?.overlay_path || 'N/A'}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default StateViewer
