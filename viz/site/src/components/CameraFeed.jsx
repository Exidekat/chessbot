import { useState, useEffect, useRef } from 'react'

/**
 * CameraFeed Component
 *
 * Displays a camera feed by polling a JPEG endpoint.
 * Updates at specified refresh interval.
 */
function CameraFeed({ endpoint, refreshInterval = 100, trigger = 0 }) {
  const [imageUrl, setImageUrl] = useState('')
  const [error, setError] = useState(false)
  const intervalRef = useRef(null)

  useEffect(() => {
    // Function to fetch new frame
    const fetchFrame = async () => {
      try {
        // Add timestamp to prevent caching
        const url = `${endpoint}?t=${Date.now()}`
        const response = await fetch(url)

        if (response.ok) {
          const blob = await response.blob()
          const objectUrl = URL.createObjectURL(blob)

          // Clean up previous object URL
          if (imageUrl) {
            URL.revokeObjectURL(imageUrl)
          }

          setImageUrl(objectUrl)
          setError(false)
        } else {
          setError(true)
        }
      } catch (err) {
        console.error('Error fetching camera frame:', err)
        setError(true)
      }
    }

    // Initial fetch
    fetchFrame()

    // Set up interval for continuous updates
    intervalRef.current = setInterval(fetchFrame, refreshInterval)

    // Cleanup
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl)
      }
    }
  }, [endpoint, refreshInterval, trigger])

  if (error) {
    return <div className="no-signal">No Signal</div>
  }

  if (!imageUrl) {
    return <div className="no-signal">Loading...</div>
  }

  return (
    <img
      src={imageUrl}
      alt="Camera feed"
      className="camera-image"
    />
  )
}

export default CameraFeed
