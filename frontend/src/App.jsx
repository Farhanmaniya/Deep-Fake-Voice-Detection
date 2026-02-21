import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

const DEFAULT_API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8001'
const WS_URL =
  import.meta.env.VITE_WS_URL ||
  DEFAULT_API_BASE.replace(/^http/i, 'ws') + '/ws/audio'
const HEALTH_URL = `${DEFAULT_API_BASE}/health`
const METRICS_URL = `${DEFAULT_API_BASE}/metrics`

function App() {
  const [connectionStatus, setConnectionStatus] = useState('disconnected')
  const [isStreaming, setIsStreaming] = useState(false)
  const [lastResult, setLastResult] = useState(null)
  const [riskHistory, setRiskHistory] = useState([])
  const [health, setHealth] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [error, setError] = useState('')

  const wsRef = useRef(null)
  const audioContextRef = useRef(null)
  const processorRef = useRef(null)
  const mediaStreamRef = useRef(null)

  useEffect(() => {
    let cancelled = false

    async function fetchHealth() {
      try {
        const res = await fetch(HEALTH_URL)
        if (!res.ok) return
        const data = await res.json()
        if (!cancelled) setHealth(data)
      } catch {}
    }

    async function fetchMetrics() {
      try {
        const res = await fetch(METRICS_URL)
        if (!res.ok) return
        const data = await res.json()
        if (!cancelled) setMetrics(data)
      } catch {}
    }

    fetchHealth()
    fetchMetrics()
    const id = setInterval(fetchMetrics, 5000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  const stopStreaming = () => {
    setIsStreaming(false)
    setConnectionStatus('disconnected')
    setError('')

    if (processorRef.current) {
      processorRef.current.disconnect()
      processorRef.current.onaudioprocess = null
      processorRef.current = null
    }
    if (audioContextRef.current) {
      audioContextRef.current.close()
      audioContextRef.current = null
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop())
      mediaStreamRef.current = null
    }
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
  }

  useEffect(() => {
    return () => {
      stopStreaming()
    }
  }, [])

  const startStreaming = async () => {
    if (isStreaming || connectionStatus === 'connecting') return

    setConnectionStatus('connecting')
    setError('')

    try {
      const ws = new WebSocket(WS_URL)
      ws.binaryType = 'arraybuffer'
      wsRef.current = ws

      ws.onopen = async () => {
        setConnectionStatus('connected')
        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
              channelCount: 1,
              sampleRate: 16000,
            },
          })
          mediaStreamRef.current = stream

          const audioContext = new AudioContext({ sampleRate: 16000 })
          audioContextRef.current = audioContext
          const source = audioContext.createMediaStreamSource(stream)
          const processor = audioContext.createScriptProcessor(4096, 1, 1)
          processorRef.current = processor

          processor.onaudioprocess = (event) => {
            if (ws.readyState !== WebSocket.OPEN) return
            const input = event.inputBuffer.getChannelData(0)
            const buffer = new ArrayBuffer(input.length * 2)
            const view = new DataView(buffer)
            for (let i = 0; i < input.length; i += 1) {
              let s = input[i]
              if (s > 1) s = 1
              if (s < -1) s = -1
              view.setInt16(i * 2, s * 0x7fff, true)
            }
            ws.send(buffer)
          }

          source.connect(processor)
          processor.connect(audioContext.destination)
          setIsStreaming(true)
        } catch (err) {
          setError('Microphone permission or audio initialisation failed')
          setConnectionStatus('error')
          stopStreaming()
        }
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (data.status !== 'success') return
          setLastResult(data)
          setRiskHistory((prev) => {
            const value = typeof data.rolling_risk === 'number' ? data.rolling_risk : data.chunk_probability
            const next = [...prev, value]
            if (next.length > 60) next.shift()
            return next
          })
        } catch {}
      }

      ws.onerror = () => {
        setError('WebSocket connection error')
        setConnectionStatus('error')
      }

      ws.onclose = () => {
        setConnectionStatus('disconnected')
        setIsStreaming(false)
      }
    } catch {
      setError('Unable to connect to backend WebSocket')
      setConnectionStatus('error')
    }
  }

  const toggleStreaming = () => {
    if (isStreaming) {
      stopStreaming()
    } else {
      startStreaming()
    }
  }

  const riskLevel = lastResult?.risk_level || 'LOW'
  const probability = lastResult?.chunk_probability ?? 0
  const rollingRisk = lastResult?.rolling_risk ?? null
  const explainability = lastResult?.explainability || null
  const attribution = lastResult?.attribution || null
  const robustness = lastResult?.robustness || null
  const consensus = lastResult?.consensus || null

  const riskColor = useMemo(() => {
    if (riskLevel === 'HIGH') return '#f97373'
    if (riskLevel === 'MEDIUM') return '#fbbf24'
    return '#4ade80'
  }, [riskLevel])

  return (
    <div className="app-root">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark">LG</span>
          <div className="brand-text">
            <h1>LiveGuard Voice Shield</h1>
            <p>Real-time deepfake voice detection and signal intelligence</p>
          </div>
        </div>
        <div className="header-meta">
          <span className="badge badge-live">Live</span>
          <span className="badge badge-pill">Enterprise Preview</span>
        </div>
      </header>

      <main className="layout">
        <section className="panel panel-primary">
          <div className="panel-header">
            <h2>Live Call Scanner</h2>
            <p>Stream audio from your microphone to the backend detector.</p>
          </div>
          <StreamController
            connectionStatus={connectionStatus}
            isStreaming={isStreaming}
            toggleStreaming={toggleStreaming}
            error={error}
          />
        </section>

        <section className="layout-grid">
          <section className="panel panel-risk">
            <RiskView
              probability={probability}
              rollingRisk={rollingRisk}
              riskLevel={riskLevel}
              riskColor={riskColor}
              riskHistory={riskHistory}
              consensus={consensus}
            />
          </section>
          <section className="panel panel-side">
            <HealthAndMetrics health={health} metrics={metrics} />
          </section>
        </section>

        <section className="layout-grid layout-grid-bottom">
          <section className="panel">
            <ExplainabilityView explainability={explainability} />
          </section>
          <section className="panel">
            <AttributionView attribution={attribution} probability={probability} />
          </section>
          <section className="panel">
            <RobustnessView robustness={robustness} />
          </section>
        </section>
      </main>
    </div>
  )
}

function StreamController({ connectionStatus, isStreaming, toggleStreaming, error }) {
  const isConnected = connectionStatus === 'connected'
  const statusLabel =
    connectionStatus === 'connected'
      ? 'Connected'
      : connectionStatus === 'connecting'
        ? 'Connecting…'
        : connectionStatus === 'error'
          ? 'Error'
          : 'Disconnected'

  return (
    <div className="stream-controller">
      <div className="stream-main">
        <button
          type="button"
          className={`primary-button ${isStreaming ? 'button-stop' : 'button-start'}`}
          onClick={toggleStreaming}
        >
          {isStreaming ? 'Stop Live Scan' : 'Start Live Scan'}
        </button>
        <div className="status-pill-row">
          <span className={`status-pill status-${connectionStatus}`}>
            <span className="status-dot" />
            {statusLabel}
          </span>
          {isStreaming && (
            <span className="status-pill status-streaming">
              <span className="status-dot" />
              Streaming microphone audio
            </span>
          )}
        </div>
        <p className="stream-helper">
          The detector runs entirely on your backend. Audio is streamed as small encrypted WebSocket chunks and never
          stored.
        </p>
      </div>
      <div className="stream-side">
        <div className="hint-row">
          <div className="hint-dot hint-safe" />
          <span>Use a real or recorded phone call to simulate production traffic.</span>
        </div>
        <div className="hint-row">
          <div className="hint-dot hint-risk" />
          <span>Watch the risk dial and attribution card for suspicious synthetic speech.</span>
        </div>
        {!isConnected && (
          <p className="hint-warning">
            Ensure the backend is running on {DEFAULT_API_BASE} and reachable from the browser.
          </p>
        )}
        {error && <p className="hint-error">{error}</p>}
      </div>
    </div>
  )
}

function RiskView({ probability, rollingRisk, riskLevel, riskColor, riskHistory, consensus }) {
  const percent = Math.round(probability * 100)
  const rollingPercent = rollingRisk != null ? Math.round(rollingRisk * 100) : null
  const cnnPercent = consensus?.cnn_score != null ? Math.round(consensus.cnn_score * 100) : null
  const lstmPercent = consensus?.lstm_score != null ? Math.round(consensus.lstm_score * 100) : null

  return (
    <div className="risk-view">
      <div className="risk-main">
        <div className="risk-gauge">
          <div className="risk-gauge-inner">
            <div className="risk-gauge-ring">
              <div
                className="risk-gauge-fill"
                style={{
                  background: `conic-gradient(${riskColor} ${percent * 1.8}deg, rgba(15,23,42,0.7) ${percent * 1.8}deg)`,
                }}
              />
              <div className="risk-gauge-center">
                <span className="risk-value">{percent}</span>
                <span className="risk-label">Deepfake score</span>
              </div>
            </div>
          </div>
        </div>
        <div className="risk-meta">
          <div className="risk-level-row">
            <span className={`chip chip-${riskLevel.toLowerCase()}`}>{riskLevel}</span>
            {rollingPercent != null && (
              <span className="chip chip-outline">
                Rolling risk {rollingPercent}
                <span className="chip-unit">%</span>
              </span>
            )}
          </div>
          <div className="risk-signals-row">
            <span className="risk-signals-label">Signals</span>
            <span className="risk-signals-values">
              <span>Chunk {percent}%</span>
              {rollingPercent != null && <span>Rolling {rollingPercent}%</span>}
              {cnnPercent != null && <span>CNN {cnnPercent}%</span>}
              {lstmPercent != null && <span>LSTM {lstmPercent}%</span>}
            </span>
          </div>
          <div className="risk-bars">
            <label className="risk-bar-label">
              CNN model
              <span className="risk-bar-value">
                {cnnPercent != null ? cnnPercent : 0}
                <span className="risk-bar-unit">%</span>
              </span>
            </label>
            <Bar
              value={consensus?.cnn_score != null ? consensus.cnn_score : 0}
              color="#38bdf8"
            />
            <label className="risk-bar-label">
              Temporal LSTM
              <span className="risk-bar-value">
                {lstmPercent != null ? lstmPercent : 0}
                <span className="risk-bar-unit">%</span>
              </span>
            </label>
            <Bar
              value={consensus?.lstm_score != null ? consensus.lstm_score : 0}
              color="#a855f7"
            />
          </div>
        </div>
      </div>
      <div className="risk-history">
        <div className="risk-history-header">
          <span>Timeline</span>
          <span className="risk-history-caption">Last {riskHistory.length} chunks</span>
        </div>
        <div className="risk-history-graph">
          {riskHistory.length === 0 && <span className="risk-history-empty">Start streaming to see live risk.</span>}
          {riskHistory.map((v, idx) => (
            <div
              key={idx}
              className="risk-history-bar"
              style={{
                height: `${Math.max(6, v * 100)}%`,
                backgroundColor:
                  v > 0.7 ? '#f97373' : v > 0.4 ? '#fbbf24' : '#4ade80',
              }}
            />
          ))}
        </div>
      </div>
    </div>
  )
}

function Bar({ value, color }) {
  const percent = Math.max(0, Math.min(1, value || 0)) * 100
  return (
    <div className="bar">
      <div
        className="bar-fill"
        style={{ width: `${percent}%`, backgroundColor: color }}
      />
    </div>
  )
}

function HealthAndMetrics({ health, metrics }) {
  const modeLabel =
    health?.model_mode === 'mock'
      ? 'Mock model'
      : health?.model_mode === 'production'
        ? 'Production model'
        : 'Unknown'

  const modeClass =
    health?.model_mode === 'mock'
      ? 'badge-soft-warning'
      : health?.model_mode === 'production'
        ? 'badge-soft-success'
        : 'badge-soft-muted'

  return (
    <div className="health-view">
      <div className="panel-header">
        <h2>Backend status</h2>
        <p>FastAPI inference service health and throughput.</p>
      </div>
      <div className="health-grid">
        <div className="health-card">
          <span className="health-label">Service</span>
          <span className={`health-badge ${health?.status === 'ok' ? 'health-ok' : 'health-degraded'}`}>
            {health?.status || 'unknown'}
          </span>
        </div>
        <div className="health-card">
          <span className="health-label">Model</span>
          <span className={`badge ${modeClass}`}>{modeLabel}</span>
        </div>
        <div className="health-card">
          <span className="health-label">Active connections</span>
          <span className="health-value">{metrics?.active_connections ?? 0}</span>
        </div>
        <div className="health-card">
          <span className="health-label">Chunks processed</span>
          <span className="health-value">{metrics?.total_chunks_processed ?? 0}</span>
        </div>
        <div className="health-card">
          <span className="health-label">Average latency</span>
          <span className="health-value">
            {metrics?.average_latency_ms != null ? `${metrics.average_latency_ms} ms` : '—'}
          </span>
        </div>
        <div className="health-card">
          <span className="health-label">Throughput</span>
          <span className="health-value">
            {metrics?.chunks_per_second != null ? `${metrics.chunks_per_second} chunks/s` : '—'}
          </span>
        </div>
      </div>
      {health?.timestamp && (
        <p className="health-footer">
          Last health check at {new Date(health.timestamp).toLocaleTimeString()}
        </p>
      )}
    </div>
  )
}

function ExplainabilityView({ explainability }) {
  const bandEnergies = explainability?.band_energies || null
  const suspiciousBand = explainability?.suspicious_band || null
  const flatness = explainability?.spectral_flatness ?? null
  const temporalRisk = explainability?.temporal_risk || []

  return (
    <div className="explain-view">
      <div className="panel-header">
        <h2>Why is this risky?</h2>
        <p>Frequency and time patterns that push the model toward deepfake.</p>
      </div>
      {!explainability && (
        <p className="placeholder-text">
          Start a scan to see explainability insights for each audio segment.
        </p>
      )}
      {explainability && (
        <>
          <div className="explain-grid">
            <div className="explain-card">
              <span className="explain-label">Suspicious band</span>
              <span className="explain-value">{suspiciousBand || 'balanced'}</span>
            </div>
            <div className="explain-card">
              <span className="explain-label">Spectral flatness</span>
              <span className="explain-value">
                {flatness != null ? flatness.toFixed(3) : '—'}
              </span>
            </div>
          </div>
          {bandEnergies && (
            <div className="band-bars">
              {Object.entries(bandEnergies).map(([band, value]) => (
                <div key={band} className="band-row">
                  <span className="band-label">{band}</span>
                  <div className="band-bar-wrapper">
                    <div
                      className="band-bar-fill"
                      style={{
                        width: `${Math.min(100, (value + 80) * 1.2)}%`,
                      }}
                    />
                  </div>
                  <span className="band-value">{value.toFixed(1)} dB</span>
                </div>
              ))}
            </div>
          )}
          {temporalRisk.length > 0 && (
            <div className="temporal-risk">
              <span className="temporal-label">Risk over time</span>
              <div className="temporal-bars">
                {temporalRisk.map((v, idx) => (
                  <div
                    key={idx}
                    className="temporal-bar"
                    style={{
                      height: `${Math.max(10, v * 100)}%`,
                    }}
                  />
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}

function AttributionView({ attribution, probability }) {
  const baseLabel = probability > 0.5 ? 'Likely generator' : 'Speaker type'

  if (!attribution) {
    return (
      <div className="attrib-view">
        <div className="panel-header">
          <h2>Who generated this?</h2>
          <p>Shows the most likely AI engine when deepfake probability is high.</p>
        </div>
        <p className="placeholder-text">
          Attribution appears when the system is confident the audio is synthetic.
        </p>
      </div>
    )
  }

  const confidencePercent = Math.round(
    (attribution.generator_confidence || 0) * 100,
  )

  return (
    <div className="attrib-view">
      <div className="panel-header">
        <h2>Who generated this?</h2>
        <p>Rule-based spectral fingerprinting of popular AI voice engines.</p>
      </div>
      <div className="attrib-main">
        <div className="attrib-primary">
          <span className="attrib-label">{baseLabel}</span>
          <span className="attrib-name">{attribution.suspected_generator}</span>
          <span className="attrib-confidence">
            Confidence {confidencePercent}
            <span className="attrib-unit">%</span>
          </span>
        </div>
        <div className="attrib-meta">
          <div className="attrib-pill-row">
            <span className="badge badge-soft-muted">
              Zero-crossing rate {attribution.zero_crossing_rate?.toFixed(3) ?? '—'}
            </span>
            <span className="badge badge-soft-muted">
              Sub-bass ratio {attribution.sub_bass_ratio?.toFixed(3) ?? '—'}
            </span>
          </div>
          <p className="attrib-note">
            This is a heuristic signal and should be used as supporting evidence, not a
            definitive attribution.
          </p>
        </div>
      </div>
    </div>
  )
}

function RobustnessView({ robustness }) {
  if (!robustness) {
    return (
      <div className="robust-view">
        <div className="panel-header">
          <h2>Audio quality</h2>
          <p>Signal health, noise and clipping diagnostics.</p>
        </div>
        <p className="placeholder-text">
          Once streaming, low-quality audio will surface here as actionable guidance.
        </p>
      </div>
    )
  }

  const {
    snr_db,
    noise_floor_db,
    clipping_percent,
    quality_score,
    is_low_quality,
    warnings,
  } = robustness

  return (
    <div className="robust-view">
      <div className="panel-header">
        <h2>Audio quality</h2>
        <p>Signal health, noise and clipping diagnostics.</p>
      </div>
      <div className="quality-score">
        <div className="quality-ring">
          <div className="quality-ring-inner">
            <span className="quality-value">{Math.round(quality_score ?? 0)}</span>
            <span className="quality-label">Quality</span>
          </div>
        </div>
        <div className="quality-meta">
          <div className="quality-row">
            <span>SNR</span>
            <span>{snr_db != null ? `${snr_db} dB` : '—'}</span>
          </div>
          <div className="quality-row">
            <span>Noise floor</span>
            <span>
              {noise_floor_db != null ? `${noise_floor_db} dB` : '—'}
            </span>
          </div>
          <div className="quality-row">
            <span>Clipping</span>
            <span>
              {clipping_percent != null ? `${clipping_percent}%` : '—'}
            </span>
          </div>
          {is_low_quality && (
            <span className="badge badge-soft-warning">Poor audio quality</span>
          )}
        </div>
      </div>
      {warnings && warnings.length > 0 && (
        <ul className="warning-list">
          {warnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default App
