import React, { useMemo, useState } from 'react'

// Types
type Mode = 'function_plot' | 'vector_addition' | 'bubble_sort_visualization'
type Tab = 'manual' | 'prompt'

interface FunctionPlotRequest {
  mode: 'function_plot'
  function_expression: string
  x_min: number
  x_max: number
  y_min: number
  y_max: number
  duration_seconds: number
  title: string
}

interface BackendResponse {
  status: string
  output_video_path?: string
  message?: string
}

type InstructionResponse = {
  status: 'ok' | 'error'
  mode: Mode | null
  instructions: any | null
  message?: string
  enhanced_prompt?: string | null
  enhanced_source?: string | null
}

interface Vector2D {
  label: string
  x: number
  y: number
}

interface VectorAdditionRequest {
  mode: 'vector_addition'
  vectors: Vector2D[]
  show_resultant: boolean
  show_tip_to_tail: boolean
  title: string
}

const defaultFunctionPlot: FunctionPlotRequest = {
  mode: 'function_plot',
  function_expression: 'sin(x)',
  x_min: -6.28,
  x_max: 6.28,
  y_min: -2,
  y_max: 2,
  duration_seconds: 6,
  title: 'Sine Wave Animation',
}

const defaultVectorAddition: VectorAdditionRequest = {
  mode: 'vector_addition',
  vectors: [
    { label: 'v1', x: 2, y: 1 },
    { label: 'v2', x: -1, y: 2 },
  ],
  show_resultant: true,
  show_tip_to_tail: true,
  title: 'Vector Addition Demo',
}

export default function App() {
  const [selectedMode, setSelectedMode] = useState<Mode>('function_plot')
  const [tab, setTab] = useState<Tab>('manual')

  // Function Plot state
  const [fp, setFp] = useState<FunctionPlotRequest>(() => ({ ...defaultFunctionPlot }))

  // Vector Addition state
  const [va, setVa] = useState<VectorAdditionRequest>(() => ({ ...defaultVectorAddition }))

  // Request/Response state
  const [requestJson, setRequestJson] = useState<object | null>(null)
  const [responseJson, setResponseJson] = useState<BackendResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Backend API base URL (editable). If your server runs on 8001, change here.
  const [apiUrl, setApiUrl] = useState<string>('http://localhost:8000')

  // Prompt Mode state
  const [prompt, setPrompt] = useState('')
  const [generatedMode, setGeneratedMode] = useState<Mode | null>(null)
  const [instructions, setInstructions] = useState<any | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [generateError, setGenerateError] = useState<string | null>(null)
  const [isRendering, setIsRendering] = useState(false)
  const [renderStatus, setRenderStatus] = useState<string | null>(null)
  const [renderOutputPath, setRenderOutputPath] = useState<string | null>(null)
  const [renderError, setRenderError] = useState<string | null>(null)
  const [enhancedPrompt, setEnhancedPrompt] = useState<string | null>(null)
  const [enhancedSource, setEnhancedSource] = useState<string | null>(null)

  async function submitFunctionPlot() {
    const body: FunctionPlotRequest = { ...fp, mode: 'function_plot' }
    setRequestJson(body)
    setIsLoading(true)
    setError(null)
    setResponseJson(null)

    try {
      const res = await fetch(`${apiUrl}/render/function_plot`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) {
        const txt = await res.text()
        throw new Error(`HTTP ${res.status}: ${txt}`)
      }
      const data: BackendResponse = await res.json()
      setResponseJson(data)
    } catch (e: any) {
      setError(e?.message || 'Unknown error')
    } finally {
      setIsLoading(false)
    }
  }

  async function submitVectorAddition() {
    const body: VectorAdditionRequest = { ...va, mode: 'vector_addition' }
    setRequestJson(body)
    setIsLoading(true)
    setError(null)
    setResponseJson(null)

    try {
      const res = await fetch(`${apiUrl}/render/vector_addition`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) {
        const txt = await res.text()
        throw new Error(`HTTP ${res.status}: ${txt}`)
      }
      const data: BackendResponse = await res.json()
      setResponseJson(data)
    } catch (e: any) {
      setError(e?.message || 'Unknown error')
    } finally {
      setIsLoading(false)
    }
  }

  async function generateInstructions() {
    setIsGenerating(true)
    setGenerateError(null)
    setGeneratedMode(null)
    setInstructions(null)
    setRenderStatus(null)
    setRenderOutputPath(null)
    setRenderError(null)
    setEnhancedPrompt(null)
    setEnhancedSource(null)

    try {
      const res = await fetch(`${apiUrl}/generate/instructions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      })
      if (!res.ok) {
        const txt = await res.text()
        throw new Error(`HTTP ${res.status}: ${txt}`)
      }
      const data: InstructionResponse = await res.json()
      if (data.status === 'ok' && data.instructions) {
        setGeneratedMode(data.mode as Mode)
        setInstructions(data.instructions)
        setEnhancedPrompt(data.enhanced_prompt ?? null)
        setEnhancedSource(data.enhanced_source ?? null)
      } else {
        setGenerateError(data.message || 'Failed to generate instructions')
      }
    } catch (e: any) {
      setGenerateError(e?.message || 'Unknown error')
    } finally {
      setIsGenerating(false)
    }
  }

  async function renderFromInstructions() {
    if (!instructions || !generatedMode) return
    setIsRendering(true)
    setRenderStatus(null)
    setRenderOutputPath(null)
    setRenderError(null)

    try {
      let endpoint = ''
      let body: any = instructions
      if (generatedMode === 'function_plot') {
        endpoint = '/render/function_plot'
      } else if (generatedMode === 'vector_addition') {
        endpoint = '/render/vector_addition'
      } else {
        // Fallback to generic for any other modes
        endpoint = '/render/any_mode'
        body = { mode: generatedMode, payload: instructions }
      }
      const res = await fetch(`${apiUrl}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) {
        const txt = await res.text()
        throw new Error(`HTTP ${res.status}: ${txt}`)
      }
      const data: BackendResponse = await res.json()
      setRenderStatus(data.status)
      setRenderOutputPath(data.output_video_path || null)
      if (data.status !== 'ok') {
        setRenderError(data.message || 'Render error')
      }
    } catch (e: any) {
      setRenderError(e?.message || 'Unknown error')
    } finally {
      setIsRendering(false)
    }
  }

  function addVectorRow() {
    setVa((prev) => ({ ...prev, vectors: [...prev.vectors, { label: `v${prev.vectors.length + 1}`, x: 0, y: 0 }] }))
  }

  function removeVectorRow(index: number) {
    setVa((prev) => {
      if (prev.vectors.length <= 2) return prev
      const next = [...prev.vectors]
      next.splice(index, 1)
      return { ...prev, vectors: next }
    })
  }

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', color: '#111', padding: 24, maxWidth: 1100, margin: '0 auto' }}>
      <header style={{ marginBottom: 16 }}>
        <h1 style={{ margin: 0 }}>Math Animation Generator (Manim + FastAPI)</h1>
        <p style={{ marginTop: 4 }}>Enter structured animation instructions and generate educational videos.</p>
      </header>

      {/* Tabs */}
      <section style={{ marginBottom: 8 }}>
        <button onClick={() => setTab('manual')} disabled={tab === 'manual'} style={{ marginRight: 8 }}>Manual Mode</button>
        <button onClick={() => setTab('prompt')} disabled={tab === 'prompt'}>Prompt Mode</button>
      </section>

      {/* Mode selector (Manual Mode only) */}
      <section style={{ marginBottom: 16 }}>
        {tab === 'manual' && (
          <>
            <label style={{ fontWeight: 600, marginRight: 12 }}>Mode:</label>
            <select value={selectedMode} onChange={(e) => setSelectedMode(e.target.value as Mode)}>
              <option value="function_plot">Function Plot</option>
              <option value="vector_addition">Vector Addition</option>
            </select>
          </>
        )}
        <div style={{ marginTop: 8 }}>
          <label style={{ fontWeight: 600, marginRight: 12 }}>Backend URL:</label>
          <input
            type="text"
            value={apiUrl}
            onChange={(e) => setApiUrl(e.target.value)}
            style={{ width: '360px' }}
            placeholder="http://localhost:8000"
          />
        </div>
      </section>
      {tab === 'manual' ? (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          {/* Form panel */}
          <section style={{ border: '1px solid #ddd', borderRadius: 8, padding: 16 }}>
            {selectedMode === 'function_plot' ? (
            <div>
              <h2 style={{ marginTop: 0 }}>Function Plot</h2>

              <label>Function Expression</label>
              <input
                type="text"
                value={fp.function_expression}
                onChange={(e) => setFp({ ...fp, function_expression: e.target.value })}
                style={{ width: '100%', marginBottom: 8 }}
              />

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
                <div>
                  <label>x_min</label>
                  <input type="number" value={fp.x_min} onChange={(e) => setFp({ ...fp, x_min: Number(e.target.value) })} style={{ width: '100%' }} />
                </div>
                <div>
                  <label>x_max</label>
                  <input type="number" value={fp.x_max} onChange={(e) => setFp({ ...fp, x_max: Number(e.target.value) })} style={{ width: '100%' }} />
                </div>
                <div>
                  <label>y_min</label>
                  <input type="number" value={fp.y_min} onChange={(e) => setFp({ ...fp, y_min: Number(e.target.value) })} style={{ width: '100%' }} />
                </div>
                <div>
                  <label>y_max</label>
                  <input type="number" value={fp.y_max} onChange={(e) => setFp({ ...fp, y_max: Number(e.target.value) })} style={{ width: '100%' }} />
                </div>
              </div>

              <label>Duration (seconds)</label>
              <input
                type="number"
                value={fp.duration_seconds}
                onChange={(e) => setFp({ ...fp, duration_seconds: Number(e.target.value) })}
                style={{ width: '100%', marginBottom: 8 }}
              />

              <label>Title</label>
              <input type="text" value={fp.title} onChange={(e) => setFp({ ...fp, title: e.target.value })} style={{ width: '100%', marginBottom: 12 }} />

              <button onClick={submitFunctionPlot} disabled={isLoading}>
                {isLoading ? 'Generating...' : 'Render Function Plot'}
              </button>
            </div>
            ) : (
            <div>
              <h2 style={{ marginTop: 0 }}>Vector Addition</h2>
              {va.vectors.map((v, i) => (
                <div key={i} style={{ border: '1px solid #eee', borderRadius: 6, padding: 8, marginBottom: 8 }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr auto', gap: 8, alignItems: 'center' }}>
                    <div>
                      <label>Label</label>
                      <input type="text" value={v.label} onChange={(e) => {
                        const next = [...va.vectors]
                        next[i] = { ...next[i], label: e.target.value }
                        setVa({ ...va, vectors: next })
                      }} style={{ width: '100%' }} />
                    </div>
                    <div>
                      <label>x</label>
                      <input type="number" value={v.x} onChange={(e) => {
                        const next = [...va.vectors]
                        next[i] = { ...next[i], x: Number(e.target.value) }
                        setVa({ ...va, vectors: next })
                      }} style={{ width: '100%' }} />
                    </div>
                    <div>
                      <label>y</label>
                      <input type="number" value={v.y} onChange={(e) => {
                        const next = [...va.vectors]
                        next[i] = { ...next[i], y: Number(e.target.value) }
                        setVa({ ...va, vectors: next })
                      }} style={{ width: '100%' }} />
                    </div>
                    <div>
                      {va.vectors.length > 2 && (
                        <button onClick={() => removeVectorRow(i)} style={{ marginTop: 18 }}>Remove</button>
                      )}
                    </div>
                  </div>
                </div>
              ))}
              <button onClick={addVectorRow} style={{ marginBottom: 12 }}>Add Vector</button>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <input type="checkbox" checked={va.show_resultant} onChange={(e) => setVa({ ...va, show_resultant: e.target.checked })} />
                  Show Resultant
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <input type="checkbox" checked={va.show_tip_to_tail} onChange={(e) => setVa({ ...va, show_tip_to_tail: e.target.checked })} />
                  Show Tip-to-Tail
                </label>
              </div>

              <label>Title</label>
              <input type="text" value={va.title} onChange={(e) => setVa({ ...va, title: e.target.value })} style={{ width: '100%', marginBottom: 12 }} />

              <button onClick={submitVectorAddition} disabled={isLoading}>
                {isLoading ? 'Generating...' : 'Render Vector Addition'}
              </button>
            </div>
            )}
          </section>

          
          <section style={{ border: '1px solid #ddd', borderRadius: 8, padding: 16 }}>
            <h2 style={{ marginTop: 0 }}>Request JSON</h2>
            <pre style={{ background: '#f9f9f9', padding: 12, borderRadius: 6, maxHeight: 240, overflow: 'auto' }}>
              {requestJson ? JSON.stringify(requestJson, null, 2) : 'Submit a request to see JSON here.'}
            </pre>

            <h2>Response</h2>
            {error && <div style={{ color: 'red', marginBottom: 8 }}>Error: {error}</div>}
            <pre style={{ background: '#f9f9f9', padding: 12, borderRadius: 6, maxHeight: 240, overflow: 'auto' }}>
              {responseJson ? JSON.stringify(responseJson, null, 2) : 'No response yet.'}
            </pre>
          </section>
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <section style={{ border: '1px solid #ddd', borderRadius: 8, padding: 16 }}>
            <h2 style={{ marginTop: 0 }}>Prompt Mode</h2>
            <label>Enter Prompt</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={8}
              style={{ width: '100%', marginBottom: 12 }}
              placeholder="e.g. Plot sin(x) from -2π to 2π and show maxima and minima"
            />
            <button onClick={generateInstructions} disabled={isGenerating || !prompt.trim()}>
              {isGenerating ? 'Generating instructions...' : 'Generate Instructions'}
            </button>
            {generateError && <div style={{ color: 'red', marginTop: 8 }}>Error: {generateError}</div>}
            {generatedMode && <p style={{ marginTop: 12 }}>Detected mode: <strong>{generatedMode}</strong></p>}
            {(enhancedPrompt || enhancedSource) && (
              <div style={{ marginTop: 12 }}>
                <h3 style={{ margin: '8px 0', display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span>Enhanced Prompt</span>
                  {enhancedSource && (
                    <span style={{ fontSize: 12, background: '#eef', color: '#224', padding: '2px 6px', borderRadius: 12 }}>
                      Enhanced by: {enhancedSource}
                    </span>
                  )}
                </h3>
                <div style={{ background: '#f9f9f9', padding: 12, borderRadius: 6, whiteSpace: 'pre-wrap' }}>
                  {enhancedPrompt}
                </div>
              </div>
            )}
            <hr style={{ margin: '16px 0' }} />
            <button
              onClick={renderFromInstructions}
              disabled={isRendering || !instructions || !generatedMode}
            >
              {isRendering ? 'Rendering...' : 'Render Animation'}
            </button>
          </section>

          <section style={{ border: '1px solid #ddd', borderRadius: 8, padding: 16 }}>
            <h2 style={{ marginTop: 0 }}>Generated Instructions (JSON)</h2>
            <pre style={{ background: '#f9f9f9', padding: 12, borderRadius: 6, maxHeight: 280, overflow: 'auto' }}>
              {instructions ? JSON.stringify(instructions, null, 2) : 'No instructions yet.'}
            </pre>
            <h2>Render Result</h2>
            {renderError && <div style={{ color: 'red', marginBottom: 8 }}>Error: {renderError}</div>}
            <div style={{ background: '#f9f9f9', padding: 12, borderRadius: 6 }}>
              <p>Status: {renderStatus ?? '—'}</p>
              <p>Output Path: {renderOutputPath ?? '—'}</p>
            </div>
          </section>
        </div>
      )}

      <footer style={{ marginTop: 24, color: '#555' }}>
        <p>
          Use Manual Mode for direct control, or Prompt Mode to generate JSON instructions via the local SLM and render them automatically.
        </p>
      </footer>
    </div>
  )
}
