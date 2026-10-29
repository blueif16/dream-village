import { Suspense, lazy, useCallback, useEffect, useState } from 'react'
import { useIndex } from './data'
import { Person } from './Person'

// PixiJS loads only when the Town view opens.
const Town = lazy(() => import('./town/Town').then((m) => ({ default: m.Town })))
import { Village } from './Village'

// Routes live in the hash so a view can be linked: #/ (village) · #/town · #/p/<pid>/<event key>
type Route = { pid: string | null; key: string | null; town?: boolean }

function parse(hash: string): Route {
  if (hash.startsWith('#/town')) return { pid: null, key: null, town: true }
  const m = /^#\/p\/([^/]+)(?:\/(.+))?$/.exec(hash)
  return m ? { pid: decodeURIComponent(m[1]), key: m[2] ? decodeURIComponent(m[2]) : null } : { pid: null, key: null }
}

function href(r: Route): string {
  if (r.town) return '#/town'
  if (!r.pid) return '#/'
  return `#/p/${encodeURIComponent(r.pid)}${r.key ? `/${encodeURIComponent(r.key)}` : ''}`
}

export default function App() {
  const { data: index, error } = useIndex()
  const [route, setRoute] = useState<Route>(() => parse(window.location.hash))

  useEffect(() => {
    const on = () => setRoute(parse(window.location.hash))
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])

  const go = useCallback((r: Route, replace = false) => {
    const h = href(r)
    if (replace) window.history.replaceState(null, '', h)
    else window.history.pushState(null, '', h)
    setRoute(r)
  }, [])

  const pid = route.pid
  const select = useCallback((key: string, replace = false) => go({ pid, key }, replace), [go, pid])

  if (error) {
    return (
      <div className="screen-msg">
        <h2 style={{ marginTop: 0 }}>No data bundle</h2>
        <p>
          Could not load <code>data/index.json</code> ({error}). Write the bundle from the repo root, then reload:
        </p>
        <pre>.venv/bin/python viz/export.py</pre>
      </div>
    )
  }
  if (!index) return <div className="placeholder">Loading the village…</div>

  const known = pid && index.principals.some((p) => p.id === pid)
  return (
    <>
      <header className="top">
        <button className="wordmark" onClick={() => go({ pid: null, key: null })}>
          Dream Village <em>inspector</em>
        </button>
        <nav className="crumbs">
          <a href="#/" onClick={(e) => (e.preventDefault(), go({ pid: null, key: null }))}>
            {index.town.name}
          </a>
          <a href="#/town" className={route.town ? 'here' : undefined}
            onClick={(e) => (e.preventDefault(), go({ pid: null, key: null, town: true }))}>
            Town
          </a>
          <span className="muted">/</span>
          <select aria-label="principal" value={known ? pid! : ''} onChange={(e) => go({ pid: e.target.value || null, key: null })}>
            <option value="">all principals</option>
            {index.principals.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </nav>
        <div className="meta">
          <span className="mono">{index.village}</span> · exported {index.generated_at.replace('T', ' ').slice(0, 16)} UTC
          <br />
          scorer: {[...new Set(Object.entries(index.scorer.files).filter(([, ok]) => ok).map(([f]) => f.split('.')[0]))].join(', ') || 'nothing yet'}
        </div>
      </header>
      {route.town ? (
        <Suspense fallback={<div className="placeholder">Loading the town…</div>}>
          <Town />
        </Suspense>
      ) : known ? (
        <Person key={pid} index={index} pid={pid!} selected={route.key} onSelect={select} />
      ) : (
        <Village index={index} onOpen={(p) => go({ pid: p, key: null })} />
      )}
    </>
  )
}
