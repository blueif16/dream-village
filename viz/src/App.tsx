import { Suspense, lazy, useCallback, useEffect, useRef, useState } from 'react'
import { useIndex } from './data'
import { Landing } from './Landing'
import { Meadow } from './Meadow'
import { Person } from './Person'
import { SiteNav } from './SiteNav'
import type { Frame } from './town/Town'

// three.js loads only when the town is on screen (the landing page's window shows it too).
const Town = lazy(() => import('./town/Town').then((m) => ({ default: m.Town })))
const Behind = lazy(() => import('./Behind').then((m) => ({ default: m.Behind })))
import { Village } from './Village'

// Routes live in the hash so a view can be linked: #/ (the landing page) · #/town (the whole town) · #/behind
// (how it works) · #/data (the village's data) · #/p/<pid>/<event key>.
type View = 'home' | 'town' | 'behind' | 'data' | 'person'
type Route = { view: View; pid: string | null; key: string | null }

function parse(hash: string): Route {
  const m = /^#\/p\/([^/]+)(?:\/(.+))?$/.exec(hash)
  if (m) return { view: 'person', pid: decodeURIComponent(m[1]), key: m[2] ? decodeURIComponent(m[2]) : null }
  const view: View = hash.startsWith('#/town') ? 'town' : hash.startsWith('#/behind') ? 'behind' : hash.startsWith('#/data') ? 'data' : 'home'
  return { view, pid: null, key: null }
}

function href(r: Route): string {
  if (r.view === 'person' && r.pid) return `#/p/${encodeURIComponent(r.pid)}${r.key ? `/${encodeURIComponent(r.key)}` : ''}`
  return { home: '#/', town: '#/town', behind: '#/behind', data: '#/data', person: '#/data' }[r.view]
}

const OPEN_MS = 900 // the window opening out to the whole town (town.css: .console-app's clip-path transition)
const reduced = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches
const WHOLE = 'inset(0px 0px 0px 0px round 0px)'
const windowClip = (r: DOMRect) =>
  `inset(${r.top}px ${window.innerWidth - r.right}px ${window.innerHeight - r.bottom}px ${r.left}px round 0px)`

export default function App() {
  const [route, setRoute] = useState<Route>(() => parse(window.location.hash))
  const routeRef = useRef(route)
  routeRef.current = route
  const { data: index, error } = useIndex()
  // the window's place on the landing page, and the open/close between it and the whole town
  const [hero, setHero] = useState<DOMRect | null>(null)
  const [phase, setPhase] = useState<'still' | 'opening' | 'closing'>('still')
  const phaseRef = useRef(phase)
  phaseRef.current = phase
  const timer = useRef(0)

  const settle = useCallback((then?: () => void) => {
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => {
      then?.()
      setPhase('still')
    }, reduced() ? 0 : OPEN_MS)
  }, [])

  useEffect(() => {
    const on = () => {
      const r = parse(window.location.hash)
      // back from the whole town to the landing page: close the town down into the window
      if (routeRef.current.view === 'town' && r.view === 'home') {
        setPhase('closing')
        settle()
      }
      setRoute(r)
    }
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [settle])

  const go = useCallback((r: Route, replace = false) => {
    const h = href(r)
    if (replace) window.history.replaceState(null, '', h)
    else window.history.pushState(null, '', h)
    setRoute(r)
  }, [])

  const enter = useCallback(() => {
    if (routeRef.current.view !== 'home' || phaseRef.current !== 'still') return
    phaseRef.current = 'opening'
    setPhase('opening')
    settle(() => go({ view: 'town', pid: null, key: null }))
  }, [go, settle])

  const pid = route.pid
  const select = useCallback((key: string, replace = false) => go({ view: 'person', pid, key }, replace), [go, pid])

  const view = route.view
  // always an inset(), so opening and closing both ease (none ↔ inset() would jump)
  const frame: Frame | undefined =
    view === 'town' ? { window: false, clip: WHOLE, animate: true, hidden: false }
      : view === 'home' ? {
        window: true,
        clip: phase === 'opening' || !hero ? WHOLE : windowClip(hero),
        animate: phase !== 'still',
        hidden: !hero && phase === 'still',
      }
        : undefined

  // the meadow: vivid round the landing page's window, soft (blurred) behind the page about the town, calm round the town
  const meadow = (
    <Meadow mode={view === 'home' && phase !== 'opening' ? 'vivid' : view === 'behind' ? 'soft' : 'calm'}
      light={view === 'home' ? hero : null} />
  )
  const town = (view === 'home' || view === 'town') && (
    <Suspense fallback={view === 'town' ? <div className="placeholder">Loading the town…</div> : null}>
      <Town frame={frame} />
    </Suspense>
  )
  if (view === 'home' || view === 'town') {
    return (
      <>
        {meadow}
        {town}
        {view === 'home' && (
          <>
            <Landing onEnter={enter} onWindow={setHero} leaving={phase === 'opening'} back={phase === 'closing'} />
          </>
        )}
      </>
    )
  }
  if (view === 'behind') {
    return (
      <>
        {meadow}
        <Suspense fallback={<div className="placeholder">Loading…</div>}>
          <Behind index={index} />
        </Suspense>
      </>
    )
  }
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

  const known = view === 'person' && pid && index.principals.some((p) => p.id === pid)
  return (
    <>
      <SiteNav compact here="data" />
      {known ? (
        <Person key={pid} index={index} pid={pid!} selected={route.key} onSelect={select} />
      ) : (
        <Village index={index} onOpen={(p) => go({ view: 'person', pid: p, key: null })} />
      )}
    </>
  )
}
