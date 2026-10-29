import './town.css'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { publishTime } from '../clock'
import { usePrincipal, useTown } from '../data'
import { residentColors } from './colors'
import { clock, monthDay, weekday } from './common'
import { PersonDetail, PlaceDetail, VehicleDetail, type PersonTab } from './Detail'
import { Expanded } from './Expanded'
import { seasonMarks } from './insight'
import { ResidentStrip } from './ResidentStrip'
import { MIN_PER_SEC, TownEngine, type Sel } from './engine'
import { EventsGlyph, Fit, MemoryGlyph, Minus, Moon, Next, Pause, PeopleGlyph, Play, PlacesGlyph, Plus, Prev, Sun } from './icons'
import { Season } from './Season'
import { Sheet, type SheetCtx, type SheetId } from './Sheets'
import { DAY_MIN } from './schedule'
import type { TownEvent } from './types'

// The town's control panel: one rounded console. The live map fills it; the season strip, the transport and the
// dock sit below. Only the map has night: everything around it stays light.

const SPEEDS = [1, 4, 16]
const DOCK: [SheetId, string, () => React.ReactNode][] = [
  ['people', 'Residents', PeopleGlyph],
  ['places', 'Places', PlacesGlyph],
  ['events', 'Events', EventsGlyph],
  ['memory', 'Memory', MemoryGlyph],
]

function Callout({ el, hover }: { el: React.RefObject<HTMLDivElement | null>; hover?: boolean }) {
  return (
    <div className={hover ? 'callout hover' : 'callout'} ref={el} hidden aria-hidden="true">
      <i className="lead" />
      <div className="card"><span className="k" /><span className="v" /></div>
    </div>
  )
}

/** How the console is shown: as the landing page's window (clipped to it, inert) or whole. */
export interface Frame {
  window: boolean // seen through the landing page's hero: no input, no keys
  clip: string // a CSS inset() the console is clipped to
  animate: boolean // ease from the last clip to this one
  hidden: boolean
}

export function Town({ frame }: { frame?: Frame }) {
  const { data: town, error } = useTown()
  const stage = useRef<HTMLDivElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const tagHover = useRef<HTMLDivElement>(null)
  const tagSel = useRef<HTMLDivElement>(null)
  const labels = useRef<HTMLDivElement>(null)
  const [engine, setEngine] = useState<TownEngine | null>(null)
  const [t, setT] = useState(7.5 * 60)
  const [sel, setSel] = useState<Sel>(null)
  const [night, setNight] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [sheet, setSheet] = useState<SheetId | null>(null)
  // a resident's card expanded over the map, and the event picked in it (or on their timeline below)
  const [expanded, setExpanded] = useState(false)
  const [picked, setPicked] = useState<string | null>(null)
  const colors = useMemo(() => residentColors((town?.principals ?? []).map((p) => p.id)), [town])
  const places = useMemo(() => new Map((town?.places ?? []).map((p) => [p.id, p])), [town])
  const maxLines = useMemo(() => Math.max(1, ...(town?.principals ?? []).flatMap((p) => p.memory.map((m) => m.lines))), [town])

  // whatever is picked, on the map or in a sheet, closes the sheet so the map shows it
  const onSelect = useCallback((s: Sel) => {
    setSel(s)
    setSheet(null)
  }, [])

  useEffect(() => {
    if (!town || !canvas.current || !stage.current || !tagHover.current || !tagSel.current || !labels.current) return
    let e: TownEngine | null = null
    let alive = true
    // the ground labels are drawn in DM Mono, so wait for it (but not for long)
    Promise.race([document.fonts.load("500 64px 'DM Mono'"), new Promise((r) => setTimeout(r, 1500))]).finally(() => {
      if (!alive) return
      try {
        e = new TownEngine({
          canvas: canvas.current!, stage: stage.current!, tagHover: tagHover.current!, tagSel: tagSel.current!, labels: labels.current!, town, colors,
          onSelect, onTick: setT, onNight: setNight,
        })
        setEngine(e)
      } catch (err) {
        console.error('the town failed to start', err)
      }
    })
    return () => {
      alive = false
      e?.dispose()
      setEngine(null)
    }
  }, [town, colors, onSelect])

  // the pages around the town show its clock
  useEffect(() => {
    if (engine) publishTime(t, night)
  }, [engine, t, night])

  // exact session times for whoever is selected, for the talks that wrote nothing; their memory for the panel
  const pid = sel?.kind === 'person' ? sel.id : null
  const { data: bundle } = usePrincipal(pid)
  const mine = bundle && bundle.principal.id === pid ? bundle : null
  useEffect(() => {
    if (engine && mine) engine.setSessions(mine.principal.id, mine.sessions.map((s) => s.start))
  }, [engine, mine])

  // their events by key, so picking one can move the town to its moment
  const marks = useMemo(() => (mine && town ? new Map(seasonMarks(mine, town.days[0]).map((m) => [m.key, m])) : null), [mine, town])
  const pick = useCallback((key: string) => {
    setPicked(key)
    setExpanded(true)
    const m = marks?.get(key)
    if (m && engine) engine.setTime(m.at)
  }, [marks, engine])
  // a new resident starts with nothing picked; no one selected, nothing expanded
  useEffect(() => setPicked(null), [pid])
  useEffect(() => {
    if (!sel) setExpanded(false)
  }, [sel])

  // picked from the memory sheet, a resident opens expanded: their memory is in the expanded card now
  const select = useCallback((s: Sel, k?: PersonTab) => {
    if (k === 'memory') setExpanded(true)
    engine?.select(s)
    if (s) engine?.focus(s)
  }, [engine])

  const setDay = useCallback((d: number) => {
    if (!engine) return
    engine.setTime(d * DAY_MIN + (engine.t % DAY_MIN))
  }, [engine])

  const step = useCallback((k: 1 | -1) => {
    if (!town) return
    const ids = town.principals.map((p) => p.id)
    const i = sel?.kind === 'person' ? ids.indexOf(sel.id) : -1
    select({ kind: 'person', id: ids[(i + k + ids.length) % ids.length] })
  }, [town, sel, select])

  const [, redraw] = useState(0)
  const togglePlay = useCallback(() => {
    if (!engine) return
    engine.setPlaying(!engine.playing)
    redraw((k) => k + 1) // paused, the clock stops ticking, so nothing else would redraw the button
  }, [engine])

  const onEvent = useCallback((e: TownEvent) => {
    if (!town) return
    const d = town.days.indexOf(e.date)
    if (d >= 0) setDay(d)
    if (e.places[0]) select({ kind: 'place', id: e.places[0] })
  }, [town, setDay, select])

  const inert = !!frame?.window
  useEffect(() => {
    if (!engine || !town || inert) return
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || e.metaKey || e.ctrlKey || e.altKey) return
      const day = Math.floor(engine.t / DAY_MIN)
      const act: Record<string, () => void> = {
        ArrowRight: () => setDay(Math.min(town.days.length - 1, day + 1)),
        ArrowLeft: () => setDay(Math.max(0, day - 1)),
        ' ': togglePlay,
        ']': () => step(1),
        '[': () => step(-1),
        f: () => engine.setFollow(!engine.follow),
        Escape: () => (expanded ? setExpanded(false) : sheet ? setSheet(null) : engine.select(null)),
        '+': () => engine.zoomBy(1.4),
        '=': () => engine.zoomBy(1.4),
        '-': () => engine.zoomBy(1 / 1.4),
        '0': () => engine.resetView(),
      }
      engine.views.forEach((v, k) => (act[String(k + 1)] = () => engine.goTo(v.id)))
      const fn = act[e.key]
      if (fn) {
        fn()
        e.preventDefault()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [engine, town, inert, sheet, expanded, step, setDay, togglePlay])

  if (error) return <div className="screen-msg">Could not load <code>data/town.json</code>: {error}. Re-run the export.</div>

  const day = Math.floor(t / DAY_MIN)
  const d = town?.days[day] ?? ''
  const person = town && sel?.kind === 'person' ? town.principals.find((p) => p.id === sel.id) ?? null : null
  const place = sel?.kind === 'place' ? places.get(sel.id) ?? null : null
  const talking = town && engine ? town.principals.filter((p) => engine.statusOf(p.id).mode === 'talking') : []
  const eventToday = town ? town.events.some((e) => e.date <= d && d <= (e.end_date ?? e.date)) : false
  const ctx: SheetCtx | null = town && engine ? {
    town, t, engine, colors, places, sel, pick: select, goTo: (v) => { engine.goTo(v); setSheet(null) }, onEvent,
  } : null
  const detail = !!(sel && engine && town)
  const playing = engine?.playing ?? true

  return (
    <div className={inert ? 'console-app window' : 'console-app'} inert={inert || undefined}
      style={frame && { clipPath: frame.clip, visibility: frame.hidden ? 'hidden' : undefined, transition: frame.animate ? undefined : 'none' }}>
      <div className="shell">
        <div className={`screen${night ? ' night' : ''}${detail ? ' has-detail' : ''}`} ref={stage}>
          <canvas ref={canvas} tabIndex={0} role="img"
            aria-label="Isometric map of the town and its residents, live through the days of the season. Drag to pan, scroll to zoom, click anyone or any building for details." />
          <div className="labels" ref={labels} aria-hidden="true" />
          <Callout el={tagHover} hover />
          <Callout el={tagSel} />
          <div className="maptools" role="group" aria-label="Map">
            <button onClick={() => engine?.zoomBy(1.4)} title="Zoom in (+)" aria-label="Zoom in"><Plus /></button>
            <button onClick={() => engine?.zoomBy(1 / 1.4)} title="Zoom out (−)" aria-label="Zoom out"><Minus /></button>
            <button onClick={() => engine?.resetView()} title="Whole town (0)" aria-label="Whole town"><Fit /></button>
          </div>
          {sheet && ctx && <Sheet key={sheet} id={sheet} c={ctx} onClose={() => setSheet(null)} />}
          {detail && (
            person ? (
              <PersonDetail key={person.id} town={town!} p={person} t={t} color={colors.get(person.id) ?? 0x888888} status={engine!.statusOf(person.id)}
                places={places} engine={engine!} bundle={mine} maxLines={maxLines} onExpand={() => setExpanded(true)}
                onSelect={select} onStep={step} onClose={() => engine!.select(null)} />
            ) : place ? (
              <PlaceDetail town={town!} place={place} t={t} engine={engine!} colors={colors} onSelect={select} onClose={() => engine!.select(null)} />
            ) : sel!.kind === 'train' || sel!.kind === 'bus' ? (
              <VehicleDetail kind={sel!.kind} places={places} onClose={() => engine!.select(null)} />
            ) : null
          )}
          {expanded && person && mine && town && (
            <Expanded bundle={mine} day0={town.days[0]} now={t} color={colors.get(person.id) ?? 0x888888} picked={picked} onPick={pick}
              onClose={() => setExpanded(false)} />
          )}
          {!engine && <p className="note">{town ? 'Drawing the town…' : 'Loading the town…'}</p>}
        </div>

        <footer className="console">
          <div className="season-row">
            <div className="clockblock" aria-live="off">
              <span className="town">{town?.town ?? 'The town'}</span>
              <span className="time">
                {clock(t)}
                {night ? <Moon /> : <Sun />}
              </span>
              <span className="date">{d ? `${weekday(d)} ${monthDay(d)} · day ${day + 1} of ${town!.days.length}` : '—'}</span>
            </div>
            {town && (person && mine
              ? <ResidentStrip town={town} t={t} bundle={mine} color={colors.get(person.id) ?? 0x888888} picked={picked} onPick={(m) => pick(m.key)} onDay={setDay} />
              : <Season town={town} t={t} person={person} color={person ? colors.get(person.id) ?? 0x888888 : 0x1b1d22} onDay={setDay} onEvent={onEvent} />)}
          </div>
          <div className="dock-row">
            <div className="transport">
              <button className="play" onClick={togglePlay} title="Play or pause (Space)" aria-label={playing ? 'Pause' : 'Play'}>
                {playing ? <Pause /> : <Play />}
              </button>
              <div className="steps">
                <button onClick={() => setDay(Math.max(0, day - 1))} title="Previous day (←)" aria-label="Previous day"><Prev /></button>
                <button onClick={() => setDay(Math.min((town?.days.length ?? 1) - 1, day + 1))} title="Next day (→)" aria-label="Next day"><Next /></button>
              </div>
              <div className="speeds" role="radiogroup" aria-label="Speed">
                {SPEEDS.map((k) => (
                  <button key={k} role="radio" aria-checked={speed === k} onClick={() => { engine?.setSpeed(k); setSpeed(k) }}
                    title={`A day in ${Math.round(DAY_MIN / MIN_PER_SEC / k)} s`}>{k}×</button>
                ))}
              </div>
            </div>
            <nav className="dock" aria-label="Panels">
              {DOCK.map(([id, label, G]) => (
                <button key={id} className={id} aria-pressed={sheet === id} onClick={() => setSheet(sheet === id ? null : id)}>
                  <G />
                  <span>{label}</span>
                  {id === 'people' && talking.length > 0 && <i className="blip live" title={`${talking.length} talking now`} />}
                  {id === 'events' && eventToday && <i className="blip event" title="An event today" />}
                </button>
              ))}
            </nav>
            <div className="outlinks">
              <a className="datalink" href="#/behind" title="How the town is made, and how we question its memory">
                <MemoryGlyph />
                <span>Behind the town</span>
              </a>
            </div>
          </div>
        </footer>
      </div>
    </div>
  )
}
