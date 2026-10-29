import '@fontsource/dm-mono/400.css'
import '@fontsource/dm-mono/500.css'
import './town.css'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { fmtWhen, ms, usePrincipal, useTown } from '../data'
import { Txt } from '../text'
import { cssHex, residentColors } from './colors'
import { MIN_PER_SEC, TownEngine, type Sel, type Status } from './engine'
import { districtName } from './palette'
import { DAY_MIN, lineIndexAt, talkTimes } from './schedule'
import type { Place, Town as TownData, TownPerson } from './types'

const KIND = (k: string) => k.replace(/_/g, ' ')
const SPEEDS = [1, 4, 16]
const pad = (n: number) => String(n).padStart(2, '0')
const clock = (t: number) => {
  const m = Math.floor(((t % DAY_MIN) + DAY_MIN) % DAY_MIN)
  return `${pad(Math.floor(m / 60))}:${pad(m % 60)}`
}
const weekday = (d: string) => fmtWhen(`${d}T12:00:00`).split(' · ')[0]

function Swatch({ color }: { color: number }) {
  return <i className="sw" style={{ background: cssHex(color) }} />
}

function statusWord(st: Status, places: Map<string, Place>): string {
  const name = (id: string | null) => (id ? places.get(id)?.name ?? id : 'somewhere')
  switch (st.mode) {
    case 'asleep':
      return 'asleep at home'
    case 'walking':
      return `on the way to ${name(st.to)}`
    case 'talking':
      return 'talking to the assistant'
    case 'dreaming':
      return 'asleep · memory dreaming'
    default:
      return `at ${name(st.at)}`
  }
}

function PersonCard({ town, p, t, color, status, places, maxLines, engine, onSelect, onClose }: {
  town: TownData
  p: TownPerson
  t: number
  color: number
  status: Status
  places: Map<string, Place>
  maxLines: number
  engine: TownEngine
  onSelect: (s: Sel) => void
  onClose: () => void
}) {
  const day = Math.floor(t / DAY_MIN)
  const d = town.days[day]
  const today = p.days[day]
  const li = lineIndexAt(p, t)
  const end = ms(`${d}T${clock(t)}:00`)
  const mem = [...p.memory].reverse().find((m) => ms(m.t) <= end)
  const talks = talkTimes(p, town.days).filter((x) => Math.floor(x / DAY_MIN) === day)
  const sessionsToday = (p.sessions[d] ?? []).length
  const [following, setFollowing] = useState(engine.follow)
  return (
    <>
      <div className="head">
        <span><Swatch color={color} /> Resident{p.minor ? '' : ` · ${p.age}`}</span>
        <button onClick={onClose} aria-label="Close details" title="Close (Esc)">×</button>
      </div>
      <div className="title">{p.name}</div>
      <div className="status" style={{ borderColor: cssHex(color) }}>{statusWord(status, places)}</div>
      <p className="role">{p.role}</p>
      {today && today.lines.length > 0 && (
        <ol className="log">
          {today.lines.map((l, i) => (
            <li key={i} className={i === li ? 'now' : undefined} style={i === li ? { borderColor: cssHex(color) } : undefined}>
              <Txt text={l.what} />
              {l.with.length > 0 && <span className="with"> · with {l.with.join(', ')}</span>}
            </li>
          ))}
        </ol>
      )}
      <dl>
        <dt>Today</dt>
        <dd>
          {today?.place ? (
            <button className="link" onClick={() => onSelect({ kind: 'place', id: today.place! })}>{places.get(today.place)?.name ?? today.place}</button>
          ) : '—'}
        </dd>
        <dt>Talked today</dt>
        <dd>{sessionsToday === 0 ? 'no' : talks.length ? talks.map(clock).join(' · ') : `${sessionsToday} session${sessionsToday > 1 ? 's' : ''}`}</dd>
        <dt>Memory</dt>
        <dd>{mem ? `${mem.lines} line${mem.lines === 1 ? '' : 's'} · ${mem.files} file${mem.files === 1 ? '' : 's'}` : 'empty'}</dd>
      </dl>
      <div className="bar">
        <span>memory lines</span>
        <div><i style={{ width: `${Math.min(100, ((mem?.lines ?? 0) / Math.max(1, maxLines)) * 100)}%`, background: cssHex(color) }} /></div>
      </div>
      <div className="foot">
        <button aria-pressed={following} onClick={() => { engine.setFollow(!following); setFollowing(!following) }} title="Follow (F)">Follow</button>
        <a className="btn" href={`#/p/${p.id}`}>Open timeline →</a>
      </div>
    </>
  )
}

function PlaceCard({ town, place, t, engine, colors, onSelect, onClose }: {
  town: TownData
  place: Place
  t: number
  engine: TownEngine
  colors: Map<string, number>
  onSelect: (s: Sel) => void
  onClose: () => void
}) {
  const day = Math.floor(t / DAY_MIN)
  const d = town.days[day]
  const here = town.principals.filter((p) => engine.statusOf(p.id).at === place.id)
  const groups: [string, TownPerson[]][] = [
    ['Here now', here],
    ['Lives here', town.principals.filter((p) => p.homes.includes(place.id))],
    ['Works or studies here', town.principals.filter((p) => p.anchor === place.id)],
  ]
  const events = town.events.filter((e) => e.places.includes(place.id))
  const now = events.find((e) => e.date <= d && d <= (e.end_date ?? e.date))
  return (
    <>
      <div className="head">
        <span>{KIND(place.kind)} · {districtName(place.district)}</span>
        <button onClick={onClose} aria-label="Close details" title="Close (Esc)">×</button>
      </div>
      <div className="title">{place.name}</div>
      <div className="status">{now ? `today · ${now.title}` : here.length ? `${here.length} resident${here.length > 1 ? 's' : ''} here now` : 'no residents here now'}</div>
      {place.description && <p className="role"><Txt text={place.description} /></p>}
      <dl>
        {groups.filter(([, ps]) => ps.length > 0).map(([label, ps]) => (
          <div key={label} className="row">
            <dt>{label}</dt>
            <dd>
              {ps.map((p) => (
                <button key={p.id} className="chip" onClick={() => onSelect({ kind: 'person', id: p.id })}>
                  <Swatch color={colors.get(p.id) ?? 0x888888} />{p.name.split(' ')[0]}
                </button>
              ))}
            </dd>
          </div>
        ))}
      </dl>
      {events.length > 0 && (
        <ul className="events">
          {events.map((e) => (
            <li key={e.id} className={e === now ? 'now' : undefined}>
              <span className="when">{e.date.slice(5)}{e.end_date ? `–${e.end_date.slice(5)}` : ''}</span>
              <span><b>{e.title}</b><br /><span className="muted">{e.description}</span></span>
            </li>
          ))}
        </ul>
      )}
    </>
  )
}

function Roster({ town, colors, engine, places, onSelect }: {
  town: TownData
  colors: Map<string, number>
  engine: TownEngine
  places: Map<string, Place>
  onSelect: (s: Sel) => void
}) {
  return (
    <>
      <div className="head"><span>Residents · {town.principals.length}</span></div>
      <p className="role">Each has an assistant whose memory we test. Click anyone on the map, or here.</p>
      <ul className="roster">
        {town.principals.map((p) => {
          const st = engine.statusOf(p.id)
          return (
            <li key={p.id}>
              <button onClick={() => onSelect({ kind: 'person', id: p.id })}>
                <Swatch color={colors.get(p.id) ?? 0x888888} />
                <span className="nm">{p.name}</span>
                <span className="at">{st.mode === 'talking' ? '● talking' : statusWord(st, places).replace(/^at /, '')}</span>
              </button>
            </li>
          )
        })}
      </ul>
    </>
  )
}

function Scrubber({ town, day, onDay }: { town: TownData; day: number; onDay: (d: number) => void }) {
  const n = town.days.length
  const ticks = useMemo(() => {
    const d0 = ms(town.days[0])
    return town.events.map((e) => ({ id: e.id, title: e.title, at: (ms(e.date) - d0) / 86_400_000 }))
  }, [town])
  return (
    <div className="scrub">
      <div className="track">
        {Array.from({ length: Math.ceil(n / 7) }, (_, w) => (
          <span key={w} className="wk" style={{ left: `${((w * 7) / (n - 1)) * 100}%` }}>W{w + 1}</span>
        ))}
        {ticks.map((k) => (
          <i key={k.id} className="ev" title={k.title} style={{ left: `${(k.at / (n - 1)) * 100}%` }} />
        ))}
        <i className="knob" style={{ left: `${(day / (n - 1)) * 100}%` }} />
      </div>
      <input type="range" min={0} max={n - 1} value={day} onChange={(e) => onDay(Number(e.target.value))} aria-label="day" />
    </div>
  )
}

export function Town() {
  const { data: town, error } = useTown()
  const stage = useRef<HTMLDivElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const tagHover = useRef<HTMLDivElement>(null)
  const tagSel = useRef<HTMLDivElement>(null)
  const labels = useRef<HTMLDivElement>(null)
  const plate = useRef<HTMLElement>(null)
  const [engine, setEngine] = useState<TownEngine | null>(null)
  const [t, setT] = useState(7.5 * 60)
  const [sel, setSel] = useState<Sel>(null)
  const [night, setNight] = useState(false)
  const [playing, setPlaying] = useState(true)
  const [speed, setSpeed] = useState(1)
  const colors = useMemo(() => residentColors((town?.principals ?? []).map((p) => p.id)), [town])
  const places = useMemo(() => new Map((town?.places ?? []).map((p) => [p.id, p])), [town])
  const maxLines = useMemo(() => Math.max(1, ...(town?.principals ?? []).flatMap((p) => p.memory.map((m) => m.lines))), [town])

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
          onSelect: setSel, onTick: setT, onNight: setNight,
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
  }, [town, colors])

  // exact session times for whoever is selected, for the talks that wrote nothing
  const pid = sel?.kind === 'person' ? sel.id : null
  const { data: bundle } = usePrincipal(pid)
  useEffect(() => {
    if (engine && bundle) engine.setSessions(bundle.principal.id, bundle.sessions.map((s) => s.start))
  }, [engine, bundle])

  const select = useCallback((s: Sel) => {
    engine?.select(s)
    if (s) engine?.focus(s)
  }, [engine])

  const setDay = useCallback((d: number) => {
    if (!engine) return
    engine.setTime(d * DAY_MIN + (engine.t % DAY_MIN))
  }, [engine])

  useEffect(() => {
    if (!engine || !town) return
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || e.metaKey || e.ctrlKey || e.altKey) return
      const day = Math.floor(engine.t / DAY_MIN)
      const ids = town.principals.map((p) => p.id)
      const i = sel?.kind === 'person' ? ids.indexOf(sel.id) : -1
      const act: Record<string, () => void> = {
        ArrowRight: () => setDay(Math.min(town.days.length - 1, day + 1)),
        ArrowLeft: () => setDay(Math.max(0, day - 1)),
        ' ': () => { engine.setPlaying(!engine.playing); setPlaying(engine.playing) },
        ']': () => select({ kind: 'person', id: ids[(i + 1) % ids.length] }),
        '[': () => select({ kind: 'person', id: ids[(i - 1 + ids.length) % ids.length] }),
        f: () => engine.setFollow(!engine.follow),
        Escape: () => engine.select(null),
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
  }, [engine, town, sel, select, setDay])

  if (error) return <div className="screen-msg">Could not load <code>data/town.json</code>: {error}. Re-run the export.</div>

  const day = Math.floor(t / DAY_MIN)
  const d = town?.days[day] ?? ''
  const person = town && sel?.kind === 'person' ? town.principals.find((p) => p.id === sel.id) : null
  const place = sel?.kind === 'place' ? places.get(sel.id) : null
  const events = town ? town.events.filter((e) => e.date <= d && d <= (e.end_date ?? e.date)) : []
  const talking = town && engine ? town.principals.filter((p) => engine.statusOf(p.id).mode === 'talking') : []
  const readout = !engine ? 'loading the town' : [
    !playing ? 'paused' : speed > 1 ? `${speed}×` : '',
    sel?.kind === 'person' && person ? `${person.name.split(' ')[0].toLowerCase()} · ${statusWord(engine.statusOf(person.id), places)}` :
      place ? place.name.toLowerCase() : talking.length ? `${talking.map((p) => p.name.split(' ')[0].toLowerCase()).join(', ')} talking to the assistant` : 'nothing selected',
  ].filter(Boolean).join(' · ')

  return (
    <figure className={`plate${night ? ' night' : ''}`} ref={plate}>
      <div className="cap">
        <span className="hi">
          Fig 1 · {d ? `${weekday(d)} · ${clock(t)}` : '—'}{night ? ' · night' : ''}
          {events.length > 0 && <span className="evt"> · ⚑ {events.map((e) => e.title).join(' · ')}</span>}
        </span>
        <nav className="sites" aria-label="Districts">
          {engine?.views.map((v, k) => (
            <button key={v.id} onClick={() => engine.goTo(v.id)} title={`${v.label} (${k + 1})`}>{v.label}</button>
          ))}
        </nav>
      </div>
      <div className="stage" ref={stage}>
        <canvas ref={canvas} tabIndex={0} role="img"
          aria-label="Isometric map of the town and its residents, live through the days of the season. Drag to pan, scroll to zoom, click anyone or any building for details." />
        <div className="tools">
          <button onClick={() => engine?.zoomBy(1.4)} title="Zoom in (+)" aria-label="Zoom in"><svg viewBox="0 0 16 16"><path d="M8 3v10M3 8h10" /></svg></button>
          <button onClick={() => engine?.zoomBy(1 / 1.4)} title="Zoom out (−)" aria-label="Zoom out"><svg viewBox="0 0 16 16"><path d="M3 8h10" /></svg></button>
          <button onClick={() => engine?.resetView()} title="Whole town (0)" aria-label="Whole town"><svg viewBox="0 0 16 16"><path d="M2.5 7.5 8 3l5.5 4.5M4 6.5V13h8V6.5" /></svg></button>
          <span className="gap" />
          <button onClick={() => { if (engine) { engine.setPlaying(!engine.playing); setPlaying(engine.playing) } }} title="Play or pause (Space)" aria-label={playing ? 'Pause' : 'Play'} aria-pressed={!playing}>
            {playing ? <svg viewBox="0 0 16 16"><path d="M5.5 3.5v9M10.5 3.5v9" /></svg> : <svg viewBox="0 0 16 16"><path d="M5 3.5v9l7-4.5z" /></svg>}
          </button>
          <button className="speed" onClick={() => { const k = SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length]; engine?.setSpeed(k); setSpeed(k) }}
            title={`Speed: a day in ${Math.round(DAY_MIN / MIN_PER_SEC / speed)} s`} aria-label="Change speed">{speed}×</button>
          <button onClick={() => setDay(Math.min((town?.days.length ?? 1) - 1, day + 1))} title="Next day (→)" aria-label="Next day"><svg viewBox="0 0 16 16"><path d="M4 3.5 9 8l-5 4.5M10 3.5 15 8l-5 4.5" /></svg></button>
        </div>
        <div className="labels" ref={labels} aria-hidden="true" />
        <div className="tag hover" ref={tagHover} hidden />
        <div className="tag" ref={tagSel} hidden />
        {town && engine && (
          <aside className="card" role="region" aria-label="Details">
            {person ? (
              <PersonCard key={person.id} town={town} p={person} t={t} color={colors.get(person.id) ?? 0x888888} status={engine.statusOf(person.id)}
                places={places} maxLines={maxLines} engine={engine} onSelect={select} onClose={() => engine.select(null)} />
            ) : place ? (
              <PlaceCard town={town} place={place} t={t} engine={engine} colors={colors} onSelect={select} onClose={() => engine.select(null)} />
            ) : sel?.kind === 'train' ? (
              <>
                <div className="head"><span>Train</span><button onClick={() => engine.select(null)} aria-label="Close details">×</button></div>
                <div className="title">Tule Line</div>
                <div className="status">running</div>
                <p className="role">Calls at {[...places.values()].filter((p) => p.kind === 'transit_station').map((p) => p.name).join(', ') || 'the station'}. The trains run on a loop of their own; only the residents follow the data.</p>
              </>
            ) : sel?.kind === 'bus' ? (
              <>
                <div className="head"><span>Bus</span><button onClick={() => engine.select(null)} aria-label="Close details">×</button></div>
                <div className="title">Town loop</div>
                <div className="status">running</div>
                <p className="role">Round the edge of town, stopping on each side. Scenery: only the residents follow the data.</p>
              </>
            ) : (
              <Roster town={town} colors={colors} engine={engine} places={places} onSelect={select} />
            )}
          </aside>
        )}
        {!engine && <p className="note">{town ? 'Drawing the town…' : 'Loading the town…'}</p>}
      </div>
      <div className="cap foot">
        {town && <Scrubber town={town} day={day} onDay={setDay} />}
        <div className="row">
          <span>Drag to pan · scroll to zoom · click anything · ← → day · space play</span>
          <span className="hi">{readout}</span>
        </div>
      </div>
    </figure>
  )
}
