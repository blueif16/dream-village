import { useCallback, useEffect, useMemo, useState } from 'react'
import { fmtWhen, ms, useTown } from '../data'
import { Txt } from '../text'
import { css, look } from './people'
import { TownCanvas, type Hover } from './TownCanvas'
import type { Place, Selection, Town as TownData, TownPerson } from './types'

const KIND = (k: string) => k.replace(/_/g, ' ')
const WHY: Record<string, string> = {
  life_log: "named in the day's life log",
  home: 'at home: the life log names no place',
  work: 'at school or work: the life log names no place',
}

function weekday(d: string) {
  return fmtWhen(`${d}T12:00:00`).split(' · ')[0]
}

function Swatch({ id }: { id: string }) {
  return <span className="swatch" style={{ background: css(look(id).body) }} />
}

function PersonCard({ town, p, day, places, onSelect }: {
  town: TownData
  p: TownPerson
  day: number
  places: Map<string, Place>
  onSelect: (s: Selection) => void
}) {
  const d = town.days[day]
  const today = p.days[day]
  const place = today?.place ? places.get(today.place) : null
  const end = ms(`${d}T23:59:59`)
  const mem = [...p.memory].reverse().find((m) => ms(m.t) <= end)
  const sessions = p.sessions[d] ?? []
  return (
    <div className="town-card">
      <div className="town-card-head">
        <Swatch id={p.id} />
        <h2>{p.name}</h2>
        <span className="muted">{p.minor ? 'minor' : p.age}</span>
      </div>
      <p className="town-role">{p.role}</p>
      <dl className="town-facts">
        <dt>where</dt>
        <dd>
          {place ? (
            <button className="linkish plain" onClick={() => onSelect({ kind: 'place', id: place.id })}>{place.name}</button>
          ) : (
            '—'
          )}
          {today?.why && <div className="muted small">{WHY[today.why]}</div>}
        </dd>
        <dt>memory</dt>
        <dd>
          {mem ? `${mem.lines} lines in ${mem.files} files` : 'empty'}{' '}
          <span className="muted small">{mem ? `at the last commit, ${fmtWhen(mem.t)}` : 'no commit yet'}</span>
        </dd>
        <dt>sessions</dt>
        <dd>
          {sessions.length === 0 && <span className="muted">none this day</span>}
          {sessions.map((sid) => (
            <a key={sid} className="session-link" href={`#/p/${p.id}/${encodeURIComponent(`s:${sid}`)}`}>
              {sid}
            </a>
          ))}
        </dd>
      </dl>
      <h3>Life log · {weekday(d)}</h3>
      {today && today.lines.length > 0 ? (
        <ul className="town-lines">
          {today.lines.map((l, i) => (
            <li key={i}>
              <Txt text={l.what} />
              {l.with.length > 0 && <span className="muted small"> with {l.with.join(', ')}</span>}
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted">No life-log entry for this day.</p>
      )}
      <a className="town-button" href={`#/p/${p.id}`}>Open timeline →</a>
    </div>
  )
}

function PlaceCard({ town, place, day, onSelect }: {
  town: TownData
  place: Place
  day: number
  onSelect: (s: Selection) => void
}) {
  const d = town.days[day]
  const groups: [string, TownPerson[]][] = [
    ['here today', town.principals.filter((p) => p.days[day]?.place === place.id)],
    ['lives here', town.principals.filter((p) => p.homes.includes(place.id))],
    ['works or studies here', town.principals.filter((p) => p.anchor === place.id)],
  ]
  const events = town.events.filter((e) => e.places.includes(place.id))
  return (
    <div className="town-card">
      <div className="town-card-head">
        <h2>{place.name}</h2>
      </div>
      <p className="town-role">
        <span className="chip kind">{KIND(place.kind)}</span> {place.description && <Txt text={place.description} />}
      </p>
      <dl className="town-facts">
        {groups.map(([label, ps]) => (
          <div key={label} style={{ display: 'contents' }}>
            <dt>{label}</dt>
            <dd>
              {ps.length === 0 && <span className="muted">nobody</span>}
              {ps.map((p) => (
                <button key={p.id} className="person-chip" onClick={() => onSelect({ kind: 'person', id: p.id })}>
                  <Swatch id={p.id} /> {p.name.split(' ')[0]}
                </button>
              ))}
            </dd>
          </div>
        ))}
      </dl>
      <h3>Events</h3>
      {events.length === 0 ? (
        <p className="muted">No town events here.</p>
      ) : (
        <ul className="town-events">
          {events.map((e) => {
            const now = e.date <= d && d <= (e.end_date ?? e.date)
            return (
              <li key={e.id} className={now ? 'now' : undefined}>
                <span className="mono">{e.date.slice(5)}{e.end_date ? `–${e.end_date.slice(5)}` : ''}</span>
                <span>
                  <b>{e.title}</b>
                  {now && <span className="badge pass">today</span>}
                  <div className="muted small">{e.description}</div>
                </span>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

export function Town() {
  const { data: town, error } = useTown()
  const [day, setDay] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [sel, setSel] = useState<Selection>(null)
  const [hover, setHover] = useState<Hover | null>(null)
  const places = useMemo(() => new Map((town?.places ?? []).map((p) => [p.id, p])), [town])
  const n = town?.days.length ?? 0
  const onSelect = useCallback((s: Selection) => setSel(s), [])

  useEffect(() => {
    if (!playing) return
    const t = window.setInterval(() => setDay((d) => (d + 1 < n ? d + 1 : (setPlaying(false), d))), 900)
    return () => window.clearInterval(t)
  }, [playing, n])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || !n) return
      if (e.key === 'ArrowRight') setDay((d) => Math.min(n - 1, d + 1))
      else if (e.key === 'ArrowLeft') setDay((d) => Math.max(0, d - 1))
      else if (e.key === ' ') setPlaying((p) => !p)
      else return
      e.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [n])

  if (error) return <div className="screen-msg">Could not load <code>data/town.json</code>: {error}. Re-run the export.</div>
  if (!town) return <div className="placeholder">Loading the town…</div>

  const d = town.days[day]
  const events = town.events.filter((e) => e.date <= d && d <= (e.end_date ?? e.date))
  const person = sel?.kind === 'person' ? town.principals.find((p) => p.id === sel.id) : null
  const place = sel?.kind === 'place' ? places.get(sel.id) : null
  const tip = hover
    ? hover.kind === 'place'
      ? { title: places.get(hover.id)?.name, sub: KIND(places.get(hover.id)?.kind ?? '') }
      : (() => {
          const p = town.principals.find((q) => q.id === hover.id)
          const at = p?.days[day]?.place
          return { title: p?.name, sub: at ? `at ${places.get(at)?.name}` : '' }
        })()
    : null

  return (
    <div className="town">
      <div className="town-main">
        <div className="town-bar">
          <button className="town-play" onClick={() => setPlaying((p) => !p)} aria-label={playing ? 'pause' : 'play'}>
            {playing ? '❚❚' : '▶'}
          </button>
          <div className="town-date">
            <b>{weekday(d)}</b>
            <span className="muted"> · week {Math.floor(day / 7) + 1} · {d.slice(0, 4)}</span>
          </div>
          <input type="range" min={0} max={n - 1} value={day} onChange={(e) => setDay(Number(e.target.value))}
            aria-label="day" className="town-slider" />
          <span className="muted small">
            <kbd>←</kbd> <kbd>→</kbd> day · <kbd>space</kbd> play
          </span>
        </div>
        <div className="town-events-today">
          {events.length === 0 ? (
            <span className="muted small">No town events today.</span>
          ) : (
            events.map((e) => (
              <button key={e.id} className="event-chip" onClick={() => e.places[0] && setSel({ kind: 'place', id: e.places[0] })}>
                ⚑ {e.title}
              </button>
            ))
          )}
        </div>
        <TownCanvas
          town={town}
          day={day}
          selection={sel}
          onSelect={onSelect}
          onHover={setHover}
          overlay={
            hover &&
            tip && (
              <div className="town-tip" style={{ left: hover.x + 12, top: hover.y + 12 }}>
                <b>{tip.title}</b>
                {tip.sub && <div className="muted">{tip.sub}</div>}
              </div>
            )
          }
        />
      </div>
      <aside className="town-side">
        {person && <PersonCard town={town} p={person} day={day} places={places} onSelect={onSelect} />}
        {place && <PlaceCard town={town} place={place} day={day} onSelect={onSelect} />}
        {!person && !place && <p className="placeholder small">Click a person or a building.</p>}
        <h3 className="roster-title">The twelve</h3>
        <ul className="roster">
          {town.principals.map((p) => {
            const at = p.days[day]?.place
            const talked = (p.sessions[d] ?? []).length > 0
            return (
              <li key={p.id}>
                <button aria-current={sel?.kind === 'person' && sel.id === p.id}
                  onClick={() => setSel({ kind: 'person', id: p.id })}>
                  <Swatch id={p.id} />
                  <span className="roster-name">{p.name}</span>
                  <span className="roster-at muted">{at ? places.get(at)?.name : ''}</span>
                  {talked ? <span className="roster-talk" title="talked to the assistant this day" /> : <span />}
                </button>
              </li>
            )
          })}
        </ul>
      </aside>
    </div>
  )
}
