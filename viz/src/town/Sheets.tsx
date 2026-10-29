// What the dock opens: everyone, every place, the season's events, and the town's memory. One sheet at a time,
// rising from the dock over the map; picking anything in it brings it up on the map.
import { useEffect, useRef } from 'react'
// Avatar.tsx and avatar.ts differ only in case: name the extension
import { Avatar } from './Avatar.tsx'
import { pickLook } from './avatar.ts'
import { cssHex } from './colors'
import { Dot, clock, isoAt, kindName, memNow, monthDay, statusWord, weekday } from './common'
import type { Sel, TownEngine } from './engine'
import { Close } from './icons'
import { memoryFeed } from './insight'
import { districtName } from './palette'
import { DAY_MIN } from './schedule'
import type { Place, Town, TownEvent, TownPerson } from './types'

export type SheetId = 'people' | 'places' | 'events' | 'memory'

export interface SheetCtx {
  town: Town
  t: number
  engine: TownEngine
  colors: Map<string, number>
  places: Map<string, Place>
  sel: Sel
  pick: (s: Sel, tab?: 'day' | 'memory' | 'talks') => void
  goTo: (district: string) => void
  onEvent: (e: TownEvent) => void
}

const colorOf = (c: SheetCtx, id: string) => c.colors.get(id) ?? 0x888888
const first = (name: string) => name.split(' ')[0]

function People({ c }: { c: SheetCtx }) {
  const iso = isoAt(c.town.days, c.t)
  return (
    <ul className="tiles">
      {c.town.principals.map((p) => {
        const st = c.engine.statusOf(p.id)
        const m = memNow(p, iso)
        const here = c.sel?.kind === 'person' && c.sel.id === p.id
        return (
          <li key={p.id}>
            <button className="tile" aria-pressed={here} onClick={() => c.pick({ kind: 'person', id: p.id })}>
              <span className="face">
                <Avatar look={pickLook(p)} color={colorOf(c, p.id)} size={28} />
                <Dot color={colorOf(c, p.id)} ring={st.mode === 'talking'} />
              </span>
              <span className="nm">{p.name}</span>
              <span className="mem">{m ? `${m.lines} ln` : '—'}</span>
              <span className={`state ${st.mode}`}>{statusWord(st, c.places)}</span>
            </button>
          </li>
        )
      })}
    </ul>
  )
}

function Places({ c }: { c: SheetCtx }) {
  const d = c.town.days[Math.floor(c.t / DAY_MIN)]
  const today = new Set(c.town.events.filter((e) => e.date <= d && d <= (e.end_date ?? e.date)).flatMap((e) => e.places))
  const at = new Map<string, TownPerson[]>()
  for (const p of c.town.principals) {
    const st = c.engine.statusOf(p.id)
    if (st.mode !== 'walking' && st.at) at.set(st.at, [...(at.get(st.at) ?? []), p])
  }
  const districts = [...new Set(c.town.places.map((p) => p.district))]
  return (
    <div className="districts">
      {districts.map((dist) => (
        <section key={dist}>
          <button className="district" onClick={() => c.goTo(dist)} title="Show on the map">
            {districtName(dist)}
          </button>
          <ul>
            {c.town.places.filter((p) => p.district === dist).map((pl) => {
              const here = c.sel?.kind === 'place' && c.sel.id === pl.id
              return (
                <li key={pl.id}>
                  <button className="row" aria-pressed={here} onClick={() => c.pick({ kind: 'place', id: pl.id })}>
                    <span className="nm">{pl.name}{today.has(pl.id) && <i className="flag" title="An event today" />}</span>
                    <span className="who">{(at.get(pl.id) ?? []).map((p) => <Dot key={p.id} color={colorOf(c, p.id)} />)}</span>
                    <span className="kind">{kindName(pl.kind)}</span>
                  </button>
                </li>
              )
            })}
          </ul>
        </section>
      ))}
    </div>
  )
}

function when(a: number, b: number, day: number): string {
  if (a <= day && day <= b) return 'Today'
  const k = a > day ? a - day : day - b
  if (a > day) return k === 1 ? 'Tomorrow' : `In ${k} days`
  return k === 1 ? 'Yesterday' : `${k} days ago`
}

function Events({ c }: { c: SheetCtx }) {
  const day = Math.floor(c.t / DAY_MIN)
  const idx = new Map(c.town.days.map((d, i) => [d, i]))
  const list = [...c.town.events].sort((x, y) => x.date.localeCompare(y.date))
  const ref = useRef<HTMLOListElement>(null)
  useEffect(() => {
    ref.current?.querySelector('.on, .next')?.scrollIntoView({ block: 'center' })
  }, [])
  let nextMarked = false
  return (
    <ol className="evlist" ref={ref}>
      {list.map((e) => {
        const a = idx.get(e.date) ?? 0
        const b = idx.get(e.end_date ?? e.date) ?? a
        const state = a <= day && day <= b ? 'on' : b < day ? 'past' : nextMarked ? '' : ((nextMarked = true), 'next')
        return (
          <li key={e.id} className={state}>
            <button onClick={() => c.onEvent(e)}>
              <span className="date">{monthDay(e.date)}{e.end_date && e.end_date !== e.date ? `–${monthDay(e.end_date).replace(/^[A-Z][a-z]+ /, '')}` : ''}</span>
              <span className="what">
                <b>{e.title}</b>
                <span>{e.places.map((id) => c.places.get(id)?.name ?? id).join(' · ')}</span>
              </span>
              <span className="rel">{when(a, b, day)}</span>
            </button>
          </li>
        )
      })}
    </ol>
  )
}

function Memory({ c }: { c: SheetCtx }) {
  const iso = isoAt(c.town.days, c.t)
  const held = c.town.principals.map((p) => ({ p, m: memNow(p, iso) }))
  const max = Math.max(1, ...c.town.principals.flatMap((p) => p.memory.map((m) => m.lines)))
  const feed = memoryFeed(c.town.principals, c.town.days, c.t, 16)
  const names = new Map(c.town.principals.map((p) => [p.id, p.name]))
  return (
    <div className="memgrid">
      <section>
        <h3>Held now</h3>
        <ul className="held">
          {held.map(({ p, m }) => (
            <li key={p.id}>
              <button onClick={() => c.pick({ kind: 'person', id: p.id }, 'memory')}>
                <span className="nm"><Dot color={colorOf(c, p.id)} />{first(p.name)}</span>
                <span className="val">{m ? `${m.lines} lines · ${m.files} files` : 'empty'}</span>
                <span className="gauge"><i style={{ width: `${((m?.lines ?? 0) / max) * 100}%`, background: cssHex(colorOf(c, p.id)) }} /></span>
              </button>
            </li>
          ))}
        </ul>
      </section>
      <section>
        <h3>Latest writes</h3>
        {feed.length === 0 ? <p className="empty">Nothing written yet.</p> : (
          <ol className="feed">
            {feed.map((e) => (
              <li key={`${e.pid}:${e.at}`}>
                <button onClick={() => c.pick({ kind: 'person', id: e.pid }, 'memory')}>
                  <span className="at">{weekday(c.town.days[Math.floor(e.at / DAY_MIN)])} {clock(e.at)}</span>
                  <span className="nm"><Dot color={colorOf(c, e.pid)} />{first(names.get(e.pid) ?? e.pid)}</span>
                  <span className="what">
                    {e.kind === 'dream' ? 'Dreamed' : 'Wrote after a talk'}
                    <span className={e.delta > 0 ? 'delta up' : e.delta < 0 ? 'delta down' : 'delta'}>
                      {e.delta > 0 ? `+${e.delta}` : e.delta < 0 ? `−${-e.delta}` : '±0'}
                    </span>
                    <span className="of">{e.lines} lines</span>
                  </span>
                </button>
              </li>
            ))}
          </ol>
        )}
      </section>
    </div>
  )
}

const TITLE: Record<SheetId, [string, (c: SheetCtx) => string]> = {
  people: ['Residents', (c) => {
    const talking = c.town.principals.filter((p) => c.engine.statusOf(p.id).mode === 'talking').length
    return `${c.town.principals.length} people, each with an assistant that remembers${talking ? ` · ${talking} talking now` : ''}`
  }],
  places: ['Places', (c) => `${c.town.places.length} places in ${new Set(c.town.places.map((p) => p.district)).size} districts · dots are who is there now`],
  events: ['Events', (c) => `${c.town.events.length} town events this season · pick one to go to that day`],
  memory: ['Memory', () => 'What each assistant holds now, and what changed most recently'],
}

export function Sheet({ id, c, onClose }: { id: SheetId; c: SheetCtx; onClose: () => void }) {
  const [title, sub] = TITLE[id]
  return (
    <section className={`sheet ${id}`} role="dialog" aria-label={title}>
      <header className="sheet-head">
        <h2>{title}</h2>
        <p>{sub(c)}</p>
        <button className="x" onClick={onClose} aria-label="Close" title="Close (Esc)"><Close /></button>
      </header>
      <div className="sheet-body">
        {id === 'people' ? <People c={c} /> : id === 'places' ? <Places c={c} /> : id === 'events' ? <Events c={c} /> : <Memory c={c} />}
      </div>
    </section>
  )
}
