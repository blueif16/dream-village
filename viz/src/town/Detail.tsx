// The panel on the right for whatever is selected. A resident's is their badge: hung on a cord over the map, their
// drawn portrait, where they are, their day, what their assistant remembers as of this moment (the lines the latest
// write added marked in their colour), and their talks.
import { useMemo, useState } from 'react'
import { commitAt, ms, parseDiff } from '../data'
import { MemLine, Txt } from '../text'
import type { PrincipalBundle } from '../types'
// Avatar.tsx and avatar.ts differ only in case: name the extension, or a case-insensitive disk picks the wrong one
import { Avatar } from './Avatar.tsx'
import { pickLook, residentNo } from './avatar.ts'
import { cssHex } from './colors'
import { Dot, capital, clock, isoAt, kindName, memNow, monthDay, statusWord, weekday } from './common'
import type { Sel, Status, TownEngine } from './engine'
import { Close, Follow, Next, Out, Prev } from './icons'
import { Barcode, Cord, Dither, onColor, roleSplit } from './IdCard'
import { districtName } from './palette'
import { DAY_MIN, lineIndexAt, talkTimes } from './schedule'
import type { Place, Town, TownPerson } from './types'

export type PersonTab = 'day' | 'memory' | 'talks'

function Head({ kicker, title, sub, onClose, nav }: {
  kicker: React.ReactNode
  title: string
  sub?: React.ReactNode
  onClose: () => void
  nav?: [() => void, () => void]
}) {
  return (
    <header className="detail-head">
      <div className="bar">
        <span className="kicker">{kicker}</span>
        <span className="acts">
          {nav && (
            <>
              <button onClick={nav[0]} aria-label="Previous resident" title="Previous resident ([)"><Prev /></button>
              <button onClick={nav[1]} aria-label="Next resident" title="Next resident (])"><Next /></button>
            </>
          )}
          <button onClick={onClose} aria-label="Close details" title="Close (Esc)"><Close /></button>
        </span>
      </div>
      <h2>{title}</h2>
      {sub && <p className="role">{sub}</p>}
    </header>
  )
}

function Readouts({ rows }: { rows: [string, React.ReactNode][] }) {
  return (
    <dl className="readouts">
      {rows.map(([k, v]) => (
        <div key={k}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  )
}

/** A memory file as the assistant holds it: frontmatter quiet, headings set, the lines just added marked. */
function MemoryText({ text, added }: { text: string; added: Set<number> }) {
  const lines = text.split('\n')
  let fm = lines[0]?.trim() === '---'
  return (
    <pre className="memtext">
      {lines.map((line, i) => {
        const inFm = fm
        if (fm && i > 0 && line.trim() === '---') fm = false
        const cls = [inFm ? 'fm' : line.startsWith('#') ? 'h' : '', added.has(i + 1) ? 'new' : ''].filter(Boolean).join(' ')
        return (
          <div key={i} className={cls || undefined}>
            {line === '' ? ' ' : inFm ? line : <MemLine text={line} />}
          </div>
        )
      })}
    </pre>
  )
}

export function MemoryTab({ bundle, iso, file, setFile }: {
  bundle: PrincipalBundle | null
  iso: string
  file: string | null
  setFile: (f: string) => void
}) {
  const ci = bundle ? commitAt(bundle, ms(iso)) : -1
  const c = bundle && ci >= 0 ? bundle.commits[ci] : null
  const diff = useMemo(() => (c ? parseDiff(c.diff) : []), [c])
  if (!bundle) return <p className="empty">Reading their memory…</p>
  if (!c) return <p className="empty">Nothing remembered yet. The assistant writes its first memory after their first talk.</p>
  const files = Object.keys(c.snapshot).sort()
  const changed = new Map(diff.map((f) => [f.path, f.add]))
  const born = new Set(c.files.filter((f) => f.status === 'A').map((f) => f.path))
  const cur = file && c.snapshot[file] !== undefined ? file : diff.find((f) => c.snapshot[f.path] !== undefined)?.path ?? files[0]
  // a file this write created is all new: mark the lines only where an existing file grew
  const added = born.has(cur) ? new Set<number>() : new Set(diff.find((f) => f.path === cur)?.lines.flatMap((l) => (l.kind === 'add' && l.new !== null ? [l.new] : [])) ?? [])
  const d = c.sim_time.slice(0, 10)
  return (
    <div className="memtab">
      <p className="asof">
        As of <b>{weekday(d)} {monthDay(d)}, {c.sim_time.slice(11, 16)}</b>
        {c.event === 'dream' ? ', consolidated overnight' : ', written after a talk'} · {c.lines} lines in {files.length} files
      </p>
      <div className="files" role="tablist" aria-label="Memory files">
        {files.map((f) => (
          <button key={f} role="tab" aria-selected={f === cur} onClick={() => setFile(f)}>
            {f.replace(/\.md$/, '')}
            {born.has(f) ? <span className="chg">new</span> : changed.has(f) && <span className="chg">{changed.get(f)! > 0 ? `+${changed.get(f)}` : '•'}</span>}
          </button>
        ))}
      </div>
      {cur && <MemoryText text={c.snapshot[cur]} added={added} />}
      {added.size > 0 && <p className="legend"><i className="swatch" /> added by this write</p>}
      {born.has(cur) && <p className="legend">This write started the file.</p>}
    </div>
  )
}

export function TalksTab({ bundle, d, iso, pid }: { bundle: PrincipalBundle | null; d: string; iso: string; pid: string }) {
  const [open, setOpen] = useState<string | null>(null)
  if (!bundle) return <p className="empty">Reading their talks…</p>
  const today = bundle.sessions.filter((s) => s.start.slice(0, 10) === d)
  if (!today.length) return <p className="empty">No talks with the assistant today.</p>
  return (
    <ol className="talks">
      {today.map((s) => {
        const later = s.start > iso
        return (
          <li key={s.id} className={later ? 'later' : undefined}>
            <button className="talk" aria-expanded={open === s.id} onClick={() => setOpen(open === s.id ? null : s.id)} disabled={later}>
              <span className="at">{s.start.slice(11, 16)}</span>
              <span className="what">
                <b>{s.task ? <Txt text={s.task} /> : 'A conversation'}</b>
                <span>{later ? 'later today' : `${s.turns.length} turns${s.task_type ? ` · ${s.task_type.replace(/_/g, ' ')}` : ''}`}</span>
              </span>
            </button>
            {open === s.id && !later && (
              <div className="turns">
                {s.turns.map((tn, i) => (
                  <p key={i} className={tn.role}><span>{tn.role === 'user' ? 'Them' : 'Assistant'}</span><Txt text={tn.text} /></p>
                ))}
                <a className="more" href={`#/p/${encodeURIComponent(pid)}/${encodeURIComponent(`s:${s.id}`)}`}>Open this talk in the data view <Out /></a>
              </div>
            )}
          </li>
        )
      })}
    </ol>
  )
}

/** Their day's life-log lines: the one the hour falls in marked, the ones still to come quiet. */
export function DayTab({ p, t, d }: { p: TownPerson; t: number; d: string }) {
  const today = p.days[Math.floor(t / DAY_MIN)]
  const li = lineIndexAt(p, t)
  if (!today || !today.lines.length) return <p className="empty">Their life log has nothing for {weekday(d)}.</p>
  return (
    <ol className="daylog">
      {today.lines.map((l, i) => (
        <li key={i} className={i === li ? 'now' : li === null || i > li ? 'later' : undefined}>
          <Txt text={l.what} />
          {l.with.length > 0 && <span className="with">with {l.with.join(', ')}</span>}
        </li>
      ))}
    </ol>
  )
}

const count = (n: number, one: string) => `${n} ${one}${n === 1 ? '' : 's'}`

export function PersonDetail({ town, p, t, color, status, places, engine, bundle, maxLines, onSelect, onStep, onClose, onExpand }: {
  town: Town
  p: TownPerson
  t: number
  color: number
  status: Status
  places: Map<string, Place>
  engine: TownEngine
  bundle: PrincipalBundle | null
  maxLines: number
  onSelect: (s: Sel) => void
  onStep: (k: 1 | -1) => void
  onClose: () => void
  onExpand: () => void // their memory, talks and timeline, in the panel over the map
}) {
  const day = Math.floor(t / DAY_MIN)
  const d = town.days[day]
  const iso = isoAt(town.days, t)
  const today = p.days[day]
  const mem = memNow(p, iso)
  const talks = talkTimes(p, town.days).filter((x) => Math.floor(x / DAY_MIN) === day)
  const sessions = (p.sessions[d] ?? []).length
  const done = talks.filter((x) => x <= t)
  const [following, setFollowing] = useState(engine.follow)
  // their file has every session's start; town.json only the ones that wrote memory
  const exact = bundle ? bundle.sessions.filter((s) => s.start.slice(0, 10) === d).map((s) => s.start) : null
  const talked = exact
    ? !exact.length ? 'No talk today' : exact.some((s) => s <= iso) ? exact.filter((s) => s <= iso).map((s) => s.slice(11, 16)).join(' · ') : `Later, ${exact[0].slice(11, 16)}`
    : sessions === 0 ? 'No talk today' : done.length ? done.map(clock).join(' · ') : talks.length ? `Later, ${clock(talks[0])}` : `${sessions} today`
  const look = useMemo(() => pickLook(p), [p])
  // the badge's facts: where they live and spend their days, when they first talked, how many talks so far
  const homes = [...new Set(p.homes.map((h) => places.get(h)?.name ?? h))]
  const anchor = p.anchor ? places.get(p.anchor)?.name ?? p.anchor : null
  const first = Object.keys(p.sessions).filter((k) => p.sessions[k].length).sort()[0] ?? null
  const before = town.days.slice(0, day).reduce((n, k) => n + (p.sessions[k]?.length ?? 0), 0)
  const soFar = before + (exact ? exact.filter((s) => s <= iso).length : done.length)
  const [given, ...family] = p.name.split(' ')
  const longest = Math.max(given.length, family.join(' ').length)
  const [head, rest] = roleSplit(capital(p.role))
  const follow = () => { engine.setFollow(!following); setFollowing(!following) }
  return (
    <aside className="detail idcard" role="region" aria-label={`${p.name}, details`}
      style={{ '--c': cssHex(color), '--c-on': onColor(color) === 'ink' ? '#1a1c1a' : '#ffffff' } as React.CSSProperties}>
      <Cord />
      <div className="idcard-paper">
        <div className="idcard-hang">
          <span className="acts">
            <button onClick={() => onStep(-1)} aria-label="Previous resident" title="Previous resident ([)"><Prev /></button>
            <button onClick={() => onStep(1)} aria-label="Next resident" title="Next resident (])"><Next /></button>
          </span>
          <span className="acts">
            <button className="follow" aria-pressed={following} onClick={follow} title="Follow on the map (F)"><Follow /> {following ? 'Following' : 'Follow'}</button>
            <button onClick={onClose} aria-label="Close details" title="Close (Esc)"><Close /></button>
          </span>
        </div>
        <header className="idcard-brand">
          <span className="town">Dream Village</span>
          <span className="pill"><Dot color={color} ring={status.mode === 'talking'} /> Resident</span>
          <span className="no" title="Resident number, from their id">{residentNo(p.id)}</span>
        </header>
        <div className="idcard-body">
          <div className="idcard-info">
            <dl>
              <div><dt>Home</dt><dd>{homes.join(' · ') || '—'}</dd></div>
              {anchor && <div><dt>Days at</dt><dd>{anchor}</dd></div>}
            </dl>
            <dl className="r">
              {first && <div><dt>{first <= d ? 'Since' : 'First talk'}</dt><dd>{monthDay(first)}, {first.slice(0, 4)}</dd></div>}
              <div><dt>Town</dt><dd>{town.town}</dd></div>
            </dl>
          </div>
          <section className="idcard-art" aria-label={`${p.name}, drawn`}>
            <Dither height={216} />
            <h2 className="idcard-name" style={{ fontSize: Math.min(46, Math.floor(152 / (0.4 * longest))) }}>
              <span>{given}</span>
              {family.length > 0 && <span>{family.join(' ')}</span>}
            </h2>
            <Avatar look={look} color={color} size={240} block="#fbf5e6" label={`A pen drawing of ${given}: a drawing, not a likeness`} />
          </section>
          <dl className={p.minor ? 'idcard-meta minor' : 'idcard-meta'}>
            {!p.minor && <div><dt>Age</dt><dd className="num">{p.age ?? '—'}</dd></div>}
            <div className="job"><dt>Role</dt><dd><b>{head}</b>{rest}</dd></div>
            <div><dt>Talks</dt><dd className="num">{soFar}<small>so far</small></dd></div>
          </dl>
          <dl className="idcard-live">
            <div><dt>Now</dt><dd>{statusWord(status, places)}</dd></div>
            <div><dt>Today</dt><dd>{today?.place ? <button className="link" onClick={() => onSelect({ kind: 'place', id: today.place! })}>{places.get(today.place)?.name ?? today.place}</button> : '—'}</dd></div>
            <div><dt>Talked</dt><dd>{talked}</dd></div>
            <div>
              <dt>Memory</dt>
              <dd>
                {mem ? `${count(mem.lines, 'line')} in ${count(mem.files, 'file')}` : 'Nothing yet'}
                <span className="gauge"><i style={{ width: `${Math.min(100, ((mem?.lines ?? 0) / maxLines) * 100)}%` }} /></span>
                {mem && <small>last written {weekday(mem.t.slice(0, 10))} {mem.t.slice(11, 16)}</small>}
              </dd>
            </div>
          </dl>
        </div>
        <footer className="idcard-stub">
          <div className="code">
            <Barcode id={p.id} />
            <span>{residentNo(p.id)} · {town.town}</span>
          </div>
          <button className="open" onClick={onExpand} title="Their memory, talks and timeline, over the map">
            <span>Expand</span>
            <Out />
          </button>
        </footer>
      </div>
    </aside>
  )
}

export function PlaceDetail({ town, place, t, engine, colors, onSelect, onClose }: {
  town: Town
  place: Place
  t: number
  engine: TownEngine
  colors: Map<string, number>
  onSelect: (s: Sel) => void
  onClose: () => void
}) {
  const d = town.days[Math.floor(t / DAY_MIN)]
  const here = town.principals.filter((p) => { const st = engine.statusOf(p.id); return st.mode !== 'walking' && st.at === place.id })
  const groups: [string, TownPerson[]][] = [
    ['Here now', here],
    ['Lives here', town.principals.filter((p) => p.homes.includes(place.id))],
    ['Works or studies here', town.principals.filter((p) => p.anchor === place.id)],
  ]
  const events = town.events.filter((e) => e.places.includes(place.id)).sort((a, b) => a.date.localeCompare(b.date))
  const now = events.find((e) => e.date <= d && d <= (e.end_date ?? e.date))
  return (
    <aside className="detail" role="region" aria-label={`${place.name}, details`}>
      <Head kicker={`${kindName(place.kind)} · ${districtName(place.district)}`} title={place.name} sub={place.description ? <Txt text={capital(place.description)} /> : undefined} onClose={onClose} />
      <div className="detail-body">
        <Readouts rows={[
          ['Here now', here.length ? `${here.length} resident${here.length > 1 ? 's' : ''}` : 'Nobody'],
          ['Today', now ? <span className="evnow">{now.title}</span> : 'No event'],
        ]} />
        {groups.filter(([, ps]) => ps.length > 0).map(([label, ps]) => (
          <section key={label} className="group">
            <h3>{label}</h3>
            <div className="chips">
              {ps.map((p) => (
                <button key={p.id} className="pchip" onClick={() => onSelect({ kind: 'person', id: p.id })}>
                  <Dot color={colors.get(p.id) ?? 0x888888} />{p.name}
                </button>
              ))}
            </div>
          </section>
        ))}
        {events.length > 0 && (
          <section className="group">
            <h3>Events here</h3>
            <ol className="evmini">
              {events.map((e) => (
                <li key={e.id} className={e === now ? 'on' : (e.end_date ?? e.date) < d ? 'past' : undefined}>
                  <span className="date">{monthDay(e.date)}</span>
                  <span><b>{e.title}</b><span><Txt text={e.description} /></span></span>
                </li>
              ))}
            </ol>
          </section>
        )}
      </div>
    </aside>
  )
}

export function VehicleDetail({ kind, places, onClose }: { kind: 'train' | 'bus'; places: Map<string, Place>; onClose: () => void }) {
  const stations = [...places.values()].filter((p) => p.kind === 'transit_station').map((p) => p.name)
  return (
    <aside className="detail short" role="region" aria-label="Details">
      <Head kicker={kind === 'train' ? 'Train' : 'Bus'} title={kind === 'train' ? 'Tule Line' : 'Town loop'} onClose={onClose}
        sub={kind === 'train'
          ? `Calls at ${stations.join(', ') || 'the station'}. The trains keep a timetable of their own; only the residents follow the data.`
          : 'Round the edge of town, stopping on each side. Scenery: only the residents follow the data.'} />
    </aside>
  )
}
