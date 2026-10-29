import { Fragment, useEffect, useMemo, useRef, useState } from 'react'
import { checkpointsOf, eventsOf, fmtDay, fmtTime, short, usePrincipal } from './data'
import { Inspector, type Ctx } from './Inspector'
import { Timeline } from './Timeline'
import type { Index, PrincipalBundle, TimelineEvent } from './types'

const FREQ: Record<string, string> = { daily: 'daily', several_per_week: 'several per week', weekly: 'weekly' }

function Glyph({ ev, b }: { ev: TimelineEvent; b: PrincipalBundle }) {
  if (ev.kind === 'session') {
    const v = b.sessions[ev.index].verify
    return v === null ? (
      <svg width="12" height="12" viewBox="-6 -6 12 12"><circle r={4} fill="none" stroke="var(--muted)" /></svg>
    ) : v.passed ? (
      <svg width="12" height="12" viewBox="-6 -6 12 12" aria-label="verified"><circle r={4.2} fill="var(--pass)" /></svg>
    ) : (
      <svg width="12" height="12" viewBox="-6 -6 12 12" aria-label="verify failed">
        <circle r={3.8} fill="none" stroke="var(--fail)" strokeWidth={1.6} />
        <path d="M-1.8,-1.8L1.8,1.8M1.8,-1.8L-1.8,1.8" stroke="var(--fail)" strokeWidth={1.3} />
      </svg>
    )
  }
  if (ev.kind === 'commit') {
    return b.commits[ev.index].event === 'dream' ? (
      <svg width="12" height="12" viewBox="-6 -6 12 12" aria-label="dream"><path d="M0,-4.5L4.5,0L0,4.5L-4.5,0Z" fill="var(--dream)" /></svg>
    ) : (
      <svg width="12" height="12" viewBox="-6 -6 12 12" aria-label="write"><rect x={-3.5} y={-3.5} width={7} height={7} fill="var(--write)" /></svg>
    )
  }
  return (
    <svg width="12" height="12" viewBox="-6 -6 12 12" aria-label="probe checkpoint"><path d="M-4.5,-3.5L4.5,-3.5L0,4Z" fill="var(--probe)" /></svg>
  )
}

function gist(ev: TimelineEvent, b: PrincipalBundle, cps: ReturnType<typeof checkpointsOf>): [string, string] {
  if (ev.kind === 'session') {
    const s = b.sessions[ev.index]
    return [s.id.split('.').pop()!, s.task ?? '']
  }
  if (ev.kind === 'commit') {
    const c = b.commits[ev.index]
    const files = c.files.map((f) => `${f.status} ${f.path}`).join(', ')
    return [`${c.event} ${short(c.sha)}`, `${c.lines} lines · ${files}`]
  }
  const cp = cps[ev.index]
  return ['probes', `${cp.probes.length} probes: ${cp.probes.map((p) => p.kind).join(', ')}`]
}

function Log({ events, b, cps, selected, onSelect }: {
  events: TimelineEvent[]
  b: PrincipalBundle
  cps: ReturnType<typeof checkpointsOf>
  selected: string | null
  onSelect: (k: string) => void
}) {
  const box = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = box.current?.querySelector('.log-row.sel')
    el?.scrollIntoView({ block: 'nearest' })
  }, [selected])
  let day = ''
  return (
    <div className="log" ref={box}>
      {events.map((ev) => {
        const d = fmtDay(ev.t)
        const head = d !== day
        day = d
        const [what, text] = gist(ev, b, cps)
        return (
          <Fragment key={ev.key}>
            {head && <div className="log-day">{new Date(ev.t).toUTCString().slice(0, 3)} {d}</div>}
            <button className={`log-row${ev.key === selected ? ' sel' : ''}`} onClick={() => onSelect(ev.key)}>
              <span className="when">{fmtTime(ev.t)}</span>
              <Glyph ev={ev} b={b} />
              <span className="what">{what}</span>
              <span className="gist">{text}</span>
            </button>
          </Fragment>
        )
      })}
    </div>
  )
}

export function Person({ index, pid, selected, onSelect }: {
  index: Index
  pid: string
  selected: string | null
  onSelect: (key: string, replace?: boolean) => void
}) {
  const { data: b, error } = usePrincipal(pid)
  const cps = useMemo(() => (b ? checkpointsOf(b) : []), [b])
  const events = useMemo(() => (b ? eventsOf(b, cps) : []), [b, cps])
  const truth = useMemo(() => new Map((b?.truth ?? []).map((e) => [e.id, e])), [b])
  const [commitTab, setCommitTab] = useState<'diff' | 'snapshot'>('diff')
  const [memFile, setMemFile] = useState<string | null>(null)
  const aside = useRef<HTMLElement>(null)
  useEffect(() => {
    aside.current?.scrollTo({ top: 0 })
  }, [selected])

  // With nothing selected, open on the first session.
  useEffect(() => {
    if (b && (!selected || !events.some((e) => e.key === selected)) && events.length) onSelect(events[0].key, true)
  }, [b, selected, events, onSelect])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
      const tag = (e.target as HTMLElement | null)?.tagName
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return
      const i = events.findIndex((ev) => ev.key === selected)
      if (i < 0) return
      const step = e.key === 'ArrowRight' ? 1 : -1
      const lane = (ev: TimelineEvent) =>
        ev.kind === 'commit' ? `commit:${b!.commits[ev.index].event}` : ev.kind
      let j = i + step
      if (e.shiftKey) while (j >= 0 && j < events.length && lane(events[j]) !== lane(events[i])) j += step
      if (j >= 0 && j < events.length) {
        e.preventDefault()
        onSelect(events[j].key, true)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [events, selected, onSelect, b])

  const summary = index.principals.find((p) => p.id === pid)
  if (error) return <div className="screen-msg">Could not load <code>data/p/{pid}.json</code>: {error}</div>
  if (!b) return <div className="placeholder">Loading {summary?.name ?? pid}…</div>

  const ctx: Ctx = { bundle: b, start: index.range.start, truth, go: (k) => onSelect(k), commitTab, setCommitTab, memFile, setMemFile }
  const p = b.principal
  const failed = b.sessions.filter((s) => s.verify && !s.verify.passed).length
  return (
    <div className="person">
      <div className="person-head">
        <h1>{p.name}</h1>
        <span className="who">
          {p.minor ? 'minor' : p.age} · {p.role} · uses the assistant {FREQ[p.usage.frequency] ?? p.usage.frequency}
        </span>
        <span className="counts">
          <span><b>{b.sessions.length}</b> sessions ({failed} failed verify)</span>
          <span><b>{b.commits.length}</b> commits</span>
          <span><b>{b.commits.at(-1)?.lines ?? 0}</b> memory lines</span>
          <span><b>{b.probes.length}</b> probes</span>
        </span>
      </div>
      <div className="left">
        <Timeline bundle={b} checkpoints={cps} range={index.range} selected={selected} onSelect={(k) => onSelect(k)} />
        <Log events={events} b={b} cps={cps} selected={selected} onSelect={(k) => onSelect(k)} />
      </div>
      <aside className="inspector" aria-label="inspector" ref={aside}>
        <Inspector selected={selected} checkpoints={cps} ctx={ctx} />
      </aside>
    </div>
  )
}
