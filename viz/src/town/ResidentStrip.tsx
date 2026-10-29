// A resident's season, in the season strip's place while they're selected: the data view's timeline, drawn the
// console's way and fitted to the strip. Four lanes: the probe checkpoints, their talks (verified or not), and the
// memory's writes and dreams, with a hairline from each talk to the write it led to. Click a mark to open it; drag
// anywhere else to move through the days, as on the town's strip.
import { useMemo, useRef, useState } from 'react'
import { checkpointsOf } from '../data'
import type { PrincipalBundle } from '../types'
import { cssHex } from './colors'
import { clock, monthDay, weekday } from './common'
import { seasonMarks, type SeasonMark } from './insight'
import { DAY_MIN } from './schedule'
import { useWidth } from './Season'
import type { Town } from './types'

const H = 94
const LANE: Record<SeasonMark['lane'], number> = { probe: 24, talk: 45, write: 66, dream: 82 }
const NAMES: [SeasonMark['lane'], string][] = [['probe', 'Questions'], ['talk', 'Talks'], ['write', 'Writes'], ['dream', 'Dreams']]
const GAP = 9 // px: marks closer than this in one lane are dodged onto a side row
const DY = [0, -5, 5]

/** Greedy side row per mark (xs ascending), so close marks in one lane sit beside each other, not on top. */
function dodge(xs: number[]): number[] {
  const last: number[] = []
  return xs.map((x) => {
    let r = last.findIndex((lx) => x - lx >= GAP)
    if (r < 0) r = last.length < DY.length ? last.length : last.indexOf(Math.min(...last))
    last[r] = x
    return r
  })
}

function Shape({ lane, verified }: { lane: SeasonMark['lane']; verified?: boolean | null }) {
  if (lane === 'probe') return <path className="shape" d="M-4.5,-3.5L4.5,-3.5L0,4Z" fill="var(--probe)" />
  if (lane === 'write') return <rect className="shape" x={-3} y={-3} width={6} height={6} fill="var(--write)" />
  if (lane === 'dream') return <path className="shape" d="M0,-4.2L4.2,0L0,4.2L-4.2,0Z" fill="var(--dream)" />
  if (verified === false) {
    return (
      <g>
        <circle className="shape" r={3.6} fill="var(--sheet)" stroke="var(--fail)" strokeWidth={1.5} />
        <path d="M-1.6,-1.6L1.6,1.6M1.6,-1.6L-1.6,1.6" stroke="var(--fail)" strokeWidth={1.2} />
      </g>
    )
  }
  return <circle className="shape" r={3.6} fill={verified ? 'var(--c)' : 'var(--sheet)'} stroke="var(--c)" strokeWidth={1.3} />
}

export function ResidentStrip({ town, t, bundle, color, picked, onPick, onDay }: {
  town: Town
  t: number
  bundle: PrincipalBundle
  color: number
  picked: string | null
  onPick: (m: SeasonMark) => void
  onDay: (d: number) => void
}) {
  const n = town.days.length
  const end = n * DAY_MIN
  const ref = useRef<HTMLDivElement>(null)
  const w = useWidth(ref)
  const [hover, setHover] = useState<SeasonMark | null>(null)
  const marks = useMemo(() => seasonMarks(bundle, town.days[0]), [bundle, town])
  const x = (at: number) => (at / end) * w

  // each mark's place: its lane, dodged sideways where marks crowd
  const placed = useMemo(() => {
    const at = new Map<string, [number, number]>()
    for (const [lane] of NAMES) {
      const ms = marks.filter((m) => m.lane === lane)
      const rows = dodge(ms.map((m) => (m.at / end) * w))
      ms.forEach((m, i) => at.set(m.key, [(m.at / end) * w, LANE[lane] + DY[rows[i]]]))
    }
    return at
  }, [marks, end, w])

  const verified = useMemo(() => new Map(bundle.sessions.map((s) => [`s:${s.id}`, s.verify ? s.verify.passed : null])), [bundle])
  const links = useMemo(
    () => bundle.commits.flatMap((c) => (c.event === 'write' && c.session_id ? [[`s:${c.session_id}`, `c:${c.sha}`]] : [])),
    [bundle],
  )
  const gist = useMemo(() => {
    const g = new Map<string, string>()
    bundle.sessions.forEach((s) => g.set(`s:${s.id}`, s.task ?? 'A conversation'))
    bundle.commits.forEach((c) => g.set(`c:${c.sha}`, `${c.lines} lines held`))
    checkpointsOf(bundle).forEach((c) => g.set(`k:${c.id}`, `${c.probes.length} questions asked`))
    return g
  }, [bundle])
  const months = useMemo(
    () => town.days.flatMap((d, i) => (i === 0 || d.endsWith('-01') ? [{ i, label: new Date(`${d}T12:00:00Z`).toLocaleDateString('en-US', { month: 'short', timeZone: 'UTC' }) }] : [])),
    [town],
  )
  const dayAt = (clientX: number) => {
    const r = ref.current!.getBoundingClientRect()
    return Math.max(0, Math.min(n - 1, Math.floor(((clientX - r.left) / r.width) * n)))
  }
  const name = bundle.principal.name.split(' ')[0]
  const lane = (l: SeasonMark['lane']) => NAMES.find(([k]) => k === l)![1].slice(0, -1)
  const tipDay = hover ? town.days[Math.floor(hover.at / DAY_MIN)] : ''

  return (
    <>
      <div className="season resident" ref={ref} role="slider" tabIndex={0} aria-label={`Day of the season, ${name}'s timeline`}
        aria-valuemin={1} aria-valuemax={n} aria-valuenow={Math.floor(t / DAY_MIN) + 1}
        style={{ '--c': cssHex(color) } as React.CSSProperties}
        onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); onDay(dayAt(e.clientX)) }}
        onPointerMove={(e) => { if (e.buttons & 1) onDay(dayAt(e.clientX)) }}
      >
        {months.map((m) => (
          <span key={m.i} className="s-month" style={{ left: `${(m.i / n) * 100}%` }}>{m.label}</span>
        ))}
        {w > 0 && (
          <svg className="r-lanes" width={w} height={H} aria-hidden="true">
            {NAMES.map(([l]) => <line key={l} className="r-lane" x1={0} x2={w} y1={LANE[l]} y2={LANE[l]} />)}
            {links.map(([a, b]) => {
              const p = placed.get(a)
              const q = placed.get(b)
              return p && q ? <line key={b} className="r-link" x1={p[0]} y1={p[1]} x2={q[0]} y2={q[1]} /> : null
            })}
          </svg>
        )}
        {w > 0 && (
          <svg className="r-marks" width={w} height={H} role="group" aria-label={`${name}'s talks, writes, dreams and questions`}>
            {marks.map((m) => {
              const [mx, my] = placed.get(m.key) ?? [x(m.at), LANE[m.lane]]
              return (
                <g key={m.key} className={`r-mark${m.at > t ? ' later' : ''}${m.key === picked ? ' sel' : ''}`} transform={`translate(${mx},${my})`}
                  onPointerDown={(e) => e.stopPropagation()} onClick={() => onPick(m)}
                  onPointerEnter={() => setHover(m)} onPointerLeave={() => setHover(null)}>
                  <circle className="hit" r={6} />
                  <Shape lane={m.lane} verified={verified.get(m.key)} />
                </g>
              )
            })}
            {picked && placed.get(picked) && <circle className="r-halo" cx={placed.get(picked)![0]} cy={placed.get(picked)![1]} r={7.5} />}
          </svg>
        )}
        <i className="s-now" style={{ left: `${(t / end) * 100}%` }} aria-hidden="true" />
        {hover && (
          <div className={`s-tip${hover.at / end > 0.8 ? ' end' : hover.at / end < 0.12 ? ' start' : ''}`} style={{ left: `${(hover.at / end) * 100}%` }} role="status">
            <b>{lane(hover.lane)} · {weekday(tipDay)} {monthDay(tipDay)}, {clock(hover.at)}</b>
            <span>{gist.get(hover.key)}</span>
          </div>
        )}
      </div>
      <ul className="s-legend resident" style={{ '--c': cssHex(color) } as React.CSSProperties} aria-label="Key">
        {NAMES.map(([l, label]) => (
          <li key={l}>
            <svg width="10" height="10" viewBox="-5 -5 10 10" aria-hidden="true"><Shape lane={l} verified={l === 'talk' ? true : undefined} /></svg>
            {label}
          </li>
        ))}
      </ul>
    </>
  )
}
