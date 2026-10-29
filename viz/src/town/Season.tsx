// The season strip: how much the town talked each day, the town's events on the baseline with the few nearest to
// now named, and where we are. With a resident selected, their assistant's memory (lines held) runs over it.
// Drag or click to move through the days; the hour of the day is kept.
import { useEffect, useId, useMemo, useRef, useState } from 'react'
import { cssHex } from './colors'
import { monthDay, weekday } from './common'
import { fitLabels, memoryFeed, talksByDay } from './insight'
import { DAY_MIN } from './schedule'
import type { Town, TownEvent, TownPerson } from './types'

const ROWS = 2
const NAMED = 6 // event names on the strip at once: the one under way, then the nearest
const LABEL_FONT = "600 12px 'Atkinson Hyperlegible Next', sans-serif"
let measure: CanvasRenderingContext2D | null = null
function textWidth(s: string): number {
  measure ??= document.createElement('canvas').getContext('2d')
  if (!measure) return s.length * 6.6
  measure.font = LABEL_FONT
  return measure.measureText(s).width
}

export function useWidth(ref: React.RefObject<HTMLElement | null>): number {
  const [w, setW] = useState(0)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(() => setW(el.clientWidth))
    ro.observe(el)
    setW(el.clientWidth)
    return () => ro.disconnect()
  }, [ref])
  return w
}

/** A resident's memory over the season as a step line: [minute, lines] at each commit, oldest first. */
function memorySteps(p: TownPerson, days: string[]): [number, number][] {
  return memoryFeed([p], days, Infinity, Infinity).reverse().map((e) => [e.at, e.lines])
}

export function Season({ town, t, person, color, onDay, onEvent }: {
  town: Town
  t: number
  person: TownPerson | null
  color: number
  onDay: (d: number) => void
  onEvent: (e: TownEvent) => void
}) {
  const n = town.days.length
  const end = n * DAY_MIN
  const day = Math.floor(t / DAY_MIN)
  const ref = useRef<HTMLDivElement>(null)
  const w = useWidth(ref)
  const [hover, setHover] = useState<number | null>(null)
  const clip = useId()

  const total = useMemo(() => {
    const sum = town.days.map(() => 0)
    for (const p of town.principals) talksByDay(p, town.days).forEach((c, i) => (sum[i] += c))
    return sum
  }, [town])
  const maxAll = Math.max(1, ...total)
  const steps = useMemo(() => (person ? memorySteps(person, town.days) : []), [person, town])
  const maxLines = Math.max(1, ...steps.map(([, l]) => l))
  const curve = useMemo(() => {
    if (!steps.length) return null
    const y = (l: number) => (100 - (l / maxLines) * 92).toFixed(2)
    let d = `M0 100 H${steps[0][0]}`
    for (const [at, l] of steps) d += ` H${at} V${y(l)}`
    return { line: `${d} H${end}`, area: `${d} H${end} V100 Z` }
  }, [steps, maxLines, end])
  const idx = useMemo(() => new Map(town.days.map((d, i) => [d, i])), [town])
  const months = useMemo(
    () => town.days.flatMap((d, i) => (i === 0 || d.endsWith('-01') ? [{ i, label: new Date(`${d}T12:00:00Z`).toLocaleDateString('en-US', { month: 'short', timeZone: 'UTC' }) }] : [])),
    [town],
  )
  const evs = useMemo(
    () => town.events.flatMap((e) => {
      const a = idx.get(e.date)
      if (a === undefined) return []
      return [{ e, a, b: idx.get(e.end_date ?? e.date) ?? a }]
    }),
    [town, idx],
  )

  // the event under way first, then the nearest to today; the rest keep their mark and show on hover
  const col = w / n
  const items = evs.map(({ e, a }) => {
    const x = (a + 0.5) * col
    const lw = textWidth(e.title) + 14
    const flip = x + lw > w - 2
    return { x: flip ? x - lw : x, w: lw, stem: x, flip }
  })
  const on = evs.map(({ a, b }) => a <= day && day <= b)
  const order = evs.map((_, i) => i).sort((i, j) => Number(on[j]) - Number(on[i]) || Math.abs(evs[i].a - day) - Math.abs(evs[j].a - day) || i - j)
  const rows = w ? fitLabels(items, order, ROWS, 14) : evs.map(() => -1)
  order.filter((i) => rows[i] >= 0).slice(NAMED).forEach((i) => (rows[i] = -1))

  const dayAt = (clientX: number) => {
    const r = ref.current!.getBoundingClientRect()
    return Math.max(0, Math.min(n - 1, Math.floor(((clientX - r.left) / r.width) * n)))
  }
  const who = person?.name.split(' ')[0]
  const tip = hover === null ? null : {
    d: town.days[hover],
    talks: total[hover],
    theirs: person ? talksByDay(person, town.days)[hover] : 0,
    lines: steps.filter(([at]) => at < (hover + 1) * DAY_MIN).at(-1)?.[1] ?? 0,
    events: evs.filter(({ a, b }) => a <= hover && hover <= b).map(({ e }) => e.title),
  }

  return (
    <>
    <div
      className={`season${person ? ' with-person' : ''}`} ref={ref} role="slider" tabIndex={0} aria-label="Day of the season"
      aria-valuemin={1} aria-valuemax={n} aria-valuenow={day + 1} aria-valuetext={`${weekday(town.days[day])} ${monthDay(town.days[day])}`}
      style={{ '--c': cssHex(color) } as React.CSSProperties}
      onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); onDay(dayAt(e.clientX)) }}
      onPointerMove={(e) => { const d = dayAt(e.clientX); setHover(d); if (e.buttons & 1) onDay(d) }}
      onPointerLeave={() => setHover(null)}
    >
      {months.map((m) => (
        <span key={m.i} className="s-month" style={{ left: `${(m.i / n) * 100}%` }}>{m.label}</span>
      ))}
      <div className="s-bars" aria-hidden="true">
        {total.map((c, i) => (
          <span key={i} className={i < day ? 'd past' : i === day ? 'd today' : 'd'}>
            <i className="b" style={{ height: `${(c / maxAll) * 100}%` }} />
          </span>
        ))}
      </div>
      {curve && (
        <svg className="s-curve" viewBox={`0 0 ${end} 100`} preserveAspectRatio="none" aria-hidden="true">
          <defs>
            <clipPath id={`${clip}p`}><rect x={0} y={-5} width={t} height={110} /></clipPath>
            <clipPath id={`${clip}f`}><rect x={t} y={-5} width={end - t} height={110} /></clipPath>
          </defs>
          <g clipPath={`url(#${clip}p)`}>
            <path className="area" d={curve.area} />
            <path className="line" d={curve.line} />
          </g>
          <g clipPath={`url(#${clip}f)`} className="later">
            <path className="area" d={curve.area} />
            <path className="line" d={curve.line} />
          </g>
        </svg>
      )}
      <i className="s-base" aria-hidden="true" />
      {evs.map(({ e, a, b }, i) => (
        <i key={e.id} aria-hidden="true"
          className={`s-mark${on[i] ? ' on' : b < day ? ' past' : ''}${b > a ? ' span' : ''}`}
          style={b > a ? { left: `${(a / n) * 100}%`, width: `${((b - a + 1) / n) * 100}%` } : { left: `${((a + 0.5) / n) * 100}%` }} />
      ))}
      {evs.map(({ e, a, b }, i) => rows[i] >= 0 && (
        <button key={e.id} type="button"
          className={`s-label${items[i].flip ? ' flip' : ''}${on[i] ? ' on' : b < day ? ' past' : ''}`}
          style={{ left: `${((a + 0.5) / n) * 100}%`, '--row': rows[i] } as React.CSSProperties}
          onPointerDown={(ev) => ev.stopPropagation()}
          onClick={() => onEvent(e)}
          title={e.description}
        >
          <span>{e.title}</span>
        </button>
      ))}
      <i className="s-now" style={{ left: `${(t / end) * 100}%` }} aria-hidden="true" />
      {tip && hover !== null && (
        <>
          <i className="s-hover" style={{ left: `${((hover + 0.5) / n) * 100}%` }} aria-hidden="true" />
          <div className={`s-tip${hover / n > 0.8 ? ' end' : hover / n < 0.12 ? ' start' : ''}`} style={{ left: `${((hover + 0.5) / n) * 100}%` }} role="status">
            <b>{weekday(tip.d)} {monthDay(tip.d)}</b>
            {person && (
              <span className="mine">
                {who}: {tip.theirs === 0 ? 'no talk' : `${tip.theirs} talk${tip.theirs > 1 ? 's' : ''}`} · {tip.lines} memory lines
              </span>
            )}
            <span>{tip.talks === 0 ? 'No talks in town' : `${tip.talks} talk${tip.talks > 1 ? 's' : ''} in town`}</span>
            {tip.events.map((x) => <span key={x} className="ev">{x}</span>)}
          </div>
        </>
      )}
    </div>
    <ul className="s-legend" style={{ '--c': cssHex(color) } as React.CSSProperties} aria-label="Key">
      <li><i className="key" />Talks per day, town</li>
      {person && <li><i className="key mine" />Memory held, {who}</li>}
      <li><i className="key event" />Town event</li>
    </ul>
    </>
  )
}
