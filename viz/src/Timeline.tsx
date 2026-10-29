import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { DAY, fmtDay, fmtWhen, ms, short } from './data'
import type { Checkpoint, PrincipalBundle } from './types'

// Lane centres (px from the top of the SVG). Writes and dreams are two tracks of the one memory lane.
const AXIS_H = 30
const Y = { probe: 50, session: 86, write: 122, dream: 144 }
const HEIGHT = 160
const PAD = 14 // px before the first day and after the last
const MIN_PPD = 4 // px per day at "fit" when the panel is narrow
const GAP = 11 // markers closer than this (px) in one lane are dodged onto a side row
const ROW_DY = [0, 9, -9]

/** Greedy row per marker (xs ascending) so close markers in one lane sit side by side instead of on top. */
function dodge(xs: number[]): number[] {
  const last: number[] = []
  return xs.map((x) => {
    let r = last.findIndex((lx) => x - lx >= GAP)
    if (r < 0) r = last.length < ROW_DY.length ? last.length : last.indexOf(Math.min(...last))
    last[r] = x
    return r
  })
}

export const LANES: [string, number][] = [
  ['probes', Y.probe],
  ['sessions', Y.session],
  ['writes', Y.write],
  ['dreams', Y.dream],
]

interface Props {
  bundle: PrincipalBundle
  checkpoints: Checkpoint[]
  range: { start: string; end: string }
  selected: string | null
  onSelect: (key: string) => void
}

function SessionShape({ passed }: { passed: boolean | null }) {
  if (passed === null) return <circle className="shape" r={4.5} fill="var(--sheet)" stroke="var(--muted)" strokeWidth={1.2} />
  if (passed) return <circle className="shape" r={4.5} fill="var(--pass)" />
  return (
    <g>
      <circle className="shape" r={4.5} fill="var(--sheet)" stroke="var(--fail)" strokeWidth={1.8} />
      <path d="M-2,-2L2,2M2,-2L-2,2" stroke="var(--fail)" strokeWidth={1.4} />
    </g>
  )
}

export function Legend() {
  return (
    <span className="legend">
      <span><svg width="10" height="10" viewBox="-5 -5 10 10"><circle r={4} fill="var(--pass)" /></svg>verified</span>
      <span>
        <svg width="10" height="10" viewBox="-5 -5 10 10"><circle r={3.6} fill="var(--sheet)" stroke="var(--fail)" strokeWidth={1.6} /></svg>
        verify failed
      </span>
      <span><svg width="10" height="10" viewBox="-5 -5 10 10"><rect x={-3.5} y={-3.5} width={7} height={7} fill="var(--write)" /></svg>write</span>
      <span><svg width="10" height="10" viewBox="-5 -5 10 10"><path d="M0,-4.5L4.5,0L0,4.5L-4.5,0Z" fill="var(--dream)" /></svg>dream</span>
      <span><svg width="10" height="10" viewBox="-5 -5 10 10"><path d="M-4.5,-3.5L4.5,-3.5L0,4Z" fill="var(--probe)" /></svg>probe checkpoint</span>
    </span>
  )
}

export function Timeline({ bundle, checkpoints, range, selected, onSelect }: Props) {
  const scroller = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(900)
  const [zoom, setZoom] = useState(1)

  useLayoutEffect(() => {
    const el = scroller.current
    if (!el) return
    const ro = new ResizeObserver(() => setWidth(el.clientWidth))
    ro.observe(el)
    setWidth(el.clientWidth)
    return () => ro.disconnect()
  }, [])

  const t0 = ms(range.start)
  const t1 = ms(range.end) + DAY
  const days = (t1 - t0) / DAY
  const ppd = Math.max(MIN_PPD, (width - 2 * PAD) / days) * zoom
  const W = Math.ceil(days * ppd + 2 * PAD)
  const x = (t: number) => PAD + ((t - t0) / DAY) * ppd

  const sessionX = useMemo(() => new Map(bundle.sessions.map((s) => [s.id, ms(s.start)])), [bundle])
  const sessionY = new Map<string, number>()
  dodge(bundle.sessions.map((s) => x(ms(s.start)))).forEach((r, i) => sessionY.set(bundle.sessions[i].id, Y.session + ROW_DY[r]))
  const cpY = new Map<string, number>()
  dodge(checkpoints.map((c) => x(ms(c.t)))).forEach((r, i) => cpY.set(checkpoints[i].id, Y.probe + ROW_DY[r]))

  // Where the selection sits, for the "now" line and to keep it in view.
  let selX: number | null = null
  let selY = 0
  if (selected) {
    const [kind, id] = [selected[0], selected.slice(2)]
    if (kind === 's') {
      const t = sessionX.get(id)
      if (t !== undefined) [selX, selY] = [x(t), sessionY.get(id)!]
    } else if (kind === 'c') {
      const c = bundle.commits.find((c) => c.sha === id)
      if (c) [selX, selY] = [x(ms(c.sim_time)), c.event === 'dream' ? Y.dream : Y.write]
    } else if (kind === 'k') {
      const c = checkpoints.find((c) => c.id === id)
      if (c) [selX, selY] = [x(ms(c.t)), cpY.get(id)!]
    }
  }

  useEffect(() => {
    const el = scroller.current
    if (!el || selX === null) return
    if (selX < el.scrollLeft + 40 || selX > el.scrollLeft + el.clientWidth - 40) {
      el.scrollTo({ left: selX - el.clientWidth / 2, behavior: 'smooth' })
    }
  }, [selX])

  const weekTicks: number[] = []
  for (let t = t0; t <= t1; t += 7 * DAY) weekTicks.push(t)
  const dayTicks: number[] = []
  for (let t = t0; t <= t1; t += DAY) dayTicks.push(t)

  const chain = bundle.commits
    .map((c) => `${x(ms(c.sim_time)).toFixed(1)},${c.event === 'dream' ? Y.dream : Y.write}`)
    .join(' ')

  return (
    <div className="timeline">
      <div className="timeline-bar">
        <Legend />
        <span className="muted">
          <kbd>←</kbd> <kbd>→</kbd> step · <kbd>shift</kbd> same lane
        </span>
        <span className="zoom" role="group" aria-label="zoom">
          {[1, 2, 4, 8].map((z) => (
            <button key={z} aria-pressed={zoom === z} onClick={() => setZoom(z)}>
              {z === 1 ? 'fit' : `${z}×`}
            </button>
          ))}
        </span>
      </div>
      <div className="timeline-body">
        <div className="lane-labels" style={{ height: HEIGHT }}>
          {LANES.map(([name, y]) => (
            <span key={name} style={{ top: y }}>{name}</span>
          ))}
        </div>
        <div className="timeline-scroll" ref={scroller}>
          <svg width={W} height={HEIGHT} role="img" aria-label="timeline of sessions, memory commits and probes">
            {weekTicks.map((t, i) =>
              i % 2 === 1 ? <rect key={t} className="tl-week-band" x={x(t)} y={0} width={7 * ppd} height={HEIGHT} /> : null,
            )}
            {ppd >= 6 && dayTicks.map((t) => <line key={t} className="tl-tick" x1={x(t)} x2={x(t)} y1={AXIS_H - 6} y2={AXIS_H} />)}
            {weekTicks.map((t, i) => (
              <g key={t}>
                <line className="tl-week-tick" x1={x(t)} x2={x(t)} y1={4} y2={HEIGHT} />
                {t1 - t >= 3 * DAY && (
                  <text className="tl-axis-label" x={x(t) + 4} y={16}>
                    <tspan className="tl-axis-label week">W{i + 1}</tspan> {fmtDay(t)}
                  </text>
                )}
              </g>
            ))}
            {LANES.map(([name, y]) => <line key={name} className="tl-lane" x1={PAD} x2={W - PAD} y1={y} y2={y} />)}

            {checkpoints.map((c) => (
              <line key={c.id} className="tl-cp-line" x1={x(ms(c.t))} x2={x(ms(c.t))} y1={Y.probe} y2={Y.dream + 8} />
            ))}
            <polyline className="tl-chain" points={chain} />
            {bundle.commits.map((c) =>
              c.event === 'write' && c.session_id && sessionX.has(c.session_id) ? (
                <line key={`l${c.sha}`} className="tl-link" x1={x(sessionX.get(c.session_id)!)} y1={sessionY.get(c.session_id)!}
                  x2={x(ms(c.sim_time))} y2={Y.write} />
              ) : null,
            )}

            {selX !== null && <line className="tl-now" x1={selX} x2={selX} y1={AXIS_H} y2={HEIGHT} />}

            {checkpoints.map((c) => (
              <g key={c.id} className="tl-mark" transform={`translate(${x(ms(c.t))},${cpY.get(c.id)})`} onClick={() => onSelect(`k:${c.id}`)}>
                <title>{`probe checkpoint ${c.id}\n${fmtWhen(c.t)} · ${c.probes.length} probes`}</title>
                <circle className="hit" r={6} />
                <path className="shape" d="M-5.5,-4.5L5.5,-4.5L0,5Z" fill="var(--probe)" />
              </g>
            ))}
            {bundle.sessions.map((s) => (
              <g key={s.id} className="tl-mark" transform={`translate(${x(ms(s.start))},${sessionY.get(s.id)})`} onClick={() => onSelect(`s:${s.id}`)}>
                <title>{`${s.id} · ${fmtWhen(s.start)}\n${s.task_type ?? ''} · verify ${s.verify ? (s.verify.passed ? 'passed' : 'FAILED') : 'n/a'}`}</title>
                <circle className="hit" r={6} />
                <SessionShape passed={s.verify ? s.verify.passed : null} />
              </g>
            ))}
            {bundle.commits.map((c) => (
              <g key={c.sha} className="tl-mark" transform={`translate(${x(ms(c.sim_time))},${c.event === 'dream' ? Y.dream : Y.write})`}
                onClick={() => onSelect(`c:${c.sha}`)}>
                <title>{`${c.event} ${short(c.sha)} · ${fmtWhen(c.sim_time)}\n${c.files.length} files changed · ${c.lines} memory lines`}</title>
                <circle className="hit" r={6} />
                {c.event === 'dream' ? (
                  <path className="shape" d="M0,-5L5,0L0,5L-5,0Z" fill="var(--dream)" />
                ) : (
                  <rect className="shape" x={-3.5} y={-3.5} width={7} height={7} fill="var(--write)" />
                )}
              </g>
            ))}

            {selX !== null && <circle className="tl-halo" cx={selX} cy={selY} r={8.5} />}
          </svg>
        </div>
      </div>
    </div>
  )
}
