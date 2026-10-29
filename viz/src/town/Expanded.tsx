// A resident's card, expanded: everything the data view has on them, in a sheet over the map beside their card.
// The log of their talks, writes, dreams and probe checkpoints on the left; whichever is picked, opened on the
// right (the talk word for word, the memory's diff and files, the questions and their verdicts). Picking here, or
// on their timeline in the console below, moves the town to that moment.
import { useMemo, useState } from 'react'
import { checkpointsOf, eventsOf, ms } from '../data'
import { Inspector, type Ctx } from '../Inspector'
import { Log } from '../Person'
import type { PrincipalBundle } from '../types'
import { cssHex } from './colors'
import { Close } from './icons'

export function Expanded({ bundle, day0, now, color, picked, onPick, onClose }: {
  bundle: PrincipalBundle
  day0: string // the season's first day
  now: number // the town's time, sim minutes
  color: number
  picked: string | null
  onPick: (key: string) => void
  onClose: () => void
}) {
  const cps = useMemo(() => checkpointsOf(bundle), [bundle])
  const events = useMemo(() => eventsOf(bundle, cps), [bundle, cps])
  const truth = useMemo(() => new Map(bundle.truth.map((e) => [e.id, e])), [bundle])
  const [commitTab, setCommitTab] = useState<'diff' | 'snapshot'>('diff')
  const [memFile, setMemFile] = useState<string | null>(null)
  // nothing picked yet: open on the latest thing that has happened by now
  const at = ms(day0) + now * 60_000
  const sel = picked ?? events.filter((e) => e.t <= at).at(-1)?.key ?? events[0]?.key ?? null
  const ctx: Ctx = { bundle, start: day0, truth, go: onPick, commitTab, setCommitTab, memFile, setMemFile }
  const p = bundle.principal
  const failed = bundle.sessions.filter((s) => s.verify && !s.verify.passed).length
  return (
    <section className="xp" role="region" aria-label={`${p.name}, everything`} style={{ '--c': cssHex(color) } as React.CSSProperties}>
      <header className="xp-head">
        <h2>{p.name}</h2>
        <span className="counts">
          <span><b>{bundle.sessions.length}</b> talks{failed ? ` (${failed} failed verify)` : ''}</span>
          <span><b>{bundle.commits.filter((c) => c.event === 'write').length}</b> writes</span>
          <span><b>{bundle.commits.filter((c) => c.event === 'dream').length}</b> dreams</span>
          <span><b>{bundle.commits.at(-1)?.lines ?? 0}</b> lines at the end</span>
          <span><b>{bundle.probes.length}</b> questions</span>
        </span>
        <button className="x" onClick={onClose} aria-label="Collapse" title="Collapse (Esc)"><Close /></button>
      </header>
      <div className="xp-body">
        <Log events={events} b={bundle} cps={cps} selected={sel} onSelect={onPick} />
        <aside className="inspector" aria-label="The picked event">
          <Inspector selected={sel} checkpoints={cps} ctx={ctx} />
        </aside>
      </div>
    </section>
  )
}
