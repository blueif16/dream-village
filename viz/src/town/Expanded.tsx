// A resident's card, expanded: the moment picked on their timeline in the console below, opened in full beside their
// card (the talk word for word, the memory's diff and files, the questions and their verdicts). Nothing picked, it
// opens on the latest thing that has happened by the town's time. The timeline is the one way to pick; the card beside
// it already has their name and their season's counts.
import { useMemo, useState } from 'react'
import { checkpointsOf, eventsOf, ms } from '../data'
import { Inspector, type Ctx } from '../Inspector'
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
  const at = ms(day0) + now * 60_000
  // a new moment opens at its top: the panel is keyed by it
  const sel = picked ?? events.filter((e) => e.t <= at).at(-1)?.key ?? events[0]?.key ?? null
  const ctx: Ctx = { bundle, start: day0, truth, go: onPick, commitTab, setCommitTab, memFile, setMemFile }
  return (
    <section className="xp" role="region" aria-label={`${bundle.principal.name}, the picked moment`} style={{ '--c': cssHex(color) } as React.CSSProperties}>
      <aside key={sel ?? ''} className="inspector" aria-label="The picked event">
        <Inspector selected={sel} checkpoints={cps} ctx={ctx} />
      </aside>
      <button className="x" onClick={onClose} aria-label="Collapse" title="Collapse (Esc)"><Close /></button>
    </section>
  )
}
