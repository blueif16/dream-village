// What the control panel reads out of town.json: the town's memory activity, talks per day, and which timeline
// labels fit; and, from a resident's own file, the marks on their season strip. Pure, so the panels stay thin. (Memory as of a moment is data.ts's commitAt and parseDiff.)
import { checkpointsOf, eventsOf, ms } from '../data'
import type { PrincipalBundle } from '../types'
import { DAY_MIN } from './schedule'
import type { TownPerson } from './types'

export interface MemoryEvent {
  pid: string
  at: number // sim minutes
  kind: 'write' | 'dream'
  lines: number
  files: number
  delta: number // lines gained (or lost) since the resident's previous commit
}

const minutesOf = (iso: string, d0: string) =>
  Math.round((Date.parse(`${iso.slice(0, 10)}T00:00:00Z`) - Date.parse(`${d0}T00:00:00Z`)) / 86_400_000) * DAY_MIN +
  Number(iso.slice(11, 13)) * 60 + Number(iso.slice(14, 16))

/** Every memory commit in town up to t, newest first, at most `limit`. The v0 replay dreams at 03:00 exactly. */
export function memoryFeed(people: TownPerson[], days: string[], t: number, limit: number): MemoryEvent[] {
  const all: MemoryEvent[] = []
  for (const p of people) {
    const ms = [...p.memory].sort((a, b) => a.t.localeCompare(b.t))
    ms.forEach((m, i) => {
      const at = minutesOf(m.t, days[0])
      if (at > t) return
      const kind = m.t.slice(11, 19) === '03:00:00' ? 'dream' : 'write'
      all.push({ pid: p.id, at, kind, lines: m.lines, files: m.files, delta: m.lines - (i ? ms[i - 1].lines : 0) })
    })
  }
  return all.sort((a, b) => b.at - a.at || a.pid.localeCompare(b.pid)).slice(0, limit)
}

/** How many sessions the resident had on each day of `days`. */
export function talksByDay(p: TownPerson, days: string[]): number[] {
  return days.map((d) => p.sessions[d]?.length ?? 0)
}

/**
 * Lay out labels [x, x + w] in priority `order`: each takes the first row it fits on (`gap` apart), or -1 if none.
 * A label below row 0 hangs from a stem (at `stem`, else its x) that runs up through the rows above it, so the stem
 * must not cross a label there, and no later label may cover it.
 */
export function fitLabels(items: { x: number; w: number; stem?: number }[], order: number[], rows: number, gap = 0): number[] {
  const out = items.map(() => -1)
  const taken: { x: number; w: number }[][] = Array.from({ length: rows }, () => [])
  const clear = (r: number, a: { x: number; w: number }) => taken[r].every((b) => a.x + a.w + gap <= b.x || b.x + b.w + gap <= a.x)
  for (const i of order) {
    const a = items[i]
    const stem = { x: a.stem ?? a.x, w: 0 }
    const row = taken.findIndex((_, r) => clear(r, a) && taken.slice(0, r).every((_, q) => clear(q, stem)))
    if (row < 0) continue
    taken[row].push(a)
    for (let q = 0; q < row; q++) taken[q].push(stem)
    out[i] = row
  }
  return out
}

/** A mark on a resident's season strip: one talk, write, dream or probe checkpoint, at its minute of the season. */
export interface SeasonMark {
  key: string // the data view's event key: s:<session> · c:<sha> · k:<checkpoint>
  lane: 'talk' | 'write' | 'dream' | 'probe'
  at: number // sim minutes since the first day's midnight
}

export function seasonMarks(b: PrincipalBundle, day0: string): SeasonMark[] {
  const t0 = ms(day0)
  return eventsOf(b, checkpointsOf(b)).map((e) => ({
    key: e.key,
    lane: e.kind === 'session' ? 'talk' : e.kind === 'checkpoint' ? 'probe' : b.commits[e.index].event === 'dream' ? 'dream' : 'write',
    at: (e.t - t0) / 60_000,
  }))
}
