import { useEffect, useState } from 'react'
import type { Town } from './town/types'
import type { Checkpoint, Index, PrincipalBundle, TimelineEvent } from './types'

const BASE = `${import.meta.env.BASE_URL}data`
const cache = new Map<string, Promise<unknown>>()

function fetchJson<T>(path: string): Promise<T> {
  let p = cache.get(path)
  if (!p) {
    p = fetch(`${BASE}/${path}`).then((r) => {
      if (!r.ok) throw new Error(`${r.status} ${r.statusText} for data/${path}`)
      return r.json()
    })
    p.catch(() => cache.delete(path)) // a failed load can be retried after the export runs
    cache.set(path, p)
  }
  return p as Promise<T>
}

type Loaded<T> = { data: T | null; error: string | null }

function useJson<T>(path: string | null): Loaded<T> {
  const [state, setState] = useState<Loaded<T> & { path: string | null }>({ data: null, error: null, path: null })
  useEffect(() => {
    if (!path) return
    let live = true
    fetchJson<T>(path).then(
      (data) => live && setState({ data, error: null, path }),
      (e: Error) => live && setState({ data: null, error: e.message, path }),
    )
    return () => {
      live = false
    }
  }, [path])
  return state.path === path ? state : { data: null, error: null }
}

export const useIndex = () => useJson<Index>('index.json')
export const usePrincipal = (pid: string | null) => useJson<PrincipalBundle>(pid ? `p/${pid}.json` : null)
export const useTown = () => useJson<Town>('town.json')

// ------------------------------------------------------------------ time (sim times are naive; read them as UTC)

export const ms = (iso: string) => Date.parse(iso.length <= 10 ? `${iso}T00:00:00Z` : `${iso}Z`)
export const DAY = 86_400_000

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
const pad = (n: number) => String(n).padStart(2, '0')

export function fmtDay(t: number): string {
  const d = new Date(t)
  return `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}`
}

export function fmtTime(t: number): string {
  const d = new Date(t)
  return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`
}

export function fmtWhen(iso: string): string {
  const t = ms(iso)
  return `${WEEKDAYS[new Date(t).getUTCDay()]} ${fmtDay(t)} · ${fmtTime(t)}`
}

export function weekOf(iso: string, start: string): number {
  return Math.floor((ms(iso) - ms(start)) / (7 * DAY)) + 1
}

export const short = (sha: string) => sha.slice(0, 7)

export function fmtPct(x: number | null | undefined): string {
  return x === null || x === undefined ? '—' : `${Math.round(x * 100)}%`
}

export function fmtUsd(x: number | null | undefined): string {
  return x === null || x === undefined ? '—' : x < 0.01 ? `$${x.toFixed(4)}` : `$${x.toFixed(2)}`
}

// ------------------------------------------------------------------ derived structure

export function checkpointsOf(b: PrincipalBundle): Checkpoint[] {
  const by = new Map<string, Checkpoint>()
  for (const p of b.probes) {
    const c = by.get(p.checkpoint) ?? { id: p.checkpoint, t: p.t, probes: [] }
    c.probes.push(p)
    by.set(p.checkpoint, c)
  }
  return [...by.values()].sort((a, b) => ms(a.t) - ms(b.t))
}

const KIND_ORDER = { session: 0, commit: 1, checkpoint: 2 } as const

export function eventsOf(b: PrincipalBundle, cps: Checkpoint[]): TimelineEvent[] {
  const ev: TimelineEvent[] = [
    ...b.sessions.map((s, i) => ({ key: `s:${s.id}`, kind: 'session' as const, t: ms(s.start), index: i })),
    ...b.commits.map((c, i) => ({ key: `c:${c.sha}`, kind: 'commit' as const, t: ms(c.sim_time), index: i })),
    ...cps.map((c, i) => ({ key: `k:${c.id}`, kind: 'checkpoint' as const, t: ms(c.t), index: i })),
  ]
  return ev.sort((a, b) => a.t - b.t || KIND_ORDER[a.kind] - KIND_ORDER[b.kind] || a.index - b.index)
}

/** The last commit at or before t: the memory a probe at t reads. */
export function commitAt(b: PrincipalBundle, t: number): number {
  let found = -1
  b.commits.forEach((c, i) => {
    if (ms(c.sim_time) <= t) found = i
  })
  return found
}

export interface DiffLine {
  kind: 'add' | 'del' | 'ctx' | 'hunk'
  text: string
  old: number | null
  new: number | null
}

export interface FileDiff {
  path: string
  add: number
  del: number
  lines: DiffLine[]
}

/** A `git diff` (no renames) split per file, with old/new line numbers. */
export function parseDiff(diff: string): FileDiff[] {
  const files: FileDiff[] = []
  let cur: FileDiff | null = null
  let inHunk = false
  let o = 0
  let n = 0
  for (const line of diff.split('\n')) {
    if (line.startsWith('diff --git ')) {
      const m = / b\/(.+)$/.exec(line)
      cur = { path: m ? m[1] : line.slice(11), add: 0, del: 0, lines: [] }
      files.push(cur)
      inHunk = false
    } else if (!cur) {
      continue
    } else if (line.startsWith('@@')) {
      const m = /^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)$/.exec(line)
      o = m ? Number(m[1]) : 0
      n = m ? Number(m[2]) : 0
      cur.lines.push({ kind: 'hunk', text: line, old: null, new: null })
      inHunk = true
    } else if (!inHunk) {
      continue // file header: index, ---, +++, mode lines
    } else if (line.startsWith('+')) {
      cur.lines.push({ kind: 'add', text: line.slice(1), old: null, new: n++ })
      cur.add++
    } else if (line.startsWith('-')) {
      cur.lines.push({ kind: 'del', text: line.slice(1), old: o++, new: null })
      cur.del++
    } else if (line.startsWith(' ')) {
      cur.lines.push({ kind: 'ctx', text: line.slice(1), old: o++, new: n++ })
    }
    // "\ No newline at end of file" and the trailing empty line are skipped
  }
  return files
}
