// A resident's day, read from town.json. Sim time t is minutes since 00:00 of the first day.
// The data gives one place per day and the order of the day's life-log lines; the hours around them are inferred:
// home at night, out between a morning leave and an evening return.
import { rng } from './rand'
import type { TownPerson } from './types'

export const DAY_MIN = 1440
const WAKE = 7 * 60 // the first life-log line starts here
const TALK_MIN = 30 // how long a talk glows; the data has its start, not its length

/** When the resident leaves home and comes back, in minutes of the day. */
export function dayPlan(p: TownPerson, day: number): { leave: number; back: number } {
  const r = rng(`${p.id}:${day}`)
  return { leave: 7 * 60 + Math.floor(r() * 151), back: 16 * 60 + 30 + Math.floor(r() * 301) }
}

/** The place id the resident is at (or heading for) at t. */
export function targetAt(p: TownPerson, t: number): string | null {
  const day = Math.floor(t / DAY_MIN)
  const d = p.days[day]
  const place = d?.place ?? null
  const home = place && p.homes.includes(place) ? place : p.homes[0] ?? null
  if (!place || place === home) return home ?? place
  const { leave, back } = dayPlan(p, day)
  const m = t - day * DAY_MIN
  return m >= leave && m < back ? place : home ?? place
}

function minutesOf(iso: string, days: string[]): number {
  const day = Math.round((Date.parse(`${iso.slice(0, 10)}T00:00:00Z`) - Date.parse(`${days[0]}T00:00:00Z`)) / 86_400_000)
  return day * DAY_MIN + Number(iso.slice(11, 13)) * 60 + Number(iso.slice(14, 16))
}

// The v0 replay commits a write right as a session starts and dreams at 03:00 exactly.
const isDream = (iso: string) => iso.slice(11, 19) === '03:00:00'

/** The moments the resident talked to the assistant (their write commits), sorted. */
export function talkTimes(p: TownPerson, days: string[]): number[] {
  return p.memory.filter((m) => !isDream(m.t)).map((m) => minutesOf(m.t, days)).sort((a, b) => a - b)
}

/** The moments the resident's memory dreamed (consolidated overnight), sorted. */
export function dreamTimes(p: TownPerson, days: string[]): number[] {
  return p.memory.filter((m) => isDream(m.t)).map((m) => minutesOf(m.t, days)).sort((a, b) => a - b)
}

/** The last time in sorted `times` at or before t, or -1. */
export function lastAtOrBefore(times: number[], t: number): number {
  let lo = 0
  let hi = times.length - 1
  let found = -1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (times[mid] <= t) {
      found = mid
      lo = mid + 1
    } else hi = mid - 1
  }
  return found
}

export function talkingAt(talks: number[], t: number): boolean {
  const i = lastAtOrBefore(talks, t)
  return i >= 0 && t < talks[i] + TALK_MIN
}

/** Which of the day's life-log lines the hour falls in: the lines share the waking day in order. */
export function lineIndexAt(p: TownPerson, t: number): number | null {
  const day = Math.floor(t / DAY_MIN)
  const n = p.days[day]?.lines.length ?? 0
  const m = t - day * DAY_MIN
  if (!n || m < WAKE) return null
  return Math.min(n - 1, Math.floor((m - WAKE) / ((DAY_MIN - WAKE) / n)))
}
