// The town's clock, for the pages around it (the landing page's time pill): the town publishes, a page reads.
import { useSyncExternalStore } from 'react'

export interface TownTime { t: number; night: boolean }

let now: TownTime | null = null
const subs = new Set<() => void>()

export function publishTime(t: number, night: boolean) {
  if (now && now.t === t && now.night === night) return
  now = { t, night }
  subs.forEach((f) => f())
}

const subscribe = (f: () => void) => {
  subs.add(f)
  return () => void subs.delete(f)
}

/** The town's time in minutes since its first midnight, or null before the town has started. */
export const useTownTime = () => useSyncExternalStore(subscribe, () => now)
