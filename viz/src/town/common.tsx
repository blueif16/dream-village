// Small pieces the console's panels share.
import { fmtWhen } from '../data'
import { cssHex } from './colors'
import type { Status } from './engine'
import { DAY_MIN } from './schedule'
import type { Place, TownPerson } from './types'

export const pad = (n: number) => String(n).padStart(2, '0')
export const kindName = (k: string) => k.replace(/_/g, ' ')
export const capital = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)

/** Minutes of sim time as the hour of its day, 24-hour. */
export const clock = (t: number) => {
  const m = Math.floor(((t % DAY_MIN) + DAY_MIN) % DAY_MIN)
  return `${pad(Math.floor(m / 60))}:${pad(m % 60)}`
}
/** The short weekday of a day, e.g. Mon. */
export const weekday = (d: string) => fmtWhen(`${d}T12:00:00`).split(' ')[0]
export const monthDay = (d: string) => new Date(`${d}T12:00:00Z`).toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' })
/** The sim moment as an ISO time without a zone, the way the bundle writes them. */
export const isoAt = (days: string[], t: number) => `${days[Math.floor(t / DAY_MIN)]}T${clock(t)}:00`

export function Dot({ color, ring }: { color: number; ring?: boolean }) {
  return <i className={ring ? 'dot ring' : 'dot'} style={{ '--c': cssHex(color) } as React.CSSProperties} />
}

export function statusWord(st: Status, places: Map<string, Place>): string {
  const name = (id: string | null) => (id ? places.get(id)?.name ?? id : 'somewhere')
  switch (st.mode) {
    case 'asleep':
      return 'Asleep at home'
    case 'walking':
      return `Walking to ${name(st.to)}`
    case 'talking':
      return 'Talking to the assistant'
    case 'dreaming':
      return 'Asleep · memory dreaming'
    default:
      return `At ${name(st.at)}`
  }
}

/** The resident's memory as town.json last saw it at t. */
export function memNow(p: TownPerson, iso: string) {
  let m: TownPerson['memory'][number] | undefined
  for (const x of p.memory) if (x.t <= iso && (!m || x.t >= m.t)) m = x
  return m
}
