// town.json, as viz/export.py writes it.

export interface Place {
  id: string
  name: string
  kind: string
  description: string
  district: string
}

export interface TownEvent {
  id: string
  date: string
  end_date: string | null
  title: string
  description: string
  places: string[]
}

export interface DayLine {
  what: string
  with: string[]
}

export interface Day {
  place: string | null
  why: 'life_log' | 'home' | 'work' | null
  lines: DayLine[]
}

export interface TownPerson {
  id: string
  name: string
  age: number | null
  minor: boolean
  role: string
  anchor: string | null
  homes: string[]
  sessions: Record<string, string[]> // day -> session ids
  memory: { t: string; lines: number; files: number }[]
  days: Day[] // aligned with Town.days
}

export interface Town {
  town: string
  days: string[]
  places: Place[]
  events: TownEvent[]
  principals: TownPerson[]
}

export type Selection = { kind: 'person' | 'place'; id: string } | null
