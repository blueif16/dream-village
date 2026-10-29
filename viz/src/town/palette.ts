// The town's colours. Grounds and buildings are muted (paper, straw, sage, clay) so the residents' own colours are
// the brightest things on the map. Night multiplies everything toward a dusk blue and lights the windows.
import { hash } from './rand'

export const GROUND = {
  land: 0xe9e3d4,
  road: 0xcdc7bb,
  lane: 0xf6f1e6,
  sidewalk: 0xe0d9c9,
  kerb: 0xd3ccbb,
  rail: 0xb9b2a5,
  sleeper: 0xa39b8d,
  sand: 0xeee3c8,
  water: 0xbfd6d6,
  waterDeep: 0xa9c8cb,
  marsh: 0xc8d4a6,
  hill: 0xcad3ac,
  hillHigh: 0xdcdcbf,
  grass: 0xc6d8a8,
  path: 0xe6dcc4,
}

// Each district's block ground. A district not listed gets a soft tint of its own.
const DISTRICT: Record<string, number> = {
  learning: 0xe9e2c4, main_street: 0xf0dccb, industry: 0xdcd7cd, homes: 0xdde4cb, civic: 0xe6dde3, parks: 0xcfe0bc, station: 0xe2e0d6,
}
const SOFT = [0xe8dfd0, 0xdfe6d8, 0xe5dde9, 0xe9e1cc, 0xd9e2e4, 0xeedbd2]
export const districtGround = (d: string) => DISTRICT[d] ?? SOFT[hash(d) % SOFT.length]

export const DISTRICT_NAME: Record<string, string> = {
  learning: 'Schools', main_street: 'Main Street', industry: 'Industry', homes: 'Homes', civic: 'Civic', parks: 'Parks', station: 'Station',
}
export const districtName = (d: string) =>
  DISTRICT_NAME[d] ?? d.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())

// Walls, roofs and accents to pick from, by a place's seed.
export const WALL = [0xf3ead8, 0xf0d9c4, 0xdfe6d2, 0xd8e2e6, 0xf2e3c6, 0xe9d6d6, 0xe4dccb]
export const ROOF = [0xb8644a, 0x8a5d4e, 0x6f7f8c, 0x9a7b5a, 0xa65a52, 0x5f6e66]
export const ACCENT = [0xd9644a, 0xe0a23c, 0x3f8f87, 0x4f6fa8, 0x9b6bb3, 0xc4506a, 0x6f9a4a, 0x2f7f9f]
export const BRICK = 0xc98b6e
export const STONE = 0xe1cfa8
export const WHITE = 0xf4f2ec
export const GLASS_LIT = 0xffd89a // a window at night
export const GLASS_DARK = 0x56606e // a window that stays dark
export const FOLIAGE = [0x8fb07a, 0x7fa56f, 0x9cba84, 0x6f9a68]
export const TRUNK = 0x8a6d55
export const EVENT = 0xe0a526

export const INK = { day: 0x6e675e, night: 0x1c2232 }
export const DETAIL = { day: 0xa9a194, night: 0x2a3142 }
export const SOFT_LINE = { day: 0xc9c1b2, night: 0x3a4357 }
export const NIGHT = 0x4b5880 // what everything is multiplied by at full night
export const DAY_GLASS = 0x4a5566 // what lit-window colours are multiplied by in daylight

export const pick = <T,>(a: T[], seed: number, k = 0) => a[(seed >>> (k * 3)) % a.length]
