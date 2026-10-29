// Procedural town layout in logical pixels (the canvas scales it by an integer zoom).
import type { Place } from './types'

export const MARGIN = 10
export const ROAD = 10
export const BLOCK_W = 120
export const BLOCK_H = 98
export const W = MARGIN * 2 + BLOCK_W * 3 + ROAD * 2
export const H = MARGIN * 2 + BLOCK_H * 2 + ROAD
const PAD = 6
const GAP = 5
const PEOPLE_ROW = 12 // the sidewalk in front of a row of buildings, where people stand

export interface Block {
  id: string
  label: string
  x: number
  y: number
  w: number
  h: number
  ground: number
}

export interface Lot {
  place: Place
  x: number
  y: number
  w: number
  h: number
  doorX: number // where people gather: the sidewalk under the building
  doorY: number
}

// Which districts share a block, in reading order: top row, then bottom row.
const BLOCKS: { id: string; label: string; districts: string[]; ground: number }[] = [
  { id: 'learning', label: 'Schools', districts: ['learning'], ground: 0xdcd8bd },
  { id: 'main', label: 'Calle Ortega', districts: ['main_street'], ground: 0xe2d4b8 },
  { id: 'industry', label: 'Ruel Road', districts: ['industry'], ground: 0xc9c6bd },
  { id: 'homes', label: 'Homes', districts: ['homes'], ground: 0xd5d6b6 },
  { id: 'civic', label: 'Civic', districts: ['civic'], ground: 0xdedad0 },
  { id: 'parks', label: 'Parks · Flats Junction', districts: ['parks', 'station'], ground: 0xc5d3a6 },
]

/** Building footprint (w, h) by kind. */
export function size(kind: string): [number, number] {
  const S: Record<string, [number, number]> = {
    high_school: [40, 24], middle_school: [28, 19], community_college: [34, 24], state_university: [38, 26],
    library: [26, 18], hospital: [34, 32], community_center: [28, 17], church: [18, 26], temple: [22, 20],
    theater: [22, 22], warehouse_employer: [70, 30], industrial_area: [40, 18], apartment_complex: [32, 26],
    house: [14, 13], park: [50, 30], transit_station: [44, 14], main_street: [26, 14], strip_mall: [34, 13],
    grocery_store: [22, 15],
  }
  return S[kind] ?? [17, 15]
}

export function blocks(): Block[] {
  return BLOCKS.map((b, i) => ({
    id: b.id,
    label: b.label,
    x: MARGIN + (i % 3) * (BLOCK_W + ROAD),
    y: MARGIN + Math.floor(i / 3) * (BLOCK_H + ROAD),
    w: BLOCK_W,
    h: BLOCK_H,
    ground: b.ground,
  }))
}

/** Lay every place out in its district's block: rows left to right, buildings standing on a shared sidewalk. */
export function layout(places: Place[]): { blocks: Block[]; lots: Map<string, Lot> } {
  const bs = blocks()
  const lots = new Map<string, Lot>()
  BLOCKS.forEach((def, i) => {
    const b = bs[i]
    const mine = places.filter((p) => def.districts.includes(p.district))
    // big things first so rows pack evenly; stable by name otherwise
    mine.sort((p, q) => size(q.kind)[0] * size(q.kind)[1] - size(p.kind)[0] * size(p.kind)[1] || p.name.localeCompare(q.name))
    let x = b.x + PAD
    let rowTop = b.y + PAD + 8 // leave room for the district label
    let row: Lot[] = []
    const flush = () => {
      const hMax = Math.max(...row.map((l) => l.h))
      for (const l of row) {
        l.y = rowTop + hMax - l.h // bottoms aligned on the row's sidewalk
        l.doorY = rowTop + hMax + 9
      }
      rowTop += hMax + PEOPLE_ROW
      row = []
      x = b.x + PAD
    }
    for (const p of mine) {
      const [w, h] = size(p.kind)
      if (row.length && x + w > b.x + b.w - PAD) flush()
      const lot: Lot = { place: p, x, y: 0, w, h, doorX: x + Math.floor(w / 2), doorY: 0 }
      row.push(lot)
      lots.set(p.id, lot)
      x += w + GAP
    }
    if (row.length) flush()
  })
  return { blocks: bs, lots }
}

/** Positions for n people in front of one lot: side by side on the sidewalk, wrapping to a second row. */
export function slots(lot: Lot, n: number): [number, number][] {
  const perRow = Math.max(1, Math.floor((lot.w + 6) / 7))
  const out: [number, number][] = []
  for (let i = 0; i < n; i++) {
    const r = Math.floor(i / perRow)
    const inRow = Math.min(perRow, n - r * perRow)
    const k = i % perRow
    out.push([lot.doorX + Math.round((k - (inRow - 1) / 2) * 7), lot.doorY - r * 3])
  }
  return out
}
