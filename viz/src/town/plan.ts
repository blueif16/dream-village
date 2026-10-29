// The town's ground plan, generated from town.json's places alone, in metres. Scene coordinates follow the iso kernel:
// +x runs down-right on screen, +y down-left. Every district gets blocks of two rows of buildings; the north row faces
// the north sidewalk, the south row the south one, so every door is on a street. Blocks sit on a grid of streets,
// the rail runs along the north edge with the stations on it, and the shore lies to the south.
import type { Place } from './types'

export const SIDEWALK = 3
export const ROAD = 14
export const BLOCK_W = 132
const GAP = 3 // between neighbouring buildings in a row
const INSET = SIDEWALK + 2 // from a block's edge to its buildings
const MIDDLE = 4 // between the backs of the two rows
const ROW_MAX = BLOCK_W - 2 * INSET

export type Arche =
  | 'house' | 'flats' | 'shop' | 'arcade' | 'mall' | 'school' | 'campus' | 'hospital' | 'library' | 'hall'
  | 'church' | 'temple' | 'theater' | 'gym' | 'warehouse' | 'works' | 'park' | 'station' | 'generic'

// Footprint (w along x, d along y) by kind. A kind not listed is a generic building with a size of its own.
const ARCHE: Record<string, [Arche, number, number]> = {
  house: ['house', 13, 12], apartment_complex: ['flats', 30, 22],
  cafe: ['shop', 13, 11], bakery: ['shop', 12, 11], taqueria: ['shop', 13, 11], pho_restaurant: ['shop', 14, 11],
  laundromat: ['shop', 13, 11], grocery_store: ['shop', 22, 15], main_street: ['arcade', 34, 13], strip_mall: ['mall', 40, 16],
  gym: ['gym', 24, 18], high_school: ['school', 46, 28], middle_school: ['school', 36, 24],
  community_college: ['campus', 46, 30], state_university: ['campus', 52, 34], library: ['library', 26, 18],
  hospital: ['hospital', 40, 34], church: ['church', 16, 26], temple: ['temple', 22, 22], community_center: ['hall', 28, 18],
  theater: ['theater', 24, 22], warehouse_employer: ['warehouse', 72, 34], industrial_area: ['works', 46, 24],
  park: ['park', 50, 32], transit_station: ['station', 48, 12],
}

function kindHash(s: string): number {
  let h = 7
  for (const c of s) h = (h * 31 + c.charCodeAt(0)) >>> 0
  return h
}

export function archetypeOf(kind: string): { arche: Arche; w: number; d: number } {
  const a = ARCHE[kind]
  if (a) return { arche: a[0], w: Math.min(a[1], ROW_MAX), d: a[2] }
  const h = kindHash(kind)
  return { arche: 'generic', w: 14 + (h % 13), d: 12 + ((h >>> 4) % 9) }
}

export interface Lot {
  place: Place
  arche: Arche
  block: string // '' for a station on the rail
  x: number
  y: number
  w: number
  d: number
  side: 'n' | 's' // which sidewalk the front faces
  door: [number, number]
}

export interface Block {
  id: string
  district: string
  first: boolean // the district's first block carries its name
  col: number
  row: number
  x: number
  y: number
  w: number
  d: number
}

/** Room left at the end of a row, for trees and small things. */
export interface Spare {
  block: string
  side: 'n' | 's'
  x: number
  y: number
  w: number
  d: number
}

export interface Plan {
  blocks: Block[]
  lots: Map<string, Lot>
  spares: Spare[]
  sidewalk: number
  road: number
  cols: number
  rows: number
  /** the grid of blocks and the streets round it: x0..x1, y0..y1 */
  grid: { x0: number; y0: number; x1: number; y1: number }
  rail: { y: number }
  promenade: { y: number } // the far sidewalk of the north road, which the stations open onto
  walk: (from: string, to: string) => [number, number][]
}

// Districts read in this order; any other district follows, alphabetically.
const ORDER = ['learning', 'main_street', 'industry', 'homes', 'civic', 'parks', 'station']

interface Draft {
  district: string
  rows: [Lot[], Lot[]]
  used: [number, number]
}

export function planTown(places: Place[]): Plan {
  const sorted = [...places].sort((a, b) => a.id.localeCompare(b.id))
  const stations = sorted.filter((p) => p.kind === 'transit_station')
  const rest = sorted.filter((p) => p.kind !== 'transit_station')

  // ---- pack each district's buildings into blocks: widest first, first row that fits
  const rank = (d: string) => (ORDER.includes(d) ? ORDER.indexOf(d) : ORDER.length)
  const districts = [...new Set(rest.map((p) => p.district))].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b))
  const drafts: Draft[] = []
  for (const district of districts) {
    const mine = rest
      .filter((p) => p.district === district)
      .map((p) => ({ p, ...archetypeOf(p.kind) }))
      .sort((a, b) => b.w - a.w || a.p.id.localeCompare(b.p.id))
    const open: Draft[] = []
    for (const { p, arche, w, d } of mine) {
      let spot: [Draft, 0 | 1] | null = null
      for (const dr of open) {
        for (const r of [0, 1] as const) {
          const need = dr.used[r] === 0 ? w : dr.used[r] + GAP + w
          if (need <= ROW_MAX) {
            spot = [dr, r]
            break
          }
        }
        if (spot) break
      }
      if (!spot) {
        const dr: Draft = { district, rows: [[], []], used: [0, 0] }
        open.push(dr)
        drafts.push(dr)
        spot = [dr, 0]
      }
      const [dr, r] = spot
      dr.used[r] = dr.used[r] === 0 ? w : dr.used[r] + GAP + w
      dr.rows[r].push({ place: p, arche, block: '', x: 0, y: 0, w, d, side: r === 0 ? 'n' : 's', door: [0, 0] })
    }
  }

  // ---- lay the blocks on a grid; a grid row is as deep as its deepest block, so the streets run straight
  const n = drafts.length
  const cols = Math.max(1, Math.min(n, Math.ceil(Math.sqrt(n * 1.4))))
  const nrows = Math.max(1, Math.ceil(n / cols))
  const depthOf = (dr: Draft) =>
    2 * INSET + MIDDLE + Math.max(0, ...dr.rows[0].map((l) => l.d)) + Math.max(0, ...dr.rows[1].map((l) => l.d))
  const rowD = Array.from({ length: nrows }, (_, r) => Math.max(...drafts.slice(r * cols, (r + 1) * cols).map(depthOf)))
  const rowY: number[] = []
  rowD.reduce((y, d, r) => ((rowY[r] = y), y + d + ROAD), 0)

  const blocks: Block[] = []
  const lots = new Map<string, Lot>()
  const spares: Spare[] = []
  const seen = new Set<string>()
  drafts.forEach((dr, i) => {
    const col = i % cols
    const row = Math.floor(i / cols)
    const b: Block = {
      id: `${dr.district}#${i}`, district: dr.district, first: !seen.has(dr.district), col, row,
      x: col * (BLOCK_W + ROAD), y: rowY[row], w: BLOCK_W, d: rowD[row],
    }
    seen.add(dr.district)
    blocks.push(b)
    for (const r of [0, 1] as const) {
      let x = b.x + INSET
      const rowDepth = Math.max(0, ...dr.rows[r].map((l) => l.d))
      for (const lot of dr.rows[r]) {
        lot.block = b.id
        lot.x = x
        lot.y = r === 0 ? b.y + INSET : b.y + b.d - INSET - lot.d
        lot.door = [x + lot.w / 2, r === 0 ? b.y + SIDEWALK / 2 : b.y + b.d - SIDEWALK / 2]
        lots.set(lot.place.id, lot)
        x += lot.w + GAP
      }
      const end = b.x + b.w - INSET
      const depth = rowDepth || (b.d - 2 * INSET - MIDDLE) / 2
      if (end - x > 6) spares.push({ block: b.id, side: r === 0 ? 'n' : 's', x, y: r === 0 ? b.y + INSET : b.y + b.d - INSET - depth, w: end - x, d: depth })
    }
  })

  const gridW = cols * BLOCK_W + (cols - 1) * ROAD
  const gridD = rowY[nrows - 1] + rowD[nrows - 1]
  const promY = -ROAD - SIDEWALK / 2
  const stationD = Math.max(12, ...stations.map((p) => archetypeOf(p.kind).d))
  const railY = promY - SIDEWALK / 2 - 2 - stationD / 2

  // ---- stations: side by side along the rail, centred on the town
  const sts = stations.map((p) => ({ p, ...archetypeOf(p.kind) }))
  let sx = gridW / 2 - (sts.reduce((a, s) => a + s.w, 0) + GAP * 4 * Math.max(0, sts.length - 1)) / 2
  for (const s of sts) {
    lots.set(s.p.id, { place: s.p, arche: s.arche, block: '', x: sx, y: railY - s.d / 2, w: s.w, d: s.d, side: 's', door: [sx + s.w / 2, promY] })
    sx += s.w + GAP * 4
  }

  // ---- the walk graph: each block's sidewalk ring, crossings at every corner, the promenade along the north road
  const s = SIDEWALK / 2
  const key = (p: [number, number]) => `${p[0].toFixed(3)},${p[1].toFixed(3)}`
  const pts = new Map<string, [number, number]>()
  const adj = new Map<string, Set<string>>()
  const link = (a: [number, number], b: [number, number]) => {
    const ka = key(a)
    const kb = key(b)
    if (ka === kb) return
    pts.set(ka, a)
    pts.set(kb, b)
    if (!adj.has(ka)) adj.set(ka, new Set())
    if (!adj.has(kb)) adj.set(kb, new Set())
    adj.get(ka)!.add(kb)
    adj.get(kb)!.add(ka)
  }
  const chain = (line: [number, number][]) => {
    const sortedLine = [...line].sort((a, b) => a[0] - b[0] || a[1] - b[1])
    for (let k = 1; k < sortedLine.length; k++) link(sortedLine[k - 1], sortedLine[k])
  }
  const corner = (b: Block, c: 'nw' | 'ne' | 'se' | 'sw'): [number, number] => [
    c === 'nw' || c === 'sw' ? b.x + s : b.x + b.w - s,
    c === 'nw' || c === 'ne' ? b.y + s : b.y + b.d - s,
  ]
  const at = new Map(blocks.map((b) => [`${b.col},${b.row}`, b]))
  const doorsOn = (b: Block, side: 'n' | 's') =>
    [...lots.values()].filter((l) => l.block === b.id && l.side === side).map((l) => l.door)
  for (const b of blocks) {
    chain([corner(b, 'nw'), ...doorsOn(b, 'n'), corner(b, 'ne')])
    chain([corner(b, 'sw'), ...doorsOn(b, 's'), corner(b, 'se')])
    link(corner(b, 'nw'), corner(b, 'sw'))
    link(corner(b, 'ne'), corner(b, 'se'))
    const right = at.get(`${b.col + 1},${b.row}`)
    if (right) {
      link(corner(b, 'ne'), corner(right, 'nw'))
      link(corner(b, 'se'), corner(right, 'sw'))
    }
    const below = at.get(`${b.col},${b.row + 1}`)
    if (below) {
      link(corner(b, 'sw'), corner(below, 'nw'))
      link(corner(b, 'se'), corner(below, 'ne'))
    }
  }
  const top = blocks.filter((b) => b.row === 0)
  const prom: [number, number][] = []
  for (const b of top) {
    for (const c of ['nw', 'ne'] as const) {
      const p = corner(b, c)
      const q: [number, number] = [p[0], promY]
      link(q, p)
      prom.push(q)
    }
  }
  for (const l of lots.values()) if (!l.block) prom.push(l.door)
  chain(prom)

  const doorOf = new Map([...lots.values()].map((l) => [l.place.id, l.door]))
  function walk(from: string, to: string): [number, number][] {
    const a = doorOf.get(from)
    const b = doorOf.get(to)
    if (!a || !b) return a ? [a] : b ? [b] : []
    const ka = key(a)
    const kb = key(b)
    if (ka === kb) return [a]
    // Dijkstra over a graph of a few hundred points
    const dist = new Map<string, number>([[ka, 0]])
    const prev = new Map<string, string>()
    const open = new Set<string>([ka])
    const done = new Set<string>()
    while (open.size) {
      let u = ''
      let best = Infinity
      for (const k of open) {
        const d = dist.get(k)!
        if (d < best) {
          best = d
          u = k
        }
      }
      open.delete(u)
      done.add(u)
      if (u === kb) break
      const pu = pts.get(u)!
      for (const v of adj.get(u) ?? []) {
        if (done.has(v)) continue
        const pv = pts.get(v)!
        const nd = best + Math.abs(pu[0] - pv[0]) + Math.abs(pu[1] - pv[1])
        if (nd < (dist.get(v) ?? Infinity)) {
          dist.set(v, nd)
          prev.set(v, u)
          open.add(v)
        }
      }
    }
    if (!prev.has(kb)) return [a, b]
    const path: [number, number][] = []
    for (let k: string | undefined = kb; k; k = prev.get(k)) path.unshift(pts.get(k)!)
    return path
  }

  return {
    blocks, lots, spares, sidewalk: SIDEWALK, road: ROAD, cols, rows: nrows,
    grid: { x0: -ROAD, y0: -ROAD, x1: gridW + ROAD, y1: gridD + ROAD },
    rail: { y: railY }, promenade: { y: promY }, walk,
  }
}
