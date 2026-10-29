import { describe, expect, it } from 'vitest'
import { planTown, type Plan } from './plan'
import type { Place } from './types'

const place = (id: string, kind: string, district: string): Place => ({ id, name: id, kind, description: '', district })

// Shaped like full13-v1: 34 places over 7 districts.
const TODAY: Place[] = [
  ...['house', 'house', 'house', 'house', 'house', 'house', 'house', 'apartment_complex', 'apartment_complex'].map((k, i) => place(`home-${i}`, k, 'homes')),
  ...['cafe', 'cafe', 'main_street', 'grocery_store', 'taqueria', 'pho_restaurant', 'strip_mall', 'gym', 'laundromat', 'bakery'].map((k, i) => place(`main-${i}`, k, 'main_street')),
  ...['high_school', 'middle_school', 'community_college', 'state_university', 'library'].map((k, i) => place(`learn-${i}`, k, 'learning')),
  ...['hospital', 'church', 'temple', 'community_center', 'theater'].map((k, i) => place(`civic-${i}`, k, 'civic')),
  place('ind-0', 'warehouse_employer', 'industry'),
  place('ind-1', 'industrial_area', 'industry'),
  place('park-0', 'park', 'parks'),
  place('park-1', 'park', 'parks'),
  place('station-0', 'transit_station', 'station'),
]

// A grown town: ten times the places, new districts and kinds the generator has never seen.
const KINDS = ['house', 'cafe', 'apartment_complex', 'hospital', 'park', 'warehouse_employer', 'skate_park', 'bowling_alley', 'transit_station']
const GROWN: Place[] = Array.from({ length: 340 }, (_, i) => place(`p${i}`, KINDS[i % KINDS.length], ['homes', 'main_street', 'harbor', 'hills', 'old_town'][i % 5]))

const overlap = (a: { x: number; y: number; w: number; d: number }, b: typeof a) =>
  a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.d && b.y < a.y + a.d

function lotsOf(plan: Plan) {
  return [...plan.lots.values()]
}

describe.each([
  ['today', TODAY],
  ['grown', GROWN],
])('a %s town', (_, places) => {
  const plan = planTown(places)

  it('places every place exactly once', () => {
    expect([...plan.lots.keys()].sort()).toEqual(places.map((p) => p.id).sort())
  })

  it('never lets two buildings overlap', () => {
    const lots = lotsOf(plan)
    for (let i = 0; i < lots.length; i++) for (let j = i + 1; j < lots.length; j++) expect(overlap(lots[i], lots[j])).toBe(false)
  })

  it("keeps each building inside a block of its own district, clear of the sidewalk", () => {
    for (const lot of lotsOf(plan).filter((l) => l.place.kind !== 'transit_station')) {
      const b = plan.blocks.find((k) => k.id === lot.block)!
      expect(b.district).toBe(lot.place.district)
      expect(lot.x).toBeGreaterThanOrEqual(b.x + plan.sidewalk)
      expect(lot.y).toBeGreaterThanOrEqual(b.y + plan.sidewalk)
      expect(lot.x + lot.w).toBeLessThanOrEqual(b.x + b.w - plan.sidewalk)
      expect(lot.y + lot.d).toBeLessThanOrEqual(b.y + b.d - plan.sidewalk)
    }
  })

  it('never lets two blocks overlap', () => {
    for (let i = 0; i < plan.blocks.length; i++)
      for (let j = i + 1; j < plan.blocks.length; j++) expect(overlap(plan.blocks[i], plan.blocks[j])).toBe(false)
  })

  it('puts every station on the rail line', () => {
    for (const lot of lotsOf(plan).filter((l) => l.place.kind === 'transit_station')) {
      expect(lot.y).toBeLessThan(plan.rail.y)
      expect(lot.y + lot.d).toBeGreaterThan(plan.rail.y)
    }
  })

  it('walks between any two places along the streets, door to door, never through a building', () => {
    const lots = lotsOf(plan)
    const pairs = lots.slice(0, 40).flatMap((a, i) => [lots[(i * 7 + 3) % lots.length], lots[(i * 13 + 1) % lots.length]].map((b) => [a, b]))
    for (const [a, b] of pairs) {
      const path = plan.walk(a.place.id, b.place.id)
      expect(path[0]).toEqual(a.door)
      expect(path.at(-1)).toEqual(b.door)
      for (let k = 1; k < path.length; k++) {
        const [p, q] = [path[k - 1], path[k]]
        expect(p[0] === q[0] || p[1] === q[1]).toBe(true) // streets run on the grid
        const seg = { x: Math.min(p[0], q[0]), y: Math.min(p[1], q[1]), w: Math.abs(p[0] - q[0]), d: Math.abs(p[1] - q[1]) }
        for (const lot of lots) {
          const inner = { x: lot.x + 0.01, y: lot.y + 0.01, w: lot.w - 0.02, d: lot.d - 0.02 }
          expect(overlap({ ...seg, w: Math.max(seg.w, 0.001), d: Math.max(seg.d, 0.001) }, inner)).toBe(false)
        }
      }
    }
  })
})

describe('the plan', () => {
  it('is the same whatever order the places come in', () => {
    const a = planTown(TODAY)
    const b = planTown([...TODAY].reverse())
    for (const [id, lot] of a.lots) expect(b.lots.get(id)).toEqual(lot)
  })

  it('gives a kind it has never seen a building of its own', () => {
    const lot = planTown([...TODAY, place('new-1', 'bowling_alley', 'main_street')]).lots.get('new-1')!
    expect(lot.w).toBeGreaterThan(0)
    expect(lot.d).toBeGreaterThan(0)
  })

  it('gives a district it has never seen a block of its own', () => {
    const plan = planTown([...TODAY, place('new-2', 'cafe', 'harbor')])
    expect(plan.blocks.some((b) => b.district === 'harbor')).toBe(true)
  })
})
