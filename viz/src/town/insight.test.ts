import { describe, expect, it } from 'vitest'
import type { PrincipalBundle } from '../types'
import { fitLabels, memoryFeed, seasonMarks, talksByDay } from './insight'
import type { TownPerson } from './types'

const DAYS = ['2025-08-25', '2025-08-26', '2025-08-27']
const H = (day: number, hour: number, min = 0) => day * 1440 + hour * 60 + min

function person(id: string, memory: TownPerson['memory'], sessions: TownPerson['sessions'] = {}): TownPerson {
  return { id, name: id, age: 30, minor: false, role: '', anchor: null, homes: [], sessions, memory, days: [] }
}

describe("the town's memory activity", () => {
  const linh = person('linh', [
    { t: '2025-08-25T07:15:00', lines: 3, files: 1 },
    { t: '2025-08-26T03:00:00', lines: 2, files: 1 },
    { t: '2025-08-26T22:40:00', lines: 6, files: 2 },
  ])
  const jun = person('jun', [{ t: '2025-08-25T12:00:00', lines: 5, files: 2 }])

  it('only shows what has happened by now, newest first', () => {
    const f = memoryFeed([linh, jun], DAYS, H(1, 12), 10)
    expect(f.map((e) => `${e.pid}@${e.at}`)).toEqual([`linh@${H(1, 3)}`, `jun@${H(0, 12)}`, `linh@${H(0, 7, 15)}`])
  })

  it("measures each write against the same resident's previous one", () => {
    const f = memoryFeed([linh, jun], DAYS, H(2, 0), 10)
    expect(f.find((e) => e.at === H(1, 22, 40))?.delta).toBe(4)
    expect(f.find((e) => e.at === H(1, 3))?.delta).toBe(-1)
    expect(f.find((e) => e.pid === 'jun')?.delta).toBe(5)
  })

  it('calls the 03:00 commit a dream and the others writes', () => {
    const f = memoryFeed([linh], DAYS, H(2, 0), 10)
    expect(f.map((e) => e.kind)).toEqual(['write', 'dream', 'write'])
  })

  it('keeps only the latest few', () => {
    expect(memoryFeed([linh, jun], DAYS, H(2, 0), 2).map((e) => e.at)).toEqual([H(1, 22, 40), H(1, 3)])
  })
})

describe('talks per day', () => {
  it('counts the sessions of each day, zero where there were none', () => {
    const p = person('linh', [], { '2025-08-25': ['s1'], '2025-08-27': ['s2', 's3'], '2025-09-30': ['s9'] })
    expect(talksByDay(p, DAYS)).toEqual([1, 0, 2])
  })
})

describe('which timeline labels fit', () => {
  it('shows labels that do not touch on one row', () => {
    expect(fitLabels([{ x: 0, w: 50 }, { x: 60, w: 50 }], [0, 1], 1)).toEqual([0, 0])
  })

  it('hides the lower-priority label of two that overlap', () => {
    expect(fitLabels([{ x: 0, w: 50 }, { x: 40, w: 50 }], [1, 0], 1)).toEqual([-1, 0])
  })

  it('moves a clashing label to the next row when there is one', () => {
    expect(fitLabels([{ x: 20, w: 50 }, { x: 0, w: 50 }], [0, 1], 2)).toEqual([0, 1])
  })

  it("keeps a lower label's stem out of the labels above it", () => {
    expect(fitLabels([{ x: 0, w: 100 }, { x: 40, w: 30 }], [0, 1], 2)).toEqual([0, -1])
  })

  it('does not lay a label over the stem of one on a lower row', () => {
    // b hangs left of its stem at 50, on row 1; c would fit beside a on row 0 but for b's stem
    const items = [{ x: 0, w: 40 }, { x: 10, w: 40, stem: 50 }, { x: 45, w: 30 }]
    expect(fitLabels(items, [0, 1, 2], 2)).toEqual([0, 1, -1])
  })

  it('keeps a gap between labels on the same row, whichever side the later one is on', () => {
    const items = [{ x: 0, w: 50 }, { x: 54, w: 50 }]
    expect(fitLabels(items, [0, 1], 1, 8)).toEqual([0, -1])
    expect(fitLabels(items, [1, 0], 1, 8)).toEqual([-1, 0])
  })
})

describe("a resident's season marks", () => {
  // only the fields the marks read: when each talk, write, dream and probe checkpoint happened
  const bundle = {
    sessions: [{ id: 'linh.s01', start: '2025-08-25T07:15:00' }],
    commits: [
      { sha: 'a1', sim_time: '2025-08-25T07:30:00', event: 'write' },
      { sha: 'b2', sim_time: '2025-08-26T03:00:00', event: 'dream' },
    ],
    probes: [
      { checkpoint: 'k1', t: '2025-08-26T12:00:00' },
      { checkpoint: 'k1', t: '2025-08-26T12:00:00' },
    ],
  } as unknown as PrincipalBundle

  it('puts each talk, write, dream and checkpoint on its own lane, at its minute of the season', () => {
    expect(seasonMarks(bundle, DAYS[0])).toEqual([
      { key: 's:linh.s01', lane: 'talk', at: H(0, 7, 15) },
      { key: 'c:a1', lane: 'write', at: H(0, 7, 30) },
      { key: 'c:b2', lane: 'dream', at: H(1, 3) },
      { key: 'k:k1', lane: 'probe', at: H(1, 12) },
    ])
  })
})
