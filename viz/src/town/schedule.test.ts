import { describe, expect, it } from 'vitest'
import { dayPlan, dreamTimes, lineIndexAt, talkTimes, talkingAt, targetAt } from './schedule'
import type { TownPerson } from './types'

const DAYS = ['2025-08-25', '2025-08-26', '2025-08-27']
const H = (day: number, hour: number, min = 0) => day * 1440 + hour * 60 + min

function person(over: Partial<TownPerson> = {}): TownPerson {
  return {
    id: 'linh',
    name: 'Linh Pham',
    age: 19,
    minor: false,
    role: 'student',
    anchor: 'pl-college',
    homes: ['home-linh'],
    sessions: { '2025-08-25': ['linh.s01'] },
    memory: [
      { t: '2025-08-25T07:15:00', lines: 1, files: 1 },
      { t: '2025-08-26T03:00:00', lines: 1, files: 1 },
      { t: '2025-08-26T22:40:00', lines: 4, files: 2 },
      { t: '2025-08-27T03:12:00', lines: 4, files: 2 }, // a night talk in the dream hour is still a talk
    ],
    days: [
      { place: 'pl-cafe', why: 'life_log', lines: [{ what: 'a', with: [] }, { what: 'b', with: [] }, { what: 'c', with: [] }] },
      { place: 'home-linh', why: 'home', lines: [{ what: 'd', with: [] }] },
      { place: 'pl-college', why: 'work', lines: [] },
    ],
    ...over,
  }
}

describe('where a resident is', () => {
  it('is at home at 3 a.m.', () => {
    expect(targetAt(person(), H(0, 3))).toBe('home-linh')
  })

  it("is at that day's place at midday", () => {
    expect(targetAt(person(), H(0, 12))).toBe('pl-cafe')
  })

  it("goes to each day's own place", () => {
    expect(targetAt(person(), H(2, 12))).toBe('pl-college')
  })

  it('stays home all day when the day is spent at home', () => {
    expect(targetAt(person(), H(1, 12))).toBe('home-linh')
  })

  it("is back home late in the evening", () => {
    expect(targetAt(person(), H(0, 23, 30))).toBe('home-linh')
  })

  it("with no home, spends the night at the day's place", () => {
    expect(targetAt(person({ homes: [] }), H(0, 3))).toBe('pl-cafe')
  })
})

describe('the day plan', () => {
  it('leaves in the morning and comes back in the evening, every day, for every resident', () => {
    for (const id of ['linh', 'arjun', 'noemi', 'resident-042', 'x']) {
      const p = person({ id, days: Array.from({ length: 91 }, () => ({ place: 'pl-cafe', why: 'life_log' as const, lines: [] })) })
      for (let d = 0; d < 91; d++) {
        const { leave, back } = dayPlan(p, d)
        expect(leave).toBeGreaterThanOrEqual(7 * 60)
        expect(leave).toBeLessThanOrEqual(9 * 60 + 30)
        expect(back).toBeGreaterThanOrEqual(16 * 60 + 30)
        expect(back).toBeLessThanOrEqual(21 * 60 + 30)
      }
    }
  })

  it('differs between residents and between days', () => {
    const many = Array.from({ length: 91 }, () => ({ place: 'pl-cafe', why: 'life_log' as const, lines: [] }))
    const a = person({ id: 'linh', days: many })
    const b = person({ id: 'arjun', days: many })
    const leaves = new Set(Array.from({ length: 91 }, (_, d) => dayPlan(a, d).leave))
    expect(leaves.size).toBeGreaterThan(10)
    expect(Array.from({ length: 91 }, (_, d) => dayPlan(a, d).leave)).not.toEqual(
      Array.from({ length: 91 }, (_, d) => dayPlan(b, d).leave),
    )
  })

  it('is the same every time it is asked', () => {
    expect(dayPlan(person(), 0)).toEqual(dayPlan(person(), 0))
  })
})

describe('talking and dreaming', () => {
  it('reads write commits as the moments a resident talked to the assistant', () => {
    expect(talkTimes(person(), DAYS)).toEqual([H(0, 7, 15), H(1, 22, 40), H(2, 3, 12)])
  })

  it('reads the 03:00 commits as dreams', () => {
    expect(dreamTimes(person(), DAYS)).toEqual([H(1, 3)])
  })

  it('glows for half an hour from the start of a talk', () => {
    const talks = talkTimes(person(), DAYS)
    expect(talkingAt(talks, H(0, 7, 10))).toBe(false)
    expect(talkingAt(talks, H(0, 7, 15))).toBe(true)
    expect(talkingAt(talks, H(0, 7, 40))).toBe(true)
    expect(talkingAt(talks, H(0, 7, 50))).toBe(false)
  })
})

describe("the life-log line for the hour", () => {
  it('walks through the day in order', () => {
    const p = person()
    expect(lineIndexAt(p, H(0, 8))).toBe(0)
    expect(lineIndexAt(p, H(0, 14))).toBe(1)
    expect(lineIndexAt(p, H(0, 21))).toBe(2)
  })

  it('has no line while the resident sleeps, or on a day with no log', () => {
    expect(lineIndexAt(person(), H(0, 4))).toBeNull()
    expect(lineIndexAt(person(), H(2, 12))).toBeNull()
  })
})
