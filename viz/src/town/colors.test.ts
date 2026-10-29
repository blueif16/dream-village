import { describe, expect, it } from 'vitest'
import { oklab, residentColors } from './colors'

const TWELVE = ['arjun', 'dulce', 'elliot', 'grace', 'isabel', 'jun', 'linh', 'marcus', 'noemi', 'rafael', 'rochelle', 'thao']
const dist = (a: number, b: number) => {
  const [p, q] = [oklab(a), oklab(b)]
  return Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2])
}

describe('resident colours', () => {
  it("keep today's twelve residents in the colours they already have", () => {
    const c = residentColors(TWELVE)
    expect(c.get('linh')).toBe(0x2a9d8f)
    expect(c.get('arjun')).toBe(0xc0573e)
    expect(new Set(c.values()).size).toBe(12)
  })

  it('give a grown town of sixty residents sixty clearly different colours', () => {
    const ids = [...TWELVE, ...Array.from({ length: 48 }, (_, i) => `resident-${i}`)]
    const c = [...residentColors(ids).values()]
    expect(c).toHaveLength(60)
    let min = Infinity
    for (let i = 0; i < c.length; i++) for (let j = i + 1; j < c.length; j++) min = Math.min(min, dist(c[i], c[j]))
    expect(min).toBeGreaterThan(0.04) // OKLab: about 0.02 is a just-noticeable difference
  })

  it('do not depend on the order residents are listed in', () => {
    const ids = [...TWELVE, 'zoe', 'amir', 'bea']
    expect([...residentColors(ids)].sort()).toEqual([...residentColors([...ids].reverse())].sort())
  })
})
