import { describe, expect, it } from 'vitest'
import { FACES, HAIRS, pickLook, type Who } from './avatar'

// today's twelve, as town.json has them
const TWELVE: Who[] = [
  { id: 'arjun', age: 20, minor: false, role: 'junior in logistics management at Bayshore State, Tamsen campus' },
  { id: 'dulce', age: null, minor: true, role: 'sophomore at Alcaraz Flats High, part-time counter worker at Taquería Colima' },
  { id: 'elliot', age: 34, minor: false, role: "Tamsen County assessor's office analyst on parental leave" },
  { id: 'grace', age: 45, minor: false, role: 'chemistry teacher at Alcaraz Flats High' },
  { id: 'isabel', age: null, minor: true, role: 'junior at Alcaraz Flats High applying to colleges' },
  { id: 'jun', age: 25, minor: false, role: 'barista at Heron & Grind starting a lumpia pop-up business' },
  { id: 'linh', age: 19, minor: false, role: 'Tamsen Valley College student, barista at Heron & Grind' },
  { id: 'marcus', age: 19, minor: false, role: 'sophomore at Bayshore State, night-shift picker at Ridgeline RF-4' },
  { id: 'noemi', age: 26, minor: false, role: 'IT support technician on the Ridgeline RF-4 help desk' },
  { id: 'rafael', age: 19, minor: false, role: 'automotive technology student at Tamsen Valley College, helps run the family pupusa stand' },
  { id: 'rochelle', age: 36, minor: false, role: 'former Ridgeline picker taking nursing prerequisites at Tamsen Valley College' },
  { id: 'thao', age: 27, minor: false, role: 'night-shift RN on 4 West at St. Anselm Valley Medical Center' },
]
const crowd = (n: number, age: number | null = 30, minor = false): Who[] =>
  Array.from({ length: n }, (_, i) => ({ id: `resident-${i}`, age, minor, role: '' }))
const outfit = (role: string) => pickLook({ id: 'x', age: 30, minor: false, role }).outfit
const band = (age: number | null, minor = false) => pickLook({ id: 'x', age, minor, role: '' }).band

describe('a resident’s look', () => {
  it('is the same every time for the same resident', () => {
    for (const p of TWELVE) expect(pickLook({ ...p })).toEqual(pickLook(p))
  })

  it('depends on who they are, not on the object it is read from', () => {
    const a = pickLook(TWELVE[0])
    const b = pickLook(TWELVE[1])
    expect(a).not.toEqual(b)
  })

  it('spreads today’s twelve across hair styles, not one style for all', () => {
    expect(new Set(TWELVE.map((p) => pickLook(p).hair)).size).toBeGreaterThanOrEqual(5)
  })

  it('offers at least eight hair styles and four face shapes', () => {
    expect(HAIRS.length).toBeGreaterThanOrEqual(8)
    expect(FACES.length).toBeGreaterThanOrEqual(4)
  })

  it('uses every hair style in a grown town, none far more than its share', () => {
    const counts = new Map<string, number>()
    for (const p of crowd(400)) counts.set(pickLook(p).hair, (counts.get(pickLook(p).hair) ?? 0) + 1)
    expect([...counts.keys()].sort()).toEqual([...HAIRS].sort())
    for (const n of counts.values()) expect(n).toBeLessThan((400 / HAIRS.length) * 2)
  })

  it('uses every face shape in a grown town', () => {
    expect(new Set(crowd(200).map((p) => pickLook(p).face))).toEqual(new Set(FACES))
  })

  it('gives a grown town a range of skin tones and hair colours', () => {
    const looks = crowd(200).map(pickLook)
    expect(new Set(looks.map((l) => l.skin)).size).toBeGreaterThanOrEqual(6)
    expect(new Set(looks.map((l) => l.hairColor)).size).toBeGreaterThanOrEqual(6)
  })

  it('puts glasses on some residents but not on all', () => {
    const withGlasses = crowd(200).filter((p) => pickLook(p).glasses !== 'none').length
    expect(withGlasses).toBeGreaterThan(20)
    expect(withGlasses).toBeLessThan(140)
  })
})

describe('the age band', () => {
  it.each([
    [15, 'young'], [19, 'young'], [20, 'adult'], [29, 'adult'], [30, 'mid'], [44, 'mid'], [45, 'senior'], [71, 'senior'],
  ] as const)('puts age %i in the %s band', (age, want) => {
    expect(band(age)).toBe(want)
  })

  it('puts a minor with no age in the young band', () => {
    expect(band(null, true)).toBe('young')
  })

  it('puts an adult with no age in the adult band', () => {
    expect(band(null, false)).toBe('adult')
  })

  it('draws a smile line from thirty on, and never before', () => {
    for (const l of crowd(60, 30).map(pickLook)) expect(l.smileLines).toBe(true)
    for (const l of crowd(60, 29).map(pickLook)) expect(l.smileLines).toBe(false)
    for (const l of crowd(60, null, true).map(pickLook)) expect(l.smileLines).toBe(false)
  })

  it('gives everyone from forty-five a grey streak or reading glasses', () => {
    for (const l of crowd(80, 45).map(pickLook)) expect(l.streak || l.glasses === 'reading').toBe(true)
  })

  it('gives nobody under forty-five a grey streak or reading glasses', () => {
    for (const l of crowd(80, 44).map(pickLook)) {
      expect(l.streak).toBe(false)
      expect(l.glasses).not.toBe('reading')
    }
  })
})

describe('the outfit, read from the role (first rule that matches wins)', () => {
  it.each([
    ['night-shift RN on 4 West at St. Anselm Valley Medical Center', 'scrubs'],
    ['registered nurse in the ER', 'scrubs'],
    ['former Ridgeline picker taking nursing prerequisites at Tamsen Valley College', 'scrubs'],
    ['barista at Heron & Grind starting a lumpia pop-up business', 'apron'],
    ['sophomore at Alcaraz Flats High, part-time counter worker at Taquería Colima', 'apron'],
    ['line cook at the taqueria on Calle Ortega', 'apron'],
    ['owner of a corner café', 'apron'],
    ['Tamsen Valley College student, barista at Heron & Grind', 'apron'],
    ['IT support technician on the Ridgeline RF-4 help desk', 'headset'],
    ['help desk lead for the county', 'headset'],
    ['automotive technology student at Tamsen Valley College, helps run the family pupusa stand', 'workshirt'],
    ['automotive technician at a tire shop', 'workshirt'],
    ['mechanic at Ruel Road Auto', 'workshirt'],
    ['chemistry teacher at Alcaraz Flats High', 'cardigan'],
    ['sophomore at Bayshore State, night-shift picker at Ridgeline RF-4', 'vest'],
    ['warehouse associate at Ridgeline', 'vest'],
    ['night shift security at the mall', 'vest'],
    ["Tamsen County assessor's office analyst on parental leave", 'collar'],
    ['office manager at a dental practice', 'collar'],
    ['junior in logistics management at Bayshore State, Tamsen campus', 'hoodie'],
    ['junior at Alcaraz Flats High applying to colleges', 'hoodie'],
    ['first-year at Tamsen Valley College', 'hoodie'],
    ['retired fisherman who walks the marsh', 'crew'],
    ['volunteer who keeps it tidy at the park', 'crew'],
    ['', 'crew'],
  ] as const)('dresses “%s” in %s', (role, want) => {
    expect(outfit(role)).toBe(want)
  })
})
