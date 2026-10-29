// How a resident is drawn on their badge. The data has no appearance at all, so the look is a drawing picked from a
// hash of the id (stable as the town grows: each trait draws from its own key, so nobody's look depends on anyone
// else's); only the age band and the outfit read the data, and neither is a claim. No facial hair: nothing to base it on.
import { hash, rng } from './rand'

export const FACES = ['oval', 'round', 'long', 'square', 'heart'] as const
export const HAIRS = ['crop', 'buzz', 'curls', 'bun', 'long', 'bob', 'ponytail', 'wavy', 'puff', 'quiff'] as const
export const BROWS = ['arch', 'flat', 'thick'] as const
export const EYES = ['dot', 'oval', 'arc', 'ring'] as const
export const NOSES = ['ell', 'button', 'hook', 'bump'] as const
export const MOUTHS = ['smile', 'grin', 'soft', 'smirk'] as const

// a wide, even range from light to deep; hair mostly dark, some brown, auburn and fair
const SKINS = ['#f6dac6', '#efc4a2', '#e2ad86', '#cf9468', '#b97c51', '#9c6342', '#7d4b31', '#5f3825']
const HAIR_COLORS = ['#1e1a19', '#2f231e', '#3e2a1f', '#583824', '#76482b', '#8b3d22', '#a77843', '#c9a165']
// the hair clip, a small accent from the meadow
const CLIPS = ['#f3b5c2', '#f6cd4c', '#f1913a', '#b9d65a']

export type Face = (typeof FACES)[number]
export type Hair = (typeof HAIRS)[number]
export type Outfit = 'scrubs' | 'apron' | 'headset' | 'cardigan' | 'vest' | 'workshirt' | 'collar' | 'hoodie' | 'crew'
/** young: under 20 (or a minor with no age) · adult: 20–29 (or an adult with no age) · mid: 30–44 · senior: 45 and over */
export type Band = 'young' | 'adult' | 'mid' | 'senior'
export type Glasses = 'none' | 'round' | 'square' | 'reading'

export interface Look {
  seed: number // the pen's wobble
  band: Band
  face: Face
  hair: Hair
  hairColor: string
  skin: string
  brows: (typeof BROWS)[number]
  eyes: (typeof EYES)[number]
  nose: (typeof NOSES)[number]
  mouth: (typeof MOUTHS)[number]
  glasses: Glasses
  freckles: boolean
  earring: boolean
  clip: string | null // the clip's colour
  streak: boolean // a grey streak in the hair
  smileLines: boolean
  flip: boolean // which side the parting, the tail and the smirk fall
  outfit: Outfit
}

export interface Who {
  id: string
  age: number | null
  minor: boolean
  role: string
}

// The role's words to an outfit, in order; the first rule that matches wins. The order settles the mixed roles:
//   1 scrubs     nurse, RN, nursing          (a nursing student, or a former picker taking nursing classes, is in scrubs)
//   2 apron      barista, counter, taquería, cafe   (a student who also works the counter wears the apron)
//   3 headset    IT, help desk, technician — not an automotive one
//   4 cardigan   teacher (with a lanyard)
//   5 vest       picker, warehouse, night shift (hi-vis; a night-shift nurse already matched 1)
//   6 workshirt  automotive, mechanic (with a name patch)
//   7 collar     analyst, office, assessor
//   8 hoodie     student, sophomore, junior, college
//   9 crew       anything else
// "IT" is matched as the capitalised acronym only, so the pronoun "it" in a role is not a help desk.
const RULES: [Outfit, (role: string, raw: string) => boolean][] = [
  ['scrubs', (r) => /\bnurse|\bnursing\b|\brn\b/.test(r)],
  ['apron', (r) => /\bbarista|\bcounter\b|\btaqueria|\bcafe\b/.test(r)],
  ['headset', (r, raw) => /\bIT\b/.test(raw) || /help ?desk/.test(r) || (/\btechnician/.test(r) && !/automotive/.test(r))],
  ['cardigan', (r) => /\bteacher/.test(r)],
  ['vest', (r) => /\bpicker|\bwarehouse|\bnight[- ]shift/.test(r) && !/\bnurse|\brn\b/.test(r)],
  ['workshirt', (r) => /\bautomotive|\bmechanic/.test(r)],
  ['collar', (r) => /\banalyst|\boffice\b|\bassessor/.test(r)],
  ['hoodie', (r) => /\bstudent|\bsophomore|\bjunior\b|\bcollege/.test(r)],
]

function outfitFor(role: string): Outfit {
  // lower case, accents off: "Taquería" and "café" match like "taqueria" and "cafe"
  const r = role.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase()
  return RULES.find(([, test]) => test(r, role))?.[0] ?? 'crew'
}

function bandOf(age: number | null, minor: boolean): Band {
  if (age === null) return minor ? 'young' : 'adult'
  return age < 20 ? 'young' : age < 30 ? 'adult' : age < 45 ? 'mid' : 'senior'
}

export function pickLook(p: Who): Look {
  const draw = (trait: string) => rng(`look:${p.id}:${trait}`)()
  const one = <T,>(trait: string, xs: readonly T[]): T => xs[Math.floor(draw(trait) * xs.length)]
  const band = bandOf(p.age, p.minor)
  const hair = one('hair', HAIRS)
  // from forty-five: reading glasses or a grey streak, the hash picks which
  const reading = band === 'senior' && draw('reading') < 0.5
  return {
    seed: hash(`look:${p.id}`) % 997,
    band,
    face: one('face', FACES),
    hair,
    hairColor: one('hairColor', HAIR_COLORS),
    skin: one('skin', SKINS),
    brows: one('brows', BROWS),
    eyes: one('eyes', EYES),
    nose: one('nose', NOSES),
    mouth: one('mouth', MOUTHS),
    glasses: reading ? 'reading' : draw('glasses') < 0.18 ? one('frames', ['round', 'square'] as const) : 'none',
    freckles: draw('freckles') < 0.25,
    earring: draw('earring') < 0.3,
    clip: hair !== 'buzz' && draw('clip') < 0.22 ? one('clipColor', CLIPS) : null,
    streak: band === 'senior' && !reading,
    smileLines: band === 'mid' || band === 'senior',
    flip: draw('flip') < 0.5,
    outfit: outfitFor(p.role),
  }
}

/** The number printed on the badge: DV· and four base-36 characters from the id, never from a position in a list. */
export function residentNo(id: string): string {
  return `DV·${(hash(`no:${id}`) % 36 ** 4).toString(36).toUpperCase().padStart(4, '0')}`
}

/** The stub's barcode as alternating bar and gap widths in modules, from the id: decorative, but the same every time. */
export function barcode(id: string, n = 46): number[] {
  const r = rng(`bar:${id}`)
  const guard = [1, 1, 1, 1]
  return [...guard, ...Array.from({ length: n }, (_, i) => (i % 2 === 0 ? 1 + Math.floor(r() * 3) : 1 + Math.floor(r() * 2))), ...guard]
}
