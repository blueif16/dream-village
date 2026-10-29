// A resident's portrait, drawn the way the references draw people: confident black ink with round ends, a little
// wobble from the hand, flat fills, and a block of colour printed slightly off behind the head. Everything it draws
// comes from the Look (avatar.ts); it is a drawing, not a likeness. Drawn facing right; a flipped look is mirrored.
import { useId } from 'react'
import type { Hair, Look } from './avatar.ts'
import { cssHex } from './colors'
import { rng } from './rand'

const INK = '#1a1c1a'
const CREAM = '#f6f0e1'
const CX = 60

type Pt = [number, number]
const n1 = (v: number) => (Math.round(v * 10) / 10).toString()
const P = ([x, y]: Pt) => `${n1(x)} ${n1(y)}`
const mirror = ([x, y]: Pt): Pt => [2 * CX - x, y]

/** Mix two #rrggbb colours, t of the way from a to b. */
export function mix(a: string, b: string, t: number): string {
  const c = (s: string, i: number) => parseInt(s.slice(1 + i * 2, 3 + i * 2), 16)
  return `#${[0, 1, 2].map((i) => Math.round(c(a, i) + (c(b, i) - c(a, i)) * t).toString(16).padStart(2, '0')).join('')}`
}

/** A smooth curve through the points (Catmull-Rom as cubic Béziers). */
function curve(pts: Pt[], closed = false): string {
  const n = pts.length
  const at = (i: number) => (closed ? pts[(i + n) % n] : pts[Math.max(0, Math.min(n - 1, i))])
  let d = `M${P(pts[0])}`
  for (let i = 0; i < (closed ? n : n - 1); i++) {
    const [p0, p1, p2, p3] = [at(i - 1), at(i), at(i + 1), at(i + 2)]
    d += `C${P([p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6])} ${P([p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6])} ${P(p2)}`
  }
  return closed ? `${d}Z` : d
}

/** Points along the same curve, about `step` apart: the base for a scalloped edge. */
function along(pts: Pt[], step: number, closed = false): Pt[] {
  const n = pts.length
  const at = (i: number) => (closed ? pts[(i + n) % n] : pts[Math.max(0, Math.min(n - 1, i))])
  const dense: Pt[] = []
  for (let i = 0; i < (closed ? n : n - 1); i++) {
    const [p0, p1, p2, p3] = [at(i - 1), at(i), at(i + 1), at(i + 2)]
    for (let k = 0; k < 12; k++) {
      const t = k / 12
      const h = (a: number, b: number, c: number, d: number) =>
        0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t * t + (-a + 3 * b - 3 * c + d) * t * t * t)
      dense.push([h(p0[0], p1[0], p2[0], p3[0]), h(p0[1], p1[1], p2[1], p3[1])])
    }
  }
  if (!closed) dense.push(pts[n - 1])
  const out: Pt[] = [dense[0]]
  let acc = 0
  for (let i = 1; i < dense.length; i++) {
    acc += Math.hypot(dense[i][0] - dense[i - 1][0], dense[i][1] - dense[i - 1][1])
    if (acc >= step) {
      out.push(dense[i])
      acc = 0
    }
  }
  if (!closed && out[out.length - 1] !== dense[dense.length - 1]) out.push(dense[dense.length - 1])
  return out
}

/** Bumps between successive points, pushed outward (the shape runs clockwise on screen). */
function scallop(pts: Pt[], bump: number, closed = false): string {
  let d = `M${P(pts[0])}`
  const n = pts.length
  for (let i = 0; i < (closed ? n : n - 1); i++) {
    const a = pts[i]
    const b = pts[(i + 1) % n]
    const [dx, dy] = [b[0] - a[0], b[1] - a[1]]
    const len = Math.hypot(dx, dy) || 1
    d += `Q${P([(a[0] + b[0]) / 2 + (dy / len) * bump, (a[1] + b[1]) / 2 - (dx / len) * bump])} ${P(b)}`
  }
  return closed ? `${d}Z` : d
}

// ---------------------------------------------------------------- the head

interface G {
  top: number
  chin: number
  hw: number // half-width at the cheeks
  tw: number // at the temples
  jw: number // at the jaw
  sq: number // a squarer jaw
  ey: number // the eye line
  es: number // half the eyes' spacing
  ny: number // the foot of the nose
  my: number // the mouth
  young: boolean
}

// half-widths at cheek, temple and jaw; top and chin
const FACE: Record<Look['face'], [number, number, number, number, number, number]> = {
  oval: [21.5, 19.5, 13.5, 24, 86, 0],
  round: [23.5, 21.5, 17, 25, 84, 0],
  long: [20, 19, 12.5, 22, 89, 0],
  square: [22.5, 21.5, 18.5, 24, 86, 1],
  heart: [22.5, 22.5, 10.5, 24, 86, 0],
}

function geom(look: Look): G {
  let [hw, tw, jw, top, chin] = FACE[look.face]
  const sq = FACE[look.face][5]
  const young = look.band === 'young'
  // the young band: a rounder, shorter face, the features a touch lower and larger
  if (young) [hw, tw, jw, top, chin] = [hw + 1.3, tw + 1, jw + 2.2, top + 1.5, chin - 4]
  const ey = top + (chin - top) * (young ? 0.53 : 0.5)
  return { top, chin, hw, tw, jw, sq, ey, es: hw * 0.43, ny: ey + (chin - ey) * (young ? 0.33 : 0.37), my: ey + (chin - ey) * 0.63, young }
}

function facePts(g: G): Pt[] {
  const right: Pt[] = [
    [CX + g.tw * 0.6, g.top + 2.2],
    [CX + g.tw, g.top + (g.ey - g.top) * 0.45],
    [CX + g.hw, g.ey + 2],
    [CX + (g.hw + g.jw) / 2 + g.sq * 1.5, g.ey + (g.chin - g.ey) * 0.58],
    [CX + g.jw * (g.sq ? 0.85 : 0.55), g.chin - (g.sq ? 2 : 3.2)],
  ]
  return [[CX, g.top], ...right, [CX, g.chin], ...right.slice().reverse().map(mirror)]
}

/** The crown of the head pushed out by o (oTop over the top), from one side at y = end, over, to the other. */
function skull(g: G, o: number, oTop: number, end: number): Pt[] {
  const right: Pt[] = [
    [CX + g.hw + o, end],
    [CX + g.tw + o, g.top + (g.ey - g.top) * 0.45 - o * 0.25],
    [CX + g.tw * 0.6 + o * 0.6, g.top + 2.2 - (o + oTop) * 0.55],
  ]
  return [...right.map(mirror), [CX, g.top - oTop], ...right.slice().reverse()]
}

// ---------------------------------------------------------------- hair

interface HairArt {
  back?: string // behind the head and shoulders
  front: string // over the forehead
  frontLine?: string // when only part of the front is outlined (curls, puff: the rest sits inside the back's silhouette)
  strands?: string[]
  buzz?: boolean
  side: number // the hair's half-width at the eyes and its top, for a headset to sit on
  crown: number
}

function hairArt(h: Hair, g: G): HairArt {
  const { top, hw, tw, ey, chin } = g
  const tY = top + (ey - top) * 0.45
  switch (h) {
    case 'crop':
      return {
        front: curve([...skull(g, 2.8, 4.2, ey - 2.5), [CX + hw - 0.2, ey - 2.5], [CX + hw - 1.4, ey - 9.5], [CX + tw * 0.72, top + 12.5],
          [CX + 8, top + 10.5], [CX + 1.5, top + 13.5], [CX - 6, top + 11], [CX - tw * 0.72, top + 13], [CX - hw + 1.4, ey - 9.5], [CX - hw + 0.2, ey - 2.5]], true),
        strands: [`M${P([CX + 1.5, top + 12.5])}Q${P([CX + 5, top + 3])} ${P([CX + 14, top + 0.5])}`],
        side: hw + 2.8, crown: top - 4.2,
      }
    case 'buzz': {
      const line: Pt[] = [[CX + hw - 0.3, ey - 3], [CX + hw - 1.2, ey - 10], [CX + tw * 0.66, top + 11.5], [CX, top + 10.5], [CX - tw * 0.66, top + 11.5], [CX - hw + 1.2, ey - 10], [CX - hw + 0.3, ey - 3]]
      return {
        front: curve([...skull(g, 1.5, 2, ey - 3), ...line], true),
        strands: [curve(line.slice(1, -1).map(([x, y]) => [x + (x - CX) * -0.06, y - 2.6]))],
        buzz: true,
        side: hw + 1.5, crown: top - 2,
      }
    }
    case 'quiff':
      return {
        front: curve([[CX - hw - 2.8, ey - 2.5], [CX - tw - 2.8, tY - 0.8], [CX - tw * 0.6 - 1.6, top - 1.5], [CX - 9, top - 6], [CX + 1, top - 9.6],
          [CX + 11, top - 9.4], [CX + tw * 0.6 + 4.5, top - 4], [CX + tw + 1.8, tY - 6], [CX + tw + 2.8, tY - 0.8], [CX + hw + 2.8, ey - 2.5],
          [CX + hw - 0.3, ey - 2.5], [CX + hw - 1.5, ey - 9.5], [CX + tw * 0.7, top + 12.5], [CX + 6, top + 11.5], [CX - 4, top + 12],
          [CX - tw * 0.72, top + 13], [CX - hw + 1.5, ey - 9.5], [CX - hw + 0.3, ey - 2.5]], true),
        strands: [`M${P([CX - 8, top + 9])}Q${P([CX - 5, top - 3])} ${P([CX + 5, top - 7.5])}`, `M${P([CX + 1, top + 10])}Q${P([CX + 6, top + 1])} ${P([CX + 14, top - 4])}`],
        side: hw + 2.8, crown: top - 9.8,
      }
    case 'bun': {
      const bx = CX + 3
      const by = top - 6.5
      return {
        back: `M${P([bx - 9.5, by])}a9.5 9.5 0 1 0 19 0a9.5 9.5 0 1 0 -19 0Z`,
        front: curve([...skull(g, 2.2, 2.6, ey - 3), [CX + hw - 0.4, ey - 3], [CX + hw - 1.8, ey - 10], [CX + tw * 0.62, top + 12], [CX + 4, top + 10.6],
          [CX - 4, top + 10.6], [CX - tw * 0.62, top + 12], [CX - hw + 1.8, ey - 10], [CX - hw + 0.4, ey - 3]], true),
        strands: [
          `M${P([CX - 9, top + 11])}Q${P([CX - 7, top + 2])} ${P([CX - 1, top - 1.8])}`,
          `M${P([CX + 8, top + 11])}Q${P([CX + 7, top + 2.5])} ${P([CX + 2.5, top - 1.8])}`,
          `M${P([bx - 4.5, by + 2])}Q${P([bx - 4, by - 5])} ${P([bx + 2, by - 4.5])}Q${P([bx + 5, by - 1])} ${P([bx + 1.5, by + 1.5])}`,
        ],
        side: hw + 2.2, crown: top - 2.6,
      }
    }
    case 'ponytail': {
      const x = CX + hw
      return {
        back: curve([[x - 6, top + 4], [x + 6, top + 0.5], [x + 14, top + 10], [x + 14.5, ey + 8], [x + 9, ey + 23], [x + 9.5, ey + 8], [x + 7, top + 16], [x - 2, top + 14]], true),
        front: curve([...skull(g, 2.2, 2.6, ey - 3), [CX + hw - 0.4, ey - 3], [CX + hw - 1.8, ey - 10], [CX + tw * 0.66, top + 12.5], [CX + 7, top + 10],
          [CX - 1, top + 12], [CX - tw * 0.62, top + 12], [CX - hw + 1.8, ey - 10], [CX - hw + 0.4, ey - 3]], true),
        strands: [`M${P([CX - 1, top + 11.5])}Q${P([CX + 3, top + 2])} ${P([CX + 12, top - 1])}`, `M${P([x + 5, top + 5])}Q${P([x + 12, top + 11])} ${P([x + 11, ey + 9])}`],
        side: hw + 2.2, crown: top - 2.6,
      }
    }
    case 'long':
      return {
        back: curve([...skull(g, 4, 4.5, ey), [CX + hw + 5.5, ey + 14], [CX + hw + 7, 92], [CX + hw + 6, 103], [CX - hw - 6, 103], [CX - hw - 7, 92], [CX - hw - 5.5, ey + 14]], true),
        front: curve([...skull(g, 3.2, 3.8, ey + 3), [CX + hw + 4.6, ey + 16], [CX + hw + 6, 96], [CX + hw + 8, 111], [CX + hw + 1.5, 110], [CX + hw - 0.3, 97],
          [CX + hw - 2.2, ey + 15], [CX + hw - 3.6, ey + 2], [CX + hw - 4.6, ey - 8], [CX + tw * 0.45, top + 10.5], [CX + 3, top + 6.5],
          [CX - tw * 0.5, top + 11], [CX - hw + 4.6, ey - 8], [CX - hw + 3.6, ey + 2], [CX - hw + 2.2, ey + 15], [CX - hw + 0.3, 97],
          [CX - hw - 1.5, 110], [CX - hw - 8, 111], [CX - hw - 6, 96], [CX - hw - 4.6, ey + 16]], true),
        strands: [`M${P([CX + hw + 1.5, ey + 4])}Q${P([CX + hw + 3.5, 90])} ${P([CX + hw + 3.6, 104])}`, `M${P([CX - hw - 1.5, ey + 4])}Q${P([CX - hw - 3.5, 90])} ${P([CX - hw - 3.4, 104])}`],
        side: hw + 3.2, crown: top - 3.8,
      }
    case 'bob':
      return {
        back: curve([...skull(g, 4, 4.5, ey), [CX + hw + 5.5, ey + 12], [CX + hw + 5, chin], [CX - hw - 5, chin], [CX - hw - 5.5, ey + 12]], true),
        front: curve([...skull(g, 3.2, 3.8, ey + 2), [CX + hw + 4.8, ey + 12], [CX + hw + 5.8, chin - 1.5], [CX + hw - 0.4, chin - 1], [CX + hw - 2.6, ey + 8],
          [CX + hw - 3.8, ey - 4], [CX + hw - 4.4, ey - 10.5], [CX + 9, ey - 11.8], [CX - 1, ey - 11.2], [CX - 11, ey - 12], [CX - hw + 4.4, ey - 10.5],
          [CX - hw + 3.8, ey - 4], [CX - hw + 2.6, ey + 8], [CX - hw + 0.4, chin - 1], [CX - hw - 5.8, chin - 1.5], [CX - hw - 4.8, ey + 12]], true),
        strands: [-7, 1, 9].map((x) => `M${P([CX + x, ey - 12.4])}L${P([CX + x - 1.5, ey - 18])}`),
        side: hw + 3.2, crown: top - 3.8,
      }
    case 'wavy':
      return {
        back: curve([...skull(g, 4, 4.5, ey), [CX + hw + 6, ey + 10], [CX + hw + 3.5, ey + 20], [CX + hw + 7.5, ey + 30], [CX + hw + 5, 100],
          [CX - hw - 5, 100], [CX - hw - 7.5, ey + 30], [CX - hw - 3.5, ey + 20], [CX - hw - 6, ey + 10]], true),
        front: curve([...skull(g, 3.2, 3.8, ey + 2), [CX + hw + 5.5, ey + 11], [CX + hw + 3.2, ey + 21], [CX + hw + 7.5, ey + 31], [CX + hw + 8.5, 102],
          [CX + hw + 0.5, 99], [CX + hw + 2, ey + 30], [CX + hw - 1.5, ey + 19], [CX + hw - 3.4, ey + 6], [CX + hw - 4.4, ey - 6], [CX + 13, ey - 13.5],
          [CX + 2, top + 14], [CX - 8, top + 10], [CX - tw * 0.75, top + 11], [CX - hw + 3.5, ey - 7], [CX - hw + 2.8, ey + 5], [CX - hw + 1.4, ey + 19],
          [CX - hw - 2, ey + 30], [CX - hw - 0.5, 99], [CX - hw - 8.5, 102], [CX - hw - 7.5, ey + 31], [CX - hw - 3.2, ey + 21], [CX - hw - 5.5, ey + 11]], true),
        strands: [`M${P([CX + 8, top + 3])}Q${P([CX - 2, top + 5])} ${P([CX - 10, top + 9])}`],
        side: hw + 3.2, crown: top - 3.8,
      }
    case 'curls': {
      const outer: Pt[] = [[CX - hw - 6, chin - 6], [CX - hw - 8.5, ey + 2], [CX - hw - 7, top + 8], [CX - 13, top - 8], [CX, top - 10], [CX + 13, top - 8],
        [CX + hw + 7, top + 8], [CX + hw + 8.5, ey + 2], [CX + hw + 6, chin - 6], [CX + hw - 2, chin - 4], [CX - hw + 2, chin - 4]]
      const line: Pt[] = [[CX + hw - 0.5, ey + 1], [CX + hw - 2.5, ey - 8], [CX + tw * 0.7, top + 13.5], [CX + 6, top + 12], [CX - 2, top + 14.5], [CX - 9, top + 12],
        [CX - tw * 0.72, top + 14], [CX - hw + 2.5, ey - 8], [CX - hw + 0.5, ey + 1]]
      return {
        back: scallop(along(outer, 7, true), 2.6, true),
        front: curve([...skull(g, 5, 7, ey + 1), ...line], true),
        frontLine: scallop(along(line, 5), 1.6),
        strands: [[CX - hw - 2, ey + 10], [CX + hw + 2.5, ey + 12], [CX - 10, top - 3], [CX + 9, top - 2]].map(
          ([x, y]) => `M${P([x - 1.8, y + 1])}q0 -2.6 2.2 -2.6q2 0 2 2`),
        side: hw + 10, crown: top - 12,
      }
    }
    case 'puff': {
      const r = hw + 9.5
      const cy = top + 15
      const ring: Pt[] = Array.from({ length: 12 }, (_, i) => {
        const a = Math.PI + (i / 12) * Math.PI * 2
        return [CX + Math.cos(a) * r, cy + Math.sin(a) * r * 0.94]
      })
      const line: Pt[] = [[CX + hw - 0.5, ey - 3], [CX + hw - 1.6, ey - 10], [CX + tw * 0.64, top + 12], [CX, top + 10.5], [CX - tw * 0.64, top + 12], [CX - hw + 1.6, ey - 10], [CX - hw + 0.5, ey - 3]]
      return {
        back: scallop(along(ring, 7.5, true), 2.8, true),
        front: curve([...skull(g, 2, 2.5, ey - 3), ...line], true),
        frontLine: scallop(along(line, 5), 1.2),
        strands: [[CX - 18, top - 2], [CX + 16, top - 5], [CX - 4, top - 10], [CX + 24, top + 9]].map(([x, y]) => `M${P([x - 1.8, y + 1])}q0 -2.6 2.2 -2.6q2 0 2 2`),
        side: Math.sqrt(r * r - (ey - cy) ** 2) + 2.8, crown: cy - r - 2.8,
      }
    }
  }
}

// ---------------------------------------------------------------- the outfit

function Outfit({ look, g, color, skin, sw }: { look: Look; g: G; color: string; skin: string; sw: number }) {
  const k = g.young ? 0.9 : 1
  const X = (dx: number) => CX + dx * k
  const torso = `M${P([X(-56), 121])}C${P([X(-55), 107])} ${P([X(-44), 100.5])} ${P([X(-13), 97.5])}Q${P([CX, 95.8])} ${P([X(13), 97.5])}C${P([X(44), 100.5])} ${P([X(55), 107])} ${P([X(56), 121])}Z`
  const o = look.outfit
  const nw = g.young ? 6.4 : 7.2
  const vNeck = o === 'scrubs'
  const low = o === 'workshirt' || o === 'collar' ? 101.5 : o === 'hoodie' ? 102 : 103.5
  // the skin the neckline leaves open: a V for scrubs, a curve to `low` for the rest
  const neck = `M${P([CX - nw + 0.4, g.chin - 12])}L${P([CX - nw, 96.6])}${vNeck ? `L${P([CX, 108.5])}L` : `Q${P([CX, 2 * low - 96.6])} `}${P([CX + nw, 96.6])}L${P([CX + nw - 0.4, g.chin - 12])}Z`
  const thin = sw * 0.62
  const under: React.ReactNode[] = []
  const over: React.ReactNode[] = []
  const crew = (
    <g key="crew">
      <path d={`M${P([CX - nw - 1.6, 96.4])}Q${P([CX, 105.4])} ${P([CX + nw + 1.6, 96.4])}`} fill="none" />
      <path d={`M${P([CX - nw - 3, 97.2])}Q${P([CX, 108.2])} ${P([CX + nw + 3, 97.2])}`} fill="none" strokeWidth={thin} />
    </g>
  )
  switch (o) {
    case 'scrubs':
      over.push(
        <path key="v" d={`M${P([CX - nw - 2.6, 96.3])}L${P([CX, 111.5])}L${P([CX + nw + 2.6, 96.3])}`} fill="none" />,
        <path key="pocket" d={`M${P([X(-31), 121])}L${P([X(-31), 107.5])}L${P([X(-18.5), 107.5])}L${P([X(-18.5), 121])}`} fill="none" strokeWidth={thin} />,
        <path key="pen" d={`M${P([X(-22), 108])}L${P([X(-22), 102.5])}`} strokeWidth={sw * 1.1} />,
        <path key="pen2" d={`M${P([X(-26.5), 108])}L${P([X(-26.5), 103.5])}`} stroke="#f1913a" strokeWidth={sw * 0.95} />,
      )
      break
    case 'apron':
      over.push(
        crew,
        <path key="bib" d={`M${P([X(-17), 107.5])}L${P([X(17), 107.5])}L${P([X(19.5), 122])}L${P([X(-19.5), 122])}Z`} fill={CREAM} />,
        <path key="sl" d={`M${P([X(-17), 107.5])}L${P([X(-12.6), 107.5])}L${P([CX - nw + 0.6, 97.4])}L${P([CX - nw - 3.4, 98])}Z`} fill={CREAM} />,
        <path key="sr" d={`M${P([X(17), 107.5])}L${P([X(12.6), 107.5])}L${P([CX + nw - 0.6, 97.4])}L${P([CX + nw + 3.4, 98])}Z`} fill={CREAM} />,
        <path key="pk" d={`M${P([X(-7), 114.5])}L${P([X(7), 114.5])}`} fill="none" strokeWidth={thin} />,
      )
      break
    case 'cardigan':
      under.push(<path key="tee" d={`M${P([X(-10.5), 96.6])}L${P([X(-1.8), 122])}L${P([X(1.8), 122])}L${P([X(10.5), 96.6])}Z`} fill={CREAM} stroke="none" />)
      over.push(
        <path key="tee-n" d={`M${P([CX - nw - 1.2, 96.4])}Q${P([CX, 105])} ${P([CX + nw + 1.2, 96.4])}`} fill="none" strokeWidth={thin} />,
        <path key="el" d={`M${P([X(-10.8), 96.4])}L${P([X(-1.6), 122])}`} fill="none" />,
        <path key="er" d={`M${P([X(10.8), 96.4])}L${P([X(1.6), 122])}`} fill="none" />,
        <circle key="b1" cx={X(-7.2)} cy={110} r={1.25} fill={INK} stroke="none" />,
        <circle key="b2" cx={X(-5.2)} cy={116.5} r={1.25} fill={INK} stroke="none" />,
        <path key="ly" d={`M${P([CX - nw + 0.5, 98])}L${P([X(1.6), 111.5])}M${P([CX + nw - 0.5, 98])}L${P([X(5.6), 111.5])}`} fill="none" strokeWidth={thin} />,
        <rect key="card" x={X(-1.4)} y={111.2} width={10} height={11} rx={1.4} fill="#ffffff" strokeWidth={thin} />,
        <rect key="stripe" x={X(-1.4)} y={111.2} width={10} height={3} rx={1.2} fill="#f1913a" strokeWidth={thin} />,
      )
      break
    case 'vest': {
      const panel = (s: number) => `M${P([X(-30 * s), 99.6])}Q${P([X(-19 * s), 97.4])} ${P([X(-10.8 * s), 97.6])}L${P([X(-4.6 * s), 122])}L${P([X(-36 * s), 122])}Z`
      const stripe = (s: number) => `M${P([X(-34.6 * s), 111])}L${P([X(-7.2 * s), 111])}L${P([X(-6.2 * s), 115.4])}L${P([X(-35.2 * s), 115.4])}Z`
      over.push(
        crew,
        <path key="vl" d={panel(1)} fill="#dfe65b" />,
        <path key="vr" d={panel(-1)} fill="#dfe65b" />,
        <path key="sl" d={stripe(1)} fill="#e9e7df" strokeWidth={thin} />,
        <path key="sr" d={stripe(-1)} fill="#e9e7df" strokeWidth={thin} />,
      )
      break
    }
    case 'workshirt':
      over.push(
        <path key="cl" d={`M${P([CX - nw - 1, 95.6])}L${P([CX - nw - 6.8, 103.8])}L${P([CX - 1.2, 102])}Z`} fill={color} />,
        <path key="cr" d={`M${P([CX + nw + 1, 95.6])}L${P([CX + nw + 6.8, 103.8])}L${P([CX + 1.2, 102])}Z`} fill={color} />,
        <path key="pl" d={`M${P([CX, 102.5])}L${P([CX, 122])}`} fill="none" strokeWidth={thin} />,
        <circle key="b1" cx={CX + 2.4} cy={108.5} r={1.15} fill={INK} stroke="none" />,
        <circle key="b2" cx={CX + 2.4} cy={115.5} r={1.15} fill={INK} stroke="none" />,
        <ellipse key="patch" cx={X(21)} cy={110.5} rx={8} ry={4.6} fill="#fbf8f0" strokeWidth={thin} />,
        <path key="name" d={`M${P([X(16.4), 111])}q1.15 -2.2 2.3 0t2.3 0t2.3 0t2.3 0`} fill="none" strokeWidth={sw * 0.5} />,
        <path key="pk" d={`M${P([X(-30), 107])}L${P([X(-18), 107])}L${P([X(-18.5), 117])}L${P([X(-29.5), 117])}Z`} fill="none" strokeWidth={thin} />,
      )
      break
    case 'collar':
      under.push(<path key="shirt" d={`M${P([X(-10.4), 96.4])}L${P([CX, 113])}L${P([X(10.4), 96.4])}Z`} fill="#fbf8f2" stroke="none" />)
      over.push(
        <path key="v" d={`M${P([X(-10.6), 96.2])}L${P([CX, 113.5])}L${P([X(10.6), 96.2])}`} fill="none" />,
        <path key="cl" d={`M${P([CX - nw - 0.4, 95.8])}L${P([CX - nw - 4.6, 104.6])}L${P([CX - 1.2, 101.6])}Z`} fill="#fbf8f2" strokeWidth={sw * 0.85} />,
        <path key="cr" d={`M${P([CX + nw + 0.4, 95.8])}L${P([CX + nw + 4.6, 104.6])}L${P([CX + 1.2, 101.6])}Z`} fill="#fbf8f2" strokeWidth={sw * 0.85} />,
      )
      break
    case 'hoodie':
      under.push(<path key="hood" d={`M${P([X(-20), 101.5])}C${P([X(-22), 91])} ${P([X(-12), 86])} ${P([CX, 86])}C${P([X(12), 86])} ${P([X(22), 91])} ${P([X(20), 101.5])}Z`} fill={mix(color, INK, 0.16)} />)
      over.push(
        <path key="op" d={`M${P([CX - nw - 3.4, 96.6])}Q${P([CX, 108])} ${P([CX + nw + 3.4, 96.6])}`} fill="none" />,
        <path key="dl" d={`M${P([CX - 4.4, 104.4])}Q${P([CX - 5.6, 110])} ${P([CX - 4.8, 116])}`} fill="none" strokeWidth={thin} />,
        <path key="dr" d={`M${P([CX + 4.4, 104.4])}Q${P([CX + 5.2, 109])} ${P([CX + 5.8, 113.4])}`} fill="none" strokeWidth={thin} />,
        <circle key="al" cx={CX - 4.8} cy={117} r={1.3} fill={CREAM} strokeWidth={sw * 0.5} />,
        <circle key="ar" cx={CX + 5.8} cy={114.4} r={1.3} fill={CREAM} strokeWidth={sw * 0.5} />,
      )
      break
    default:
      over.push(crew)
  }
  return (
    <g>
      <path d={torso} fill={color} />
      {under}
      <path d={neck} fill={skin} stroke="none" />
      <path d={`M${P([CX - nw + 0.4, g.chin - 9])}L${P([CX - nw, 97])}M${P([CX + nw - 0.4, g.chin - 9])}L${P([CX + nw, 97])}`} fill="none" />
      {over}
    </g>
  )
}

// ---------------------------------------------------------------- the face

function Features({ look, g, sw }: { look: Look; g: G; sw: number }) {
  const r = g.young ? 1.12 : 1
  const xs = [CX - g.es, CX + g.es]
  const by = g.ey - (g.young ? 7.4 : 8)
  const out: React.ReactNode[] = []
  // brows
  xs.forEach((x, i) => {
    const m = i === 0 ? -1 : 1
    const d =
      look.brows === 'flat' ? `M${P([x - 4.2 * m, by + 0.2])}L${P([x + 4.2 * m, by - 0.6])}`
      : look.brows === 'thick' ? `M${P([x - 3.3 * m, by + 0.6])}Q${P([x, by - 1.6])} ${P([x + 3.3 * m, by + 0.2])}`
      : `M${P([x - 4.4 * m, by + 1.4])}Q${P([x - 0.4 * m, by - 2.6])} ${P([x + 4.4 * m, by + 0.6])}`
    out.push(<path key={`b${i}`} d={d} fill="none" strokeWidth={look.brows === 'thick' ? sw * 1.7 : sw * 0.95} />)
  })
  // eyes
  xs.forEach((x, i) => {
    const m = i === 0 ? 1 : -1
    if (look.eyes === 'dot') out.push(<circle key={`e${i}`} cx={x} cy={g.ey} r={2.3 * r} fill={INK} stroke="none" />)
    else if (look.eyes === 'oval') {
      out.push(<ellipse key={`e${i}`} cx={x} cy={g.ey} rx={2 * r} ry={2.9 * r} fill={INK} stroke="none" />)
      out.push(<circle key={`h${i}`} cx={x + 0.7} cy={g.ey - 1.1} r={0.75} fill="#fff" stroke="none" />)
    } else if (look.eyes === 'arc') out.push(<path key={`e${i}`} d={`M${P([x - 3.4 * r, g.ey + 1])}Q${P([x, g.ey - 3.4 * r])} ${P([x + 3.4 * r, g.ey + 1])}`} fill="none" />)
    else {
      out.push(<circle key={`e${i}`} cx={x} cy={g.ey} r={3.5 * r} fill="#fff" strokeWidth={sw * 0.8} />)
      out.push(<circle key={`p${i}`} cx={x + 0.6 * m} cy={g.ey + 0.3} r={1.8 * r} fill={INK} stroke="none" />)
    }
  })
  // nose, drawn as one line the way a pen would
  const { ey, ny } = g
  const nose =
    look.nose === 'button' ? `M${P([CX - 3, ny - 0.8])}Q${P([CX, ny + 3])} ${P([CX + 3, ny - 0.8])}`
    : look.nose === 'hook' ? `M${P([CX - 0.5, ey + 4])}Q${P([CX + 4.8, ny - 1.5])} ${P([CX + 1.8, ny + 1.3])}Q${P([CX - 0.3, ny + 2.3])} ${P([CX - 1.8, ny + 0.6])}`
    : look.nose === 'bump' ? `M${P([CX + 1.4, ny - 5])}Q${P([CX - 4, ny - 0.6])} ${P([CX + 1.6, ny + 1.4])}`
    : `M${P([CX + 1, ey + 3.5])}Q${P([CX - 1.2, ny - 2])} ${P([CX - 1.4, ny + 0.6])}Q${P([CX + 0.2, ny + 2.2])} ${P([CX + 3.2, ny + 0.6])}`
  out.push(<path key="n" d={nose} fill="none" strokeWidth={sw * 0.9} />)
  // mouth
  const my = g.my
  const w = g.young ? 0.9 : 1
  if (look.mouth === 'grin')
    out.push(<path key="m" d={`M${P([CX - 7 * w, my - 1.3])}Q${P([CX, my - 0.2])} ${P([CX + 7 * w, my - 1.3])}Q${P([CX + 0.6, my + 9 * w])} ${P([CX - 7 * w, my - 1.3])}Z`} fill="#8a3a36" strokeWidth={sw * 0.9} />)
  else {
    const d =
      look.mouth === 'soft' ? `M${P([CX - 4.2 * w, my])}Q${P([CX, my + 3.4])} ${P([CX + 4.2 * w, my])}`
      : look.mouth === 'smirk' ? `M${P([CX - 5.5 * w, my + 0.8])}Q${P([CX + 1.5, my + 4.4])} ${P([CX + 6.2 * w, my - 1.4])}`
      : `M${P([CX - 6.5 * w, my - 0.6])}Q${P([CX, my + 5.6])} ${P([CX + 6.5 * w, my - 0.6])}`
    out.push(<path key="m" d={d} fill="none" />)
  }
  if (look.smileLines)
    for (const m of [-1, 1]) out.push(<path key={`sl${m}`} d={`M${P([CX + 10 * m, my - 4.4])}Q${P([CX + 11.8 * m, my - 0.4])} ${P([CX + 9.4 * m, my + 2.8])}`} fill="none" strokeWidth={sw * 0.55} />)
  return <g>{out}</g>
}

function Glasses({ look, g, sw }: { look: Look; g: G; sw: number }) {
  const xs = [CX - g.es, CX + g.es]
  const t = sw * 0.85
  const arm = (x: number, m: number) => <path key={`a${m}`} d={`M${P([x, g.ey - 1])}L${P([CX + (g.hw - 0.4) * m, g.ey - 2.6])}`} fill="none" strokeWidth={t} />
  if (look.glasses === 'round')
    return (
      <g>
        {xs.map((x) => <circle key={x} cx={x} cy={g.ey} r={5.7} fill="rgb(255 255 255 / 0.22)" strokeWidth={t} />)}
        <path d={`M${P([xs[0] + 5.5, g.ey - 1.2])}Q${P([CX, g.ey - 3.8])} ${P([xs[1] - 5.5, g.ey - 1.2])}`} fill="none" strokeWidth={t} />
        {arm(xs[0] - 5.7, -1)}
        {arm(xs[1] + 5.7, 1)}
      </g>
    )
  if (look.glasses === 'square')
    return (
      <g>
        {xs.map((x) => <rect key={x} x={x - 6.4} y={g.ey - 4.6} width={12.8} height={9.6} rx={2.6} fill="rgb(255 255 255 / 0.22)" strokeWidth={t} />)}
        <path d={`M${P([xs[0] + 6.4, g.ey - 1.6])}Q${P([CX, g.ey - 3.6])} ${P([xs[1] - 6.4, g.ey - 1.6])}`} fill="none" strokeWidth={t} />
        {arm(xs[0] - 6.4, -1)}
        {arm(xs[1] + 6.4, 1)}
      </g>
    )
  // reading glasses: half-moons low on the nose
  const y = g.ey + 2.6
  return (
    <g>
      {xs.map((x) => <path key={x} d={`M${P([x - 6, y])}L${P([x + 6, y])}Q${P([x + 5.6, y + 5.6])} ${P([x, y + 5.6])}Q${P([x - 5.6, y + 5.6])} ${P([x - 6, y])}Z`} fill="rgb(255 255 255 / 0.22)" strokeWidth={t} />)}
      <path d={`M${P([xs[0] + 6, y + 0.4])}Q${P([CX, y - 1.6])} ${P([xs[1] - 6, y + 0.4])}`} fill="none" strokeWidth={t} />
      <path d={`M${P([xs[0] - 6, y])}L${P([CX - g.hw + 0.4, g.ey - 1])}M${P([xs[1] + 6, y])}L${P([CX + g.hw - 0.4, g.ey - 1])}`} fill="none" strokeWidth={t} />
    </g>
  )
}

// ---------------------------------------------------------------- the portrait

/**
 * The resident drawn in ink. `color` dresses them (their resident colour); `block` is the colour printed off-register
 * behind the head (a tint of theirs unless the caller has a better one, e.g. paper on a coloured card).
 */
export function Avatar({ look, color, size = 120, block, label }: { look: Look; color: number; size?: number; block?: string; label?: string }) {
  const uid = useId().replace(/[^a-zA-Z0-9_-]/g, '')
  const g = geom(look)
  const h = hairArt(look.hair, g)
  const outfit = cssHex(color)
  // the pen's weight: 2.2 at 120 px and in proportion above; thumbnails keep a line that still reads
  const sw = 2.2 * Math.max(1, (72 / size) ** 0.75)
  const jitter = rng(`pen:${look.seed}`)
  const blob: Pt[] = Array.from({ length: 9 }, (_, i) => {
    const a = (i / 9) * Math.PI * 2
    const rr = 37 + (jitter() - 0.5) * 3.4
    return [CX - 5 + Math.cos(a) * rr, g.ey - 8 + Math.sin(a) * rr * 0.96]
  })
  const face = curve(facePts(g).map(([x, y]) => [x + (jitter() - 0.5) * 0.7, y + (jitter() - 0.5) * 0.7] as Pt), true)
  const hairFill = h.buzz ? mix(look.hairColor, look.skin, 0.2) : look.hairColor
  const showEarring = look.earring && !['long', 'bob', 'wavy'].includes(look.hair)
  const ear = (m: number) =>
    `M${P([CX + (g.hw - 1.4) * m, g.ey - 3.8])}C${P([CX + (g.hw + 6.4) * m, g.ey - 7])} ${P([CX + (g.hw + 7) * m, g.ey + 7.5])} ${P([CX + (g.hw - 1.4) * m, g.ey + 7])}`
  const hs = h.side
  const y0 = g.ey - 1
  const yc = (4 * (h.crown - 1.6) - y0) / 3
  const blush = mix(look.skin, '#e2626b', 0.32)
  return (
    // a thumbnail closes in on the head, the way a small profile photo does
    <svg className="avatar" viewBox={size < 48 ? '16 8 88 88' : '0 0 120 120'} width={size} height={size} role={label ? 'img' : undefined} aria-label={label} aria-hidden={label ? undefined : true}>
      <defs>
        <filter id={`w${uid}`} x="-8%" y="-8%" width="116%" height="116%">
          <feTurbulence type="fractalNoise" baseFrequency="0.028" numOctaves={1} seed={look.seed} />
          <feDisplacementMap in="SourceGraphic" scale={2.6} xChannelSelector="R" yChannelSelector="G" />
        </filter>
        <clipPath id={`h${uid}`}>
          <path d={h.front} />
        </clipPath>
      </defs>
      {/* a thumbnail is too small to show the wobble: it skips the filter, so a long list of faces stays cheap */}
      <g filter={size >= 48 ? `url(#w${uid})` : undefined} transform={look.flip ? `matrix(-1 0 0 1 ${2 * CX} 0)` : undefined}
        stroke={INK} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round">
        <path d={curve(blob, true)} fill={block ?? mix(outfit, '#ffffff', 0.62)} stroke="none" />
        {h.back && <path d={h.back} fill={hairFill} />}
        <Outfit look={look} g={g} color={outfit} skin={look.skin} sw={sw} />
        <path d={ear(-1)} fill={look.skin} />
        <path d={ear(1)} fill={look.skin} />
        <path d={`M${P([CX + g.hw + 1.4, g.ey - 0.6])}Q${P([CX + g.hw + 3.8, g.ey + 1.6])} ${P([CX + g.hw + 1.2, g.ey + 3.6])}`} fill="none" strokeWidth={sw * 0.6} />
        {showEarring && (
          <>
            <circle cx={CX + g.hw + 2.6} cy={g.ey + 9.6} r={2.4} fill="none" strokeWidth={sw * 0.62} />
            <circle cx={CX + g.hw + 2.4} cy={g.ey + 7.4} r={1.2} fill="#f6cd4c" strokeWidth={sw * 0.4} />
          </>
        )}
        <path d={face} fill={look.skin} />
        {[-1, 1].map((m) => <ellipse key={m} cx={CX + (g.es + 1.8) * m} cy={g.ey + 7.6} rx={3.8} ry={2.2} fill={blush} opacity={0.55} stroke="none" />)}
        {look.freckles && [-1, 1].flatMap((m) => [[-2.2, 0], [0.6, 1.3], [2.8, -0.3]].map(([dx, dy], i) => (
          <circle key={`${m}${i}`} cx={CX + (g.es + dx) * m} cy={g.ey + 5.6 + dy} r={0.8} fill={mix(look.skin, '#3a1c10', 0.5)} stroke="none" />
        )))}
        <Features look={look} g={g} sw={sw} />
        <path d={h.front} fill={hairFill} stroke={h.frontLine ? 'none' : undefined} />
        {look.streak && (
          <g clipPath={`url(#h${uid})`}>
            <path d={`M${P([CX + 4, g.top + 14])}Q${P([CX + 8, g.top + 1])} ${P([CX + 20, g.top - 5])}`} fill="none" stroke="#dcd8cf" strokeWidth={6.5} />
          </g>
        )}
        {look.streak && !h.frontLine && <path d={h.front} fill="none" />}
        {h.frontLine && <path d={h.frontLine} fill="none" />}
        {h.buzz
          ? h.strands?.map((d) => <path key={d} d={d} fill="none" strokeWidth={sw * 0.7} strokeDasharray={`0.1 ${n1(sw * 1.5)}`} />)
          : h.strands?.map((d) => <path key={d} d={d} fill="none" strokeWidth={sw * 0.62} />)}
        {look.clip && (
          <rect x={CX - g.tw * 0.72 - 4.6} y={g.top + 8.6} width={9.2} height={3.6} rx={1.8} fill={look.clip} strokeWidth={sw * 0.62}
            transform={`rotate(-28 ${n1(CX - g.tw * 0.72)} ${n1(g.top + 10.4)})`} />
        )}
        {look.glasses !== 'none' && <Glasses look={look} g={g} sw={sw} />}
        {look.outfit === 'headset' && (
          <g>
            <path d={`M${P([CX - hs, y0])}C${P([CX - hs, yc])} ${P([CX + hs, yc])} ${P([CX + hs, y0])}`} fill="none" strokeWidth={sw * 1.5} />
            <rect x={CX - hs - 4} y={g.ey - 5} width={7.4} height={12} rx={3.2} fill="#33363b" />
            <rect x={CX + hs - 3.4} y={g.ey - 5} width={7.4} height={12} rx={3.2} fill="#33363b" />
            <path d={`M${P([CX + hs, g.ey + 6])}Q${P([CX + hs - 3, g.my + 6])} ${P([CX + 9, g.my + 2.4])}`} fill="none" strokeWidth={sw * 0.8} />
            <circle cx={CX + 8.4} cy={g.my + 2.4} r={2.3} fill={INK} stroke="none" />
          </g>
        )}
      </g>
    </svg>
  )
}
