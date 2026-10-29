// The building kit and the land. Every place is drawn from its archetype, varied by a seed from its own id, so the
// same place always looks the same and a new place never changes an old one.
import * as THREE from 'three'
import { FRONT, SIDE, W } from './iso'
import { Part } from './part'
import {
  ACCENT, BRICK, EVENT, FOLIAGE, GLASS_DARK, GLASS_LIT, GROUND, ROOF, STONE, TRUNK, WALL, WHITE, districtGround, pick,
} from './palette'
import type { Lot, Plan } from './plan'
import { hash, rng } from './rand'


// ---------------------------------------------------------------- shared detail

/** Window quads over the two faces the camera sees: the front (y = max) and the side (x = max). */
function windows(p: Part, x: number, y: number, z: number, w: number, d: number, h: number, seed: string, o: { floor?: number; ww?: number; wh?: number; step?: number; skipGround?: boolean } = {}) {
  const { floor = 3, ww = 1.2, wh = 1.4, step = 3, skipGround = false } = o
  const r = rng(`win:${seed}`)
  const floors = Math.max(1, Math.floor(h / floor))
  const lit = () => (r() < 0.62 ? GLASS_LIT : GLASS_DARK)
  const F = FRONT(x, y + d, z + h)
  const S = SIDE(x + w, y + d, z + h)
  for (let f = skipGround ? 1 : 0; f < floors; f++) {
    const v = h - (f + 1) * floor + (floor - wh) / 2
    for (let u = (w % step) / 2 + (step - ww) / 2; u + ww <= w - 0.3; u += step) p.quad(F, u, v, ww, wh, lit(), { glow: true })
    for (let u = (d % step) / 2 + (step - ww) / 2; u + ww <= d - 0.3; u += step) p.quad(S, u, v, ww, wh, lit(), { glow: true })
  }
}

function door(p: Part, x: number, y: number, z: number, w: number, d: number, h: number, at = 0.5, hex = 0x6b5444) {
  const F = FRONT(x, y + d, z + h)
  p.quad(F, w * at - 0.6, h - 2.2, 1.2, 2.2, hex, { outline: 'ink' })
}

/** A striped awning sloping out of the front face. */
function awning(p: Part, x0: number, x1: number, y: number, z: number, hex: number) {
  const a = W(x0, y, z)
  const b = W(x1, y, z)
  const c = W(x1, y + 1.4, z - 0.9)
  const e = W(x0, y + 1.4, z - 0.9)
  p.poly([a, b, c, e], hex)
  p.seg('ink', a, b).seg('ink', b, c).seg('ink', c, e).seg('ink', e, a)
  const n = Math.max(2, Math.round((x1 - x0) / 0.9))
  for (let i = 1; i < n; i++) {
    const t = x0 + ((x1 - x0) * i) / n
    p.seg('detail', W(t, y, z), W(t, y + 1.4, z - 0.9))
  }
}

/** A street lamp: a post and a head that lights after dusk. */
export function lamp(p: Part, x: number, y: number) {
  p.seg('ink', W(x, y, 0), W(x, y, 4.4))
  const q = (pts: [number, number, number][]) => p.poly(pts.map((v) => W(...v)), GLASS_LIT, true, 'glow')
  const [a, b, h] = [x - 0.3, x + 0.3, 4.4]
  q([[a, y - 0.3, h], [b, y - 0.3, h], [b, y + 0.3, h], [a, y + 0.3, h]])
  q([[a, y + 0.3, h], [b, y + 0.3, h], [b, y + 0.3, h - 0.5], [a, y + 0.3, h - 0.5]])
  q([[b, y - 0.3, h], [b, y + 0.3, h], [b, y + 0.3, h - 0.5], [b, y - 0.3, h - 0.5]])
}

export function tree(p: Part, x: number, y: number, seed: number, scale = 1) {
  const h = (2.2 + (seed % 5) * 0.35) * scale
  const r = (1.5 + ((seed >>> 3) % 4) * 0.25) * scale
  p.cylZ(x, y, 0, 0.18 * scale, h, TRUNK, 5, { lines: false })
  const g = new THREE.IcosahedronGeometry(r, 0)
  g.translate(x, h + r * 0.7, y)
  geo(p, g, pick(FOLIAGE, seed))
}

export function pine(p: Part, x: number, y: number, seed: number, scale = 1) {
  const h = (5 + (seed % 4)) * scale
  const g = new THREE.ConeGeometry(1.6 * scale, h, 6)
  g.translate(x, h / 2 + 0.8, y)
  p.cylZ(x, y, 0, 0.16 * scale, 1, TRUNK, 4, { lines: false })
  geo(p, g, pick([0x6f9a74, 0x7aa47c, 0x5f8a6a], seed))
}

/** Any three.js geometry, toon-toned by facet, every facet edge a hairline. */
export function geo(p: Part, g: THREE.BufferGeometry, hex: number) {
  const ng = g.index ? g.toNonIndexed() : g
  const pos = ng.attributes.position
  const a = new THREE.Vector3()
  const b = new THREE.Vector3()
  const c = new THREE.Vector3()
  for (let i = 0; i < pos.count; i += 3) {
    a.fromBufferAttribute(pos, i)
    b.fromBufferAttribute(pos, i + 1)
    c.fromBufferAttribute(pos, i + 2)
    p.poly([a.clone(), b.clone(), c.clone()], hex)
  }
  const e = new THREE.EdgesGeometry(ng, 20).attributes.position
  for (let i = 0; i < e.count; i += 2) p.seg('ink', a.fromBufferAttribute(e, i).clone(), b.fromBufferAttribute(e, i + 1).clone())
}

// ---------------------------------------------------------------- the kit, one builder per archetype

type Build = (p: Part, l: Lot, s: number, id: string) => number // returns the building's height, for tags

const house: Build = (p, l, s, id) => {
  const x = l.x + 1
  const y = l.y + 1.5
  const w = l.w - 2
  const d = l.d - 3.5
  p.rectGround(l.x, l.y, l.w, l.d, 0.02, 0xd6e2c0)
  const h = s % 3 === 0 ? 5.8 : 3.4
  p.box(x, y, 0, w, d, h, pick(WALL, s))
  p.gable(x, y, h, w, d, 2.6, s % 2 ? 'x' : 'y', pick(ROOF, s, 1))
  windows(p, x, y, 0, w, d, h, id, { floor: h > 4 ? 2.9 : 3.4, ww: 1.1, wh: 1.2, step: 3.2 })
  door(p, x, y, 0, w, d, Math.min(h, 3.4), 0.3)
  if (s % 4 === 1) p.box(x + w - 2.2, y + 1, h, 0.9, 0.9, 2.6, BRICK) // a chimney
  tree(p, l.x + l.w - 1.2, l.y + l.d - 0.8, s, 0.75)
  return h + 2.6
}

const flats: Build = (p, l, s, id) => {
  const floors = 3 + (s % 3)
  const h = floors * 3
  const x = l.x + 1
  const y = l.y + 1
  const w = l.w - 2
  const d = l.d - 2
  const wall = pick([0xf1ebdf, 0xe9e0d0, 0xe6e3da, 0xefe1cf], s)
  p.box(x, y, 0, w, d, h, wall)
  p.box(x - 0.2, y - 0.2, h, w + 0.4, d + 0.4, 0.6, 0xd9d1c2) // parapet
  windows(p, x, y, 0, w, d, h, id, { floor: 3, step: 3.4 })
  const F = FRONT(x, y + d, h)
  const acc = pick(ACCENT, s, 2)
  for (let f = 1; f < floors; f++) for (let u = 1.2; u + 2.6 < w; u += 6.8) p.quad(F, u, f * 3 - 0.5, 2.6, 0.5, acc, { outline: 'detail' }) // balconies
  p.box(x + w * 0.6, y + d * 0.3, h + 0.6, 4, 3, 2.2, 0xd4ccbd)
  p.cylZ(x + w * 0.25, y + d * 0.4, h + 0.6, 1.1, 2, 0xc8c0b0, 10)
  door(p, x, y, 0, w, d, 3, 0.5)
  return h + 3
}

const shop: Build = (p, l, s, id) => {
  const two = s % 3 !== 0
  const h = two ? 6.2 : 4
  const x = l.x + 0.5
  const y = l.y + 0.5
  const w = l.w - 1
  const d = l.d - 2
  const wall = pick(WALL, s, 1)
  const acc = pick(ACCENT, s)
  p.rectGround(l.x, l.y + l.d - 2, l.w, 2, 0.03, 0xe8dfcf)
  p.box(x, y, 0, w, d, h, wall)
  p.box(x, y + d - 0.1, h, w, 0.25, 0.9, wall) // a false front
  const F = FRONT(x, y + d, h)
  p.quad(F, 0.8, h - 3.2, w - 1.6, 2.3, GLASS_LIT, { glow: true, outline: 'ink' })
  p.quad(F, 0.4, two ? 1 : 0.15, w - 0.8, 0.8, acc, { outline: 'detail' }) // the sign band
  awning(p, x + 0.6, x + w - 0.6, y + d, h - 0.6 - (two ? 3 : 0), acc)
  if (two) windows(p, x, y, 3.2, w, d, 3, id, { floor: 3, step: 3 })
  const S = SIDE(x + w, y + d, h)
  p.quad(S, 1, h - 3.2, Math.max(1, d - 3), 2.2, GLASS_LIT, { glow: true, outline: 'ink' })
  return h + 0.9
}

const arcade: Build = (p, l, s) => {
  const n = 3
  const pw = (l.w - 1) / n
  let top = 0
  for (let i = 0; i < n; i++) {
    const sub: Lot = { ...l, x: l.x + 0.5 + i * pw, w: pw }
    top = Math.max(top, shop(p, sub, hash(`${s}:${i}`), `${l.place.id}:${i}`))
  }
  return top
}

const mall: Build = (p, l, s, id) => {
  const h = 5
  const x = l.x + 0.5
  const y = l.y + 0.5
  const w = l.w - 1
  const d = l.d * 0.55
  p.rectGround(l.x, y + d, l.w, l.d - d - 0.5, 0.03, 0xd9d3c6, 'soft') // parking
  for (let u = l.x + 2; u < l.x + l.w - 2; u += 2.8) p.seg('soft', W(u, y + d + 1, 0.05), W(u, y + d + 3.5, 0.05))
  p.box(x, y, 0, w, d, h, 0xebe3d3)
  const F = FRONT(x, y + d, h)
  const units = Math.max(3, Math.round(w / 9))
  for (let i = 0; i < units; i++) {
    const uw = w / units
    p.quad(F, i * uw + 0.6, 1.6, uw - 1.2, 2.8, GLASS_LIT, { glow: true, outline: 'ink' })
    p.quad(F, i * uw + 0.6, 0.3, uw - 1.2, 0.8, pick(ACCENT, hash(`${id}:${i}`)), { outline: 'detail' })
  }
  p.box(x, y + d, h - 0.8, w, 1.6, 0.4, 0xd8cfbf)
  void s
  return h
}

const school: Build = (p, l, s, id) => {
  const bd = Math.min(15, l.d * 0.52)
  const floors = 2 + (s % 2)
  const h = floors * 3.4
  const x = l.x + 0.5
  const y = l.y + 0.5
  const w = l.w - 1
  // the playing field in front: a track round a green
  const fy = y + bd + 1
  const fd = l.d - bd - 2
  p.rectGround(l.x + 1, fy, l.w - 2, fd, 0.03, 0xc2a07f)
  p.rectGround(l.x + 2.4, fy + 1.4, l.w - 4.8, fd - 2.8, 0.05, 0xb5cf95, 'soft')
  p.seg('soft', W(l.x + l.w / 2, fy + 1.4, 0.07), W(l.x + l.w / 2, fy + fd - 1.4, 0.07))
  p.box(x, y, 0, w, bd, h, BRICK, { top: 0xd8c6b2 })
  p.box(x - 0.1, y - 0.1, h, w + 0.2, bd + 0.2, 0.5, WHITE)
  const F = FRONT(x, y + bd, h)
  p.quad(F, 0, h - 3.4 * 1 + 0.2, w, 0.25, WHITE)
  windows(p, x, y, 0, w, bd, h, id, { floor: 3.4, ww: 1.8, wh: 1.6, step: 3 })
  // the entrance block and its flagpole
  p.box(x + w / 2 - 3, y + bd - 0.2, 0, 6, 2, 4.4, WHITE)
  p.seg('ink', W(x + w / 2 + 5, y + bd + 1, 0), W(x + w / 2 + 5, y + bd + 1, 9))
  p.poly([W(x + w / 2 + 5, y + bd + 1, 9), W(x + w / 2 + 7.2, y + bd + 1, 8.3), W(x + w / 2 + 5, y + bd + 1, 7.6)], pick(ACCENT, s), true)
  return h + 0.5
}

const campus: Build = (p, l, s, id) => {
  const x = l.x + 0.5
  const y = l.y + 0.5
  p.rectGround(l.x, l.y, l.w, l.d, 0.02, 0xd4e2bd)
  // the quad's cross paths
  p.rectGround(l.x + l.w / 2 - 1, l.y, 2, l.d, 0.04, GROUND.path)
  p.rectGround(l.x, l.y + l.d * 0.62, l.w, 2, 0.04, GROUND.path)
  const hw = l.w * 0.55
  const hd = l.d * 0.4
  const h = 10
  p.box(x, y, 0, hw, hd, h, STONE, { top: 0x9fb39a })
  windows(p, x, y, 0, hw, hd, h, id, { floor: 3.3, ww: 1, wh: 2, step: 2.6 })
  const F = FRONT(x, y + hd, h)
  for (let u = hw * 0.3; u < hw * 0.7; u += 1.6) p.draw(F, [u, 2, u, h], 'ink') // the portico's columns
  p.gable(x + hw * 0.28, y + hd - 1.5, h, hw * 0.44, 3, 2.2, 'y', STONE, 0.2)
  // the clock tower
  p.box(x + hw + 2, y + 2, 0, 5, 5, 20, STONE)
  p.pyramid(x + hw + 1.8, y + 1.8, 20, 5.4, 5.4, 4, 0x7e8f87)
  const C = FRONT(x + hw + 2, y + 7, 20)
  p.quad(C, 1.3, 2, 2.4, 2.4, WHITE, { outline: 'ink' })
  // a newer glass hall
  const gx = x + hw + 9
  const gw = Math.max(8, l.w - hw - 11)
  p.box(gx, y + 1, 0, gw, l.d * 0.45, 13, 0xcfdfe4, { top: 0xe5ecee })
  windows(p, gx, y + 1, 0, gw, l.d * 0.45, 13, `${id}:g`, { floor: 2.6, ww: 2, wh: 2.1, step: 2.4 })
  for (let i = 0; i < 5; i++) tree(p, l.x + 3 + i * (l.w / 5), l.y + l.d - 2, hash(`${id}:t${i}`), 0.9)
  void s
  return 24
}

const hospital: Build = (p, l, s, id) => {
  const x = l.x + 0.5
  const y = l.y + 0.5
  const w = l.w - 1
  const d = l.d - 1
  const h = 15
  p.box(x, y, 0, w * 0.62, d, h, WHITE, { top: 0xe6e6e0 })
  p.box(x + w * 0.62, y + d * 0.35, 0, w * 0.38, d * 0.65, 7, 0xe9ecea)
  windows(p, x, y, 0, w * 0.62, d, h, id, { floor: 3, ww: 2.2, wh: 1.3, step: 3 })
  windows(p, x + w * 0.62, y + d * 0.35, 0, w * 0.38, d * 0.65, 7, `${id}:w`, { floor: 3.4, ww: 2.2, wh: 1.4, step: 3.2 })
  // a red cross high on the side, and the helipad on the roof
  const S = SIDE(x + w * 0.62, y + d, h)
  p.quad(S, d * 0.5 - 0.6, 1, 1.2, 3.6, 0xd2463c).quad(S, d * 0.5 - 1.8, 2.2, 3.6, 1.2, 0xd2463c)
  const T = new THREE.Matrix4().makeBasis(W(1, 0, 0), W(0, 1, 0), W(0, 0, 1)).setPosition(W(x + w * 0.31, y + d * 0.5, h + 0.06))
  const ring: number[] = []
  for (let i = 0; i < 24; i++) {
    const a0 = (i / 24) * Math.PI * 2
    const a1 = ((i + 1) / 24) * Math.PI * 2
    ring.push(Math.cos(a0) * 5, Math.sin(a0) * 5, Math.cos(a1) * 5, Math.sin(a1) * 5)
  }
  p.draw(T, [...ring, -1.4, -2, -1.4, 2, 1.4, -2, 1.4, 2, -1.4, 0, 1.4, 0], 'ink', 0)
  void s
  return h
}

const library: Build = (p, l, s, id) => {
  const x = l.x + 1
  const y = l.y + 1
  const w = l.w - 2
  const d = l.d - 3
  const h = 8
  const wall = pick([0x9fc1bd, 0xb9cfc4, 0xc9d4cf], s)
  p.box(x, y, 0, w, d, h, wall, { top: 0xdfe3d8 })
  p.box(x - 0.3, y + d, 0, w + 0.6, 1.6, 0.5, 0xd8d0c2).box(x + 0.3, y + d + 0.6, 0, w - 0.6, 1, 0.25, 0xd8d0c2) // steps
  const F = FRONT(x, y + d, h)
  for (let u = 1.6; u < w - 1; u += 2.4) p.quad(F, u, 1.4, 1.2, h - 2.4, GLASS_LIT, { glow: true, outline: 'ink' })
  p.quad(F, 0, 0, w, 0.9, WHITE, { outline: 'ink' })
  windows(p, x, y, 0, w, d, h, id, { floor: 4, ww: 1.4, wh: 2.4, step: 3 })
  return h
}

const hall: Build = (p, l, s, id) => {
  const x = l.x + 1
  const y = l.y + 1
  const w = l.w - 2
  const d = l.d - 3
  const h = 5.5
  p.box(x, y, 0, w, d, h, pick([0xe7c98e, 0xe8cfa6, 0xdcbf94], s))
  p.gable(x, y, h, w, d, 2, 'x', pick(ROOF, s))
  windows(p, x, y, 0, w, d, h, id, { floor: 5.5, ww: 2, wh: 2, step: 4 })
  door(p, x, y, 0, w, d, h, 0.5)
  // a banner
  p.quad(FRONT(x, y + d, h), w * 0.15, 0.6, 1.4, 3, pick(ACCENT, s, 1), { outline: 'detail' })
  return h + 2
}

const church: Build = (p, l, s, id) => {
  const x = l.x + 2
  const y = l.y + 5
  const w = l.w - 4
  const d = l.d - 6
  const h = 7
  p.box(x, y, 0, w, d, h, WHITE)
  p.gable(x, y, h, w, d, 5, 'y', pick([0x6f7f8c, 0x5f6e66, 0x8a5d4e], s))
  // the tower over the door, with its spire
  const tx = x + w / 2 - 2
  const ty = y + d - 1.5
  p.box(tx, ty, 0, 4, 4, 14, WHITE)
  p.pyramid(tx - 0.2, ty - 0.2, 14, 4.4, 4.4, 6, pick([0x6f7f8c, 0x5f6e66], s))
  p.quad(FRONT(tx, ty + 4, 14), 1.4, 1.2, 1.2, 2, GLASS_DARK, { outline: 'ink' })
  door(p, tx, ty, 0, 4, 4, 14 > 3 ? 3 : 14, 0.5, 0x5a3f33)
  const S = SIDE(x + w, y + d, h)
  for (let u = 2; u < d - 2; u += 3.4) p.quad(S, u, 1.6, 1.1, 3.2, GLASS_LIT, { glow: true, outline: 'ink' })
  void id
  return 20
}

const temple: Build = (p, l, s, id) => {
  const x = l.x + 2
  const y = l.y + 2
  const w = l.w - 4
  const d = l.d - 4
  const red = pick([0xb8483a, 0xa9403a, 0xc0583e], s)
  p.box(x - 1, y - 1, 0, w + 2, d + 2, 1, 0xdcd2c0) // the plinth
  p.box(x + 1, y + 1, 1, w - 2, d - 2, 4.5, 0xf0e2c8)
  p.pyramid(x - 0.6, y - 0.6, 5.5, w + 1.2, d + 1.2, 2.6, red)
  p.box(x + w * 0.25, y + d * 0.25, 6.6, w * 0.5, d * 0.5, 3, 0xf0e2c8)
  p.pyramid(x + w * 0.2, y + d * 0.2, 9.6, w * 0.6, d * 0.6, 2.8, red)
  p.seg('ink', W(x + w / 2, y + d / 2, 12.4), W(x + w / 2, y + d / 2, 14.4))
  door(p, x + 1, y + 1, 1, w - 2, d - 2, 4.5, 0.5, red)
  void id
  return 14
}

const theater: Build = (p, l, s, id) => {
  const x = l.x + 1
  const y = l.y + 1
  const w = l.w - 2
  const d = l.d - 3
  const h = 10
  const plum = pick([0x7a4a6a, 0x6a4a7a, 0x84506a], s)
  p.box(x, y, 0, w, d, h, plum, { top: 0xb79aae })
  const F = FRONT(x, y + d, h)
  p.quad(F, 1, h - 4.4, w - 2, 1.2, GLASS_LIT, { glow: true, outline: 'ink' }) // the marquee
  p.box(x + 1, y + d, 4.4, w - 2, 1.4, 0.4, 0xe7d3c3)
  p.box(x + w - 2.4, y + d + 0.2, 5, 1, 1, 8, EVENT) // the blade sign
  p.quad(F, w * 0.3, h - 2.6, w * 0.4, 2.6, 0x3e2b38, { outline: 'ink' })
  windows(p, x, y, 0, w, d, h, id, { floor: 5, ww: 1.2, wh: 2, step: 4, skipGround: true })
  return h + 3
}

const gym: Build = (p, l, s, id) => {
  const x = l.x + 1
  const y = l.y + 1
  const w = l.w - 2
  const d = l.d - 3
  const h = 7
  p.box(x, y, 0, w, d, h, pick([0x9fb0c4, 0xa9b8c9, 0x93a7b8], s))
  p.gable(x, y, h, w, d, 1.6, 'y', 0xdfe4e8)
  const F = FRONT(x, y + d, h)
  p.quad(F, 1, 1, w - 2, h - 2.4, GLASS_LIT, { glow: true, outline: 'ink' })
  for (let u = 1 + (w - 2) / 4; u < w - 1.5; u += (w - 2) / 4) p.draw(F, [u, 1, u, h - 1.4], 'ink')
  void id
  return h + 1.6
}

const warehouse: Build = (p, l, s, id) => {
  const x = l.x + 0.5
  const y = l.y + 0.5
  const w = l.w - 1
  const d = l.d - 6
  const h = 10
  p.rectGround(l.x, y + d, l.w, 5.5, 0.03, 0xd2cdc3, 'soft') // the yard
  p.box(x, y, 0, w, d, h, 0xdedbd3, { top: 0xe9e7e1 })
  const F = FRONT(x, y + d, h)
  const S = SIDE(x + w, y + d, h)
  const ribs: number[] = []
  for (let u = 1.5; u < w; u += 1.5) ribs.push(u, 0, u, h)
  p.draw(F, ribs, 'soft')
  const sr: number[] = []
  for (let u = 1.5; u < d; u += 1.5) sr.push(u, 0, u, h)
  p.draw(S, sr, 'soft')
  for (let u = 3; u + 4 < w; u += 7) p.quad(F, u, h - 4.6, 4, 4.6, pick([0x8c94a0, 0x9aa1aa], hash(`${id}:${u}`)), { outline: 'ink' })
  p.quad(F, 0, 0.6, w, 0.6, pick(ACCENT, s, 1)) // a livery stripe
  for (let u = 4; u < w - 4; u += 10) p.box(x + u, y + d * 0.2, h, 6, d * 0.6, 0.35, 0xcfd8dc) // skylights
  // a box truck at the dock
  p.box(x + 6, y + d + 1.4, 0, 7, 2.4, 3.2, 0xf2f0ea).box(x + 13, y + d + 1.4, 0, 2.4, 2.4, 2.4, pick(ACCENT, s))
  return h
}

const works: Build = (p, l, s, id) => {
  const x = l.x + 0.5
  const y = l.y + 0.5
  const w = l.w - 1
  const d = l.d - 1
  const rust = pick([0xc07a58, 0xb46a4e, 0xa8745c], s)
  p.box(x, y, 0, w * 0.6, d, 7, rust, { top: 0xd8b59d })
  p.gable(x, y, 7, w * 0.6, d, 2, 'x', 0x9b6a55)
  windows(p, x, y, 0, w * 0.6, d, 7, id, { floor: 7, ww: 2.4, wh: 2.6, step: 4 })
  p.cylZ(x + w * 0.72, y + d * 0.35, 0, 1.2, 18, 0xd8d1c6, 12)
  p.box(x + w * 0.72 - 1.3, y + d * 0.35 - 1.3, 17.6, 2.6, 2.6, 0.6, 0xc4503f, { lines: false })
  p.cylZ(x + w * 0.85, y + d * 0.7, 0, 3, 5, 0xe7e3da, 16)
  p.cylZ(x + w * 0.66, y + d * 0.75, 0, 2.4, 4.2, 0xe7e3da, 16)
  return 18
}

const park: Build = (p, l, s, id) => {
  p.rectGround(l.x, l.y, l.w, l.d, 0.02, GROUND.grass)
  // paths: a loop and a crossing
  const r = rng(`park:${id}`)
  const cx = l.x + l.w / 2
  const cy = l.y + l.d / 2
  const loop: [number, number][] = []
  for (let i = 0; i < 28; i++) {
    const a = (i / 28) * Math.PI * 2
    loop.push([cx + Math.cos(a) * l.w * 0.36, cy + Math.sin(a) * l.d * 0.34])
  }
  for (let i = 0; i < loop.length; i++) {
    const [a, b] = [loop[i], loop[(i + 1) % loop.length]]
    p.seg('soft', W(a[0], a[1], 0.06), W(b[0], b[1], 0.06))
  }
  p.rectGround(cx - 0.8, l.y, 1.6, l.d, 0.04, GROUND.path)
  // a pond
  const pond: [number, number][] = []
  const px = cx + l.w * 0.16
  const py = cy - l.d * 0.08
  for (let i = 0; i < 18; i++) {
    const a = (i / 18) * Math.PI * 2
    pond.push([px + Math.cos(a) * l.w * 0.12, py + Math.sin(a) * l.d * 0.13])
  }
  p.ground(pond, 0.05, GROUND.water, 'detail')
  // a bandstand or a play frame, then trees round the edge
  if (s % 2) {
    p.cylZ(cx - l.w * 0.2, cy, 0, 2.6, 0.5, 0xe8e0cf, 8)
    for (let i = 0; i < 6; i++) {
      const a = (i / 6) * Math.PI * 2
      p.seg('ink', W(cx - l.w * 0.2 + Math.cos(a) * 2.3, cy + Math.sin(a) * 2.3, 0.5), W(cx - l.w * 0.2 + Math.cos(a) * 2.3, cy + Math.sin(a) * 2.3, 3.2))
    }
    p.pyramid(cx - l.w * 0.2 - 2.8, cy - 2.8, 3.2, 5.6, 5.6, 1.8, pick(ACCENT, s, 2))
  } else {
    p.box(cx - l.w * 0.24, cy - 1, 0, 4, 2, 2.2, pick(ACCENT, s), { lines: true })
  }
  for (let i = 0; i < 9; i++) {
    const a = r() * Math.PI * 2
    const k = 0.42 + r() * 0.06
    tree(p, cx + Math.cos(a) * l.w * k, cy + Math.sin(a) * l.d * k, hash(`${id}:${i}`), 0.9 + r() * 0.4)
  }
  return 6
}

const station: Build = (p, l, s, id) => {
  const x = l.x
  const y = l.y
  const w = l.w
  p.box(x, y + l.d / 2 + 2, 0, w, l.d / 2 - 2, 1, 0xd9d3c6) // the platform, south of the line
  p.rectGround(x, y + l.d / 2 + 2.05, w, 0.5, 1.02, 0xe3b740)
  for (let u = 3; u < w; u += 6) p.seg('ink', W(x + u, y + l.d - 1.4, 1), W(x + u, y + l.d - 1.4, 4.6))
  p.box(x + 1, y + l.d / 2 + 1.6, 4.6, w - 2, l.d / 2 - 1.4, 0.35, pick([0x5f7f8f, 0x4f6f8a], s))
  p.box(x + w / 2 - 5, y + l.d - 2.8, 1, 10, 2.6, 3.2, 0xf0e6d4) // the ticket hall
  p.quad(FRONT(x + w / 2 - 5, y + l.d - 0.2, 4.2), 1, 0.6, 8, 1.6, GLASS_LIT, { glow: true, outline: 'ink' })
  void id
  return 5
}

const generic: Build = (p, l, s, id) => {
  const floors = 1 + (s % 3)
  const h = floors * 3.4
  const x = l.x + 0.8
  const y = l.y + 0.8
  const w = l.w - 1.6
  const d = l.d - 1.6
  p.box(x, y, 0, w, d, h, pick(WALL, s, 2))
  p.box(x - 0.1, y - 0.1, h, w + 0.2, d + 0.2, 0.5, pick(ACCENT, s))
  windows(p, x, y, 0, w, d, h, id, { floor: 3.4 })
  door(p, x, y, 0, w, d, 3.4, 0.5)
  return h + 0.5
}

const KIT: Record<string, Build> = {
  house, flats, shop, arcade, mall, school, campus, hospital, library, hall, church, temple, theater, gym, warehouse, works, park, station, generic,
}

/** Draw one place into a part; returns the height of its top, where its name tag hangs. */
export function buildLot(p: Part, l: Lot): number {
  const s = hash(l.place.id)
  return (KIT[l.arche] ?? generic)(p, l, s, l.place.id)
}

// ---------------------------------------------------------------- land: blocks, streets, rail, hills, shore

export function buildLand(plan: Plan): { land: Part; hills: Part; extent: { x0: number; y0: number; x1: number; y1: number }; light: [number, number, number] } {
  const land = new Part()
  const { grid, road } = plan
  const S = plan.sidewalk
  const north = plan.rail.y - 26
  const south = grid.y1 + 40
  const x0 = grid.x0 - 60
  const x1 = grid.x1 + 60

  // the streets under everything, then each block on them
  land.rectGround(x0, plan.rail.y - 6, x1 - x0, grid.y1 - plan.rail.y + 6, 0, GROUND.road)
  land.rectGround(x0, plan.promenade.y - S / 2, x1 - x0, S, 0.02, GROUND.sidewalk, 'soft')
  land.rectGround(x0, plan.rail.y - 6, x1 - x0, plan.promenade.y - S / 2 - plan.rail.y + 6, 0.01, GROUND.land)
  for (const b of plan.blocks) {
    land.rectGround(b.x, b.y, b.w, b.d, 0.01, GROUND.sidewalk)
    land.rectGround(b.x + S, b.y + S, b.w - 2 * S, b.d - 2 * S, 0.02, districtGround(b.district), 'soft')
    land.ground([[b.x, b.y], [b.x + b.w, b.y], [b.x + b.w, b.y + b.d], [b.x, b.y + b.d]], 0.015, GROUND.sidewalk, 'detail')
  }
  // lane dashes down the middle of every street, zebras at the corners
  const dash = (ax: number, ay: number, bx: number, by: number) => {
    const len = Math.hypot(bx - ax, by - ay)
    for (let t = 2; t < len - 2; t += 6) {
      const f0 = t / len
      const f1 = Math.min(1, (t + 2.6) / len)
      land.seg('detail', W(ax + (bx - ax) * f0, ay + (by - ay) * f0, 0.03), W(ax + (bx - ax) * f1, ay + (by - ay) * f1, 0.03))
    }
  }
  const colX = [...new Set(plan.blocks.map((b) => b.x))].sort((a, b) => a - b)
  const rowY = [...new Set(plan.blocks.map((b) => b.y))].sort((a, b) => a - b)
  const xs = [grid.x0 + road / 2, ...colX.slice(1).map((x) => x - road / 2), grid.x1 - road / 2]
  const rowEnd = rowY.map((y) => y + Math.max(...plan.blocks.filter((b) => b.y === y).map((b) => b.d)))
  const ys = [grid.y0 + road / 2, ...rowY.slice(1).map((y) => y - road / 2), grid.y1 - road / 2]
  for (const x of xs) dash(x, grid.y0, x, grid.y1)
  for (const y of ys) dash(x0, y, x1, y)
  for (const x of xs)
    for (const y of ys)
      for (const [dx, dy, horiz] of [[-road / 2 - 2.5, 0, false], [road / 2 + 0.5, 0, false], [0, -road / 2 - 2.5, true], [0, road / 2 + 0.5, true]] as [number, number, boolean][]) {
        for (let k = -road / 2 + 1.5; k < road / 2 - 1; k += 1.6) {
          if (horiz) land.rectGround(x + k, y + dy, 0.8, 2, 0.035, GROUND.lane)
          else land.rectGround(x + dx, y + k, 2, 0.8, 0.035, GROUND.lane)
        }
      }
  void rowEnd

  // the rail: ballast, sleepers, two rails
  land.rectGround(x0, plan.rail.y - 2.4, x1 - x0, 4.8, 0.03, GROUND.rail)
  for (let x = x0; x < x1; x += 1.6) land.seg('detail', W(x, plan.rail.y - 1.6, 0.05), W(x, plan.rail.y + 1.6, 0.05))
  land.seg('ink', W(x0, plan.rail.y - 0.75, 0.1), W(x1, plan.rail.y - 0.75, 0.1))
  land.seg('ink', W(x0, plan.rail.y + 0.75, 0.1), W(x1, plan.rail.y + 0.75, 0.1))

  // the shore: a promenade, the beach, the marsh and the bay
  const sy = grid.y1
  land.rectGround(x0, sy, x1 - x0, 6, 0.02, GROUND.path, 'soft')
  land.rectGround(x0, sy + 6, x1 - x0, 12, 0.01, GROUND.sand)
  land.rectGround(x0, sy + 18, x1 - x0, south - sy - 18 + 220, 0, GROUND.water)
  const r = rng('shore')
  for (let i = 0; i < 70; i++) {
    const x = x0 + r() * (x1 - x0)
    const y = sy + 19 + r() * 7
    for (let k = 0; k < 4; k++) land.seg('detail', W(x + k * 0.5, y, 0.02), W(x + k * 0.5 + 0.3, y, 0.9 + r() * 0.6)) // marsh reeds
  }
  for (let i = 0; i < 26; i++) {
    const x = x0 + r() * (x1 - x0)
    const y = sy + 30 + r() * 90
    land.seg('soft', W(x, y, 0.02), W(x + 6 + r() * 8, y, 0.02)) // the bay's ripples
  }
  // a pier, and a lighthouse at its end
  const pierX = grid.x0 + (grid.x1 - grid.x0) * 0.72
  land.box(pierX, sy + 14, 0, 4, 40, 0.8, 0xc9b597)
  for (let y = sy + 16; y < sy + 54; y += 4) land.seg('ink', W(pierX, y, 0), W(pierX, y, -1.4)).seg('ink', W(pierX + 4, y, 0), W(pierX + 4, y, -1.4))
  const lx = pierX + 2
  const ly = sy + 56
  land.cylZ(lx, ly, 0, 3.2, 1, 0xd9d2c4, 14)
  for (let k = 0; k < 4; k++) land.cylZ(lx, ly, 1 + k * 3, 1.7 - k * 0.12, 3, k % 2 ? 0xc4503f : WHITE, 14)
  land.cylZ(lx, ly, 13, 1.05, 1.8, GLASS_LIT, 10, { lines: true })
  land.pyramid(lx - 1.3, ly - 1.3, 14.8, 2.6, 2.6, 1.6, 0x3e4651)
  // street lamps at every block's corners and along the promenade
  for (const b of plan.blocks)
    for (const [x, y] of [[b.x - 1, b.y - 1], [b.x + b.w + 1, b.y - 1], [b.x - 1, b.y + b.d + 1], [b.x + b.w + 1, b.y + b.d + 1]]) lamp(land, x, y)
  for (let x = x0 + 20; x < x1 - 10; x += 36) lamp(land, x, sy + 5)

  // trees and benches in the room left at the ends of rows
  for (const sp of plan.spares) {
    const rr = rng(`spare:${sp.block}:${sp.side}`)
    const n = Math.max(1, Math.floor(sp.w / 6))
    for (let i = 0; i < n; i++) {
      const x = sp.x + 3 + i * ((sp.w - 6) / Math.max(1, n - 1 || 1))
      const y = sp.y + sp.d * (0.3 + rr() * 0.4)
      if (rr() < 0.75) tree(land, x, y, hash(`${sp.block}:${i}:${sp.side}`), 0.8 + rr() * 0.4)
      else land.box(x - 1, y, 0, 2, 0.6, 0.5, 0xb59b7e)
    }
  }
  // the woods at either end of town
  for (const [a, b] of [[x0 + 4, grid.x0 - 6], [grid.x1 + 6, x1 - 4]]) {
    const rr = rng(`woods:${a}`)
    for (let i = 0; i < 40; i++) {
      const x = a + rr() * (b - a)
      const y = plan.rail.y + 6 + rr() * (grid.y1 - plan.rail.y - 10)
      if (rr() < 0.5) pine(land, x, y, hash(`wp:${a}:${i}`), 0.8 + rr() * 0.5)
      else tree(land, x, y, hash(`wt:${a}:${i}`), 0.9 + rr() * 0.4)
    }
  }

  // the foothills behind the rail: rounded hills on a hairline grid, rising from the rail to the back
  const hills = new Part()
  const cell = 14
  const hy0 = north - 130
  const hy1 = plan.rail.y - 5
  const nx = Math.ceil((x1 - x0) / cell)
  const ny = Math.ceil((hy1 - hy0) / cell)
  const hr = rng('hills')
  const bumps: { x: number; y: number; r: number; h: number }[] = []
  for (let x = x0 - 40; x < x1 + 60; x += 70 + hr() * 50)
    bumps.push({ x, y: hy0 + 25 + hr() * (hy1 - hy0 - 60), r: 45 + hr() * 50, h: 12 + hr() * 24 })
  const zAt = (x: number, y: number) => {
    let z = 0
    for (const b of bumps) z = Math.max(z, b.h * Math.exp(-((x - b.x) ** 2 + (y - b.y) ** 2) / (b.r * b.r)))
    const near = Math.max(0, Math.min(1, (hy1 - y) / 30)) // the land flattens toward the rail
    return z * near
  }
  const pt = (i: number, j: number) => {
    const x = x0 + i * cell
    const y = hy0 + j * cell
    return W(x, y, zAt(x, y))
  }
  for (let i = 0; i < nx; i++)
    for (let j = 0; j < ny; j++) {
      const [a, b, c, d] = [pt(i, j), pt(i + 1, j), pt(i + 1, j + 1), pt(i, j + 1)]
      const zt = (a.y + c.y) / 2
      const hex = zt > 24 ? GROUND.hillHigh : GROUND.hill
      hills.poly([a, b, c], hex).poly([a, c, d], hex)
      hills.seg('soft', a, b).seg('soft', a, d)
    }
  const pr = rng('hill-pines')
  for (let i = 0; i < 130; i++) {
    const x = x0 + pr() * (x1 - x0)
    const y = hy0 + 10 + pr() * (hy1 - hy0 - 20)
    const g = new THREE.ConeGeometry(1.5, 4.5, 5)
    g.translate(x, zAt(x, y) + 2.4, y)
    geo(hills, g, 0x7f9f7c)
  }
  return { land, hills, extent: { x0, y0: hy0, x1, y1: south }, light: [grid.x0 + (grid.x1 - grid.x0) * 0.72 + 2, grid.y1 + 56, 13.9] }
}

/** Bunting and a flag for a town event at a place. */
export function buildEventDressing(p: Part, l: Lot, top: number) {
  const x0 = l.x
  const x1 = l.x + l.w
  const y = l.y + l.d + 0.5
  const z = Math.min(top, 6) + 0.5
  p.seg('ink', W(x0, y, 0), W(x0, y, z)).seg('ink', W(x1, y, 0), W(x1, y, z))
  const n = Math.max(4, Math.round(l.w / 1.6))
  for (let i = 0; i < n; i++) {
    const a = x0 + (l.w * i) / n
    const b = x0 + (l.w * (i + 1)) / n
    const sag = (t: number) => z - Math.sin(Math.PI * ((t - x0) / l.w)) * 1.2
    p.poly([W(a, y, sag(a)), W(b, y, sag(b)), W((a + b) / 2, y, sag((a + b) / 2) - 0.9)], pick(ACCENT, i * 7 + 3), true)
    p.seg('detail', W(a, y, sag(a)), W(b, y, sag(b)))
  }
  const fx = x1 - 0.5
  p.seg('ink', W(fx, y - 1, top), W(fx, y - 1, top + 5))
  p.poly([W(fx, y - 1, top + 5), W(fx + 2.6, y - 1, top + 4.3), W(fx, y - 1, top + 3.6)], EVENT, true)
}
