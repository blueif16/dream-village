// Figures and vehicles. A figure is a body part plus two legs that swing from the hip; residents are drawn larger,
// in their own colour, with a pin that floats over the roofs so they can always be found.
import * as THREE from 'three'
import { LineMaterial } from 'three/addons/lines/LineMaterial.js'
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js'
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js'
import { W } from './iso'
import { Part, type Materials } from './part'
import { ACCENT, pick } from './palette'
import { hash } from './rand'

const SKIN = [0xf0c8a0, 0xd9a47a, 0xb7825a, 0x8d5a3b, 0xe8b98f]
const HAIR = [0x2b2118, 0x3b2a1e, 0x1d1a17, 0x5a3b24, 0x6b4a2f, 0x8a6a4a]
const TROUSERS = [0x3c4454, 0x4a4036, 0x2f3a3f, 0x5a5048]
const FOLK = [0xb9b2a6, 0xa8a49b, 0xc3b8a6, 0x9fa6a8, 0xb5aa9c, 0xa9b0a0] // townsfolk wear quiet colours

export interface Figure {
  group: THREE.Group
  legs: [THREE.Group, THREE.Group]
}

/** A person standing at the origin, facing +x. */
export function buildFigure(m: Materials, id: string, coat: number | null, scale: number): Figure {
  const s = hash(id)
  const body = new Part()
  const col = coat ?? pick(FOLK, s)
  body.box(-0.16, -0.27, 0.86, 0.32, 0.54, 0.72, col) // torso
  body.box(-0.1, -0.36, 0.95, 0.2, 0.1, 0.56, col).box(-0.1, 0.26, 0.95, 0.2, 0.1, 0.56, col) // arms
  body.box(-0.15, -0.15, 1.6, 0.3, 0.3, 0.32, pick(SKIN, s, 1)) // head
  body.box(-0.17, -0.17, 1.84, 0.34, 0.34, 0.12, pick(HAIR, s, 2), { lines: false })
  const g = body.build(m)
  const leg = (dy: number) => {
    const p = new Part()
    p.box(-0.08, dy - 0.08, -0.86, 0.16, 0.16, 0.86, pick(TROUSERS, s, 3), { seams: false })
    const lg = p.build(m)
    lg.position.copy(W(0, 0, 0.86))
    return lg
  }
  const legs: [THREE.Group, THREE.Group] = [leg(-0.11), leg(0.11)]
  const group = new THREE.Group()
  group.add(g, ...legs)
  group.scale.setScalar(scale)
  return { group, legs }
}

/** The floating pin over a resident: a small diamond in their colour, drawn over everything. */
export function buildPin(color: number): { group: THREE.Group; fill: THREE.MeshBasicMaterial; line: LineMaterial } {
  const fill = new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true })
  const geo = new THREE.OctahedronGeometry(0.9, 0)
  geo.scale(1, 1.5, 1)
  const mesh = new THREE.Mesh(geo, fill)
  mesh.renderOrder = 20
  const line = new LineMaterial({ color: 0x2b2a30, linewidth: 1, worldUnits: false, depthTest: false, transparent: true })
  const edges = new THREE.EdgesGeometry(geo).attributes.position
  const pos: number[] = []
  for (let i = 0; i < edges.count; i++) pos.push(edges.getX(i), edges.getY(i), edges.getZ(i))
  const lines = new LineSegments2(new LineSegmentsGeometry().setPositions(pos), line)
  lines.renderOrder = 21
  lines.raycast = () => {}
  const group = new THREE.Group()
  group.add(mesh, lines)
  return { group, fill, line }
}

/** A flat ring on the ground, for the selection and for a talk's pulse. */
export function buildRing(color: number, r0: number, r1: number, opacity = 1): THREE.Mesh {
  const geo = new THREE.RingGeometry(r0, r1, 40)
  geo.rotateX(-Math.PI / 2)
  const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthTest: false, side: THREE.DoubleSide })
  const m = new THREE.Mesh(geo, mat)
  m.renderOrder = 15
  m.raycast = () => {}
  return m
}

// ---------------------------------------------------------------- vehicles, drawn facing +x from their centre

const CAR = [0xd9d4ca, 0x8f9aa6, 0x5d6b7a, 0xb8644a, 0xe7dfcf, 0x3e4651, 0x9aa88f, 0xc9b28a]

export function buildCar(m: Materials, id: string): THREE.Group {
  const s = hash(id)
  const p = new Part()
  const c = pick(CAR, s)
  p.box(-2.1, -0.9, 0.35, 4.2, 1.8, 0.75, c)
  p.box(-1.1, -0.82, 1.1, 2.1, 1.64, 0.6, 0xc9d4da, { top: c })
  for (const [x, y] of [[-1.3, -0.95], [1.3, -0.95], [-1.3, 0.75], [1.3, 0.75]]) p.box(x - 0.35, y, 0, 0.7, 0.2, 0.55, 0x2c2f33, { lines: false })
  return p.build(m)
}

export function buildBus(m: Materials): THREE.Group {
  const p = new Part()
  p.box(-5.5, -1.25, 0.4, 11, 2.5, 2.7, 0xf2efe6, { top: 0xe6e2d8 })
  p.box(-5.5, -1.26, 0.9, 11, 2.52, 0.35, ACCENT[2], { lines: false })
  for (let x = -4.6; x < 4.6; x += 1.5) p.box(x, -1.27, 1.6, 1.1, 2.54, 1, 0x9fb4bf, { lines: false })
  for (const x of [-3.6, 3.6]) for (const y of [-1.3, 1.1]) p.box(x - 0.45, y, 0, 0.9, 0.2, 0.7, 0x2c2f33, { lines: false })
  return p.build(m)
}

/** The Tule Line: three cars, coupled along +x. */
export function buildTrain(m: Materials): { group: THREE.Group; length: number } {
  const p = new Part()
  const car = 14
  for (let i = 0; i < 3; i++) {
    const x = -((3 * car) / 2) + i * (car + 0.6)
    p.box(x, -1.4, 0.8, car, 2.8, 3.1, 0xf4f1ea, { top: 0xd8d4ca })
    p.box(x, -1.41, 1.2, car, 2.82, 0.4, 0x2f7f9f, { lines: false })
    for (let u = 1; u < car - 1.2; u += 1.8) p.box(x + u, -1.42, 2.1, 1.3, 2.84, 1.1, 0x8fa4b0, { lines: false })
  }
  return { group: p.build(m), length: 3 * car + 1.2 }
}

/** A small sailing boat, bow along +x. */
export function buildBoat(m: Materials, id: string): THREE.Group {
  const s = hash(id)
  const p = new Part()
  p.extrude([[-2.6, -0.9, 0], [2.2, -0.9, 0], [3.2, 0, 0], [2.2, 0.9, 0], [-2.6, 0.9, 0]], [0, 0, 0.8], pick([0xf4f1ea, 0x2f5f7a, 0xb8644a], s))
  p.seg('ink', W(0, 0, 0.8), W(0, 0, 6.4))
  p.poly([W(0.1, 0, 6.2), W(0.1, 0, 1.3), W(2.6, 0, 1.3)], 0xf7f4ec, true)
  p.poly([W(-0.1, 0, 5.4), W(-0.1, 0, 1.3), W(-2.2, 0, 1.3)], pick(ACCENT, s, 1), true)
  return p.build(m)
}

/** Place a group at scene (x, y, z) facing heading h (radians, in the scene's x-y plane). */
export function pose(o: THREE.Object3D, x: number, y: number, h = 0, z = 0) {
  o.position.set(x, z, y)
  o.rotation.y = -h
}
