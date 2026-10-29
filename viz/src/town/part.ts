// Parts: boxes, prisms and flat detail collected into a few draw calls, after Factory Yard's Part (MIT, see
// LICENSE.factory-yard). Where Factory Yard fills every face from a handful of grey tokens, a face here carries its
// own colour, toon-shaded by which way it faces (top light, left mid, right dark) and baked into vertex colours, so a
// whole building of many colours is still one mesh. Edges are 1-px hairlines at any zoom.
import * as THREE from 'three'
import { LineMaterial } from 'three/addons/lines/LineMaterial.js'
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js'
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js'
import { W, type P3 } from './iso'

export type LineKey = 'ink' | 'detail' | 'soft'

// Toon tones by facing, and the cool the shade leans to.
const TONE = { top: 1, left: 0.86, right: 0.7 }
const SHADE = new THREE.Color(0x4c5a7a)

export interface Materials {
  fill: THREE.MeshBasicMaterial
  glow: THREE.MeshBasicMaterial // windows: dark glass by day, lit after dusk
  line: Record<LineKey, LineMaterial>
}

export function makeMaterials(): Materials {
  const fill = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 })
  const glow = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 0.5, polygonOffsetUnits: 0.5 })
  const line = {
    ink: new LineMaterial({ linewidth: 1, worldUnits: false }),
    detail: new LineMaterial({ linewidth: 1, worldUnits: false }),
    soft: new LineMaterial({ linewidth: 1, worldUnits: false }),
  }
  return { fill, glow, line }
}

const tmpC = new THREE.Color()
const v = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z)

/** The toon tone of a face with normal n (three.js, y-up), as a linear colour. */
function toned(hex: number, n: THREE.Vector3, flat: boolean): THREE.Color {
  const c = new THREE.Color(hex)
  if (flat) return c
  const f = n.y > 0.55 ? TONE.top : n.z >= n.x ? TONE.left : TONE.right
  c.multiplyScalar(f)
  return c.lerp(tmpC.copy(SHADE).multiplyScalar(f), (1 - f) * 0.35)
}

export interface ExtrudeOpts {
  lines?: boolean
  seams?: boolean
  top?: number // the up-facing faces' colour, if not the body's
  flat?: boolean // no toon tones (ground, water)
  ink?: LineKey
}

export class Part {
  private pos: number[] = []
  private col: number[] = []
  private gpos: number[] = []
  private gcol: number[] = []
  private lines: Record<LineKey, number[]> = { ink: [], detail: [], soft: [] }

  private push(dst: 'fill' | 'glow', a: THREE.Vector3, b: THREE.Vector3, c: THREE.Vector3, col: THREE.Color) {
    const P = dst === 'fill' ? this.pos : this.gpos
    const C = dst === 'fill' ? this.col : this.gcol
    P.push(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z)
    for (let i = 0; i < 3; i++) C.push(col.r, col.g, col.b)
  }

  /** A flat polygon (three.js coordinates), fan-triangulated, toon-toned by its normal. */
  poly(p: THREE.Vector3[], hex: number, flat = false, dst: 'fill' | 'glow' = 'fill'): this {
    if (p.length < 3) return this
    const n = v(0, 0, 0).crossVectors(p[1].clone().sub(p[0]), p[2].clone().sub(p[0])).normalize()
    if (n.y < 0) n.negate()
    // a face turned away from the camera's side is still toned by its outward side
    const col = dst === 'glow' ? new THREE.Color(hex) : toned(hex, n.y > 0.55 ? n : v(Math.abs(n.x), n.y, Math.abs(n.z)), flat)
    for (let i = 1; i < p.length - 1; i++) this.push(dst, p[0], p[i], p[i + 1], col)
    return this
  }

  seg(k: LineKey, a: THREE.Vector3, b: THREE.Vector3): this {
    this.lines[k].push(a.x, a.y, a.z, b.x, b.y, b.z)
    return this
  }

  /** A polygon in scene coordinates swept along e: a solid with its faces turned outward and toned. */
  extrude(pts: P3[], e: P3, hex: number, o: ExtrudeOpts = {}): this {
    const { lines = true, seams = true, top, flat = false, ink = 'ink' } = o
    const A = pts.map((p) => W(...p))
    const E = W(...e)
    const B = A.map((p) => p.clone().add(E))
    const mid = [...A, ...B].reduce((s, p) => s.add(p), v(0, 0, 0)).multiplyScalar(1 / (A.length * 2))
    const face = (q: THREE.Vector3[]) => {
      const n = v(0, 0, 0).crossVectors(q[1].clone().sub(q[0]), q[2].clone().sub(q[0])).normalize()
      const c = q.reduce((s, p) => s.add(p), v(0, 0, 0)).multiplyScalar(1 / q.length)
      if (n.dot(c.sub(mid)) < 0) n.negate()
      if (n.y < -0.82) return // the camera never sees a face turned this far down
      const col = toned(n.y > 0.55 && top !== undefined ? top : hex, n, flat)
      for (let i = 1; i < q.length - 1; i++) this.push('fill', q[0], q[i], q[i + 1], col)
    }
    face(A)
    face(B)
    for (let i = 0; i < A.length; i++) {
      const j = (i + 1) % A.length
      face([A[i], A[j], B[j], B[i]])
    }
    if (lines)
      for (let i = 0; i < A.length; i++) {
        const j = (i + 1) % A.length
        this.seg(ink, A[i], A[j]).seg(ink, B[i], B[j])
        if (seams) this.seg(ink, A[i], B[i])
      }
    return this
  }

  /** A box at (x, y, z), w along x, d along y, h up. */
  box(x: number, y: number, z: number, w: number, d: number, h: number, hex: number, o?: ExtrudeOpts): this {
    return this.extrude([[x, y, z], [x + w, y, z], [x + w, y + d, z], [x, y + d, z]], [0, 0, h], hex, o)
  }

  /** A gable roof over (x, y, w, d) from height z, rising h, its ridge along x or y. */
  gable(x: number, y: number, z: number, w: number, d: number, h: number, ridge: 'x' | 'y', hex: number, over = 0.4): this {
    const [x0, x1, y0, y1] = [x - over, x + w + over, y - over, y + d + over]
    if (ridge === 'x') {
      const ym = (y0 + y1) / 2
      return this.extrude([[x0, y0, z], [x0, y1, z], [x0, ym, z + h]], [x1 - x0, 0, 0], hex)
    }
    const xm = (x0 + x1) / 2
    return this.extrude([[x0, y0, z], [x1, y0, z], [xm, y0, z + h]], [0, y1 - y0, 0], hex)
  }

  /** A hipped (pyramid) roof. */
  pyramid(x: number, y: number, z: number, w: number, d: number, h: number, hex: number): this {
    const c = W(x + w / 2, y + d / 2, z + h)
    const q = [W(x, y, z), W(x + w, y, z), W(x + w, y + d, z), W(x, y + d, z)]
    for (let i = 0; i < 4; i++) {
      const a = q[i]
      const b = q[(i + 1) % 4]
      this.poly([a, b, c], hex)
      this.seg('ink', a, b).seg('ink', a, c)
    }
    return this
  }

  cylZ(cx: number, cy: number, z: number, r: number, h: number, hex: number, n = 12, o?: ExtrudeOpts): this {
    const p: P3[] = []
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2
      p.push([cx + r * Math.cos(a), cy + r * Math.sin(a), z])
    }
    return this.extrude(p, [0, 0, h], hex, { seams: false, ...o })
  }

  /** A flat rectangle drawn in a face's local (u, v) units, lifted off the face so it wins the depth test. */
  quad(M: THREE.Matrix4, u: number, w0: number, w: number, h: number, hex: number, o: { lift?: number; glow?: boolean; outline?: LineKey | null } = {}): this {
    const { lift = 0.04, glow = false, outline = null } = o
    const off = v(0, 0, 0).setFromMatrixColumn(M, 2).normalize().multiplyScalar(-lift)
    const at = (a: number, b: number) => v(a, b, 0).applyMatrix4(M).add(off)
    const q = [at(u, w0), at(u + w, w0), at(u + w, w0 + h), at(u, w0 + h)]
    this.poly(q, hex, true, glow ? 'glow' : 'fill')
    if (outline) for (let i = 0; i < 4; i++) this.seg(outline, q[i], q[(i + 1) % 4])
    return this
  }

  /** Hairlines in a face's local units: [u0, v0, u1, v1, …]. */
  draw(M: THREE.Matrix4, segs: number[], k: LineKey = 'detail', lift = 0.05): this {
    const off = v(0, 0, 0).setFromMatrixColumn(M, 2).normalize().multiplyScalar(-lift)
    const at = (a: number, b: number) => v(a, b, 0).applyMatrix4(M).add(off)
    for (let i = 0; i < segs.length; i += 4) this.seg(k, at(segs[i], segs[i + 1]), at(segs[i + 2], segs[i + 3]))
    return this
  }

  /** A flat ground shape in scene coordinates at height z. */
  ground(pts: [number, number][], z: number, hex: number, outline: LineKey | null = null): this {
    const q = pts.map(([x, y]) => W(x, y, z))
    this.poly(q, hex, true)
    if (outline) for (let i = 0; i < q.length; i++) this.seg(outline, q[i], q[(i + 1) % q.length])
    return this
  }

  rectGround(x: number, y: number, w: number, d: number, z: number, hex: number, outline: LineKey | null = null): this {
    return this.ground([[x, y], [x + w, y], [x + w, y + d], [x, y + d]], z, hex, outline)
  }

  get empty(): boolean {
    return this.pos.length === 0 && this.gpos.length === 0 && Object.values(this.lines).every((l) => l.length === 0)
  }

  build(m: Materials, name?: string): THREE.Group {
    const g = new THREE.Group()
    if (name) g.name = name
    const mesh = (pos: number[], col: number[], mat: THREE.Material) => {
      const geo = new THREE.BufferGeometry()
      geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
      geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3))
      geo.computeBoundingSphere()
      return new THREE.Mesh(geo, mat)
    }
    if (this.pos.length) g.add(mesh(this.pos, this.col, m.fill))
    if (this.gpos.length) {
      const w = mesh(this.gpos, this.gcol, m.glow)
      w.raycast = () => {}
      g.add(w)
    }
    for (const k of Object.keys(this.lines) as LineKey[]) {
      if (!this.lines[k].length) continue
      const l = new LineSegments2(new LineSegmentsGeometry().setPositions(this.lines[k]), m.line[k])
      l.raycast = () => {}
      l.userData.line = k
      g.add(l)
    }
    return g
  }
}
